"""Bounded core-only engineering benchmark; no cloud calls or held-out scoring."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/local-training/v1'
OUT = ROOT / 'data/models/mac-training-preflight-v1'
SEED = 20261001
STEPS = 200


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def network(features):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(features, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))


def validate_panel(x, y, samples, genes):
    if (x.ndim != 2 or x.shape != (len(samples), len(genes)) or y.shape != (len(samples), 1)
            or len(set(samples)) != len(samples) or len(set(genes)) != len(genes)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Invalid training-only panel')


def prepare():
    from core_baseline import inner_splits
    from run_core_baselines import load_data, fit_preparation, preparation_signature, INPUT, LABELS
    started = time.perf_counter()
    data = load_data()
    outer = np.flatnonzero(data['study_ids'] != 'GSE8917')
    tr, va = next(inner_splits(data['study_ids'][outer], data['split_group_ids'][outer]))
    train, validation = outer[tr], outer[va]
    if set(data['split_group_ids'][train]) & set(data['split_group_ids'][validation]):
        raise ValueError('Donor overlap')
    q, genes, matrix, preparations = fit_preparation(data, train)
    signature = preparation_signature(q, preparations)
    altered = dict(data)
    nontraining = np.ones(len(data['sample_ids']), bool); nontraining[train] = False
    altered['values'] = data['values'].copy(); altered['values'][:, nontraining] = np.nan
    altered['detected'] = data['detected'].copy(); altered['detected'][:, nontraining] = False
    altered['y'] = data['y'].copy(); altered['y'][nontraining] = 123456.
    q2, _, _, p2 = fit_preparation(altered, train)
    if signature != preparation_signature(q2, p2):
        raise ValueError('Nontraining data affected fitted state')
    prep = preparations[500]
    x = prep.transform(matrix, genes, True).astype(np.float32)
    y_raw = data['y'][train]
    y_mean, y_scale = float(y_raw.mean()), float(y_raw.std())
    if y_scale <= 0:
        raise ValueError('Constant training target')
    y = ((y_raw-y_mean)/y_scale).astype(np.float32)[:, None]
    selected_genes = np.asarray(prep.gene_ids)[prep.selected]
    samples = data['sample_ids'][train]
    validate_panel(x, y, samples, selected_genes)
    q.save(OUT/'reference.npz'); prep.save(OUT/'preparation.npz')
    np.savez_compressed(OUT/'panel.npz', x=x, y=y, sample_ids=samples, gene_ids=selected_genes)
    result = {'training_samples':len(train), 'features':x.shape[1],
        'training_groups':sorted(set(data['split_group_ids'][train])), 'training_sample_ids':samples.tolist(),
        'studies':sorted(set(data['study_ids'][train])), 'excluded_outer_study':'GSE8917',
        'nontraining_perturbation_unchanged':True, 'training_signature':signature,
        'target_mean_gy':y_mean, 'target_scale_gy':y_scale,
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [INPUT,LABELS]},
        'preparation_and_isolation_check_seconds':time.perf_counter()-started,
        'panel_sha256':sha(OUT/'panel.npz'), 'no_heldout_scoring':True}
    save(LOG/'prepared.json',result)
    return result


def worker(device):
    import psutil
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no implicit CPU substitution')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('MPS fallback must be explicitly disabled')
    preparation = json.loads((LOG/'prepared.json').read_text())
    if sha(OUT/'panel.npz') != preparation['panel_sha256']:
        raise ValueError('Panel checksum mismatch')
    with np.load(OUT/'panel.npz', allow_pickle=False) as z:
        x0, y0, samples, genes = z['x'],z['y'],z['sample_ids'],z['gene_ids']
    validate_panel(x0,y0,samples,genes)
    torch.manual_seed(SEED)
    initial = cpu_tree(network(x0.shape[1]).state_dict())
    proc = psutil.Process()
    samples_rss = [proc.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.02):
            samples_rss.append(proc.memory_info().rss)
    watch = threading.Thread(target=monitor, daemon=True); watch.start()
    def sync():
        if device == 'mps': torch.mps.synchronize()
    def create():
        model = network(x0.shape[1]); model.load_state_dict(initial); model = model.to(device)
        return model, torch.optim.Adam(model.parameters(), lr=.001, foreach=False)
    def train(model, optimizer, count):
        model.train()
        for _ in range(count):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x),y)
            loss.backward(); optimizer.step()
        return loss
    def predictions(model, values):
        model.eval()
        with torch.inference_mode():
            return model(values).detach().cpu().numpy()

    first_start=time.perf_counter()
    x=torch.from_numpy(x0).to(device); y=torch.from_numpy(y0).to(device)
    model,optimizer=create()
    initial_pred=predictions(model,x)
    cpu_model=network(x0.shape[1]);cpu_model.load_state_dict(initial)
    cpu_initial=predictions(cpu_model,torch.from_numpy(x0))
    initial_difference=float(np.max(abs(initial_pred-cpu_initial)))
    if not np.allclose(initial_pred,cpu_initial,atol=1e-4,rtol=1e-4):
        raise ValueError('Initial CPU/device predictions disagree')
    train(model,optimizer,20);sync()
    first_use=time.perf_counter()-first_start
    timings=[]; losses=[]
    for repeat in range(3):
        sync(); before=time.perf_counter()
        model,optimizer=create()
        train(model,optimizer,STEPS);sync()
        timings.append(time.perf_counter()-before)
        pred=predictions(model,x)
        final_loss=float(np.mean((pred-y0)**2)); initial_loss=float(np.mean((initial_pred-y0)**2))
        if not np.isfinite(pred).all() or not np.isfinite(final_loss) or final_loss >= initial_loss:
            raise ValueError('Training smoke check failed')
        losses.append(final_loss)

    # Same-device optimizer resume, independently initialized from a saved midpoint.
    model,optimizer=create(); train(model,optimizer,STEPS//2);sync()
    checkpoint={'schema_version':1,'engineering_only':True,'deployment_allowed':False,
        'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),
        'steps_completed':STEPS//2,'seed':SEED,'panel_sha256':preparation['panel_sha256']}
    checkpoint_path=OUT/f'{device}_checkpoint.pt';torch.save(checkpoint,checkpoint_path)
    train(model,optimizer,STEPS//2);sync(); uninterrupted=predictions(model,x)
    resumed,resumed_optimizer=create()
    checkpoint_reload=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint_reload['panel_sha256'] != preparation['panel_sha256']:
        raise ValueError('Checkpoint belongs to another input')
    resumed.load_state_dict(checkpoint_reload['model']);resumed_optimizer.load_state_dict(checkpoint_reload['optimizer'])
    train(resumed,resumed_optimizer,STEPS//2);sync(); resumed_pred=predictions(resumed,x)
    resume_difference=float(np.max(abs(uninterrupted-resumed_pred)))
    if not np.allclose(uninterrupted,resumed_pred,atol=1e-4,rtol=1e-4):
        raise ValueError('Optimizer resume changed predictions')
    weights_path=OUT/f'{device}_weights.pt';torch.save(cpu_tree(resumed.state_dict()),weights_path)
    reloaded_cpu=network(x0.shape[1]);reloaded_cpu.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
    host_x=torch.from_numpy(x0)
    batch=predictions(reloaded_cpu,host_x)
    one=np.concatenate([predictions(reloaded_cpu,row[None,:]) for row in host_x])
    reverse=predictions(reloaded_cpu,torch.from_numpy(x0[::-1].copy()))[::-1]
    checks={'same_device_resume_max_abs':resume_difference,
        'cpu_reload_max_abs':float(np.max(abs(resumed_pred-batch))),
        'individual_vs_batch_max_abs':float(np.max(abs(one-batch))),
        'reverse_vs_batch_max_abs':float(np.max(abs(reverse-batch))),
        'initial_cpu_device_max_abs':initial_difference}
    for a in [resumed_pred,one,reverse]:
        if not np.allclose(a,batch,atol=1e-4,rtol=1e-4):
            raise ValueError('Portable inference replay check failed')
    gpu_memory=None
    if device=='mps':
        gpu_memory={'current_allocated_bytes_after_checks':torch.mps.current_allocated_memory(),
            'driver_allocated_bytes_after_checks':torch.mps.driver_allocated_memory(),
            'note':'End-of-check readings, not peak or additional-to-RSS memory.'}
    stop.set();watch.join()
    result={'device':device,'torch_version':str(torch.__version__),'cpu_threads':torch.get_num_threads(),
        'mps_available':torch.backends.mps.is_available(),'implicit_mps_cpu_fallback':False,
        'samples':len(x0),'features':x0.shape[1],'parameters':sum(p.numel() for p in model.parameters()),
        'steps_per_fit':STEPS,'repeats':3,'first_use_setup_and_20_steps_seconds':first_use,
        'warm_fit_seconds':timings,'median_warm_fit_seconds':float(np.median(timings)),
        'training_loss_smoke_only':losses,'heldout_metrics':None,
        'replay_checks_standardized_target_units':checks,
        'sampled_rss_max_bytes':max(samples_rss),'os_process_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'mps_memory':gpu_memory,'checkpoint_sha256':sha(checkpoint_path),'weights_sha256':sha(weights_path),
        'status':'passed','engineering_only':True,'deployment_allowed':False}
    save(LOG/f'{device}.json',result)
    print(json.dumps({'device':device,'seconds_per_200_steps':result['median_warm_fit_seconds'],'checks':'passed'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['cpu','mps']);args=parser.parse_args()
    if args.worker:
        worker(args.worker);return
    if (LOG/'summary.json').exists() or OUT.exists():
        raise FileExistsError('Preserve this preflight; use a new version for reruns')
    LOG.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True)
    if not (LOG/'METHOD.md').exists(): raise FileNotFoundError('Prespecified method missing')
    protected=list((ROOT/'data/models/core-baselines-v1').glob('*'))
    protected += [p for p in (ROOT/'output/final_proposal').iterdir() if p.is_file()]
    protected += [ROOT/'scripts'/name for name in ['core_baseline.py','run_core_baselines.py','core_transfer.py','masked_quantile.py','assay_qc.py']]
    original={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    save(LOG/'protected-before.json',original)
    start=time.perf_counter();prepared=prepare()
    env={**os.environ,'PYTORCH_ENABLE_MPS_FALLBACK':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','VECLIB_MAXIMUM_THREADS':'4'}
    import torch
    capability={'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),
        'torch':str(torch.__version__),'python':sys.version,'platform':platform.platform(),
        'physical_memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))}
    save(LOG/'capability.json',capability)
    devices=['cpu']+(['mps'] if capability['mps_available'] else [])
    results=[]
    for device in devices:
        print('Benchmarking '+device,flush=True)
        with (LOG/f'{device}.log').open('w') as f:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',device],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        if p.returncode:
            raise RuntimeError(f'{device} failed; inspect its log')
        results.append(json.loads((LOG/f'{device}.json').read_text()))
        print(f'{device}: {results[-1]["median_warm_fit_seconds"]:.3f} s per 200-step fit; replay checks pass',flush=True)
    after={p:sha(ROOT/p) for p in original}
    if after!=original: raise ValueError('Protected prior artifacts changed')
    summary={'completed_utc':datetime.now(timezone.utc).isoformat(),'scope':'Local engineering preflight only; no held-out scoring or deployment',
        'capability':capability,'prepared':prepared,'device_results':results,
        'fastest_tested_device_for_this_fixture':min(results,key=lambda r:r['median_warm_fit_seconds'])['device'],
        'total_seconds':time.perf_counter()-start,'protected_files_unchanged':len(original),
        'method_sha256':sha(LOG/'METHOD.md'),'code_sha256':sha(Path(__file__)),
        'no_cloud_writes_no_Colab_no_paid_model_API':True,'cloud_worker_implemented':False,
        'limitations':['Small A-like MLP fixture only; not A-D validation','No eligible auxiliary pipeline benchmark',
            'Only a prepared 96-profile subset; not full raw-data peak memory','CPU export checked locally; not deployed to Cloud Run']}
    save(LOG/'summary.json',summary)
    with (LOG/'environment-freeze.txt').open('w') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    save(LOG/'verification.json',{'protected_unchanged':after,'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()},
        'replay_and_fold_isolation_passed':True,'engineering_only':True})


if __name__=='__main__':
    main()
