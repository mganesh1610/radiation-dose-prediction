"""One-shot locked inference and frozen donor statistics. No training entry point."""
import csv
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy
import torch
from audit_locked_metadata import ROOT, sha
from verify_locked_release import main as verify_entry
from final_model_bundle import predict_panel
from run_abcd_experiments import completed
from locked_evaluation_stats import donor_errors,bootstrap_interval,paired_comparison,holm_three

LOG=ROOT/'context/locked-evaluation/v1'
MODEL=ROOT/'data/models/final-selection-v2'
ELIG=ROOT/'context/locked-eligibility/v1'


def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def save(name,x):(LOG/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def tsv(name,rows):
    with (LOG/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)


def run():
    start=time.perf_counter();LOG.mkdir(parents=True,exist_ok=True)
    if (LOG/'run-start.json').exists():raise ValueError('One-time evaluation already started; inspect preserved records, do not rerun')
    verify_entry()
    stats_plan=read(ROOT/'context/final-selection/v2/statistical-plan.json')
    if (stats_plan['alternative']!='delta_less_than_zero' or stats_plan['family_size']!=3 or
        stats_plan['primary_seed']!=20261002 or stats_plan['descriptive_bootstrap']['seed']!=20261003):
        raise ValueError('Unexpected frozen statistical contract')
    entry=read(ELIG/'evaluation-entry-freeze.json');index=read(MODEL/'artifact-index.json')
    bundle=read(ROOT/'context/final-training/v1/model-bundle-freeze.json')
    labels=read(ELIG/'eligible-samples.json');ids=[r['gsm'] for r in labels]
    donors=[r['donor_group'] for r in labels];actual=np.array([r['dose_gy'] for r in labels])
    if len(set(ids))!=52 or len(set(donors))!=12:raise ValueError('Unexpected eligible cohort')
    artifacts=index['artifacts']
    if len(artifacts)!=18 or len({r['path'] for r in artifacts})!=18:raise ValueError('Incomplete final model inventory')
    for a in artifacts:
        path=(MODEL/a['path']).resolve()
        if not path.is_relative_to(MODEL) or not completed(path,index['identity']) or sha(path/'complete.json')!=a['complete_sha256']:
            raise ValueError('Invalid trusted model artifact')
    with np.load(ROOT/'data/processed_locked/GSE90909/v1/frozen-panels.npz',allow_pickle=False) as z:
        if list(z['sample_ids'])!=ids:raise ValueError('Prediction membership/order mismatch')
        panels={k:z[k].copy() for k in ('log2','rank')}
    if any(x.shape!=(52,500) or not np.isfinite(x).all() for x in panels.values()):raise ValueError('Invalid frozen panels')
    protected=read(ELIG/'protected-hashes.json');protected.update(entry['sha256'])
    for p in [*ELIG.rglob('*'),*(ROOT/'context/final-training/v1').rglob('*'),*(ROOT/'context/final-selection/v2').rglob('*')]:
        if p.is_file():protected[str(p.relative_to(ROOT))]=sha(p)
    for raw in read(ELIG/'raw-integrity.json')['files']:protected[raw['path']]=raw['sha256']
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('Changed pre-evaluation evidence')
    code={p:sha(ROOT/p) for p in ['scripts/run_locked_evaluation.py','scripts/locked_evaluation_stats.py',
          'scripts/final_model_bundle.py','scripts/verify_locked_release.py','tests/test_locked_evaluation.py']}
    save('protected-before.json',protected)
    # Exclusive ledger is written before the first model invocation. A partial run is never silently restarted.
    with (LOG/'run-start.json').open('x') as f:
        json.dump(dict(utc=utc(),source_sha256=code,entry_freeze_sha256=sha(ELIG/'evaluation-entry-freeze.json'),
           model_bundle_freeze_sha256=sha(ROOT/'context/final-training/v1/model-bundle-freeze.json'),
           statistical_plan_sha256=sha(ROOT/'context/final-selection/v2/statistical-plan.json'),
           test_execution_authorized_by_current_user=True,profiles=52,donors=12,
           donor_independence_basis='12 explicit GEO donor IDs; inference conditional on distinct independent volunteers; public IDs do not exclude cross-study person reuse',
           unblinding='first model invocation follows this record',runtime=dict(python=sys.version,numpy=np.__version__,
            scipy=scipy.__version__,torch=str(torch.__version__),platform=platform.platform(),device='cpu',threads=4)),f,indent=2)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    predictions=[];donor_rows=[];summaries=[];strata=[];vectors={};(LOG/'predictions-by-model').mkdir()
    donor_order=sorted(set(donors));ix=np.random.default_rng(20261003).integers(0,12,(10000,12))
    np.save(LOG/'bootstrap-donor-indices.npy',ix)
    for a in artifacts:
        path=MODEL/a['path'];name=a['model'];seed=a['seed'];key=f'{name}:{seed}'
        if name=='median_null':
            model=read(path/'model.json');pred=np.full(52,model['prediction_gy']);rep='none'
        else:
            saved=read(path/'result.json');rep=saved['params']['representation']
            if saved['preprocessing']!='folds/full-development' or saved['model']!=name or saved['seed']!=seed:
                raise ValueError('Model/frozen-preparation mismatch')
            pred=predict_panel(path,panels[rep],index['identity'])
        if pred.shape!=(52,) or not np.isfinite(pred).all():raise ValueError('Invalid complete-cohort prediction')
        rows=[dict(model=name,seed=seed,seed_role=a['seed_role'],representation=rep,profile_id=r['gsm'],
              donor_group=r['donor_group'],actual_gy=r['dose_gy'],predicted_gy=float(p)) for r,p in zip(labels,pred)]
        save(f'predictions-by-model/{name}-{seed}.json',rows);predictions+=rows
        dr,metrics=donor_errors(actual,pred,donors)
        if [r['donor_group'] for r in dr]!=donor_order:raise ValueError('Unpaired donor ordering')
        vectors[key]=np.array([r['mae'] for r in dr]);metrics['mae_bootstrap_ci']=bootstrap_interval(vectors[key],ix)
        summaries.append(dict(model=name,seed=seed,seed_role=a['seed_role'],representation=rep,**metrics))
        donor_rows += [dict(model=name,seed=seed,seed_role=a['seed_role'],**r) for r in dr]
        for title,mask in [('control',actual==0),('above_2gy',actual>2)]+[(f'dose_{d:g}gy',actual==d) for d in sorted(set(actual))]:
            _,m=donor_errors(actual[mask],pred[mask],np.array(donors)[mask])
            strata.append(dict(model=name,seed=seed,seed_role=a['seed_role'],stratum=title,**m))
        print('COMPLETE',key,'52 profiles / 12 donors',flush=True)
    comparisons={j:paired_comparison(vectors['A:20261002'],vectors[f'{j}:20261002'],ix) for j in 'BCD'}
    comparisons=holm_three(comparisons)
    differences=[dict(donor_group=d,**{f'A_minus_{j}':float(vectors['A:20261002'][i]-vectors[f'{j}:20261002'][i]) for j in 'BCD'}) for i,d in enumerate(donor_order)]
    tsv('predictions.tsv',predictions);tsv('donor-metrics.tsv',donor_rows);tsv('stratum-metrics.tsv',strata);tsv('paired-donor-differences.tsv',differences)
    save('model-metrics.json',summaries);save('primary-comparisons.json',comparisons)
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('Protected input changed during evaluation')
    result=dict(status='COMPLETE_ONE_TIME_LOCKED_EVALUATION',utc=utc(),profiles=52,donors=12,model_versions=18,
       prediction_rows=len(predictions),donor_metric_rows=len(donor_rows),stratum_metric_rows=len(strata),
       primary_comparisons=3,bootstrap_resamples=10000,protected_files_unchanged=len(protected),
       seconds=time.perf_counter()-start,new_fits=0,excluded_profiles=0,clipping=False,
       all_three_primary_rejections=all(r['holm_reject'] and r['mean_difference']<0 for r in comparisons.values()),
       primary_claim_assumptions='Donor independence and paired-t approximation remain conditional; inspect QQ diagnostic and show all donors.',
       cloud_execution=False)
    save('run-complete.json',result)
    outputs={str(p.relative_to(LOG)):sha(p) for p in LOG.rglob('*') if p.is_file() and p.name not in ['run-execution.log','result-freeze.json']}
    save('result-freeze.json',dict(utc=utc(),sha256=outputs,purpose='Immutable one-time results; later analyses must be labelled exploratory'))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    try:run()
    except Exception:
        if LOG.exists():
            with (LOG/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(utc=utc(),traceback=traceback.format_exc()))+'\n')
        raise
