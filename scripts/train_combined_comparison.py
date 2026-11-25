"""Select and freeze the 567-profile dose predictors using development data only."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from abcd_models import balanced_weights, fit_network, predict, tensor
from clustering_models import ClusterRegressor
from core_baseline import donor_metrics
from extended_models import fit_estimator
from run_abcd_experiments import completed
from weighted_elastic_net import fit_regularized
from weighted_pls import WeightedPLS

SEED = 20261002
REUSED = ('B', 'C', 'D', 'gradient_boosting')
RELEASE = 'combined-567-gse90909-v1'
FINAL = Path('data/models/final-selection-v2')
SPLITS = Path('context/final-selection/v2/splits.json')
META = Path('data/model_ready/development_raw_f6022cfaaca6d1d9/metadata.json')
DEV = Path('data/experiments/combined-development-v2')
BOOST = Path('context/retrospective-gb-wbc-2026-11-14')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


@lru_cache(2)
def labels(root):
    return {r['profile_id']: r for r in read(root / META)['profiles']}


@lru_cache(12)
def panel(root, fold, representation):
    base = root / FINAL / 'folds' / fold
    info = read(base / 'fit.json')
    with np.load(base / (representation + '-panel.npz'), allow_pickle=False) as z:
        core, aux, val = (z[k].tolist() for k in ('training_ids', 'auxiliary_ids', 'validation_ids'))
        assert core == info['training'] and val == info['validation']
        x, validation = np.concatenate((z['train'], z['aux'])), z['validation']
    lookup = labels(root)
    records = [[lookup[i] for i in ids] for ids in (core, aux)]
    w = np.concatenate([0.5 * balanced_weights([r['study'] for r in rows], [r['donor_group'] for r in rows]) for rows in records])
    target = np.array([lookup[i]['dose_gy'] for i in core + aux])
    assert x.shape == (len(target), 500) and np.isfinite(x).all() and len(aux) == 422
    assert np.isclose(w[:len(core)].sum(), .5) and np.isclose(w[len(core):].sum(), .5)
    return x, (target-info['target_mean']) / info['target_scale'], w, validation, val, info


def fit_model(model, params, x, y, w, seed):
    if model == 'A':
        net, _ = fit_network(dict(core_x=tensor(x), core_y=tensor(y), core_weights=tensor(w)), params, seed)
        return net
    if model == 'median_null':
        return float(np.median(y))
    if model in ('elastic_net', 'lasso'):
        return fit_regularized(x, y, w, params['alpha'], params.get('l1_ratio', 1.))
    if model == 'pls':
        return WeightedPLS(params['n_components']).fit(x, y, w)
    if model in ('pca_ridge', 'kmeans_ridge', 'gmm_ridge', 'cluster_median'):
        options = {k:v for k,v in params.items() if k not in ('id', 'representation')}
        return ClusterRegressor(**options, seed=seed).fit(x, y, w)
    return fit_estimator(params, x, y, w, seed)


def predict_gy(model, estimator, x, info):
    if model == 'A':
        return np.asarray(predict(estimator, x, info['target_mean'], info['target_scale']))
    if model == 'median_null':
        y = np.full(len(x), estimator)
    elif model in ('elastic_net', 'lasso'):
        coef, intercept = estimator
        y = x @ coef + intercept
    else:
        y = np.asarray(estimator.predict(x)).reshape(-1)
    return y * info['target_scale'] + info['target_mean']


def freeze(root, run):
    root = root.resolve()
    cfg, dev = read(root / DEV / 'protocol.json'), read(root / DEV / 'summary.json')
    assert sha(root / DEV / 'protocol.json') == dev['protocol_sha256'] and not dev['final_test_included']
    models = sorted(r['model'] for r in dev['regression'])
    assert len(set(models)) == 20 and set(REUSED) <= set(models)
    split = read(root / SPLITS)
    all_labels = labels(root)
    core, aux = set(split['core_profiles']), set(split['auxiliary_profiles'])
    assert len(core) == 145 and len(aux) == 422 and core.isdisjoint(aux) and core | aux == set(all_labels)
    assert all(all_labels[i]['role'] == 'core_development' for i in core)
    assert all(all_labels[i]['role'] == 'auxiliary_development' for i in aux)
    assert {all_labels[i]['dose_gy'] for i in aux} == {0, 2}
    assert all(all_labels[i]['study'] != 'GSE90909' for i in core | aux)
    full = split['full_refit']
    assert set(full['core_training']) == core and set(full['auxiliary_training']) == aux and full['primary_seed'] == SEED
    idx = read(root / FINAL / 'artifact-index.json')
    dirs = root / FINAL / 'folds'
    pins = {}
    def pin(path):
        pins[str(path.relative_to(root))] = sha(path)
    for rel in (META, SPLITS, DEV/'protocol.json', DEV/'summary.json', FINAL/'artifact-index.json', BOOST/'protocol.json', BOOST/'model-freeze.json', BOOST/'training-complete.json'):
        pin(root / rel)
    seen = []
    folds = split['folds']
    for f in folds + [dict(fold_id='full-development', training=full['core_training'], validation=[], auxiliary_training=full['auxiliary_training'])]:
        directory = dirs / f['fold_id']
        assert completed(directory, idx['identity'])
        pin(directory/'complete.json')
        fit = read(directory/'fit.json')
        assert fit['training'] == f['training'] and fit['validation'] == f['validation'] and fit['nontraining_perturbation_invariant']
        tr, va = set(f['training']), set(f['validation'])
        assert tr | va == core and not tr & va and set(f['auxiliary_training']) == aux
        assert not {all_labels[i]['donor_group'] for i in tr | aux} & {all_labels[i]['donor_group'] for i in va}
        seen.extend(f['validation'])
        for rep in ('log2','rank'):
            with np.load(directory / (rep+'-panel.npz'),allow_pickle=False) as z:
                assert z['training_ids'].tolist() == f['training'] and z['validation_ids'].tolist() == f['validation']
                assert set(z['auxiliary_ids'].tolist()) == aux and z['aux'].shape == (422,500)
    assert sorted(seen) == sorted(core) and len(folds) == 5
    reuse = {}
    for name in REUSED:
        if name == 'gradient_boosting':
            rec = read(root / BOOST / 'model-freeze.json')
            path = root / BOOST / 'models' / f'combined-{SEED}.joblib'
            assert sha(path) == rec['sha256'][str(path.relative_to(root / BOOST))]
            assert sha(root / BOOST / 'training-complete.json') == rec['training_complete_sha256']
            assert sha(root / BOOST / 'protocol.json') == rec['protocol_sha256']
            prot = read(root / BOOST / 'protocol.json')
            assert prot['training']['combined_profiles'] == 567 and prot['primary_seed'] == SEED
            params = prot['estimator']
        else:
            entry = next(a for a in idx['artifacts'] if a['model'] == name and a['seed'] == SEED)
            directory = root / FINAL / entry['path']
            assert completed(directory, idx['identity']) and sha(directory/'complete.json') == entry['complete_sha256']
            result = read(directory/'result.json')
            assert result['phase'] == 'full_refit' and set(result['training_ids']) == core and set(result['auxiliary_training_ids']) == aux
            params = result['params']
            path = directory/'model.pt'
        pin(path)
        reuse[name] = dict(params=params, path=str(path.relative_to(root)), sha256=sha(path))
    sources = Path(__file__).resolve().parent
    code = {f:sha(sources/f) for f in ('train_combined_comparison.py','abcd_models.py','clustering_models.py','core_baseline.py','extended_models.py','weighted_pls.py','weighted_elastic_net.py')}
    method = dict(release=RELEASE, source_hashes=code, input_hashes=pins, development=dev['release_id'],
                  training_profiles=567, core_profiles=145, wbc_profiles=422, models=models, primary_seed=SEED,
                  reuse=reuse, candidates={m:cfg['candidates'][m] for m in models if m not in (*REUSED,'median_null')},
                  selection='For missing models, minimum donor-balanced MAE over all 145 out-of-fold development profiles in the original five donor-grouped selection folds; ties use lowest candidate ID.',
                  weighting='Half of total weight per tissue, equal study and donor within tissue, except unweighted median. Archived neural strategies retain their own WBC objectives.',
                  test='Previously evaluated GSE90909; retrospective inference on all 52 eligible profiles after model freeze; no test-dependent choices.')
    path = run / 'protocol.json'
    if path.exists() and read(path) != method:
        raise ValueError('Inputs or protocol changed. Use a separate release directory.')
    if not path.exists():
        write(path, method)
        write(run/'started.json',dict(utc=utc(), protocol_sha256=sha(path)))
    return method, sha(path)


def trial(args):
    root, run, model, params, identity = args
    path = run/'trials'/model/(str(params['id'])+'.json')
    if path.exists():
        saved = read(path)
        if saved['identity'] != identity or saved['params'] != params:
            raise ValueError('Mismatched cached candidate')
        return
    ids, predictions = [], []
    for f in read(root/SPLITS)['folds']:
        x,y,w,val,valid,info = panel(root, f['fold_id'], params['representation'])
        estimator = fit_model(model, params, x, y, w, f['seed'])
        preds = predict_gy(model, estimator, val, info)
        ids.extend(valid)
        predictions.extend(preds.tolist())
    assert sorted(ids) == sorted(read(root/SPLITS)['core_profiles'])
    records = labels(root)
    score, _ = donor_metrics([records[i]['dose_gy'] for i in ids], predictions, [records[i]['donor_group'] for i in ids])
    write(path, dict(identity=identity, params=params, ids=ids, predictions_gy=predictions, mae_gy=score['mae']))


def initialize():
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.project.resolve()
    run = root/'data/experiments'/RELEASE
    method, identity = freeze(root, run)
    jobs = [(root,run,m,p,identity) for m,grid in method['candidates'].items() for p in grid]
    print(f'Protocol frozen. {len(jobs)} candidates, five development folds each; four archived fits.', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        futures = [pool.submit(trial, j) for j in jobs]
        for n, future in enumerate(as_completed(futures),1):
            future.result()
            if n % 12 == 0: print('Candidates complete',n,'/',len(jobs),flush=True)
    chosen = {}
    for m, grid in method['candidates'].items():
        trials = [read(run/'trials'/m/(str(p['id'])+'.json')) for p in grid]
        chosen[m] = min(trials,key=lambda t:(t['mae_gy'],t['params']['id']))['params']
    chosen['median_null'] = dict(representation='log2',statistic='unweighted training median')
    write(run/'selected.json',dict(protocol_sha256=identity, settings=chosen))
    initialize()
    artifacts = {}
    for m in method['models']:
        if m in REUSED:
            artifacts[m] = dict(method['reuse'][m], source='archived')
            continue
        p = chosen[m]
        x,y,w,_,_,info = panel(root, 'full-development', p['representation'])
        assert len(y) == 567
        model = fit_model(m, p, x, y, w, SEED)
        dest = run / 'models' / (m+'.joblib')
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.tmp')
        joblib.dump(model, temp, compress=3)
        os.replace(temp,dest)
        artifacts[m] = dict(params=p, source='new_full_refit', path=str(dest.relative_to(root)), sha256=sha(dest))
    assert len(artifacts) == 20
    write(run/'model-freeze.json', dict(utc=utc(),protocol_sha256=identity, selection_sha256=sha(run/'selected.json'),
        training_profiles=567, selection_fits=len(jobs)*5, new_final_fits=len(chosen), models=artifacts))
    print('Frozen',len(artifacts),'models before GSE90909 inference.',flush=True)


if __name__ == '__main__':
    main()
