"""Rebuild the combined-cohort nested validation from verified development-only panels."""
import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy import __version__ as scipy_version
import sklearn
import torch
import xgboost
from threadpoolctl import threadpool_limits

from abcd_models import balanced_weights, tensor, fit_network, predict
from clustering_models import ClusterRegressor
from core_baseline import donor_metrics
from extended_models import CLASSIFICATION, REGRESSION, STOCHASTIC, binary_metrics, fit_estimator
from matched_classical import fit_classical
from weighted_pls import WeightedPLS
from weighted_elastic_net import fit_regularized
import osqp

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'combined-development-v2'
OUT = ROOT / 'data/experiments' / RELEASE
META = ROOT / 'data/model_ready/development_raw_f6022cfaaca6d1d9/metadata.json'
FOLDS = ROOT / 'data/models/abcd-v1/folds'
SOURCE = ROOT / 'context/model-abcd/v1'
REUSED = ('B', 'C', 'D')
CLUSTERS = ('pca_ridge', 'kmeans_ridge', 'gmm_ridge', 'cluster_median')
ORDINARY = ('gradient_boosting', 'adaboost', 'random_forest', 'xgboost', 'A', 'svr_linear',
            'elastic_net', 'pls', *CLUSTERS, *(m for m in REGRESSION if m not in
            ('adaboost', 'random_forest', 'xgboost', 'svr_linear')), *CLASSIFICATION)
COUNTS = {'GSE102971': 100, 'GSE65292': 20, 'GSE8917': 25}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    partial.replace(path)


@lru_cache(None)
def inputs():
    split = read(SOURCE / 'splits.json')
    labels = {r['profile_id']: r for r in read(META)['profiles']}
    configs = {}
    for group in ('model-abcd', 'classical-comparison', 'extended-models', 'clustering'):
        configs.update(read(ROOT / 'context' / group / 'v1/config.json')['candidates'])
    return split, labels, configs


def freeze():
    split, labels, configs = inputs()
    core = {k for k, r in labels.items() if r['role'] == 'core_development'}
    aux = {k for k, r in labels.items() if r['role'] == 'auxiliary_development'}
    if (len(labels), len(core), len(aux)) != (567, 145, 422) or core != set(split['core_profiles']) or aux != set(split['auxiliary_training']):
        raise ValueError('Development membership mismatch')
    if any(r['study'] == 'GSE90909' or not np.isfinite(r['dose_gy']) or r['exposure'] != int(r['dose_gy'] > 0) for r in labels.values()):
        raise ValueError('Unexpected study or unverified target')
    if {labels[i]['dose_gy'] for i in aux} != {0, 2}:
        raise ValueError('Unexpected auxiliary labels')
    pins = {}
    def pin(path, expected=None):
        signature = digest(path)
        if expected is not None and signature != expected:
            raise ValueError(f'Invalid source digest {path}')
        pins[str(path.relative_to(ROOT))] = signature
    for name in ('predictions.tsv', 'verification.json', 'splits.json', 'config.json'):
        path = SOURCE / name
        record = next(r for r in read(SOURCE / 'artifact-manifest.json')['files'] if r['path'] == str(path.relative_to(ROOT)))
        pin(path, record['sha256'])
    for pkg in ('classical-comparison', 'extended-models', 'clustering'):
        path = ROOT / 'context' / pkg / 'v1/config.json'
        pin(path, read(path.with_name('protocol-freeze.json'))['sha256']['config.json'])
    pin(META)
    for name in ('run_combined_development', 'weighted_pls', 'weighted_elastic_net', 'abcd_models', 'extended_models', 'matched_classical', 'clustering_models', 'core_baseline'):
        pin(ROOT / 'scripts' / (name + '.py'))
    counts = {}
    for outer in split['outer']:
        study = outer['study']
        val = {k for k in core if labels[k]['study'] == study}
        if set(outer['validation']) != val or set(outer['training']) != core - val or len(val) != COUNTS[study]:
            raise ValueError('Invalid outer split')
        inner_validation = []
        for fold in outer['inner'] + [dict(name=study+'/outer', training=outer['training'], validation=outer['validation'])]:
            directory = FOLDS / fold['name']
            manifest = read(directory / 'complete.json')['files']
            tr, va = set(fold['training']), set(fold['validation'])
            if tr & va or ({labels[i]['donor_group'] for i in tr | aux} & {labels[i]['donor_group'] for i in va}):
                raise ValueError('Training and validation overlap')
            if fold['name'] != study+'/outer':
                if tr | va != core-val or (tr | va) & val:
                    raise ValueError('Invalid inner study isolation')
                inner_validation.extend(va)
            for name in ('fit.json', 'rank-panel.npz', 'log2-panel.npz'):
                pin(directory/name, manifest[name])
            fit = read(directory/'fit.json')
            if fit['training'] != fold['training'] or fit['validation'] != fold['validation'] or not fit['nontraining_perturbation_invariant']:
                raise ValueError('Invalid preprocessing membership')
            for rep in ('rank', 'log2'):
                with np.load(directory/(rep+'-panel.npz'), allow_pickle=False) as panel:
                    if (panel['training_ids'].tolist() != fit['training'] or panel['validation_ids'].tolist() != fit['validation']
                            or len(panel['auxiliary_ids']) != 422 or set(panel['auxiliary_ids']) != aux):
                        raise ValueError('Prepared panel membership mismatch')
                    for key, size in [('train',len(tr)), ('aux',422), ('validation',len(va))]:
                        if panel[key].shape != (size,500) or not np.isfinite(panel[key]).all():
                            raise ValueError('Invalid feature matrix')
        if sorted(inner_validation) != sorted(outer['training']):
            raise ValueError('Inner folds must cover outer training once')
        counts[study] = {'whole_blood_training':145-len(val), 'wbc_training':422, 'training':567-len(val), 'validation':len(val)}
    historic = read(SOURCE/'verification.json')
    if historic['auxiliary_profiles'] != 422 or historic['locked_test_opened'] or historic['checkpoint_raw_replays'] != 36:
        raise ValueError('Historical combined predictions lack verified source')
    protocol = dict(release=RELEASE, inputs=pins, membership=counts, candidates={m:configs[m] for m in ORDINARY},
                    reused_primary_predictions=list(REUSED), task='predict held-out whole blood; auxiliary WBC training only',
                    weighting='half of mass per tissue, then equal study and donor within tissue; class-balanced for classifiers',
                    median='unweighted median of all training doses', selection='pooled inner donor-balanced MAE or 1 - balanced accuracy, then lowest candidate id',
                    versions=dict(osqp=osqp.__version__,numpy=np.__version__,torch=str(torch.__version__),sklearn=sklearn.__version__,scipy=scipy_version,xgboost=xgboost.__version__),
                    final_test_included=False)
    path=OUT/'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Refusing to mix runs with a changed protocol')
    if not path.exists():
        save(path,protocol)
        save(OUT/'started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(path)})
    return digest(path)


@lru_cache(12)
def matrix(fold, representation, binary):
    _, labels, _ = inputs()
    folder = FOLDS / fold
    fit = read(folder/'fit.json')
    with np.load(folder/(representation+'-panel.npz'), allow_pickle=False) as z:
        ids = z['training_ids'].tolist() + z['auxiliary_ids'].tolist()
        x = np.concatenate([z['train'],z['aux']])
        validation = z['validation']
    tr = [labels[i] for i in ids]
    parts = [tr[:len(fit['training'])],tr[len(fit['training']):]]
    weights = np.concatenate([.5*balanced_weights([r['study'] for r in rows],[r['donor_group'] for r in rows],
                              [r['exposure'] for r in rows] if binary else None) for rows in parts])
    target = np.array([r['exposure'] if binary else r['dose_gy'] for r in tr])
    if not binary:
        target = (target-fit['target_mean'])/fit['target_scale']
    return x, target, weights, validation, fit


def run_fit(model, params, fold, seed):
    binary = model in CLASSIFICATION
    x,y,w,val,f = matrix(fold,params['representation'],binary)
    if model == 'A':
        batch = dict(core_x=tensor(x),core_y=tensor(y),core_weights=tensor(w))
        net,_ = fit_network(batch,params,seed)
        return predict(net,val,f['target_mean'],f['target_scale']).tolist()
    if model == 'pls':
        estimator = WeightedPLS(params['n_components']).fit(x,y,w)
    elif model in ('elastic_net','lasso'):
        coef,intercept = fit_regularized(x,y,w,params['alpha'],params.get('l1_ratio',1.))
        return ((val@coef+intercept)*f['target_scale']+f['target_mean']).tolist()
    elif model=='gradient_boosting':
        estimator,_ = fit_classical(params,x,y,w,seed)
    elif model in CLUSTERS:
        estimator = ClusterRegressor(**{k:v for k,v in params.items() if k not in ('id','representation')},seed=seed).fit(x,y,w)
    else:
        estimator = fit_estimator(params,x,y,w,seed)
    if binary:
        return estimator.predict_proba(val)[:,1].tolist()
    return (np.asarray(estimator.predict(val)).reshape(-1)*f['target_scale']+f['target_mean']).tolist()


def score(model, ids, predictions):
    labels = inputs()[1]
    donors = [labels[i]['donor_group'] for i in ids]
    if model in CLASSIFICATION:
        return binary_metrics([labels[i]['exposure'] for i in ids],predictions,donors)
    return donor_metrics([labels[i]['dose_gy'] for i in ids],predictions,donors)[0]


def trial(task):
    model,study,params,identity = task
    name = str(params['id'])
    path = OUT/'trials'/study/model/(name+'.json')
    if path.exists() and read(path)['identity'] == identity:
        return
    outer = next(o for o in inputs()[0]['outer'] if o['study']==study)
    ids, preds = [], []
    for fold in outer['inner']:
        preds.extend(run_fit(model,params,fold['name'],fold['seed']))
        ids.extend(fold['validation'])
    if sorted(ids) != sorted(outer['training']):
        raise ValueError('Incomplete inner scores')
    metrics = score(model,ids,preds)
    loss = 1-metrics['balanced_accuracy'] if model in CLASSIFICATION else metrics['mae']
    save(path,dict(identity=identity,params=params,validation_ids=ids,predictions=preds,loss=loss))


def outer_fit(task):
    model,study,identity = task
    path = OUT/'outer'/study/(model+'.json')
    if path.exists() and read(path)['identity'] == identity:
        return
    splits, labels, configs = inputs()
    o = next(o for o in splits['outer'] if o['study']==study)
    if model in REUSED:
        with (SOURCE/'predictions.tsv').open() as file:
            rows = {r['profile_id']:r for r in csv.DictReader(file,delimiter='\t') if r['strategy']==model and r['study']==study and r['seed_index']=='0'}
        if set(rows) != set(o['validation']) or any(float(r['actual_gy'])!=labels[i]['dose_gy'] or r['donor_group']!=labels[i]['donor_group'] for i,r in rows.items()):
            raise ValueError('Historical holdout prediction mismatch')
        params = {'id': int(next(iter(rows.values()))['candidate_id'])}
        seeds = [o['seeds'][0]]
        values = [[float(rows[i]['predicted_gy']) for i in o['validation']]]
    elif model=='median_null':
        ids = o['training'] + splits['auxiliary_training']
        params, seeds = {}, [0]
        values = [[float(np.median([labels[i]['dose_gy'] for i in ids]))]*len(o['validation'])]
    else:
        candidates = [read(OUT/'trials'/study/model/(str(p['id'])+'.json')) for p in configs[model]]
        if any(c['identity']!=identity for c in candidates):
            raise ValueError('Mixed candidate protocols')
        params = min(candidates,key=lambda c:(c['loss'],c['params']['id']))['params']
        seeds = o['seeds'] if model in (*STOCHASTIC,'A','gradient_boosting','gmm_ridge','kmeans_ridge','cluster_median') else o['seeds'][:1]
        values = [run_fit(model,params,study+'/outer',s) for s in seeds]
    save(path,dict(identity=identity,model=model,study=study,params=params,seeds=seeds,validation_ids=o['validation'],predictions=values,
                   training_profiles=len(o['training'])+422,source='verified existing' if model in REUSED else 'refit'))


def summarize(identity):
    models = [*ORDINARY, *REUSED, 'median_null']
    regression, classification = [], []
    splits, labels, _ = inputs()
    for model in models:
        ids, preds, by_study = [], [], {}
        for o in splits['outer']:
            row = read(OUT/'outer'/o['study']/(model+'.json'))
            if row['identity'] != identity or row['validation_ids'] != o['validation'] or row['training_profiles']+len(o['validation'])!=567:
                raise ValueError('Invalid outer result')
            values = row['predictions'][0]
            by_study[o['study']] = score(model,o['validation'],values)
            ids.extend(o['validation']); preds.extend(values)
        if len(ids)!=145 or set(ids)!=set(splits['core_profiles']) or len({labels[i]['donor_group'] for i in ids})!=30:
            raise ValueError('Incomplete held-out coverage')
        overall = score(model,ids,preds)
        if model in CLASSIFICATION:
            for study, metric in [('pooled',overall), *by_study.items()]:
                classification.append(dict(model=model,study=study,profiles=145 if study=='pooled' else COUNTS[study],**metric))
        else:
            regression.append(dict(model=model,mae_gy=overall['mae'],rmse_gy=overall['rmse'],bias_gy=overall['bias'],
                                   worst_study_mae_gy=max(m['mae'] for m in by_study.values()),
                                   equal_study_mae_gy=float(np.mean([m['mae'] for m in by_study.values()])),by_study=by_study))
    if (len(regression),len(classification))!=(20,12):
        raise ValueError('Incomplete model comparison')
    save(OUT/'summary.json',dict(release_id=RELEASE,completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=identity,
         eligible_profiles=567,whole_blood_profiles=145,wbc_profiles=422,held_out_profiles=145,donor_groups=30,
         final_test_included=False,folds=read(OUT/'protocol.json')['membership'], reused=list(REUSED),
         regression=sorted(regression,key=lambda r:r['mae_gy']),classification=classification))


def initialize():
    torch.set_num_threads(1)
    threadpool_limits(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    identity=freeze()
    _,_,configs=inputs()
    with ProcessPoolExecutor(args.workers,initializer=initialize) as pool:
        for fn, tasks in [(trial,[(m,s,p,identity) for m in ORDINARY for p in configs[m] for s in COUNTS]),
                          (outer_fit,[(m,s,identity) for m in (*ORDINARY,*REUSED,'median_null') for s in COUNTS])]:
            futures={pool.submit(fn,t):t[:2] for t in tasks}
            failures=[]
            for idx,future in enumerate(as_completed(futures),1):
                try:
                    future.result()
                except Exception as exc:
                    failures.append((futures[future],str(exc)))
                    print('FAILED',futures[future],repr(exc),flush=True)
                if idx%20==0: print(fn.__name__,idx,'/',len(tasks),flush=True)
            if failures:
                raise RuntimeError(str(failures))
    summarize(identity)
    print('Verified',OUT/'summary.json',flush=True)


if __name__=='__main__':
    main()
