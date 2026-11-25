"""Engineering validation and full refits on the pinned 567-profile development pool."""
import platform
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

from cloud_io import encode
from core_baseline import donor_metrics
from portable_model import estimator_spec, numeric_predict
from training_contract import CORE
from weighted_pls import WeightedPLS

SCOPES = (*CORE, 'full')
SEED = 20260930


def verify_bundle(data):
    ids, studies, groups, dose = (data[k] for k in ('profile_ids','studies','groups','dose'))
    if (len(set(ids)) != 567 or any(a.shape != (567,) for a in (ids,studies,groups,dose))
            or len(set(studies))!=6 or any(not str(g).startswith(str(s)+':') for g,s in zip(groups,studies))
            or set(CORE)-set(studies) or 'GSE90909' in studies or not np.isfinite(dose).all() or np.any(dose<0)):
        raise ValueError('Invalid development cohort')
    aux = ~np.isin(studies,list(CORE))
    if aux.sum()!=422 or len(set(groups[~aux]))!=30 or set(dose[aux])!={0,2}:
        raise ValueError('Invalid tissue membership')
    for scope in SCOPES:
        tr, va = data[scope+'_train'], data[scope+'_test']
        test = np.flatnonzero(studies==scope)
        if (len(va)!=CORE.get(scope,0) or not np.array_equal(np.sort(va),test)
                or not np.array_equal(np.sort(np.r_[tr,va]),np.arange(567))
                or set(groups[tr])&set(groups[va])):
            raise ValueError('Invalid fold membership')
        x, v = data[scope+'_x'],data[scope+'_v']
        if x.shape!=(len(tr),100) or v.shape!=(len(va),100) or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError('Invalid prepared engineering features')
        if len(set(data[scope+'_genes']))!=100 or np.any(data[scope+'_scale']<=0):
            raise ValueError('Invalid selected gene panel')


def fit_weights(studies, groups):
    weights = np.empty(len(studies))
    for mask in (np.isin(studies,list(CORE)),~np.isin(studies,list(CORE))):
        sources=np.unique(studies[mask])
        for study in sources:
            members = studies==study
            donors, inverse, counts = np.unique(groups[members],return_inverse=True,return_counts=True)
            weights[members] = .5 / (len(sources)*len(donors)*counts[inverse])
    return weights


def fit(job, data, scope):
    idx = data[scope+'_train']
    x, dose = data[scope+'_x'],data['dose'][idx]
    w = fit_weights(data['studies'][idx],data['groups'][idx])
    method, params = job['method'],job['parameters']
    if method=='median_null':
        model=None
    elif method=='pls':
        model=WeightedPLS(params['n_components']).fit(x,dose,w)
    elif method=='elastic_net':
        model=ElasticNet(alpha=params['alpha'],l1_ratio=params['l1_ratio'],tol=1e-6,max_iter=100000).fit(x,dose,sample_weight=w)
        if model.n_iter_>=model.max_iter:
            raise ValueError('Elastic Net did not converge')
    else:
        model=GradientBoostingRegressor(n_estimators=params['n_estimators'],max_depth=params['max_depth'],
                   random_state=SEED,learning_rate=.05,min_samples_leaf=3).fit(x,dose,sample_weight=w)
    if method=='pls':
        estimator={'kind':'linear','coef':model.coef_.tolist(),'intercept':model.intercept_}
    else:
        estimator=estimator_spec(method,model,dose)
    v = data[scope+'_v'] if scope!='full' else x[:5]
    expected=np.full(len(v),np.median(dose)) if model is None else np.asarray(model.predict(v)).reshape(-1)
    predicted = numeric_predict(estimator,v)
    single = np.array([numeric_predict(estimator,row[None])[0] for row in v])
    if not all(np.allclose(t,predicted,rtol=1e-8,atol=1e-8) for t in (expected,single,numeric_predict(estimator,v[::-1])[::-1])):
        raise ValueError('Combined predictor replay failed')
    spec = dict(schema_version=2,method=method,estimator=estimator,fit_samples=data['profile_ids'][idx].tolist(),
                parameters=params,dataset=job['dataset'],output_unit='Gy',engineering_only=True,deployment_allowed=False,
                input_scale='positive_per_array_background_corrected_linear_intensity',scope=scope)
    return spec, predicted, float(np.max(abs(predicted-expected)))


def train(job, data, folder):
    started=time.perf_counter()
    verify_bundle(data)
    folder.mkdir(parents=True,exist_ok=True)
    held=job['holdout_study']
    selected, predictions, check = fit(job,data,held)
    full, sample_replay, full_check = fit(job,data,'full')
    idx=data[held+'_test']
    metrics,_=donor_metrics(data['dose'][idx],predictions,data['groups'][idx])
    rows=[dict(job_id=job['job_id'],sample_id=str(data['profile_ids'][i]),group_id=str(data['groups'][i]),
          observed_gy=float(data['dose'][i]),predicted_gy=float(y),evaluation_scope='engineering_smoke_single_study') for i,y in zip(idx,predictions)]
    (folder/'model.json').write_bytes(encode(full))
    selected.update(input_scale='prepared_standardized_log2_panel',gene_ids=data[held+'_genes'].tolist())
    (folder/'validation-model.json').write_bytes(encode(selected))
    (folder/'reference.npz').write_bytes(data['full_reference'].tobytes())
    np.savez_compressed(folder/'preparation.npz',**{k:data['full_'+k] for k in ('genes','medians','mean','scale')})
    np.savez_compressed(folder/'replay.npz',validation_panel=data[held+'_v'],validation_expected_gy=predictions,
                        refit_panel=data['full_x'][:5],refit_expected_gy=sample_replay)
    import scipy, sklearn
    result=dict(job_id=job['job_id'],method=job['method'],holdout_study=held,train_samples=len(data[held+'_train']),
                validation_samples=len(idx),refit_samples=len(data['full_train']),wbc_training_profiles=422,
                training_seconds=time.perf_counter()-started,purpose=job['purpose'],deployment_allowed=False,
                metrics=metrics,predictions=rows,checks=dict(fold_isolation=True,validation_replay_max_abs_gy=check,refit_replay_max_abs_gy=full_check),
                runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    (folder/'result.json').write_bytes(encode(result))
    return result
