"""Bounded outbound Mac worker for core engineering jobs stored in GCS.

Runs one job; mac_worker_loop discovers work. No public port or cloud training.
"""
import argparse
import io
import json
from pathlib import Path
import time
import uuid
import platform
import http.client
import urllib.error

import numpy as np
from cloud_io import Cloud, Conflict, PREFIX, encode, digest
from training_contract import PRESETS, SCHEMAS, identifier, validate_job, validate_data, validate_rows
from core_baseline import fit_estimator, donor_metrics
from run_core_baselines import fit_preparation, preparation_signature
from portable_model import estimator_spec, PortablePredictor

ROOT=Path(__file__).resolve().parents[1]


from job_queue import Queue, job_name


class WorkerCloud(Cloud):
    def immutable(self, name, data):
        # A dropped upload may already have committed. The base implementation
        # verifies bytes on conflict, so retrying this immutable write is safe.
        for attempt in range(3):
            try:
                return super().immutable(name, data)
            except (http.client.RemoteDisconnected, urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)


def train(job, data, folder):
    validate_job(job);validate_data(data)
    folder.mkdir(parents=True,exist_ok=True)
    held=job['holdout_study'];train_idx=np.flatnonzero(data['study_ids']!=held);val=np.flatnonzero(data['study_ids']==held)
    if set(data['split_group_ids'][train_idx])&set(data['split_group_ids'][val]): raise ValueError('Group leakage')
    started=time.perf_counter()
    transfer,genes,matrix,preps=fit_preparation(data,train_idx)
    # Positive isolation check on actual inputs, not only synthetic tests.
    changed=dict(data);changed['values']=data['values'].copy();changed['values'][:,val]=np.nan
    changed['detected']=data['detected'].copy();changed['detected'][:,val]=False
    changed['y']=data['y'].copy();changed['y'][val]=99999
    transfer2,_,_,preps2=fit_preparation(changed,train_idx)
    if preparation_signature(transfer,preps)!=preparation_signature(transfer2,preps2): raise ValueError('Validation influenced fitting')
    method=job['method'];prep=preps[100]
    panel=prep.transform(matrix,genes,method!='gradient_boosting')
    estimator=None if method=='median_null' else fit_estimator(method,job['parameters'],panel,data['y'][train_idx])
    transfer.save(folder/'reference.npz');prep.save(folder/'preparation.npz')
    spec={'schema_version':1,'method':method,'estimator':estimator_spec(method,estimator,data['y'][train_idx]),
          'input_scale':'positive_per_array_background_corrected_linear_intensity',
          'output_unit':'Gy','engineering_only':True,'deployment_allowed':False,'dataset':job['dataset'],
          'fit_samples':data['sample_ids'][train_idx].tolist(),'parameters':job['parameters']}
    (folder/'model.json').write_bytes(encode(spec))
    predictor=PortablePredictor(folder);values=data['values'][:,val]
    predicted=predictor.predict(values,data['probe_ids'])
    vg,vx,_=transfer.transform(values,np.isfinite(values),data['probe_ids'])
    vpanel=prep.transform(vx.T,vg,method!='gradient_boosting')
    expected=np.full(len(val),np.median(data['y'][train_idx])) if estimator is None else np.asarray(estimator.predict(vpanel)).reshape(-1)
    single=np.array([predictor.predict(values[:,i:i+1],data['probe_ids'])[0] for i in range(len(val))])
    reverse=predictor.predict(values[:,::-1],data['probe_ids'])[::-1]
    checks={'portable_vs_sklearn_max_abs_gy':float(np.max(abs(predicted-expected))),
            'single_vs_batch_max_abs_gy':float(np.max(abs(predicted-single))),
            'reordered_max_abs_gy':float(np.max(abs(predicted-reverse))), 'fold_isolation':True}
    for other in (expected,single,reverse):
        if not np.allclose(predicted,other,atol=1e-8,rtol=1e-8): raise ValueError('Portable inference mismatch')
    metrics,_=donor_metrics(data['y'][val],predicted,data['split_group_ids'][val])
    rows=[dict(job_id=job['job_id'],sample_id=str(data['sample_ids'][i]),group_id=str(data['split_group_ids'][i]),
               observed_gy=float(data['y'][i]),predicted_gy=float(p),evaluation_scope='engineering_smoke_single_study') for i,p in zip(val,predicted)]
    np.savez_compressed(folder/'replay.npz',values=values,probe_ids=data['probe_ids'],expected_gy=predicted)
    result=dict(job_id=job['job_id'],method=method,holdout_study=held,train_samples=len(train_idx),
        validation_samples=len(val),training_seconds=time.perf_counter()-started,checks=checks,metrics=metrics,
        purpose='engineering_smoke',deployment_allowed=False,predictions=rows)
    import scipy, sklearn
    result['runtime']={'python':platform.python_version(),'numpy':np.__version__,
                       'scipy':scipy.__version__,'sklearn':sklearn.__version__}
    result['code_sha256']={name:digest((ROOT/'scripts'/name).read_bytes()) for name in
        ('training_worker.py','portable_model.py','training_contract.py','core_transfer.py','core_baseline.py',
         'run_core_baselines.py','assay_qc.py','masked_quantile.py')}
    (folder/'result.json').write_bytes(encode(result))
    return result


def publish_warehouse(cloud,job_id):
    state,_=cloud.read_json(job_name(job_id))
    if state['status']!='SUCCEEDED': raise ValueError('Job has no committed result')
    ref=state['result']
    committed=json.loads(cloud.read(ref['object'],ref['generation'],ref['sha256']))
    job=state['request'];result=committed['result']
    release=job['dataset']['object'].split('/')[3]
    tables={'model_runs':[dict(job_id=job_id,release_id=release,method=job['method'],holdout_study=job['holdout_study'],
        purpose='engineering_smoke',status='SUCCEEDED',deployment_allowed=False,train_samples=result['train_samples'],
        validation_samples=result['validation_samples'],training_seconds=result['training_seconds'],artifact_manifest=ref['object'])],
        'predictions':result['predictions'],
        'model_metrics':[dict(job_id=job_id,metric=k,value=float(result['metrics'][k]),unit='Gy',
            evaluation_scope='engineering_smoke_single_study') for k in ('mae','rmse','bias')]}
    loaded={}
    for table,rows in tables.items():
        validate_rows(table,rows)
        raw=b''.join(encode(r)+b'\n' for r in rows)
        source=cloud.immutable(PREFIX+'results/'+job_id+'/'+table+'.jsonl',raw)
        load=cloud.load_table(table,SCHEMAS[table],source,job_id+'_'+table,len(rows))
        loaded[table]={'rows':len(rows),'job':load['jobReference']}
    cloud.immutable(PREFIX+'results/'+job_id+'/warehouse.json',encode(loaded))
    return loaded


def run(cloud,job_id):
    if job_id.startswith('web_'):
        from training_bridge import job_month,validate_approval,worker_hashes
        from dispatch_control import DispatchControl
        control,_=DispatchControl(cloud,job_month(job_id)).read()
        entry=control['jobs'].get(job_id)
        if entry and entry.get('bridge_approval_sha256'):
            validate_approval(cloud,entry,worker_hashes(ROOT/'scripts'))
    queue=Queue(cloud);claimed=queue.claim(job_id)
    if claimed is None: return {'already_completed':True,'warehouse':publish_warehouse(cloud,job_id)}
    try:
        return run_claimed(cloud,job_id,claimed)
    except Exception:
        # Only release this attempt's lease. Cancellation, takeover, and a
        # completion whose response was lost remain protected by the CAS check.
        try: queue.fail(job_id,claimed[0]['lease']['id'],'Training/publication interrupted; retry pending')
        except Exception: pass # An unreachable cloud still recovers on lease expiry.
        raise


def run_claimed(cloud,job_id,claimed):
    queue=Queue(cloud)
    state,generation=claimed;job=state['request'];ref=job['dataset']
    cloud.preflight(15_000_000)
    cache=ROOT/'data/training_cache'/ref['sha256'];cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists() or digest(cache.read_bytes())!=ref['sha256']:
        raw=cloud.read(ref['object'],ref['generation'],ref['sha256'])
        tmp=cache.with_suffix('.'+str(uuid.uuid4())+'.partial')
        try:
            tmp.write_bytes(raw);tmp.replace(cache)
        finally:
            tmp.unlink(missing_ok=True)
    with np.load(io.BytesIO(cache.read_bytes()),allow_pickle=False) as z:data={k:z[k] for k in z.files}
    state,generation=queue.update(state,generation,'training')
    folder=ROOT/'data/worker_runs'/job_id/state['lease']['id']
    result=train(job,data,folder)
    state,generation=queue.update(state,generation,'publishing')
    artifacts={}
    # Explicit allowlist: no context, logs, credentials or arbitrary workspace uploads.
    for name in ('reference.npz','preparation.npz','model.json','replay.npz'):
        artifacts[name]=cloud.immutable(PREFIX+'artifacts/'+job_id+'/'+state['lease']['id']+'/'+name,(folder/name).read_bytes())
    committed={'schema_version':1,'request_sha256':state['request_sha256'],'artifacts':artifacts,'result':result}
    result_ref=cloud.immutable(PREFIX+'artifacts/'+job_id+'/'+state['lease']['id']+'/result.json',encode(committed))
    state,generation=queue.update(state,generation,'completed',result_ref)
    warehouse=publish_warehouse(cloud,job_id)
    receipt={'job_id':job_id,'result':result_ref,'warehouse':warehouse,'requests':cloud.requests,
             'local_run':str(folder.relative_to(ROOT))}
    evidence=ROOT/'context/cloud/integration-v1';evidence.mkdir(parents=True,exist_ok=True)
    (evidence/(job_id+'.json')).write_bytes(encode(receipt))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('submit');a.add_argument('--release-receipt',required=True);a.add_argument('--method',choices=PRESETS,required=True);a.add_argument('--job-id',required=True);a.add_argument('--holdout-study',default='GSE8917')
    a=sub.add_parser('run');a.add_argument('--job-id',required=True)
    args=p.parse_args();cloud=WorkerCloud()
    if args.command=='submit':
        release=json.loads(Path(args.release_receipt).read_text())
        job=dict(schema_version=1,job_id=args.job_id,dataset=release['dataset'],method=args.method,
                 holdout_study=args.holdout_study,purpose='engineering_smoke',parameters=PRESETS[args.method])
        print(Queue(cloud).submit(job))
    else: print(json.dumps(run(cloud,args.job_id),indent=2))
