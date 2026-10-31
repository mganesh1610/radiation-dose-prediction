"""Approved engineering-job adapter. No model fitting or LLM-controlled writes.

The authenticated website mints approval; Evidence only reads owned job status.
The existing outbound worker checks the approved code and request before claiming.
"""
import json
import math
import re
import time
from pathlib import Path
from cloud_io import PREFIX, Conflict, digest, encode
from dispatch_control import DispatchControl
from job_queue import Queue, job_name
from training_contract import PRESETS, CORE, validate_job

BASE=PREFIX+'control/bridge/'
WORKER_FILES=('training_worker.py','portable_model.py','training_contract.py','core_transfer.py',
              'core_baseline.py','run_core_baselines.py','assay_qc.py','masked_quantile.py',
              'job_queue.py','cloud_io.py','dispatch_control.py','mac_worker_loop.py','training_bridge.py')


def worker_hashes(directory):
    return {name:digest((Path(directory)/name).read_bytes()) for name in WORKER_FILES}


def job_month(job_id):
    match=re.fullmatch(r'web_(\d{4})(0[1-9]|1[0-2])_[a-f0-9]{32}',str(job_id))
    if not match:raise ValueError('Invalid application job ID')
    return match[1]+'-'+match[2]


def validate_approval(cloud,entry,code_hashes=None):
    """Approval is read from the trusted store, never accepted from model input."""
    job=entry['request'];validate_job(job)
    if digest(encode(job))!=entry['sha256']:raise ValueError('Admission/request mismatch')
    approval,_=cloud.read_json(BASE+job['job_id']+'.json')
    plan=approval['plan']
    if (approval['owner']!=entry['owner'] or plan['job']!=job
            or digest(encode(plan))!=approval['approval_sha256']
            or entry.get('bridge_approval_sha256')!=approval['approval_sha256']
            or plan['owner_binding']!=digest(entry['owner'].encode())
            or plan['procedure']!='approved-engineering-v1'
            or plan['limits']!={'attempt_seconds':720,'max_attempts':3,'deployment_allowed':False}):
        raise ValueError('Training approval integrity failed')
    if code_hashes is not None and plan['worker_code_sha256']!=code_hashes:
        raise ValueError('Worker changed since approval; prepare and approve a new job')
    return approval


class TrainingBridge:
    def __init__(self,cloud,owner,dataset,policy,*,reviewer=False,month=None,clock=time.time):
        self.cloud,self.owner,self.dataset,self.policy=cloud,owner,dataset,policy
        self.reviewer,self.clock=reviewer,clock
        self.control=DispatchControl(cloud,month)
        if set(policy)!=set(WORKER_FILES) or any(not re.fullmatch('[a-f0-9]{64}',h) for h in policy.values()):
            raise ValueError('Invalid protected training policy')

    def prepare(self,body):
        if not isinstance(body,dict) or set(body)!={'job_id','method','holdout_study'}:
            raise ValueError('Choose one fixed method and one core holdout')
        if job_month(body['job_id'])!=self.control.month:raise ValueError('Use the current admission month')
        if not isinstance(body['method'],str) or body['method'] not in PRESETS:raise ValueError('Unsupported engineering preset')
        job=dict(body,schema_version=1,purpose='engineering_smoke',dataset=self.dataset,parameters=PRESETS[body['method']])
        validate_job(job)
        plan={'procedure':'approved-engineering-v1','job':job,'owner_binding':digest(self.owner.encode()),
              'worker_code_sha256':self.policy,'seed':20260930,
              'limits':{'attempt_seconds':720,'max_attempts':3,'deployment_allowed':False},
              'evaluation':'single core study holdout; preprocessing fitted on remaining core studies; engineering only',
              'compute':'Existing Mac worker; no cloud training. May wait while the Mac is offline.',
              'cost':'Consumes one of 12 monthly jobs (reviewers: 2). Cloud storage/API usage remains metered.'}
        return {'plan':plan,'approval_sha256':digest(encode(plan)),'status':'approval_required'}

    def approve(self,body):
        if not isinstance(body,dict) or set(body)!={'job_id','method','holdout_study','approval_sha256'}:
            raise ValueError('An exact reviewed plan hash is required')
        prepared=self.prepare({k:v for k,v in body.items() if k!='approval_sha256'})
        if prepared['approval_sha256']!=body['approval_sha256']:raise ValueError('Plan changed; review it again')
        job=prepared['plan']['job'];key=BASE+job['job_id']+'.json'
        approval={'plan':prepared['plan'],'approval_sha256':body['approval_sha256'],'owner':self.owner,'approved_at':self.clock()}
        try:self.cloud.state(key,approval,0)
        except Conflict:
            previous,_=self.cloud.read_json(key)
            if any(previous[k]!=approval[k] for k in ('plan','approval_sha256','owner')):
                raise ValueError('Approval ID already bound to another plan')
        # Approval alone is not dispatch. Durable admission precedes queue creation;
        # a retry or the outbound worker can recover an uncertain queue response.
        self.control.admit(job,self.owner,2 if self.reviewer else None,approval_sha256=body['approval_sha256'])
        Queue(self.cloud).submit(job)
        return self.status(job['job_id'])

    def owned(self,job_id):
        control,_=DispatchControl(self.cloud,job_month(job_id)).read()
        entry=control['jobs'].get(job_id)
        if not entry or entry['owner']!=self.owner:raise FileNotFoundError()
        validate_job(entry['request'])
        if entry['request']['dataset']!=self.dataset or digest(encode(entry['request']))!=entry['sha256']:
            raise ValueError('Job/release integrity failure')
        if entry.get('bridge_approval_sha256'):validate_approval(self.cloud,entry)
        return entry

    def status(self,job_id):
        entry=self.owned(job_id);job=entry['request']
        try:state,_=self.cloud.read_json(job_name(job_id))
        except FileNotFoundError:state={'status':'QUEUED','stage':'admitted','attempt':0,'request':job,'request_sha256':entry['sha256']}
        if state['request']!=job or state['request_sha256']!=entry['sha256']:raise ValueError('Queue/request mismatch')
        try:worker,_=self.cloud.read_json(PREFIX+'control/worker.json')
        except FileNotFoundError:worker={}
        answer={'observed_at':self.clock(),'job_id':job_id,'status':state['status'],'stage':state.get('stage'),
                'attempt':state['attempt'],'method':job['method'],'holdout_study':job['holdout_study'],
                'purpose':'engineering_smoke','deployment_allowed':False,'request_sha256':entry['sha256'],
                'approval_sha256':entry.get('bridge_approval_sha256'),
                'worker_last_checked_at':worker.get('checked_at'),
                'notice':'Engineering integration test, not a tuned comparison or a published prediction model. A past worker check-in does not prove current availability.'}
        if state['status']=='QUEUED':answer['execution']='Waiting for the outbound Mac worker; completion is not guaranteed.'
        if state['status']=='CANCELLED':answer['execution']='Completion fenced. An already running local calculation may stop only at its next checkpoint.'
        if state['status']=='SUCCEEDED':
            ref=state['result']
            if not re.fullmatch(re.escape(PREFIX+'artifacts/'+job_id+'/')+r'[a-f0-9-]{36}/result.json',ref['object']):
                raise ValueError('Invalid result reference')
            committed=json.loads(self.cloud.read(ref['object'],ref['generation'],ref['sha256'],limit=250000))
            result=committed['result']
            if (committed['request_sha256']!=entry['sha256'] or result['job_id']!=job_id
                    or result['method']!=job['method'] or result['holdout_study']!=job['holdout_study']
                    or result['purpose']!='engineering_smoke' or result['deployment_allowed'] is not False
                    or result['train_samples']!=145-CORE[job['holdout_study']] or result['validation_samples']!=CORE[job['holdout_study']]):
                raise ValueError('Result/request integrity failure')
            metrics={k:result['metrics'][k] for k in ('mae','rmse','bias')}
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in metrics.values()):raise ValueError('Invalid metrics')
            answer.update(metrics_gy=metrics,train_profiles=result['train_samples'],validation_profiles=result['validation_samples'],
                          checks=result['checks'],result_sha256=ref['sha256'])
        return answer

    def cancel(self,job_id):
        entry=self.owned(job_id)
        Queue(self.cloud).submit(entry['request'])
        Queue(self.cloud).cancel(job_id)
        return self.status(job_id)
