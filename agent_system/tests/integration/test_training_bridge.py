"""Synthetic cloud only: approval, dispatch recovery, isolation and fencing."""
import asyncio
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'agent_system/tests/tools')]
from test_prediction_tool import Source
from cloud_io import Conflict,digest,encode
from dispatch_control import DispatchControl,BudgetExceeded
from job_queue import Queue,job_name
from training_bridge import TrainingBridge,WORKER_FILES,BASE,validate_approval
from mac_worker_loop import process_entry
from app.server import DATASET,create_app
from research_agents.tools.training_tool import TrainingTool
from research_agents.policies.request_contract import compile_contract

POLICY={k:'b'*64 for k in WORKER_FILES}
MONTH='2026-10'

def body(n=1,**kwargs):return dict(job_id='web_202610_'+f'{n:032x}',method='median_null',holdout_study='GSE8917',**kwargs)

def approved(bridge,b=None):
    b=b or body();p=bridge.prepare(b)
    return dict(b,approval_sha256=p['approval_sha256'])

@pytest.fixture
def env():
    class AtomicSource(Source):
        def __init__(self):super().__init__();self.lock=threading.RLock()
        def state(self,*a):
            with self.lock:
                super().state(*a)
                return self.rows[a[0]][1]
    source=AtomicSource();bridge=TrainingBridge(source,'alice',DATASET,POLICY,month=MONTH)
    return source,bridge


def test_prepare_is_read_only_exact_preset_and_no_test_study(env):
    source,b=env;p=b.prepare(body());assert not source.rows
    assert p['status']=='approval_required' and p['plan']['limits']['deployment_allowed'] is False
    for change in [{'method':'A'},{'holdout_study':'GSE90909'},{'parameters':{}},{'job_id':'web_202611_'+'a'*32}]:
        with pytest.raises(ValueError):b.prepare(dict(body(),**change))


def test_hash_binds_owner_request_and_worker(env):
    source,b=env;a=approved(b)
    for change in [{'method':'elastic_net'},{'holdout_study':'GSE102971'},{'approval_sha256':'c'*64}]:
        with pytest.raises(ValueError):b.approve(dict(a,**change))
    other=TrainingBridge(source,'bob',DATASET,POLICY,month=MONTH)
    with pytest.raises(ValueError):other.approve(a)
    changed=TrainingBridge(source,'alice',DATASET,{**POLICY,'training_worker.py':'c'*64},month=MONTH)
    with pytest.raises(ValueError):changed.approve(a)
    assert not source.rows


def test_retries_concurrency_one_admission_one_job(env):
    source,b=env;a=approved(b)
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda _:b.approve(a),range(4)))
    assert all(r['status']=='QUEUED' for r in rows)
    assert len(b.control.read()[0]['jobs'])==1
    assert sum(k.startswith('training/v1/jobs/') for k in source.rows)==1
    assert source.rows[job_name(body()['job_id'])][0]['attempt']==0


def test_unknown_postcommit_response_recovers_same_job(env,monkeypatch):
    source,b=env;a=approved(b);state=source.state;lost=[False]
    def loss(key,value,generation):
        result=state(key,value,generation)
        if key.startswith('training/v1/jobs/') and not lost[0]:lost[0]=True;raise TimeoutError()
        return result
    monkeypatch.setattr(source,'state',loss)
    with pytest.raises(TimeoutError):b.approve(a)
    assert b.approve(a)['status']=='QUEUED' and len(b.control.read()[0]['jobs'])==1


def test_worker_repairs_admission_without_duplicate_and_checks_approval(env,monkeypatch):
    source,b=env;b.approve(approved(b));entry=b.control.read()[0]['jobs'][body()['job_id']]
    source.rows.pop(job_name(body()['job_id']));calls=[]
    import training_bridge
    monkeypatch.setattr(training_bridge,'worker_hashes',lambda _:POLICY)
    assert process_entry(source,entry,calls.append)=='PUBLISHED' and calls==[body()['job_id']]
    # Changed code must fail before execute or queue mutation.
    monkeypatch.setattr(training_bridge,'worker_hashes',lambda _:{**POLICY,'training_worker.py':'f'*64})
    with pytest.raises(ValueError):process_entry(source,entry,calls.append)
    assert len(calls)==1


def test_approval_tampering_and_missing_fail_closed(env):
    source,b=env;b.approve(approved(b));entry=b.control.read()[0]['jobs'][body()['job_id']]
    validate_approval(source,entry,POLICY)
    source.rows[BASE+body()['job_id']+'.json'][0]['plan']['limits']['max_attempts']=99
    with pytest.raises(ValueError):validate_approval(source,entry,POLICY)
    source.rows.pop(BASE+body()['job_id']+'.json')
    with pytest.raises(FileNotFoundError):validate_approval(source,entry,POLICY)


def test_owner_isolation_no_read_cancel_or_id_reuse(env):
    source,b=env;b.approve(approved(b));other=TrainingBridge(source,'bob',DATASET,POLICY,month=MONTH)
    for action in (other.status,other.cancel):
        with pytest.raises(FileNotFoundError):action(body()['job_id'])
    with pytest.raises(ValueError):other.approve(approved(other))
    assert b.status(body()['job_id'])['status']=='QUEUED'


def test_cancel_fences_running_worker_and_duplicate_submit(env):
    source,b=env;a=approved(b);b.approve(a);q=Queue(source);state,generation=q.claim(body()['job_id'])
    assert b.cancel(body()['job_id'])['status']=='CANCELLED'
    with pytest.raises(Conflict):q.update(state,generation,'completed',{'object':'ignored'})
    assert b.approve(a)['status']=='CANCELLED'
    assert b.cancel(body()['job_id'])['status']=='CANCELLED'


def test_queue_offline_and_stale_heartbeat_not_online_claim(env):
    source,b=env;b.approve(approved(b));source.state('training/v1/control/worker.json',{'checked_at':1,'status':'waiting'},0)
    r=b.status(body()['job_id']);assert r['status']=='QUEUED' and r['attempt']==0
    assert 'completion is not guaranteed' in r['execution'] and 'does not prove current availability' in r['notice']


def test_reviewer_and_total_quotas_not_reset(env):
    source,_=env;b=TrainingBridge(source,'reviewer',DATASET,POLICY,month=MONTH,reviewer=True)
    for i in range(2):b.approve(approved(b,body(i)))
    with pytest.raises(BudgetExceeded):b.approve(approved(b,body(3)))
    assert len(b.control.read()[0]['jobs'])==2
    for i in range(2,12):
        other=TrainingBridge(source,'owner'+str(i),DATASET,POLICY,month=MONTH);other.approve(approved(other,body(i+10)))
    with pytest.raises(BudgetExceeded):other.approve(approved(other,body(99)))


def complete(env):
    source,b=env;b.approve(approved(b));state,gen=Queue(source).claim(body()['job_id'])
    result=dict(job_id=body()['job_id'],method='median_null',holdout_study='GSE8917',purpose='engineering_smoke',deployment_allowed=False,
                train_samples=120,validation_samples=25,metrics={'mae':1.,'rmse':2.,'bias':-.5},checks={'fold_isolation':True})
    raw=encode({'result':result,'request_sha256':state['request_sha256']});ref=dict(object='training/v1/artifacts/'+body()['job_id']+'/'+'a'*8+'-'+'b'*4+'-'+'c'*4+'-'+'d'*4+'-'+'e'*12+'/result.json',generation='1',sha256=digest(raw))
    source.read=lambda name,generation,expected_sha,limit:raw if digest(raw)==expected_sha else (_ for _ in ()).throw(ValueError())
    Queue(source).update(state,gen,'completed',ref);return raw


def test_completed_result_bound_to_request_and_never_deployed(env):
    raw=complete(env);source,b=env;r=b.status(body()['job_id'])
    assert r['metrics_gy']['mae']==1 and r['deployment_allowed'] is False
    assert b.cancel(body()['job_id'])['status']=='SUCCEEDED'
    source.rows[job_name(body()['job_id'])][0]['request_sha256']='c'*64
    with pytest.raises(ValueError):b.status(body()['job_id'])


def test_http_auth_csrf_pause_and_exact_approval(env,monkeypatch,tmp_path):
    # Match the fixture job IDs without depending on the month the suite runs.
    monkeypatch.setattr('dispatch_control.month_key',lambda now=None:MONTH)
    source,b=env;policy=tmp_path/'policy.json';policy.write_text(json.dumps(POLICY))
    monkeypatch.setenv('TRAINING_BRIDGE_ENABLED','true');monkeypatch.setenv('TRAINING_BRIDGE_POLICY',str(policy))
    def identity(headers):
        if headers.get('X-Test-User') not in ('alice','bob'):raise PermissionError()
        return headers['X-Test-User']
    client=create_app(lambda **kw:source,identity).test_client()
    h={'X-Test-User':'alice','X-Defender-Request':'1','Origin':'https://localhost'}
    assert client.get('/api/training/catalog').status_code==403
    assert client.post('/api/training/prepare',json=body(),headers={'X-Test-User':'alice'}).status_code==403
    stale=client.post('/api/training/prepare',json=dict(body(),job_id='web_202609_'+'a'*32),headers=h)
    assert stale.status_code==400 and 'current admission month' in stale.json['error']
    plan=client.post('/api/training/prepare',json=body(),headers=h);assert plan.status_code==200
    a=dict(body(),approval_sha256=plan.json['approval_sha256'])
    monkeypatch.setenv('TRAINING_ENABLED','false');assert client.post('/api/training/approve',json=a,headers=h).status_code==403
    monkeypatch.setenv('TRAINING_ENABLED','true');assert client.post('/api/training/approve',json=a,headers=h).status_code==202
    assert client.get('/api/training/jobs/'+body()['job_id'],headers={'X-Test-User':'bob'}).status_code==404
    assert client.post('/api/training/jobs/'+body()['job_id']+'/cancel',json={},headers=h).json['status']=='CANCELLED'


def test_tool_read_only_explicit_current_id_and_receipt(env):
    source,b=env;b.approve(approved(b));receipts=[];scope=SimpleNamespace(principal=SimpleNamespace(user_id='alice'),run_id='r',fence='f',contract={})
    class Guard:
        ledger=SimpleNamespace(reserve=lambda *a:None,progress=lambda *a:receipts.append(a[-1]))
        async def authorize(self):return scope
    tool=TrainingTool(lambda owner:TrainingBridge(source,owner,DATASET,POLICY,month=MONTH),Guard(),lambda p:p.user_id)
    assert asyncio.run(tool.run(body()['job_id']))['status']=='clarification_required'
    scope.contract=compile_contract('Status for '+body()['job_id']);before=copy.deepcopy(source.rows)
    assert asyncio.run(tool.run(body()['job_id']))['answer']['status']=='QUEUED'
    assert source.rows==before and receipts[0]['tool']=='training_status'
    scope.principal.user_id='bob';assert asyncio.run(tool.run(body()['job_id']))['status']=='not_found'
