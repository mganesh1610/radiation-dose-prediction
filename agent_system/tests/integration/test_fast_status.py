"""Thin status still validates every identity/grant and delegates original data RPCs."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import pytest
from starlette.testclient import TestClient
from research_agents.policies.access import IAP_ISSUER,user_key
from research_agents.policies.runtime import ChatConfig
from research_agents.services.fast_status import FastStatusApp,LiveBindings


def fixture(*,fail=False,gate=None):
    config=ChatConfig('test','core','chat','https://ui.test','https://evidence.run.app')
    settings={'iap_audience':'/projects/1/locations/global/backendServices/1','analysis_enabled':True,'memory_enabled':True,'protocol_rag_enabled':True}
    policy_data={'workspace':'test','revision':'p1','releases':['core'],'members':[{'email':u+'@example.test','enabled':True} for u in ['alice','bob']]}
    class Policy:
        def read(self):return copy.deepcopy(policy_data)
        def emails(self):return [m['email'] for m in policy_data['members'] if m['enabled']]
    class Bindings:
        def __init__(self):
            self.records={user_key(IAP_ISSUER,u):{'user_id':user_key(IAP_ISSUER,u),'email':u+'@example.test'} for u in ['alice','bob']};self.reads=0;self.closed=False
        def read(self,user):self.reads+=1;return copy.deepcopy(self.records.get(user))
        def close(self):self.closed=True
    bindings=Bindings();loads=[];events=[];started=threading.Event();ready={'execution_ready':False,'startup_state':'warming'}
    def verify(token,audience):
        if token not in ['alice','bob']:raise ValueError('private verifier detail')
        return {'iss':IAP_ISSUER,'aud':audience,'sub':token,'email':token+'@example.test','exp':time.time()+60,'iat':time.time()}
    @asynccontextmanager
    async def lifespan(app):
        events.append('startup')
        try:yield
        finally:events.append('shutdown')
    class Full:
        state=SimpleNamespace(protected_chat_config=config,execution_readiness=lambda:ready.copy())
        router=SimpleNamespace(lifespan_context=lifespan)
        async def __call__(self,scope,receive,send):
            body=b''
            while True:
                m=await receive();body+=m.get('body',b'')
                if not m.get('more_body'):break
            payload={'delegated':scope['path'],'method':scope['method'],'body':body.decode(),'headers':dict((k.decode(),v.decode()) for k,v in scope['headers'])}
            data=json.dumps(payload).encode();await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]});await send({'type':'http.response.body','body':data})
    def load():
        loads.append(True);started.set()
        if gate is not None:assert gate.wait(3)
        if fail:raise ValueError('private loader error')
        return Full()
    app=FastStatusApp(settings,config,Policy(),[user_key(IAP_ISSUER,'alice')],bindings,verify,load)
    return app,bindings,policy_data,loads,events,started,ready


def headers(user='alice'):return {'X-Research-IAP-Assertion':user}


def test_bound_status_reads_live_binding_every_time_without_loading_framework():
    app,bindings,policy,loads,events,started,ready=fixture()
    with TestClient(app) as client:
        for _ in range(2):
            r=client.get('/api/copilot/status',headers=headers());assert r.status_code==200
            assert not r.json()['execution_ready'] and r.json()['startup_state']=='warming'
            assert r.headers['cache-control']=='no-store' and 'cp_auth' in r.headers['server-timing']
        assert bindings.reads==2 and not loads
        assert client.get('/api/copilot/status',headers=headers('bob')).json()['memory_enabled'] is False
        assert client.get('/api/copilot/status',headers=headers()).json()['memory_enabled'] is True
    assert bindings.closed and not events


@pytest.mark.parametrize('change',['disable','release','workspace','binding_email','binding_user','extra_field','delete_then_bad_token'])
def test_revocation_and_binding_changes_do_not_use_cached_authorization(change):
    app,b,p,loads,*_=fixture()
    with TestClient(app) as c:
        assert c.get('/api/copilot/status',headers=headers()).status_code==200
        key=user_key(IAP_ISSUER,'alice')
        if change=='disable':p['members'][0]['enabled']=False
        elif change=='release':p['releases']=[]
        elif change=='workspace':p['workspace']='elsewhere'
        elif change=='binding_email':b.records[key]['email']='bob@example.test'
        elif change=='binding_user':b.records[key]['user_id']=user_key(IAP_ISSUER,'bob')
        elif change=='extra_field':b.records[key]['unknown']='value'
        else:del b.records[key]
        r=c.get('/api/copilot/status',headers=headers('invalid' if change=='delete_then_bad_token' else 'alice'))
        assert r.status_code==403 and r.json()=={'error':'Access denied'} and not loads


def test_missing_binding_uses_original_application_without_creating_a_fast_binding():
    app,b,p,loads,events,*_=fixture();b.records.pop(user_key(IAP_ISSUER,'alice'))
    with TestClient(app) as c:
        r=c.get('/api/copilot/status',headers=headers());assert r.status_code==200 and r.json()['delegated']=='/api/copilot/status'
        assert r.json()['headers']['x-research-iap-assertion']=='alice'
        assert len(loads)==1 and user_key(IAP_ISSUER,'alice') not in b.records
    assert events==['startup','shutdown']


def test_first_data_requests_initialize_once_without_blocking_status_or_losing_body():
    gate=threading.Event();app,b,p,loads,events,started,ready=fixture(gate=gate)
    try:
        with TestClient(app) as c,ThreadPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(c.post,'/api/copilot/chats',headers=headers(),json={'request_id':'fixture'}) for _ in range(2)]
            assert started.wait(1)
            r=c.get('/api/copilot/status',headers=headers());assert r.status_code==200 and not r.json()['execution_ready']
            assert c.get('/api/copilot/runtime-health').json()=={'status':'ok','execution_ready':False,'startup_state':'warming'}
            assert len(loads)==1 and all(not f.done() for f in pending)
            gate.set()
            for f in pending:
                r=f.result(2);assert r.status_code==200 and json.loads(r.json()['body'])=={'request_id':'fixture'}
            ready.update(execution_ready=True,startup_state='ready')
            assert c.get('/api/copilot/status',headers=headers()).json()['execution_ready'] is True
    finally:gate.set()
    assert events==['startup','shutdown']


def test_loader_failure_is_sanitized_and_not_retried():
    app,b,p,loads,*_=fixture(fail=True)
    with TestClient(app) as c:
        for _ in range(2):
            r=c.get('/api/copilot/chats',headers=headers());assert r.status_code==503 and 'private' not in r.text
        r=c.get('/api/copilot/status',headers=headers());assert r.status_code==200 and r.json()['startup_state']=='failed'
        assert len(loads)==1


def test_missing_or_invalid_signature_denied_before_binding_read():
    app,b,p,loads,*_=fixture()
    with TestClient(app) as c:
        assert c.get('/api/copilot/status').status_code==403
        assert c.get('/api/copilot/status',headers=headers('forged')).status_code==403
        assert b.reads==0 and not loads


def test_entrypoint_import_has_no_framework_storage_sdk_or_network():
    code='''
import socket,sys
def blocked(*a,**k):raise AssertionError('networkduringimport')
socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
import research_agents.services.fast_status
assert not any(k=='fastapi' or k.startswith('fastapi.') or k=='google.cloud.storage' or k.startswith('google.cloud.storage.') or k=='google.adk' or k.startswith('google.adk.') for k in sys.modules)
'''
    subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=10)


@pytest.mark.parametrize('kind',['ok','missing','forbidden','malformed','oversize','not_object'])
def test_live_binding_reader_is_bounded_fresh_scoped_and_read_only(kind):
    import io
    user=user_key(IAP_ISSUER,'alice');calls=[];closes=[]
    class Raw:
        def read(self,size,decode_content):
            assert size==1_000_001 and decode_content
            return (b'x'*size if kind=='oversize' else b'bad' if kind=='malformed' else b'[]' if kind=='not_object' else json.dumps({'user_id':user,'email':'alice@example.test'}).encode())
    class Reply:
        status_code=404 if kind=='missing' else 403 if kind=='forbidden' else 200
        raw=Raw()
        def __enter__(self):return self
        def __exit__(self,*args):closes.append(True)
    class Session:
        def get(self,url,**kwargs):calls.append((url,kwargs));return Reply()
        def close(self):pass
    reader=LiveBindings('protected-bucket','test',None);reader.local.session=Session()
    if kind in ['forbidden','malformed','oversize','not_object']:
        with pytest.raises(Exception):reader.read(user)
    else:
        value=reader.read(user);assert value is None if kind=='missing' else value['user_id']==user
        reader.read(user);assert len(calls)==2
    assert all('/copilot%2Fv1%2Ftest%2Fidentities%2F' in u or 'copilot%2Fv1%2Ftest%2Fidentities%2F' in u for u,k in calls)
    assert all(k['headers']['Cache-Control']=='no-cache' and k['headers']['Accept-Encoding']=='identity' and k['allow_redirects'] is False and k['timeout']==(2,5) for u,k in calls)
    assert closes
