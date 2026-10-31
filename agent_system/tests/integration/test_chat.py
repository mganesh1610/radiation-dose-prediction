"""Offline identity/storage doubles; no claim of actual reviewer or model access."""
import asyncio
import time
import uuid
import pytest
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event
from google.genai import types
from fastapi.testclient import TestClient
from research_agents.policies.access import AccessPolicy,Grant,Principal
from research_agents.policies.runtime import ChatConfig
from research_agents.policies.run_scope import session_user
from research_agents.persistence.control_store import MemoryControlStore,Conflict,QuotaExceeded
from research_agents.persistence.ledger import Ledger
from research_agents.services.chat_service import ChatService
from research_agents.services.chat_api import create_app


def key():return uuid.uuid4().hex


class TestSessions(InMemorySessionService):
    __test__ = False
    async def create_session(self, *, expire_time=None, ttl=None, **kwargs):
        return await super().create_session(**kwargs)


def setup(executor=None):
    sessions=TestSessions();store=MemoryControlStore();ledger=Ledger(store,'workspace')
    enabled={'alice','bob'};calls=[]
    access=AccessPolicy(lambda u,w:Grant(u,w,frozenset({'core_test'}),'r1') if u in enabled and w=='workspace' else None)
    config=ChatConfig('workspace','core_test','copilot','https://research.test','http://127.0.0.1:8766')
    async def execute(scope,session):
        calls.append(scope.run_id)
        await sessions.append_event(session,Event(author='research_coordinator',invocation_id=scope.run_id,
            content=types.Content(role='model',parts=[types.Part(text='Fixture answer')]),timestamp=time.time()))
    service=ChatService(config,ledger,sessions,access,executor or execute)
    class Auth:
        def authenticate(self,headers):
            name=headers.get('test-identity')
            if name not in enabled:raise PermissionError()
            return Principal(name,time.time()+600)
    client=TestClient(create_app(service,Auth()))
    return client,service,calls,enabled


def headers(user='alice'):
    return {'test-identity':user,'X-Defender-Request':'1','Origin':'https://research.test'}


def create(client,user='alice',request_id=None):
    r=client.post('/api/copilot/chats',json={'request_id':request_id or key()},headers=headers(user))
    assert r.status_code==200,r.text
    return r.json()


def test_chat_resume_exactly_once_and_owner_isolation():
    client,service,calls,_=setup();k=key();chat=create(client,request_id=k)
    assert create(client,request_id=k)==chat
    url=f"/api/copilot/chats/{chat['chat_id']}/messages";request={'request_id':key(),'text':'What is the recorded dose?'}
    first=client.post(url,json=request,headers=headers());assert first.status_code==200,first.text
    assert first.json()['status']=='completed'
    second=client.post(url,json=request,headers=headers());assert second.json()==first.json();assert len(calls)==1
    history=client.get(url,headers=headers()).json()
    assert [m['role'] for m in history['messages']]==['user','assistant']
    assert client.get(url,headers=headers('bob')).status_code==404
    assert client.get('/api/copilot/chats',headers=headers('bob')).json()=={'chats':[]}
    rid=first.json()['run_id']
    for suffix in ['', '/events','/receipt']:
        assert client.get('/api/copilot/runs/'+rid+suffix,headers=headers('bob')).status_code==404
    assert client.post(url,json={**request,'text':'Different request'},headers=headers()).status_code==409
    restarted=ChatService(service.config,Ledger(service.ledger.store,'workspace'),service.sessions,service.access,service.executor)
    messages=asyncio.run(restarted.messages(Principal('alice',time.time()+60),chat['chat_id']))
    assert messages['messages']==history['messages']


@pytest.mark.parametrize('payload',[{'user_id':'bob'},{'workspace_id':'elsewhere'},{'session_id':'other'},{'plan':{}},{'text':'x'*2001}])
def test_model_and_browser_cannot_inject_scope(payload):
    c,_,_,_=setup();chat=create(c)
    body={'request_id':key(),'text':'Question',**payload}
    assert c.post(f"/api/copilot/chats/{chat['chat_id']}/messages",json=body,headers=headers()).status_code==400


def test_auth_csrf_and_revocation():
    c,_,_,enabled=setup()
    assert c.get('/api/copilot/chats').status_code==403
    assert c.post('/api/copilot/chats',json={'request_id':key()},headers={'test-identity':'alice'}).status_code==403
    h={**headers(),'Origin':'https://evil.test'}
    assert c.post('/api/copilot/chats',json={'request_id':key()},headers=h).status_code==403
    chat=create(c);enabled.remove('alice')
    assert c.get(f"/api/copilot/chats/{chat['chat_id']}/messages",headers=headers()).status_code==403


def test_deletion_tombstone_blocks_history_and_retries():
    c,service,_,_=setup();chat=create(c);url=f"/api/copilot/chats/{chat['chat_id']}"
    assert c.request('DELETE',url,json={},headers=headers('bob')).status_code==404
    assert c.request('DELETE',url,json={},headers=headers()).status_code==200
    assert c.get(url+'/messages',headers=headers()).status_code==404
    assert c.post(url+'/messages',json={'request_id':key(),'text':'x'},headers=headers()).status_code==404
    assert c.get('/api/copilot/chats',headers=headers()).json()=={'chats':[]}


def test_expired_run_is_interrupted_without_replay():
    c,service,calls,_=setup();chat=create(c)
    run=service.ledger.admit('alice',chat['chat_id'],key(),'question');service.ledger.claim('alice',run['run_id'])
    service.ledger.clock=lambda:time.time()+181
    with pytest.raises(Conflict):service.ledger.check('alice',run['run_id'],run['fence'])
    new=service.ledger.admit('alice',chat['chat_id'],key(),'next')
    assert new['run_id']!=run['run_id'] and service.ledger.run('alice',run['run_id'])['status']=='interrupted'
    assert calls==[]


def test_same_chat_lease_and_global_limits():
    c,s,_,_=setup();a=create(c);b=create(c,'bob');third=create(c)
    r=s.ledger.admit('alice',a['chat_id'],key(),'one')
    with pytest.raises(Conflict):s.ledger.admit('alice',a['chat_id'],key(),'second')
    s.ledger.admit('bob',b['chat_id'],key(),'other')
    with pytest.raises(QuotaExceeded):s.ledger.admit('alice',third['chat_id'],key(),'third')
    for i in range(6):s.ledger.reserve('alice',r['run_id'],r['fence'],'model',input_bytes=100)
    with pytest.raises(QuotaExceeded):s.ledger.reserve('alice',r['run_id'],r['fence'],'model')
    s.ledger.cancel('alice',r['run_id'])
    with pytest.raises(Conflict):s.ledger.check('alice',r['run_id'],r['fence'])
    with pytest.raises(Conflict):s.ledger.mark_deleted('alice',a['chat_id'])


def test_session_namespace_separates_users_and_workspaces():
    assert len({session_user('w1','a'),session_user('w2','a'),session_user('w1','b')})==3


def test_partial_and_thought_text_not_shown():
    c,s,_,_=setup();chat=create(c)
    async def add():
        _,session=await s.session(Principal('alice',time.time()+60),chat['chat_id'])
        await s.sessions.append_event(session,Event(author='model',invocation_id='i',
            content=types.Content(parts=[types.Part(text='private thought',thought=True),types.Part(text='Visible')]),timestamp=time.time()))
    asyncio.run(add())
    rows=c.get(f"/api/copilot/chats/{chat['chat_id']}/messages",headers=headers()).json()['messages']
    assert [r['text'] for r in rows]==['Visible']


def test_executor_failure_stays_failed_on_retry():
    async def fail(scope,session):raise RuntimeError('credential-like provider detail must not reach UI')
    c,s,_,_=setup(fail);chat=create(c);body={'request_id':key(),'text':'question'}
    path=f"/api/copilot/chats/{chat['chat_id']}/messages"
    first=c.post(path,json=body,headers=headers());assert first.json()['status']=='failed'
    assert 'credential-like' not in first.text
    assert c.post(path,json=body,headers=headers()).json()==first.json()
