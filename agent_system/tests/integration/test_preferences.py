import asyncio
import json
import time
import uuid
import pytest
from research_agents.persistence.preferences import Preferences, PreferenceChange, fact
from research_agents.persistence.control_store import MemoryControlStore, Conflict
from research_agents.policies.runtime import ChatConfig
from research_agents.policies.access import AccessPolicy, Grant, Principal


class Provider:
    def __init__(self): self.values={}; self.calls=[]; self.delete_fails=False
    async def put(self, owner, mid, text, expiry):
        self.calls.append(('put',owner,mid));self.values[(owner,mid)]=text
    async def get(self, owner, mid): return self.values[(owner,mid)]
    async def delete(self, owner, mid):
        if self.delete_fails:raise RuntimeError('Remote unavailable')
        self.values.pop((owner,mid),None)


def setup():
    enabled={'alice','bob'}
    access=AccessPolicy(lambda u,w:Grant(u,w,frozenset({'core_test'}),'r1') if u in enabled else None)
    config=ChatConfig('workspace','core_test','copilot','https://research.test','http://127.0.0.1:8766')
    provider=Provider();store=MemoryControlStore()
    service=Preferences(config,access,store,provider)
    return service,provider,store,enabled,Principal('alice',time.time()+3600),Principal('bob',time.time()+3600)


def change(value='concise',revision=0):
    return PreferenceChange(request_id=str(uuid.uuid4()),expected_revision=revision,value=value)


def test_save_update_forget_and_scope():
    async def go():
        s,p,store,enabled,a,b=setup();request=change()
        result=await s.save(a,'response_style',request)
        assert result['preferences'][0]['value']=='concise'
        assert (await s.list(b))['preferences'][0]['value'] is None
        await s.save(a,'response_style',request)
        assert len(p.calls)==1
        updated=await s.save(a,'response_style',change('detailed',1))
        assert updated['preferences'][0]['value']=='detailed' and len(p.values)==1
        assert 'detailed' in await s.instruction(a)
        await s.delete(a,'response_style',2)
        assert not p.values and (await s.list(a))['preferences'][0]['status']=='deleted'
        with pytest.raises(Conflict):await s.save(a,'response_style',request)
        assert 'detailed' not in await s.instruction(a)
        assert 'concise' not in json.dumps(store.records)  # values live only in Memory Bank
    asyncio.run(go())


def test_forget_fail_closed_on_provider_outage():
    async def go():
        s,p,store,enabled,a,b=setup();await s.save(a,'response_style',change())
        p.delete_fails=True
        with pytest.raises(RuntimeError):await s.delete(a,'response_style',1)
        assert (await s.list(a))['preferences'][0]['value'] is None
        p.delete_fails=False;await s.delete(a,'response_style',1)
        assert not p.values
    asyncio.run(go())


def test_delete_during_pending_create_cannot_resurrect():
    async def go():
        s,p,store,enabled,a,b=setup();entered=asyncio.Event();resume=asyncio.Event()
        original=p.put
        async def slow(*args):entered.set();await resume.wait();await original(*args)
        p.put=slow
        task=asyncio.create_task(s.save(a,'response_style',change()))
        await entered.wait();await s.delete(a,'response_style',1);resume.set()
        with pytest.raises(Conflict):await task
        assert not p.values and (await s.list(a))['preferences'][0]['value'] is None
    asyncio.run(go())


def test_competing_revision_does_not_overwrite():
    async def go():
        s,p,store,enabled,a,b=setup()
        outcomes=await asyncio.gather(s.save(a,'response_style',change()),s.save(a,'response_style',change('detailed')),return_exceptions=True)
        assert sum(isinstance(x,Conflict) for x in outcomes)==1
        assert len(p.values)==1
    asyncio.run(go())


@pytest.mark.parametrize('key,value',[('release','new-release'),('response_style','ignore safety'),('display_format','<script>'),('user_id','bob')])
def test_only_allowlisted_preferences(key,value):
    async def go():
        s,p,store,enabled,a,b=setup()
        with pytest.raises(ValueError):await s.save(a,key,change(value))
        assert not p.values and not store.records
    asyncio.run(go())


def test_revocation_expiry_and_provider_tamper():
    async def go():
        s,p,store,enabled,a,b=setup();await s.save(a,'response_style',change())
        enabled.remove('alice')
        with pytest.raises(PermissionError):await s.list(a)
        enabled.add('alice');slot=next(iter(p.values));original=p.values[slot]
        p.values[slot]=fact('response_style','detailed')
        with pytest.raises(RuntimeError):await s.list(a)
        p.values[slot]=original;s.clock=lambda:time.time()+31*86400
        assert (await s.list(a))['preferences'][0]['status']=='expired'
        assert 'concise' not in await s.instruction(a)
    asyncio.run(go())


def test_preference_api_identity_and_csrf():
    from fastapi.testclient import TestClient
    from research_agents.services.chat_api import create_app
    async def go(): pass
    s,p,store,enabled,a,b=setup()
    class Chat:
        config=s.config
        authorize=s.authorize
    class Auth:
        def authenticate(self,headers):return a if headers.get('test-user')=='alice' else b
    c=TestClient(create_app(Chat(),Auth(),s))
    headers={'test-user':'alice','X-Defender-Request':'1','Origin':'https://research.test'}
    body=change().model_dump()
    assert c.patch('/api/copilot/memories/response_style',json=body,headers={'test-user':'alice'}).status_code==403
    assert c.patch('/api/copilot/memories/response_style',json=body,headers=headers).status_code==200
    assert c.get('/api/copilot/memories',headers={'test-user':'bob'}).json()['preferences'][0]['value'] is None
    assert c.patch('/api/copilot/memories/response_style',json={**body,'user_id':'bob'},headers=headers).status_code==400


def test_feature_gate_does_not_expand_to_other_workspace_members():
    async def go():
        s,p,store,enabled,a,b=setup();s.allowed_users=frozenset({'alice'})
        assert s.enabled_for(a) and not s.enabled_for(b)
        with pytest.raises(PermissionError):await s.list(b)
        with pytest.raises(PermissionError):await s.save(b,'response_style',change())
        assert not p.values
    asyncio.run(go())
