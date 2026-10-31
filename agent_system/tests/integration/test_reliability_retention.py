"""Controlled replica/cleanup failures. No live crash or provider-erasure claim."""
import asyncio
import copy
import time
import pytest
from test_chat import setup, create, key
from test_preferences import setup as preference_setup, change
from research_agents.services.chat_service import ChatService
from research_agents.persistence.ledger import Ledger
from research_agents.persistence.preferences import Preferences
from research_agents.persistence.control_store import Conflict
from research_agents.policies.access import Principal


def test_two_replicas_share_inflight_fence_and_reject_competing_turn():
    async def go():
        entered=asyncio.Event();release=asyncio.Event();calls=[]
        async def execute(scope,session):
            calls.append(scope.run_id);entered.set();await release.wait()
        _,a,_,_=setup(execute)
        b=ChatService(a.config,Ledger(a.ledger.store,'workspace'),a.sessions,a.access,execute)
        principal=Principal('alice',time.time()+600);chat=await a.create(principal,key());request=key()
        first=asyncio.create_task(a.send(principal,chat['chat_id'],request,'fixture'))
        await entered.wait()
        same=await b.send(principal,chat['chat_id'],request,'fixture')
        assert same['status']=='running' and same['run_id']==calls[0]
        with pytest.raises(Conflict):await b.send(principal,chat['chat_id'],key(),'different')
        release.set();finished=await first
        assert finished['status']=='completed' and len(calls)==1
        assert await b.send(principal,chat['chat_id'],request,'fixture')==finished
        messages=await b.messages(principal,chat['chat_id'])
        assert len(messages['messages'])==1 # One user event; fixture executor produces no text.
    asyncio.run(go())


def test_restarted_reader_closes_expired_run_without_executing_again(monkeypatch):
    async def go():
        _,a,calls,_=setup();p=Principal('alice',time.time()+600);chat=await a.create(p,key());request=key()
        run=a.ledger.admit('alice',chat['chat_id'],request,'fixture');a.ledger.claim('alice',run['run_id'])
        a.ledger.change(lambda s:s['runs'][run['run_id']].update(deadline=time.time()-1))
        b=ChatService(a.config,Ledger(a.ledger.store,'workspace'),a.sessions,a.access,a.executor)
        current=await b.get_run(p,run['run_id'])
        assert current['status']=='interrupted' and calls==[]
        replay=await b.send(p,chat['chat_id'],request,'fixture')
        assert replay['status']=='interrupted' and calls==[]
        assert b.ledger.read()['chats'][chat['chat_id']]['active_run'] is None
    asyncio.run(go())


def test_chat_expiry_exact_boundary_hides_history_and_runs_but_preserves_audit():
    async def go():
        _,s,_,_=setup();p=Principal('alice',time.time()+600);c=await s.create(p,key())
        r=await s.send(p,c['chat_id'],key(),'fixture');before=copy.deepcopy(s.ledger.read())
        s.ledger.clock=lambda:c['expires_at']-0.001
        assert (await s.messages(p,c['chat_id']))['messages']
        s.ledger.clock=lambda:c['expires_at']
        with pytest.raises(FileNotFoundError):await s.messages(p,c['chat_id'])
        with pytest.raises(FileNotFoundError):await s.get_run(p,r['run_id'])
        assert await s.list(p)==[] and s.ledger.read()==before
    asyncio.run(go())


def test_failed_session_cleanup_stays_hidden_across_restart_then_retries():
    async def go():
        _,s,_,_=setup();p=Principal('alice',time.time()+600);c=await s.create(p,key())
        await s.send(p,c['chat_id'],key(),'fixture');original=s.sessions.delete_session
        async def outage(**kw):raise RuntimeError('provider temporarily unavailable')
        s.sessions.delete_session=outage
        with pytest.raises(RuntimeError):await s.delete(p,c['chat_id'])
        b=ChatService(s.config,Ledger(s.ledger.store,'workspace'),s.sessions,s.access,s.executor)
        assert await b.list(p)==[]
        with pytest.raises(FileNotFoundError):await b.messages(p,c['chat_id'])
        s.sessions.delete_session=original
        assert (await b.delete(p,c['chat_id']))['deleted']
        assert (await b.delete(p,c['chat_id']))['deleted']
        assert len(b.ledger.read()['runs'])==1 # Audit retained, not counted as erased.
    asyncio.run(go())


def test_memory_expiry_and_forget_failure_survive_service_recreation():
    async def go():
        s,provider,store,_,a,b=preference_setup()
        result=await s.save(a,'display_format',change('table'))
        expires=next(i['expires_at'] for i in result['preferences'] if i['key']=='display_format')
        restarted=Preferences(s.config,s.access,store,provider,clock=lambda:expires)
        assert next(i for i in (await restarted.list(a))['preferences'] if i['key']=='display_format')['status']=='expired'
        provider.delete_fails=True
        with pytest.raises(RuntimeError):await restarted.delete(a,'display_format',1)
        again=Preferences(s.config,s.access,store,provider)
        item=next(i for i in (await again.list(a))['preferences'] if i['key']=='display_format')
        assert item['status']=='deleted' and item['value'] is None
        assert all(i['value'] is None for i in (await again.list(b))['preferences'])
        provider.delete_fails=False;await again.delete(a,'display_format',1)
        assert not provider.values
    asyncio.run(go())
