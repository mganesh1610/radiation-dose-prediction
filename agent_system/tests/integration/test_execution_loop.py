import asyncio
import threading
import time
import httpx
import pytest
from research_agents.services.execution_loop import AgentExecutionLoop
from research_agents.policies.run_scope import current_run,RunScope
from research_agents.policies.access import Principal
from test_chat import setup,key,headers


def test_blocking_agent_startup_does_not_starve_authenticated_http_controls():
    async def go():
        entered=threading.Event();finished=threading.Event();seen=[]
        async def sdk(scope,session):
            assert current_run.get() is scope
            seen.append(threading.get_ident());entered.set()
            try:
                time.sleep(.8)  # Reproduce synchronous startup/I/O inside SDK.
                await asyncio.Event().wait()
            finally:finished.set()
        executor=AgentExecutionLoop(sdk)
        client,service,_,_=setup(executor)
        p=Principal('alice',time.time()+60)
        chat=await service.create(p,key())
        task=asyncio.create_task(service.send(p,chat['chat_id'],key(),'fixture'))
        try:
            for _ in range(100):
                if entered.is_set():break
                await asyncio.sleep(.01)
            assert entered.is_set() and seen[0]!=threading.get_ident()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app),base_url='https://research.test') as api:
                start=time.monotonic()
                assert (await api.get('/api/copilot/status',headers=headers())).status_code==200
                assert time.monotonic()-start<.4
                run=service.ledger.chat('alice',chat['chat_id'])['active_run']
                assert (await api.post('/api/copilot/runs/'+run+'/cancel',json={},headers=headers('bob'))).status_code==404
                start=time.monotonic()
                result=await api.post('/api/copilot/runs/'+run+'/cancel',json={},headers=headers())
                assert result.status_code==200 and result.json()['cancel_requested']
                assert time.monotonic()-start<.4
            assert (await asyncio.wait_for(task,3))['status']=='cancelled'
            assert finished.is_set()
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True);await executor.aclose()
        assert not executor._thread.is_alive()
    asyncio.run(go())


def test_worker_contexts_exceptions_and_close_are_scoped():
    async def go():
        users=[]
        async def sdk(scope,session):
            await asyncio.sleep(.01)
            users.append(current_run.get().principal.user_id)
            if scope.principal.user_id=='bob':raise ValueError('fixture failure')
            return 'ok'
        executor=AgentExecutionLoop(sdk)
        async def invoke(user):
            scope=RunScope(Principal(user,time.time()+30),'w','run-'+user,'chat-'+user,'f','s','e')
            token=current_run.set(scope)
            try:return await executor(scope,None)
            finally:current_run.reset(token)
        try:
            results=await asyncio.gather(invoke('alice'),invoke('bob'),return_exceptions=True)
            assert results[0]=='ok' and isinstance(results[1],ValueError)
            assert sorted(users)==['alice','bob']
        finally:await executor.aclose()
        with pytest.raises(RuntimeError):await executor(None,None)
    asyncio.run(go())
