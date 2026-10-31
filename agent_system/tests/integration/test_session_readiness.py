"""First-use SDK work belongs to startup, before accepting HTTP requests."""
import asyncio
import threading
import pytest
from google.auth.credentials import AnonymousCredentials
from research_agents.persistence.managed_sessions import ManagedSessions
from test_chat import setup, headers


def test_readiness_prepares_sessions_before_first_authenticated_request():
    client,service,_,_=setup()
    ready=[]
    async def prepare():
        await asyncio.sleep(.01)
        ready.append(threading.get_ident())
    service.sessions.prepare=prepare
    with client:
        assert len(ready)==1
        assert client.get('/api/copilot/status',headers=headers()).status_code==200
        assert client.get('/api/copilot/status').status_code==403
    assert len(ready)==1


def test_failed_session_preparation_blocks_readiness_and_closes_worker():
    client,service,_,_=setup()
    closed=[]
    async def prepare():raise RuntimeError('SDK initialization failed')
    class Worker:
        async def aclose(self):closed.append(True)
    service.sessions.prepare=prepare
    service.executor=Worker()
    with pytest.raises(RuntimeError,match='SDK initialization failed'):
        with client:pytest.fail('An unprepared service must not start')
    assert closed==[True]


def test_prepared_sdk_surface_reuses_same_loop_client_without_provider_calls(monkeypatch):
    # The entire suite blocks non-loopback sockets. Anonymous credentials also
    # ensure this fixture cannot authenticate a cloud operation.
    service=ManagedSessions(project='fixture-only',location='us-west1',engine_id='0',
                            credentials=AnonymousCredentials())
    async def run():
        await service.prepare()
        client=service._get_api_client()
        assert client.sessions.events is not None
        import importlib
        original=importlib.import_module
        def no_late_sessions(name,*args,**kwargs):
            assert name!='.sessions','Sessions must already be imported'
            return original(name,*args,**kwargs)
        monkeypatch.setattr(importlib,'import_module',no_late_sessions)
        await service.prepare()
        assert service._get_api_client() is client
        await client.aclose()
    asyncio.run(run())
