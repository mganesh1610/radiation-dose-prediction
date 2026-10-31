"""Blocked SDK startup must not block authenticated control operations."""
import asyncio
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from research_agents.policies.access import Principal
from research_agents.services.background_runtime import BackgroundRuntime, DeferredSessions, DeferredExecutor
from research_agents.services.chat_service import ChatService
from research_agents.services.chat_api import create_app
from test_chat import setup, headers, key


def test_control_entrypoint_imports_without_agent_sdk_or_network():
    code = '''
import socket,sys,time
def blocked(*a,**kw):raise AssertionError('Import tried network I/O')
socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
import research_agents.services.control_bootstrap
assert not any(x=='google.adk' or x.startswith('google.adk.') for x in sys.modules)
assert 'agentplatform' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True, timeout=20,
                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})


def fixture(*, fail=False):
    _, actual, calls, enabled = setup()
    started, release = threading.Event(), threading.Event()
    loads, closes = [], []
    original = actual.executor
    class Executor:
        async def __call__(self, scope, session):return await original(scope, session)
        async def aclose(self):closes.append(True)
    actual.executor = Executor()
    async def prepare():pass
    actual.sessions.prepare = prepare
    def loader():
        loads.append(True); started.set()
        if not release.wait(5):raise RuntimeError('Test did not release startup')
        if fail:raise ValueError('Private diagnostic should not enter HTTP responses')
        return SimpleNamespace(config=actual.config, sessions=actual.sessions, chat_service=lambda: actual)
    runtime = BackgroundRuntime(loader, expected_config=actual.config, wait_seconds=3)
    facade = ChatService(actual.config, actual.ledger, DeferredSessions(runtime), actual.access, DeferredExecutor(runtime))
    class Auth:
        def authenticate(self, values):
            user = values.get('test-identity')
            if user not in enabled:raise PermissionError()
            return Principal(user, time.time()+60)
    client = TestClient(create_app(facade, Auth()))
    return client, actual, runtime, started, release, loads, closes, calls


def test_controls_and_ownership_work_before_execution_is_ready():
    client, actual, runtime, started, release, loads, closes, calls = fixture()
    p = Principal('alice', time.time()+60)
    chat = asyncio.run(actual.create(p, key()))
    prior = actual.ledger.admit('alice', chat['chat_id'], key(), 'Old active request')
    actual.ledger.claim('alice', prior['run_id'])
    try:
        with client:
            assert started.wait(1) and not release.is_set()
            assert client.get('/api/copilot/status').status_code == 403
            status = client.get('/api/copilot/status', headers=headers()).json()
            assert status['enabled'] and status['execution_ready'] is False and status['startup_state'] == 'warming'
            assert client.get('/healthz').json()['execution_ready'] is False
            path = '/api/copilot/runs/' + prior['run_id']
            receipt = client.get(path + '/receipt', headers=headers())
            assert receipt.status_code == 200 and receipt.json()['model_calls'] == 0
            for suffix in ['', '/events', '/receipt']:
                denied = client.get(path + suffix, headers=headers('bob'))
                assert denied.status_code == 404 and denied.json() == {'error': 'Record not found'}
            cancelled = client.post(path + '/cancel', json={}, headers=headers())
            assert cancelled.status_code == 200 and cancelled.json()['cancel_requested']
            assert not release.is_set() and runtime.service is None and not calls and len(loads) == 1
            release.set()
            assert client.get('/api/copilot/chats', headers=headers()).status_code == 200
            ready = client.get('/api/copilot/status', headers=headers()).json()
            assert ready['execution_ready'] is True and ready['startup_state'] == 'ready'
            assert len(loads) == 1 and not calls
    finally:release.set()
    assert len(closes) == 1


def test_private_runtime_health_has_only_readiness_and_no_user_data():
    client,actual,runtime,started,release,loads,closes,calls=fixture()
    baseline=actual.ledger.read()
    try:
        with client:
            assert started.wait(1)
            response=client.get('/api/copilot/runtime-health')
            assert response.status_code==200
            assert response.json()=={'status':'ok','execution_ready':False,'startup_state':'warming'}
            assert client.get('/api/copilot/status').status_code==403
            assert client.post('/api/copilot/runtime-health',json={}).status_code==403
            assert actual.ledger.read()==baseline and not calls
            release.set()
    finally:release.set()


def test_concurrent_history_reads_wait_for_one_startup_without_blocking_status():
    client, actual, runtime, started, release, loads, closes, calls = fixture()
    p = Principal('alice', time.time()+60)
    chat = asyncio.run(actual.create(p, key()))
    baseline = actual.ledger.read()
    try:
        with client, concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            assert started.wait(1)
            reads = [pool.submit(client.get, '/api/copilot/chats/'+chat['chat_id']+'/messages', headers=headers()) for _ in range(2)]
            status = client.get('/api/copilot/status', headers=headers())
            assert status.status_code == 200 and not status.json()['execution_ready']
            assert not release.is_set() and all(not r.done() for r in reads)
            assert actual.ledger.read() == baseline
            release.set()
            assert all(r.result(timeout=3).status_code == 200 for r in reads)
            assert len(loads) == 1 and not calls and actual.ledger.read() == baseline
    finally:release.set()
    assert len(closes) == 1


def test_failed_startup_does_not_retry_or_admit_a_model_run():
    client, actual, runtime, started, release, loads, closes, calls = fixture(fail=True)
    baseline = actual.ledger.read()
    try:
        with client:
            assert started.wait(1); release.set()
            response = client.post('/api/copilot/chats', json={'request_id': key()}, headers=headers())
            assert response.status_code == 503 and 'Private diagnostic' not in response.text
            for _ in range(2):
                status = client.get('/api/copilot/status', headers=headers()).json()
                assert status['startup_state'] == 'failed' and not status['execution_ready']
            assert len(loads) == 1 and not calls and actual.ledger.read() == baseline
    finally:release.set()
    assert not closes
