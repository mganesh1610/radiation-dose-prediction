"""Opt-in disposable Cloud Run Job crash probe; never an application endpoint.

Requires current image settings, a separately named workspace and at most one
platform retry. Uses real GCS/managed Sessions and the unchanged ChatService;
the executor is deterministic and makes no model or scientific-data call.
"""
import asyncio
import json
import os
from pathlib import Path
import re
import signal
import time
import uuid


async def main():
    if os.getpid() == 1:
        raise RuntimeError('Run this fixture as a supervised application child; PID 1 ignores self SIGKILL')
    execution = os.environ['CLOUD_RUN_EXECUTION']
    attempt = int(os.environ['CLOUD_RUN_TASK_ATTEMPT'])
    workspace = os.environ['RECOVERY_WORKSPACE']
    if not re.fullmatch(r'recovery_20261030_[a-f0-9]{12}', workspace):
        raise ValueError('A disposable recovery workspace is required')
    if attempt not in (0, 1):
        raise ValueError('At most one platform retry is permitted')
    from google.cloud import storage
    from google.adk.events import Event
    from google.genai import types
    from research_agents.persistence.control_store import GcsControlStore, mutate
    import google.auth
    from research_agents.persistence.ledger import Ledger
    from research_agents.persistence.managed_sessions import ManagedSessions
    from research_agents.policies.access import AccessPolicy, Grant, Principal
    from research_agents.policies.run_scope import session_user
    from research_agents.policies.runtime import ChatConfig
    from research_agents.services.chat_service import ChatService

    settings = json.loads(Path(os.environ['COPILOT_CONFIG']).read_text())
    assert workspace != settings['chat']['workspace']
    credentials = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])[0]
    backing = GcsControlStore(settings['control_bucket'], 'copilot/v1/' + settings['chat']['workspace'] + '/',
        client=storage.Client(project=settings['project'], credentials=credentials))
    class FixtureStore:
        # Existing IAM boundary; all fixture keys live in a new isolated subdirectory.
        def read(self, key):
            return backing.read('recovery/' + workspace + '/' + key)
        def write(self, key, value, generation):
            return backing.write('recovery/' + workspace + '/' + key, value, generation)
    store = FixtureStore()
    ledger = Ledger(store, workspace, max_runs=2, max_chats=1)
    sessions = ManagedSessions(project=settings['project'], location=settings['session_location'],
        engine_id=settings['engine_id'], credentials=credentials, ledger=ledger)
    principal = Principal('disposable-cloud-recovery', time.time() + 600)
    config = ChatConfig(workspace, 'core_test', 'recovery_probe', 'https://fixture.test',
        'https://unused.run.app', max_runs=2, max_chats=1)
    access = AccessPolicy(lambda u, w: Grant(u, w, frozenset({'core_test'}), 'fixture-v1')
        if u == principal.user_id and w == workspace else None)

    async def execute(scope, session):
        def record(old):
            old = old or {'execution': execution, 'calls': []}
            assert old['execution'] == execution
            old['calls'].append({'run_id': scope.run_id, 'attempt': attempt})
            return old
        await asyncio.to_thread(mutate, store, 'probe.json', record)
        if attempt == 0:
            # Only this disposable fixture deadline differs; production stays180s.
            for read_attempt in range(8):
                fresh = await sessions.get_session(app_name=config.app_name,
                    user_id=session.user_id, session_id=session.id)
                if sum(e.author == 'user' and e.invocation_id == scope.run_id for e in fresh.events) == 1:
                    break
                await asyncio.sleep(1)
            else:
                raise AssertionError('Persisted user event not visible after bounded reads')
            deadline = time.time() + 30
            await asyncio.to_thread(ledger.change,
                lambda state: state['runs'][scope.run_id].update(deadline=deadline))
            await asyncio.to_thread(mutate, store, 'checkpoint.json', lambda old: {
                'execution': execution, 'attempt': attempt, 'run_id': scope.run_id,
                'session_id': session.id, 'chat_id': scope.chat_id,
                'request_id': 'crash-original-request', 'text': 'Synthetic cloud recovery question',
                'deadline': deadline, 'claimed': ledger.run(principal.user_id, scope.run_id)['claimed'],
                'user_event_persisted': True, 'pid': os.getpid(), 'at': time.time()})
            print(json.dumps({'event': 'cloud_recovery_checkpoint', 'attempt': attempt,
                'run_id': scope.run_id, 'claimed': True, 'user_event_persisted': True,
                'signal': 'SIGKILL', 'pid': os.getpid()}), flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError('SIGKILL returned')
        await sessions.append_event(session, Event(author='research_coordinator',
            invocation_id=scope.run_id, content=types.Content(role='model',
            parts=[types.Part(text='Synthetic cloud recovery response')]), timestamp=time.time()))

    service = ChatService(config, ledger, sessions, access, execute)
    if attempt == 0:
        assert not ledger.read()['runs']
        chat = await service.create(principal, 'crash-probe-chat')
        await service.send(principal, chat['chat_id'], 'crash-original-request',
            'Synthetic cloud recovery question')
        raise AssertionError('First process was not terminated')
    checkpoint = store.read('checkpoint.json')[0]
    assert checkpoint and checkpoint['execution'] == execution and checkpoint['claimed']
    if checkpoint['deadline'] > time.time():
        await asyncio.sleep(checkpoint['deadline'] - time.time() + 1)
    old = await service.get_run(principal, checkpoint['run_id'])
    assert old['status'] == 'interrupted'
    replay = await service.send(principal, checkpoint['chat_id'], checkpoint['request_id'], checkpoint['text'])
    assert replay['run_id'] == old['run_id'] and replay['status'] == 'interrupted'
    before = await service.messages(principal, checkpoint['chat_id'])
    assert len(before['messages']) == 1 and before['messages'][0]['role'] == 'user'
    healthy = await service.send(principal, checkpoint['chat_id'], uuid.uuid4().hex,
        'Fresh synthetic cloud recovery question')
    assert healthy['status'] == 'completed'
    for read_attempt in range(8):
        after = await service.messages(principal, checkpoint['chat_id'])
        if [m['role'] for m in after['messages']] == ['user', 'user', 'assistant']:
            break
        await asyncio.sleep(1)
    else:
        raise AssertionError('Completed history not visible after bounded reads')
    calls = store.read('probe.json')[0]['calls']
    assert len(calls) == 2 and len({c['run_id'] for c in calls}) == 2
    state = ledger.read()
    assert len(state['runs']) == 2 and all(r['model_calls'] == 0 for r in state['runs'].values())
    owner = session_user(workspace, principal.user_id)
    await sessions.delete_session(app_name=config.app_name, user_id=owner, session_id=checkpoint['session_id'])
    assert await sessions.get_session(app_name=config.app_name, user_id=owner, session_id=checkpoint['session_id']) is None
    await asyncio.to_thread(ledger.mark_deleted, principal.user_id, checkpoint['chat_id'])
    result = {'execution': execution, 'workspace': workspace, 'platform_attempt': attempt,
        'prior_actual_SIGKILL_checkpoint': checkpoint, 'old_status': old['status'],
        'replay_status': replay['status'], 'replay_same_run': True, 'executor_calls': calls,
        'fresh_status': healthy['status'], 'message_roles': [m['role'] for m in after['messages']],
        'managed_session_deleted_and_absent': True, 'new_model_calls': 0,
        'test_deadline_seconds': 30, 'production_deadline_seconds': 180,
        'scope': 'Isolated job actual process failure, GCS and managed Sessions; deterministic executor and fixture grant. Not a live HTTP/IAP/A2A/provider outage test.',
        'finished_at': time.time()}
    await asyncio.to_thread(mutate, store, 'result.json', lambda old: result)
    print(json.dumps({'event': 'cloud_recovery_pass', 'attempt': attempt,
        'interrupted_same_key_replay': True, 'fresh_completed': True,
        'managed_session_deleted_and_absent': True, 'new_model_calls': 0}), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
