"""Fault injection for concurrent CAS writes and uncertain commit recovery."""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from google.api_core.exceptions import ServiceUnavailable, PreconditionFailed, NotFound
from research_agents.persistence import control_store as cs
from research_agents.persistence.ledger import Ledger
from test_repair_controls import ObjectStore


def marker(n):
    return f'{n:032x}'


@pytest.mark.parametrize('error', [PreconditionFailed, NotFound])
@pytest.mark.parametrize('committed', [False, True])
def test_uncertain_write_with_exhausted_recovery_reads_never_replays(error, committed, monkeypatch):
    monkeypatch.setattr(cs, 'backoff', lambda *args: None)
    backend = ObjectStore([])
    backend.value[cs.CONTROL_OPS] = [marker(1)]
    uploads, invocations, races = [], [], []
    original_download = backend.download_as_bytes

    def upload(data, **kwargs):
        uploads.append(1)
        assert kwargs['if_generation_match'] == backend.generation
        if committed:
            backend.value = json.loads(data)
            backend.generation += 1
        raise TimeoutError('Write response lost')

    def download(**kwargs):
        if uploads and len(races) < 4:
            races.append(1)
            backend.value['unrelated_writes'] = len(races)
            backend.value[cs.CONTROL_OPS].append(marker(100 + len(races)))
            backend.generation += 1
            raise error('Generation replaced between metadata and content')
        return original_download(**kwargs)

    backend.upload_from_string = upload
    backend.download_as_bytes = download

    def increment(state):
        invocations.append(1)
        state['counter'] += 1
        return state

    with pytest.raises(cs.StorageUnavailable):
        cs.mutate(backend.store, 'ledger.json', increment)
    assert len(uploads) == len(invocations) == 1
    assert len(races) == 4
    assert backend.value['counter'] == int(committed)
    history = backend.value[cs.CONTROL_OPS]
    assert len(history) == len(set(history))


def test_lost_commit_recovers_after_three_generation_races(monkeypatch):
    monkeypatch.setattr(cs, 'backoff', lambda *args: None)
    backend = ObjectStore(['committed'])
    original_download = backend.download_as_bytes
    races, invocations = [], []

    def download(**kwargs):
        if backend.writes and len(races) < 3:
            races.append(1)
            backend.generation += 1
            backend.value[cs.CONTROL_OPS].append(marker(100 + len(races)))
            raise PreconditionFailed('Concurrent generation change')
        return original_download(**kwargs)

    backend.download_as_bytes = download

    def increment(state):
        invocations.append(1)
        state['counter'] += 1
        return state

    assert cs.mutate(backend.store, 'ledger.json', increment)['counter'] == 1
    assert backend.value['counter'] == 1
    assert len(invocations) == len(backend.writes) == 1
    assert len(races) == 3


def test_lost_write_and_competing_commit_rebases_once(monkeypatch):
    monkeypatch.setattr(cs, 'backoff', lambda *args: None)
    backend = ObjectStore([])
    backend.value[cs.CONTROL_OPS] = [marker(1), marker(2)]
    original = backend.upload_from_string
    calls = []
    def upload(data, **kwargs):
        if not calls:
            calls.append('competing writer')
            backend.value['other_counter'] = 1
            backend.value[cs.CONTROL_OPS].append(marker(3))
            backend.generation += 1
            raise ServiceUnavailable('Our request did not commit; another request won')
        original(data, **kwargs)
    backend.upload_from_string = upload
    invocations = []
    def increment(state):
        invocations.append(1)
        state['counter'] += 1
        return state
    cs.mutate(backend.store, 'ledger.json', increment)
    assert backend.value['counter'] == backend.value['other_counter'] == 1
    assert len(invocations) == 2
    assert backend.value[cs.CONTROL_OPS][:3] == [marker(1), marker(2), marker(3)]


@pytest.mark.parametrize('committed', [False, True])
def test_aged_out_ancestor_leaves_unknown_outcome_closed(committed, monkeypatch):
    monkeypatch.setattr(cs, 'backoff', lambda *args: None)
    backend = ObjectStore([])
    backend.value[cs.CONTROL_OPS] = [marker(1), marker(2)]
    attempts = []
    def upload(data, **kwargs):
        attempts.append(1)
        if committed:
            backend.value = json.loads(data)
        # Enough compliant later writes to evict both ancestor and our marker.
        backend.value[cs.CONTROL_OPS] = [marker(i) for i in range(100, 228)]
        backend.generation += 129
        raise TimeoutError('Outcome can no longer be proved')
    backend.upload_from_string = upload
    def increment(state):
        state['counter'] += 1
        return state
    with pytest.raises(cs.StorageUnavailable):
        cs.mutate(backend.store, 'ledger.json', increment)
    assert len(attempts) == 1 and backend.value['counter'] == int(committed)


@pytest.mark.parametrize('observed', [[], None, [marker(2)], [marker(2), marker(2)],
    [marker(9), marker(2), marker(3)], [marker(1), marker(2), 'bad'],
    [marker(1), marker(2), marker(4)]])
def test_missing_corrupt_or_committed_histories_are_not_noncommit_proof(observed):
    value = {cs.CONTROL_OPS: [marker(1), marker(2), marker(4)]}
    assert not cs.competing_write_proven(value, {cs.CONTROL_OPS: observed})


def test_rolling_window_retains_sufficient_ancestor_proof():
    candidate = {cs.CONTROL_OPS: [marker(i) for i in range(1, 128)] + [marker(999)]}
    latest = {cs.CONTROL_OPS: [marker(i) for i in range(10, 138)]}
    assert cs.competing_write_proven(candidate, latest)


@pytest.mark.parametrize('commit_loser', [False, True])
def test_parallel_reservations_keep_both_counts_exactly_once(commit_loser, monkeypatch):
    monkeypatch.setattr(cs, 'backoff', lambda *args: None)
    lock = threading.Lock()
    state = {'value': None, 'generation': 0, 'armed': False, 'entered': 0}
    barrier = threading.Barrier(2)
    winner_done = threading.Event()
    def store():
        result = object.__new__(cs.GcsControlStore)
        def read(key, deadline):
            with lock:
                return copy.deepcopy(state['value']), str(state['generation'])
        def upload(data, **kwargs):
            with lock:
                position = state['entered'] if state['armed'] else 99
                if state['armed']:
                    state['entered'] += 1
            if position < 2:
                barrier.wait(timeout=5)
            if position == 1:
                assert winner_done.wait(timeout=5)
                if not commit_loser:
                    raise ServiceUnavailable('Competing CAS won while this request failed')
            with lock:
                if kwargs['if_generation_match'] != state['generation']:
                    raise PreconditionFailed('CAS conflict')
                state['value'] = json.loads(data)
                state['generation'] += 1
                if position == 0 and commit_loser:
                    # Winner committed but its response was lost. A later writer
                    # retains the winner marker while making an unrelated change.
                    state['value']['unrelated'] = True
                    state['value'][cs.CONTROL_OPS] = [*state['value'][cs.CONTROL_OPS][-127:], marker(999)]
                    state['generation'] += 1
            if position == 0:
                winner_done.set()
                if commit_loser:
                    raise TimeoutError('Committed reservation response lost')
        result._read = read
        result.blob = lambda key: SimpleNamespace(upload_from_string=upload)
        return result
    first, second = Ledger(store(), 'w'), Ledger(store(), 'w')
    runs = []
    for i in range(2):
        chat = first.create_chat('owner', str(i), 'core_test')
        first.ready('owner', chat['chat_id'])
        runs.append(first.admit('owner', chat['chat_id'], str(i), 'question'))
    state['armed'] = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(ledger.reserve, 'owner', run['run_id'], run['fence'], 'model', input_bytes=100)
                for ledger, run in zip((first, second), runs)]
        for job in jobs:
            job.result(timeout=10)
    saved = first.read()['runs']
    assert [saved[r['run_id']]['model_calls'] for r in runs] == [1, 1]
    assert [saved[r['run_id']]['input_bytes'] for r in runs] == [100, 100]
