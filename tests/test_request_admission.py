"""Durable admission under concurrency/failure; only synthetic in-memory data."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'), str(Path(__file__).resolve().parents[1])]
from cloud_io import CloudUnavailable, Conflict
from dispatch_control import BudgetExceeded, DispatchControl
from app.request_admission import RequestAdmission


class Store:
    def __init__(self):
        self.value = {'requests': 0, 'jobs': {'existing': {'owner': 'alice'}}, 'extra': 'preserved'}
        self.generation = 1
        self.writes = 0
        self.lock = threading.Lock()

    def read_json(self, name):
        with self.lock:
            return copy.deepcopy(self.value), str(self.generation)

    def state(self, name, value, generation):
        with self.lock:
            if generation != str(self.generation):
                raise Conflict('changed')
            self.value = copy.deepcopy(value)
            self.generation += 1
            self.writes += 1


def parallel(batcher, store, owners):
    barrier = threading.Barrier(len(owners))
    def run(owner):
        barrier.wait(timeout=3)
        return batcher.reserve(DispatchControl(store), owner)
    with ThreadPoolExecutor(max_workers=len(owners)) as pool:
        return list(pool.map(run, owners))


def test_eight_concurrent_requests_share_one_confirmed_write_and_separate_snapshots():
    store = Store()
    results = parallel(RequestAdmission(window=.05), store, ['alice', 'bob'] * 4)
    assert store.writes == 1 and store.value['requests'] == 8
    assert store.value['requests_by_owner'] == {'alice': 4, 'bob': 4}
    assert store.value['jobs'] == {'existing': {'owner': 'alice'}}
    assert store.value['extra'] == 'preserved'
    results[0]['jobs'].clear()
    assert all(r['jobs'] for r in results[1:]) and store.value['jobs']


def test_mixed_owner_and_global_caps_never_overadmit_or_charge_denials():
    store = Store()
    store.value.update(requests=597, requests_by_owner={'alice': 2})
    state, errors = DispatchControl(store).reserve_requests([
        ('alice', 2), ('bob', 2), ('bob', 2), ('bob', 2), ('carol', None), ('dave', None)])
    assert [e is None for e in errors] == [False, True, True, False, True, False]
    assert state['requests'] == 600 and store.writes == 1
    assert state['requests_by_owner'] == {'alice': 2, 'bob': 2, 'carol': 1}
    with pytest.raises(BudgetExceeded):
        RequestAdmission(window=0).reserve(DispatchControl(store), 'new owner')
    assert store.writes == 1  # Exhaustion is read-only, including after restart.


def test_cas_conflict_recomputes_entire_batch_and_keeps_other_writer_state():
    store = Store()
    original = store.state
    def collide(name, state, generation):
        if store.generation == 1:
            store.value.update(requests=599, requests_by_owner={'other': 599})
            store.value['jobs']['external'] = {'owner': 'other'}
            store.generation = 2
        return original(name, state, generation)
    store.state = collide
    with patch('dispatch_control.time.sleep'):
        state, errors = DispatchControl(store).reserve_requests([('alice', None), ('bob', None)])
    assert state['requests'] == 600 and errors[0] is None and errors[1] is not None
    assert state['jobs']['external'] == {'owner': 'other'} and store.writes == 1


def test_two_independent_batchers_preserve_limits_with_cross_process_style_cas():
    store = Store()
    store.value['requests'] = 599
    barrier = threading.Barrier(2)
    original = store.read_json
    reads = 0
    lock = threading.Lock()
    def read(name):
        nonlocal reads
        result = original(name)
        with lock:
            reads += 1
            first = reads <= 2
        if first:
            barrier.wait(timeout=3)
        return result
    store.read_json = read
    def run(_):
        try:
            RequestAdmission(window=0).reserve(DispatchControl(store), 'alice')
            return 'accepted'
        except BudgetExceeded:
            return 'denied'
    with patch('dispatch_control.time.sleep'), ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    assert sorted(results) == ['accepted', 'denied'] and store.value['requests'] == 600


def test_ambiguous_commit_fails_entire_batch_without_replay_or_refund():
    store = Store()
    original = store.state
    def uncertain(*args):
        original(*args)
        raise CloudUnavailable('storage_post', 503)
    store.state = uncertain
    batcher = RequestAdmission(window=.05)
    barrier = threading.Barrier(2)
    def run(owner):
        barrier.wait(timeout=3)
        with pytest.raises(CloudUnavailable):
            batcher.reserve(DispatchControl(store), owner)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, ['alice', 'bob']))
    assert store.writes == 1 and store.value['requests'] == 2


def test_waiters_cannot_enter_handler_before_commit_and_queued_timeout_is_not_charged():
    store = Store()
    started, release = threading.Event(), threading.Event()
    original = store.state
    def delayed(*args):
        started.set()
        assert release.wait(timeout=3)
        return original(*args)
    store.state = delayed
    batcher = RequestAdmission(window=0, queue_timeout=.05, spacing=0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(batcher.reserve, DispatchControl(store), 'alice')
        assert started.wait(timeout=1) and not first.done()
        second = pool.submit(batcher.reserve, DispatchControl(store), 'bob')
        with pytest.raises(CloudUnavailable):
            second.result(timeout=1)
        assert store.value['requests'] == 0
        release.set()
        assert first.result(timeout=1)['requests'] == 1
    assert store.value['requests_by_owner'] == {'alice': 1}


def test_queue_is_bounded_and_recovers_after_storage_failure():
    store = Store()
    started, release = threading.Event(), threading.Event()
    original = store.state
    def failure(*args):
        started.set()
        assert release.wait(timeout=3)
        raise CloudUnavailable('storage_post', 429)
    store.state = failure
    batcher = RequestAdmission(window=0, spacing=0, max_pending=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(batcher.reserve, DispatchControl(store), 'alice')
        assert started.wait(timeout=1)
        second = pool.submit(batcher.reserve, DispatchControl(store), 'bob')
        deadline = time.monotonic() + 1
        while len(batcher._pending) != 1 and time.monotonic() < deadline:
            time.sleep(.001)
        with pytest.raises(CloudUnavailable):
            batcher.reserve(DispatchControl(store), 'carol')
        release.set()
        for result in (first, second):
            with pytest.raises(CloudUnavailable):
                result.result(timeout=2)
    store.state = original
    assert batcher.reserve(DispatchControl(store), 'dave')['requests'] == 1


def test_month_and_policy_changes_are_not_batched_together():
    store = Store()
    names = []
    original = store.state
    def write(name, *args):
        names.append(name)
        return original(name, *args)
    store.state = write
    batcher = RequestAdmission(window=.03, spacing=0)
    controls = [DispatchControl(store, '2026-10'), DispatchControl(store, '2026-11')]
    barrier = threading.Barrier(2)
    def run(control):
        barrier.wait(timeout=3)
        return batcher.reserve(control, 'alice')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, controls))
    assert len(names) == 2 and len(set(names)) == 2


def test_pacing_separates_confirmed_writes_without_preallocating_quota():
    store = Store()
    batcher = RequestAdmission(window=0, spacing=.06)
    batcher.reserve(DispatchControl(store), 'alice')
    assert store.value['requests'] == 1
    start = time.monotonic()
    batcher.reserve(DispatchControl(store), 'alice')
    assert time.monotonic() - start >= .055
    assert store.value['requests'] == 2


def test_flask_batching_preserves_auth_and_owner_scoped_views(monkeypatch):
    from app.server import create_app
    store = Store()
    original_read = store.read_json
    def read(name):
        if name.endswith('/worker.json'):
            raise FileNotFoundError()
        return original_read(name)
    store.read_json = read
    monkeypatch.setenv('REVIEWER_EMAILS', 'bob')
    def identity(headers):
        if headers.get('Test-Owner') not in ('alice', 'bob'):
            raise PermissionError('Unauthorized')
        return headers['Test-Owner']
    app = create_app(lambda **kwargs: store, identity)
    assert app.test_client().get('/api/bootstrap').status_code == 403
    assert store.writes == 0
    barrier = threading.Barrier(2)
    def run(owner):
        barrier.wait(timeout=3)
        response = app.test_client().get('/api/bootstrap', headers={'Test-Owner': owner})
        assert response.status_code == 200
        return response.json
    with ThreadPoolExecutor(max_workers=2) as pool:
        alice, bob = list(pool.map(run, ['alice', 'bob']))
    assert alice['jobs'] == ['existing'] and bob['jobs'] == []
    assert alice['access']['role'] == 'researcher' and bob['access']['role'] == 'reviewer'
    assert alice['access']['requests_used'] == bob['access']['requests_used'] == 1
    assert store.writes == 1


@pytest.mark.parametrize('base_limit,new_limit', [(450,500),(500,550),(550,750)])
def test_reviewer_extension_preserves_usage_and_stops_at_new_limit(monkeypatch,base_limit,new_limit):
    from app.server import create_app
    monkeypatch.setenv('DEFENDER_REVIEWER_MONTHLY_API_LIMIT', str(new_limit))
    monkeypatch.setenv('DEFENDER_MONTHLY_API_LIMIT', '2600')
    monkeypatch.setenv('REVIEWER_EMAILS', 'bob')
    store = Store()
    initial_total=1795+base_limit
    store.value.update(requests=initial_total, requests_by_owner={'alice':1754, 'bob':base_limit, 'other':1})
    old_jobs = copy.deepcopy(store.value['jobs'])
    original = store.read_json
    def read(name):
        if name.endswith('/worker.json'):raise FileNotFoundError()
        return original(name)
    store.read_json = read
    app = create_app(lambda **kwargs:store, lambda headers:headers['Test-Owner'])
    # A restart does not reset usage. Each of the 50 additional requests is
    # charged against both the existing global and per-reviewer counters.
    client = app.test_client()
    with patch('app.request_admission.time.sleep'):
        response = client.get('/api/bootstrap', headers={'Test-Owner':'bob'})
    assert response.status_code == 200
    assert store.value['requests_by_owner']['bob'] == base_limit+1
    for _ in range(new_limit-base_limit-1):DispatchControl(store).reserve_request('bob', new_limit)
    before = copy.deepcopy(store.value)
    restarted = create_app(lambda **kwargs:store, lambda headers:headers['Test-Owner']).test_client()
    assert restarted.get('/api/bootstrap', headers={'Test-Owner':'bob'}).status_code == 429
    assert store.value == before and store.value['requests'] == initial_total+new_limit-base_limit
    assert store.value['jobs'] == old_jobs
    assert store.value['requests_by_owner']['alice'] == 1754
