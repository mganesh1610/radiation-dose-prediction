"""CAS-controlled monthly admission, request limits, and discoverable work.

Limits persist across API restarts. A request reservation is intentionally consumed
on downstream failure; retries cannot multiply work or escape the monthly cap.
"""
from datetime import datetime, timezone
import os
import random
import time
from cloud_io import Conflict, PREFIX, digest, encode
from training_contract import identifier

MAX_HTTP = 600
MAX_JOBS = 12
CONTROL_PREFIX = PREFIX + 'control/'


class BudgetExceeded(RuntimeError):
    pass


def month_key(now=None):
    return (now or datetime.now(timezone.utc)).strftime('%Y-%m')


class DispatchControl:
    def __init__(self, cloud, month=None):
        self.cloud = cloud
        raw_limit = os.environ.get('DEFENDER_MONTHLY_API_LIMIT', str(MAX_HTTP))
        if not raw_limit.isascii() or not raw_limit.isdecimal() or not 1 <= int(raw_limit) <= 5000:
            raise ValueError('Monthly API allowance must be an integer from 1 to 5000')
        self.http_limit = int(raw_limit)
        self.month = month or month_key()
        self.name = CONTROL_PREFIX + self.month + '.json'

    def read(self):
        try:
            return self.cloud.read_json(self.name)
        except FileNotFoundError:
            return {'month': self.month, 'requests': 0, 'jobs': {}}, '0'

    def mutate(self, operation):
        for attempt in range(6):
            try:
                state, generation = self.read()
                operation(state)
                self.cloud.state(self.name, state, generation)
                return state
            except Conflict:
                if attempt < 5:
                    time.sleep(min(.2 * 2**attempt, 2) + random.uniform(0, .15))
                continue
        raise Conflict('Concurrent update; retry the same request')

    def reserve_request(self, owner=None, owner_limit=None):
        state, errors = self.reserve_requests([(owner, owner_limit)])
        if errors[0] is not None:
            raise BudgetExceeded(errors[0])
        return state

    def reserve_requests(self, requests):
        """Atomically charge each accepted request, in FIFO order, once.

        Batch membership is fixed before I/O. A CAS conflict recomputes all
        decisions from fresh state; an uncertain write is never rebased.
        Denied entries consume nothing and cannot block another eligible owner.
        No request is released to its handler until the write is confirmed.
        """
        requests = tuple(requests)
        if not 1 <= len(requests) <= 8:
            raise ValueError('Admission batch must contain 1 to 8 requests')
        for attempt in range(6):
            try:
                state, generation = self.read()
                errors = []
                usage = state.setdefault('requests_by_owner', {})
                for owner, owner_limit in requests:
                    if state['requests'] >= self.http_limit:
                        errors.append('Monthly application request limit reached')
                    elif owner_limit is not None and usage.get(owner, 0) >= owner_limit:
                        errors.append('Your monthly demo request allowance is used up')
                    else:
                        errors.append(None)
                        state['requests'] += 1
                        if owner is not None:
                            usage[owner] = usage.get(owner, 0) + 1
                if any(error is None for error in errors):
                    self.cloud.state(self.name, state, generation)
                return state, errors
            except Conflict:
                if attempt < 5:
                    time.sleep(min(.2 * 2**attempt, 2) + random.uniform(0, .15))
        raise Conflict('Concurrent update; retry the same request')

    def admit(self, job, owner, owner_limit=None, *, approval_sha256=None):
        identifier(job['job_id'])
        fingerprint = digest(encode(job))
        def update(state):
            old = state['jobs'].get(job['job_id'])
            if old:
                if old['sha256'] != fingerprint or old['owner'] != owner or old.get('bridge_approval_sha256') != approval_sha256:
                    raise ValueError('Job ID already used for another request')
                return
            if owner_limit is not None and sum(v.get('owner') == owner for v in state['jobs'].values()) >= owner_limit:
                raise BudgetExceeded('Your monthly demo training allowance is used up')
            if len(state['jobs']) >= MAX_JOBS:
                raise BudgetExceeded('Monthly training-job limit reached')
            state['jobs'][job['job_id']] = {'request': job, 'sha256': fingerprint,
                                           'owner': owner}
            if approval_sha256 is not None:
                state['jobs'][job['job_id']]['bridge_approval_sha256'] = approval_sha256
        return self.mutate(update)
