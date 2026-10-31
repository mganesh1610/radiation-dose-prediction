"""Generation-conditional control records; never a second transcript store."""
import copy
import json
import re
import threading
import time
import random
import uuid
from builtins import ConnectionError as BuiltinConnectionError

CONTROL_OPS = "_control_ops"


def record_limit(key):
    # Finite 240-turn pilot ledger needs room for retained audit receipts.
    # Other control documents keep their original cap; no unbounded growth.
    return 2_000_000 if key == "ledger.json" else 1_000_000


def competing_write_proven(value, latest):
    """Prove non-commit from the retained, append-only mutation history.

    Every mutation writer retains the last 128 operation IDs. If an ancestor
    immediately before our operation still exists, our committed operation
    could not already have aged out. Missing or inconsistent history is not
    evidence of non-commit. Identity bindings do not use this recovery path.
    """
    candidate = value.get(CONTROL_OPS, [])
    observed = (latest or {}).get(CONTROL_OPS, [])
    for history in (candidate, observed):
        if (not isinstance(history, list) or not 1 <= len(history) <= 128
                or any(not isinstance(op, str) or not re.fullmatch(r'[0-9a-f]{32}', op) for op in history)
                or len(set(history)) != len(history)):
            return False
    if len(candidate) < 2 or candidate[-1] in observed:
        return False
    previous = candidate[:-1]
    if previous[-1] not in observed:
        return False
    end = observed.index(previous[-1]) + 1
    return (end < len(observed) and end <= len(previous)
            and observed[:end] == previous[-end:])


class StorageUnavailable(RuntimeError):
    """Bounded recovery failed; the logical mutation must not be blindly replayed."""


def transient(exc):
    from google.api_core import exceptions as g
    from requests.exceptions import ConnectionError, Timeout
    return isinstance(exc, (g.TooManyRequests, g.ServerError, g.DeadlineExceeded,
                            ConnectionError, Timeout, TimeoutError, BuiltinConnectionError)) or getattr(exc, "code", None) == 408


def retry_deadline():
    from ..policies.run_scope import current_run
    scope = current_run.get(None)
    return min(time.time()+20, scope.deadline) if scope and scope.deadline else time.time()+20


def remaining(deadline):
    from ..policies.run_scope import current_run
    scope = current_run.get(None)
    if (scope and scope.stop.is_set()) or deadline <= time.time():
        raise StorageUnavailable("Control storage recovery deadline reached")
    return min(10, deadline-time.time())


def backoff(attempt, deadline):
    from ..policies.run_scope import current_run
    delay = min(1.1 * 2**attempt + random.uniform(0, .25), remaining(deadline))
    scope = current_run.get(None)
    if scope:
        scope.stop.wait(delay)
    else:
        time.sleep(delay)
    remaining(deadline)


class Conflict(RuntimeError):
    """A rejected mutation can be rebased; never an uncertain write outcome."""


class QuotaExceeded(RuntimeError):
    pass


class GcsControlStore:
    def __init__(self, bucket, prefix, *, client=None):
        if not re.fullmatch(r"copilot/v1/[a-z0-9_-]+/", prefix):
            raise ValueError("A separate copilot control prefix is required")
        if client is None:
            from google.cloud import storage
            client = storage.Client()
        self.bucket, self.prefix = client.bucket(bucket), prefix

    def blob(self, key):
        if not re.fullmatch(r"[a-z0-9_-]+(?:/[a-z0-9_-]+)*\.json", key):
            raise ValueError("Invalid control key")
        return self.bucket.blob(self.prefix + key)

    def read(self, key):
        return self._read(key, retry_deadline())

    def _read(self, key, deadline):
        from google.api_core.exceptions import NotFound,PreconditionFailed
        for attempt in range(4):
            blob = self.blob(key)
            try:
                try:blob.reload(timeout=remaining(deadline), retry=None)
                except NotFound:return None, "0"
                if blob.size > record_limit(key):raise ValueError("Control record exceeds limit")
                try:data = blob.download_as_bytes(if_generation_match=blob.generation, timeout=remaining(deadline), retry=None)
                except (NotFound,PreconditionFailed):
                    # A disappearing old generation is not an empty ledger.
                    continue
            except Exception as exc:
                if not transient(exc):raise
                if attempt == 3:raise StorageUnavailable("Control storage read unavailable") from exc
                backoff(attempt, deadline)
                continue
            if len(data) > record_limit(key):raise ValueError("Control response exceeds limit")
            return json.loads(data), str(blob.generation)
        # No stable snapshot is not proof that a preceding upload failed.
        # In particular, write() may be reconciling a committed response loss.
        # Only a proven CAS rejection may escape as Conflict to mutate().
        raise StorageUnavailable('Control record changed during read; outcome unavailable')

    def write(self, key, value, generation):
        from google.api_core.exceptions import PreconditionFailed, Conflict as ProviderConflict
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        if len(data) > record_limit(key):
            raise ValueError("Control record exceeds limit")
        deadline = retry_deadline()
        uncertain = False
        operation_id = value.get(CONTROL_OPS, [None])[-1]
        for attempt in range(4):
            try:
                self.blob(key).upload_from_string(data, content_type="application/json",
                    if_generation_match=int(generation), timeout=remaining(deadline), retry=None)
                return
            except (PreconditionFailed, ProviderConflict):
                if not uncertain:raise Conflict("Concurrent control update") from None
            except Exception as exc:
                if not transient(exc):raise
                uncertain = True
            # A response may be lost after commit. Its atomic marker survives
            # subsequent writers, preventing duplicate budget/call increments.
            latest, current_generation = self._read(key, deadline)
            if (operation_id and operation_id in (latest or {}).get(CONTROL_OPS, [])) or latest == value:
                return
            if current_generation != generation:
                if competing_write_proven(value, latest):
                    # Another CAS writer won, and the retained ancestor proves
                    # this operation never committed. Rebase the logical change.
                    raise Conflict("Competing control write verified")
                raise StorageUnavailable("Control write outcome uncertain; do not replay mutation")
            if attempt == 3:raise StorageUnavailable("Control storage write unavailable")
            backoff(attempt, deadline)


class MemoryControlStore:
    """Explicit test double. Production entry points never select this store."""
    def __init__(self):
        self.records, self.lock = {}, threading.Lock()

    def read(self, key):
        with self.lock:
            value, generation = self.records.get(key, (None, 0))
            return copy.deepcopy(value), str(generation)

    def write(self, key, value, generation):
        with self.lock:
            if int(generation) != self.records.get(key, (None, 0))[1]:
                raise Conflict("Concurrent control update")
            self.records[key] = copy.deepcopy(value), int(generation) + 1


def mutate(store, key, operation, *, record_operation=True):
    operation_id = uuid.uuid4().hex
    deadline = retry_deadline()
    for attempt in range(4):
        remaining(deadline)
        state, generation = store.read(key)
        previous_ops = (state or {}).get(CONTROL_OPS, [])
        result = operation(state)
        if record_operation:
            result[CONTROL_OPS] = [*previous_ops[-127:], operation_id]
        try:
            store.write(key, result, generation)
            return result
        except Conflict:
            if attempt < 3:
                backoff(attempt, deadline)
            continue
    raise Conflict("Concurrent update; retry the same request")
