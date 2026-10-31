"""Small synchronous batches for the website's durable monthly request cap.

Only queued reservations live in memory: no cached quota, preallocated credits,
background writer or cross-user response cache. CAS remains authoritative across
processes/restarts. Pacing is local to this app instance, so other writers can
still cause the existing bounded Cloud Storage retry path.
"""
from copy import deepcopy
from dataclasses import dataclass
import threading
import time

from cloud_io import CloudUnavailable
from dispatch_control import BudgetExceeded


@dataclass(eq=False)
class _Reservation:
    control: object
    owner: str
    limit: int | None
    deadline: float
    state: dict | None = None
    error: BaseException | None = None
    done: bool = False


class RequestAdmission:
    def __init__(self, *, window=.025, spacing=1.05, queue_timeout=5.0, max_pending=64):
        if window < 0 or spacing < 0 or queue_timeout <= window or max_pending < 1:
            raise ValueError('Invalid admission queue settings')
        self.window, self.spacing = window, spacing
        self.queue_timeout, self.max_pending = queue_timeout, max_pending
        self._condition = threading.Condition()
        self._pending = []
        self._flushing = False
        self._next_flush = 0.0

    def reserve(self, control, owner=None, owner_limit=None):
        ticket = _Reservation(control, owner, owner_limit, time.monotonic() + self.queue_timeout)
        with self._condition:
            if len(self._pending) >= self.max_pending:
                raise CloudUnavailable('admission_queue')
            self._pending.append(ticket)
            self._condition.notify_all()
        while True:
            with self._condition:
                if ticket.done:
                    if ticket.error is not None:
                        raise ticket.error
                    return ticket.state
                remaining = ticket.deadline - time.monotonic()
                if remaining <= 0 and ticket in self._pending:
                    self._pending.remove(ticket)
                    raise CloudUnavailable('admission_queue')
                if self._flushing:
                    # Once submitted, the durable outcome must be awaited. A
                    # timed-out *queued* request is removed before submission.
                    self._condition.wait(max(.001, remaining) if ticket in self._pending else None)
                    continue
                self._flushing = True
                ready_at = max(self._next_flush, time.monotonic() + self.window)
                while time.monotonic() < ready_at:
                    self._condition.wait(ready_at - time.monotonic())
                now = time.monotonic()
                for pending in self._pending[:]:
                    if pending.deadline <= now:
                        self._pending.remove(pending)
                        pending.error = CloudUnavailable('admission_queue')
                        pending.done = True
                batch = []
                if self._pending:
                    first = self._pending[0].control
                    for pending in self._pending[:8]:
                        if (pending.control.name, pending.control.http_limit) != (first.name, first.http_limit):
                            break  # Never mix monthly ledgers or policy versions.
                        batch.append(pending)
                    del self._pending[:len(batch)]
            try:
                if batch:
                    state, errors = batch[0].control.reserve_requests([(v.owner, v.limit) for v in batch])
                    for pending, error in zip(batch, errors):
                        if error is None:
                            pending.state = deepcopy(state)
                        else:
                            pending.error = BudgetExceeded(error)
            except BaseException as error:
                for pending in batch:
                    pending.error = error if isinstance(error, Exception) else CloudUnavailable('admission_aborted')
                if not isinstance(error, Exception):
                    raise
            finally:
                with self._condition:
                    for pending in batch:
                        pending.done = True
                    if batch:
                        self._next_flush = time.monotonic() + self.spacing
                    self._flushing = False
                    self._condition.notify_all()
