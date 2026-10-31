"""Process-local bounded LRU/TTL cache. Stores immutable matrices, not user answers."""
from collections import OrderedDict
from dataclasses import dataclass
import threading
import time


class CacheBusy(RuntimeError):
    pass


@dataclass
class Entry:
    frame: object
    size: int
    expires_at: float


class MatrixCache:
    def __init__(self, *, max_bytes=64 * 1024**2, max_entries=2, ttl_seconds=300,
                 lock_timeout=5, clock=time.monotonic):
        if not 0 < max_bytes <= 64 * 1024**2 or not 1 <= max_entries <= 2 or not 0 < ttl_seconds <= 300:
            raise ValueError("Cache limits outside the pilot bounds")
        self.max_bytes, self.max_entries, self.ttl = max_bytes, max_entries, ttl_seconds
        self.lock_timeout, self.clock = lock_timeout, clock
        self._entries, self._lock = OrderedDict(), threading.Lock()
        self.hits = self.misses = self.evictions = 0

    def _bytes(self):
        return sum(e.size for e in self._entries.values())

    def _evict(self, key):
        self._entries.pop(key)
        self.evictions += 1

    def read(self, key, reserve_bytes, loader, consumer):
        """Keep ownership within this lock; consumer returns only bounded copied output.

        One load at a time prevents duplicate downloads and bounds load reservations.
        Evict before loading, so a new allocation does not overlap an overfull cache.
        """
        if not 0 < reserve_bytes <= self.max_bytes:
            raise ValueError("Matrix reservation exceeds cache budget")
        if not self._lock.acquire(timeout=self.lock_timeout):
            raise CacheBusy("Matrix cache is busy; retry later")
        try:
            for old in list(self._entries):
                if self._entries[old].expires_at <= self.clock():
                    self._evict(old)
            hit = key in self._entries
            if hit:
                self.hits += 1
                self._entries.move_to_end(key)
            else:
                self.misses += 1
                while self._entries and (self._bytes() + reserve_bytes > self.max_bytes or len(self._entries) >= self.max_entries):
                    self._evict(next(iter(self._entries)))
                frame = loader()
                if not 0 < frame.resident_bytes <= reserve_bytes:
                    raise ValueError("Decoded matrix exceeds its memory reservation")
                self._entries[key] = Entry(frame, frame.resident_bytes, self.clock() + self.ttl)
            output = consumer(self._entries[key].frame)
            return output, {"hit": hit, "entries": len(self._entries),
                            "accounted_resident_bytes": self._bytes(), "ttl_seconds": self.ttl}
        finally:
            self._lock.release()

    def invalidate(self, workspace, release):
        # Fail-closed callers may not wait indefinitely behind an in-progress load.
        if not self._lock.acquire(timeout=self.lock_timeout):
            return False
        try:
            for key in list(self._entries):
                if key[:2] == (workspace, release): self._evict(key)
            return True
        finally:
            self._lock.release()

    def clear(self):
        with self._lock:
            self._entries.clear()
