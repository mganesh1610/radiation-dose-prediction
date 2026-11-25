CREATE TABLE IF NOT EXISTS guest_work (
  id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL, request_key TEXT NOT NULL,
  input TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued', created INTEGER NOT NULL,
  updated INTEGER NOT NULL, available INTEGER NOT NULL DEFAULT 0, fence TEXT,
  lease INTEGER NOT NULL DEFAULT 0, result TEXT, UNIQUE(owner, request_key)
);
CREATE INDEX IF NOT EXISTS guest_owner ON guest_work(owner, kind, created DESC);
CREATE INDEX IF NOT EXISTS guest_queue ON guest_work(state, available, created);
CREATE INDEX IF NOT EXISTS guest_expiry ON guest_work(created);
CREATE INDEX IF NOT EXISTS guest_daily_ai ON guest_work(kind, created, owner);
