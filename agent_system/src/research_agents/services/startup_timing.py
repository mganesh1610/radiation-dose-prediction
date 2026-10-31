"""Numeric startup diagnostics only; never log settings, paths or identities."""
import json
import time


def phase(name, started):
    print(json.dumps({'event': 'copilot_startup', 'phase': name,
                      'duration_ms': round((time.perf_counter()-started)*1000, 3)}),
          flush=True)
