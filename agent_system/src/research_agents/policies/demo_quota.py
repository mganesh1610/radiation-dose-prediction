"""Temporary Gemini allowances, counted from midnight in Phoenix (UTC−07)."""
from datetime import datetime, timezone
import json
from pathlib import Path


POLICY = json.loads(Path(__file__).with_suffix('.json').read_text())
START = int(datetime.fromisoformat(POLICY['starts_at']).timestamp())
END = int(datetime.fromisoformat(POLICY['ends_at']).timestamp())
DAY = 86400


def day_for(stamp):
    if START <= stamp < END:
        midnight = START + int((stamp - START) // DAY) * DAY
        return datetime.fromtimestamp(midnight, timezone.utc).date().isoformat()
    return None


def next_midnight(stamp):
    return START + (int((stamp - START) // DAY) + 1) * DAY
