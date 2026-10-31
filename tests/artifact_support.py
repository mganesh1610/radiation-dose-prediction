"""Artifact checks skip only when explicitly absent; strict mode fails closed."""
import os
from pathlib import Path
import unittest


def require_artifacts(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        message = "Separately provisioned research artifacts required: " + ", ".join(missing)
        if os.environ.get("DEFENDER_REQUIRE_DATA_TESTS") == "1":
            raise AssertionError(message)
        raise unittest.SkipTest(message)
