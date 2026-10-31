"""Coordinator startup must not import the separately deployed Evidence server."""
import os
import subprocess
import sys
from pathlib import Path


def test_coordinator_import_does_not_load_evidence_server_or_connect():
    source = str(Path(__file__).resolve().parents[2] / 'src')
    code = '''
import socket, sys
def blocked(*args, **kwargs):
    raise AssertionError('Import attempted a network connection')
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
import research_agents.services.bootstrap as bootstrap
assert callable(bootstrap.coordinator_app)
assert callable(bootstrap.evidence_app)
assert 'research_agents.services.evidence_a2a' not in sys.modules
assert 'research_agents.agents.evidence_specialist.agent' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30,
                   capture_output=True, text=True,
                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONPATH':source})
