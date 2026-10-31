"""Offline SDK/card checks: no model inference or external cloud access."""
import importlib.util
from pathlib import Path
import socket

from fastapi.testclient import TestClient


def test_starter_card_and_agent_topology_without_external_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline preflight attempted an external connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    root = Path(__file__).resolve().parent / "fixtures/sdk_starter"
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    evidence = load("preflight_evidence", root / "remote_evidence.py")
    coordinator = load("preflight_coordinator", root / "research_coordinator/agent.py")
    assert len(coordinator.root_agent.sub_agents) == 1
    assert coordinator.root_agent.sub_agents[0].name == "evidence_service"
    assert evidence.get_approved_fact("not-in-snapshot")["status"] == "not_recorded"
    with TestClient(evidence.a2a_app) as client:
        response = client.get(coordinator.AGENT_CARD_WELL_KNOWN_PATH)
        assert response.status_code == 200
        assert response.json()["name"] == "radiation_evidence"
