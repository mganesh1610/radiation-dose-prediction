"""Trusted per-request context, never reconstructed from LLM/session fields."""
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event
from .access import Principal


@dataclass(frozen=True)
class RunScope:
    principal: Principal
    workspace: str
    run_id: str
    chat_id: str
    fence: str
    session_user: str
    evidence_session: str
    contract: dict = field(default_factory=dict)
    deadline: float = 0
    stop: Event = field(default_factory=Event, compare=False)


current_run = ContextVar("copilot_trusted_run")


def session_user(workspace, user):
    import hashlib
    return "scope-" + hashlib.sha256((workspace+"\0"+user).encode()).hexdigest()
