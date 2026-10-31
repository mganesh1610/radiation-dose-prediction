"""An execution-only view of a durable conversation, bound to one trusted run.

History remains in the underlying service for UI reads and audit. The Runner
and its A2A proxy see only this invocation, including its current tool pairs.
This prevents unanswered/failed old tasks from becoming new delegated work and
keeps historical receipts out of a new scientific request. No model summaries,
stored-event deletion, extra provider read or enlarged context limit is used.
"""
from google.adk.sessions.base_session_service import BaseSessionService


class CurrentTurnSessions(BaseSessionService):
    def __init__(self, service, *, app_name, user_id, session_id, invocation_id):
        self.service = service
        self.identity = (app_name, user_id, session_id)
        self.invocation_id = invocation_id

    def _check(self, app_name, user_id, session_id):
        if (app_name, user_id, session_id) != self.identity:
            raise PermissionError("Execution session differs from the trusted run")

    async def get_session(self, *, app_name, user_id, session_id, config=None):
        self._check(app_name, user_id, session_id)
        session = await self.service.get_session(app_name=app_name, user_id=user_id, session_id=session_id, config=config)
        if session is None:
            return None
        # Also validate provider output. Never filter/mutate a provider-owned
        # instance: other readers must retain the full transcript and state.
        self._check(session.app_name, session.user_id, session.id)
        view = session.model_copy(deep=True)
        view.events = [e for e in view.events if e.invocation_id == self.invocation_id]
        return view

    async def append_event(self, session, event):
        self._check(session.app_name, session.user_id, session.id)
        if event.invocation_id != self.invocation_id:
            raise PermissionError("Event differs from the trusted run")
        # The underlying provider remains the sole writer, with its original
        # fencing, state-delta and persistence behavior. No history replacement.
        return await self.service.append_event(session, event)

    async def flush(self):
        return await self.service.flush()

    async def create_session(self, **kwargs):
        raise PermissionError("Execution view cannot create sessions")

    async def list_sessions(self, **kwargs):
        raise PermissionError("Execution view cannot list sessions")

    async def delete_session(self, **kwargs):
        raise PermissionError("Execution view cannot delete sessions")
