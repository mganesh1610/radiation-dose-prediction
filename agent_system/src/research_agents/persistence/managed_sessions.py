"""Explicit Google managed Sessions adapter; no production local fallback."""
import asyncio
import datetime
from google.adk.sessions import VertexAiSessionService
from google.genai import types

from .credentials import OperatorCredentials, cloud_credentials


class ManagedSessions(VertexAiSessionService):
    def __init__(self, *, project, location, engine_id, credentials=None, ledger=None):
        super().__init__(project=project, location=location, agent_engine_id=engine_id)
        self.credentials = credentials or cloud_credentials()
        self.ledger = ledger

    def _build_api_client(self):
        import agentplatform
        return agentplatform.Client(project=self._project, location=self._location,
            credentials=self.credentials, http_options=types.HttpOptions(timeout=15000,
            retry_options=types.HttpRetryOptions(attempts=1))).aio

    async def prepare(self):
        """Load the lazy Sessions surface before the HTTP server is ready.

        This does not call the provider or create a session. The client remains
        owned by the serving event loop; imports must not block a first history
        request while status/cancellation responses are being delivered.
        """
        self._get_api_client().sessions.events

    async def create_session(self, **kwargs):
        if "ttl" not in kwargs and "expire_time" not in kwargs:
            kwargs["ttl"] = "2592000s"
        return await super().create_session(**kwargs)

    async def append_event(self, session, event):
        from ..policies.run_scope import current_run
        scope=current_run.get(None)
        if scope and self.ledger:
            await asyncio.to_thread(self.ledger.check,scope.principal.user_id,scope.run_id,scope.fence)
        return await super().append_event(session,event)


async def ensure_session(service, *, app_name, owner, session_id, state, expires_at):
    session = await service.get_session(app_name=app_name, user_id=owner, session_id=session_id)
    if session is not None:
        return session
    try:
        return await service.create_session(app_name=app_name, user_id=owner, session_id=session_id,
            state=state, expire_time=datetime.datetime.fromtimestamp(expires_at,datetime.timezone.utc).isoformat())
    except Exception:
        # Creation may have committed before a timeout. Reconcile the same ID.
        session = await service.get_session(app_name=app_name, user_id=owner, session_id=session_id)
        if session is None: raise
        return session
