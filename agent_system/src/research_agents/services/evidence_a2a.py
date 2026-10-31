"""Authenticated A2A boundary with server-resolved run ownership and sessions."""
import asyncio
import hmac
import json
import time
from urllib.parse import urlsplit

from google.adk import Runner
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.converters.request_converter import AgentRunRequest
from google.adk.agents.run_config import RunConfig
from google.genai import types
from starlette.responses import JSONResponse

from ..policies.access import Principal
from ..policies.run_scope import RunScope,current_run,session_user
from ..persistence.managed_sessions import ensure_session
from ..persistence.a2a_tasks import ScopedTaskStore


class ServiceAuthentication:
    def __init__(self, audience, allowed_service_email, *, local_token=None):
        self.audience,self.email,self.local_token=audience,allowed_service_email,local_token
        if local_token and urlsplit(audience).hostname not in {"127.0.0.1","localhost"}:
            raise ValueError("Local token authentication is loopback-only")

    def verify(self, headers):
        auth=headers.get("authorization","")
        if self.local_token:
            if not hmac.compare_digest(auth,"Bearer "+self.local_token):raise PermissionError("Service identity required")
            return
        from google.oauth2.id_token import verify_oauth2_token
        from google.auth.transport.requests import Request
        if not auth.startswith("Bearer "):raise PermissionError("Service identity required")
        claims=verify_oauth2_token(auth[7:],Request(),audience=self.audience)
        if claims.get("email")!=self.email or claims.get("email_verified") is not True:
            raise PermissionError("Unexpected service identity")


def create_app(agent, config, sessions, ledger, store, access, service_auth):
    runner=Runner(app_name=config.app_name+"_evidence",agent=agent,session_service=sessions)
    def convert(request, part_converter):
        s=current_run.get()
        parts=[]
        for part in request.message.parts:
            converted=part_converter(part)
            parts.extend(converted if isinstance(converted,list) else [converted])
        if not parts or any(not p.text or p.function_call or p.function_response or p.inline_data or p.file_data for p in parts):
            raise ValueError("Only text delegation is allowed")
        if sum(len(p.text.encode()) for p in parts)>8000:raise ValueError("Delegation too large")
        return AgentRunRequest(user_id=s.session_user,session_id=s.evidence_session,
            invocation_id="remote-"+s.run_id,new_message=types.Content(role="user",parts=parts),
            run_config=RunConfig(max_llm_calls=4))
    parsed=urlsplit(config.evidence_url)
    port=parsed.port or (443 if parsed.scheme=="https" else 80)
    a2a=to_a2a(agent,host=parsed.hostname,port=port,protocol=parsed.scheme,runner=runner,
        task_store=ScopedTaskStore(store,ledger),
        agent_executor_factory=lambda r:A2aAgentExecutor(runner=r,config=A2aAgentExecutorConfig(request_converter=convert)))

    class SecuredEvidence:
        async def __call__(self, scope, receive, send):
            if scope["type"]!="http":return await a2a(scope,receive,send)
            from starlette.datastructures import Headers
            headers=Headers(scope=scope)
            try:
                await asyncio.to_thread(service_auth.verify,headers)
                run_id=headers.get("x-copilot-run","")
                fence=headers.get("x-copilot-fence","")
                data=await asyncio.to_thread(ledger.read)
                run=data["runs"].get(run_id)
                if not run:raise PermissionError()
                principal=Principal(run["owner"],min(run["deadline"],run['identity_expires_at']))
                await asyncio.to_thread(access.authorize,principal,config.workspace,config.release_id)
                await asyncio.to_thread(ledger.check,principal.user_id,run_id,fence)
                chat=await asyncio.to_thread(ledger.chat,principal.user_id,run["chat_id"])
                s=RunScope(principal,config.workspace,run_id,run["chat_id"],fence,
                    session_user(config.workspace,principal.user_id),"evidence-"+run_id[4:],
                    run.get('request_contract',{}),run['deadline'])
                await ensure_session(sessions,app_name=runner.app_name,owner=s.session_user,
                    session_id=s.evidence_session,state={"selected_release_id":config.release_id},expires_at=chat["expires_at"])
            except Exception:
                return await JSONResponse({"error":"Authorized active delegation required"},status_code=403)(scope,receive,send)
            token=current_run.set(s)
            try:
                await a2a(scope,receive,send)
            finally:
                s.stop.set()
                current_run.reset(token)
    return SecuredEvidence()
