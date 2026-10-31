"""Per-turn authenticated remote client; calls only the configured Evidence origin."""
import asyncio
from urllib.parse import urlsplit
import httpx
from google.adk import Runner
from google.adk.agents.run_config import RunConfig
from ..agents.research_coordinator.agent import create_agent
from ..persistence.current_turn_sessions import CurrentTurnSessions


class CoordinatorExecutor:
    def __init__(self, config, sessions, model_factory, guard, token_provider, *, preferences=None, protocol_enabled=False, feature_users=None, analysis_enabled=False,prediction_enabled=False,training_enabled=False):
        self.config,self.sessions,self.model_factory,self.guard,self.token_provider=config,sessions,model_factory,guard,token_provider
        self.preferences,self.protocol_enabled=preferences,protocol_enabled
        self.feature_users=feature_users
        self.analysis_enabled=analysis_enabled
        self.prediction_enabled=prediction_enabled
        self.training_enabled=training_enabled

    async def __call__(self, scope, session):
        token=await asyncio.to_thread(self.token_provider)
        expected=urlsplit(self.config.evidence_url)
        async def restrict(request):
            actual=urlsplit(str(request.url))
            if (actual.scheme,actual.hostname,actual.port or (443 if actual.scheme=="https" else 80)) != (
                    expected.scheme,expected.hostname,expected.port or (443 if expected.scheme=="https" else 80)):
                raise PermissionError("Remote endpoint changed")
        async with httpx.AsyncClient(timeout=95,follow_redirects=False,
            headers={"Authorization":"Bearer "+token,"X-Serverless-Authorization":"Bearer "+token,
                "X-Copilot-Run":scope.run_id,"X-Copilot-Fence":scope.fence},
            event_hooks={"request":[restrict]}) as client:
            agent=create_agent(self.model_factory(),self.config.evidence_url,client,self.guard,
                training_enabled=self.training_enabled,preferences=self.preferences,analysis_enabled=self.analysis_enabled,prediction_enabled=self.prediction_enabled,protocol_enabled=self.protocol_enabled and
                    (self.feature_users is None or scope.principal.user_id in self.feature_users))
            execution_sessions=CurrentTurnSessions(self.sessions,app_name=self.config.app_name,
                user_id=session.user_id,session_id=session.id,invocation_id=scope.run_id)
            runner=Runner(app_name=self.config.app_name,agent=agent,session_service=execution_sessions)
            final=False
            async for event in runner.run_async(user_id=session.user_id,session_id=session.id,new_message=None,
                    invocation_id=scope.run_id,run_config=RunConfig(max_llm_calls=6)):
                await self.guard.authorize()
                if event.error_code or event.error_message:raise RuntimeError('Agent execution failed')
                if event.is_final_response() and event.content and any(p.text and not p.thought for p in event.content.parts or []):
                    final=True
            task_record,_=await asyncio.to_thread(self.guard.ledger.store.read,'tasks/'+scope.run_id+'.json')
            for task in (task_record or {}).get('tasks',{}).values():
                if task.get('status',{}).get('state') not in {'TASK_STATE_COMPLETED'}:
                    raise RuntimeError('Evidence task did not complete successfully')
            if not final:raise RuntimeError("No completed agent answer")
