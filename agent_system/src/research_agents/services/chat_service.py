"""Request-lifetime chat execution with managed history and durable run fences."""
import asyncio
import time

from ..persistence.control_store import Conflict
from ..persistence.ledger import TERMINAL
from ..policies.run_scope import RunScope,current_run,session_user
from .chat_titles import TITLE_KEY, normalize_title, first_question_title, session_title


class ChatService:
    def __init__(self, config, ledger, sessions, access, executor):
        self.config,self.ledger,self.sessions,self.access,self.executor=config,ledger,sessions,access,executor

    def authorize(self, principal):
        self.access.authorize(principal,self.config.workspace,self.config.release_id)

    async def create(self, principal, request_id):
        await asyncio.to_thread(self.authorize,principal)
        ready=getattr(self.sessions,'wait_ready',None)
        if ready is not None:await ready()
        from ..persistence.managed_sessions import ensure_session
        chat=await asyncio.to_thread(self.ledger.create_chat,principal.user_id,request_id,self.config.release_id)
        owner=session_user(self.config.workspace,principal.user_id)
        await ensure_session(self.sessions,app_name=self.config.app_name,owner=owner,session_id=chat["session_id"],
            state={"selected_release_id":self.config.release_id},expires_at=chat["expires_at"])
        await asyncio.to_thread(self.authorize,principal)
        await asyncio.to_thread(self.ledger.ready,principal.user_id,chat["chat_id"])
        return self.public_chat(chat)

    @staticmethod
    def public_chat(chat, title=None):
        return {**{k:chat[k] for k in ("chat_id","session_id","release_id","created_at","expires_at","active_run")},
                "title":normalize_title(title)}

    async def list(self, principal):
        await asyncio.to_thread(self.authorize,principal)
        owner=session_user(self.config.workspace,principal.user_id)
        sessions=await self.sessions.list_sessions(app_name=self.config.app_name,user_id=owner)
        titles={s.id:normalize_title(s.state.get(TITLE_KEY)) for s in sessions.sessions if s.user_id==owner}
        await asyncio.to_thread(self.authorize,principal)
        # Managed Sessions also contains Evidence sessions. Join only the current
        # owner's unexpired, ready chat references after the provider read.
        return [self.public_chat(c,titles.get(c['session_id']))
                for c in await asyncio.to_thread(self.ledger.list_chats,principal.user_id) if c["ready"]]

    async def session(self, principal, chat_id):
        await asyncio.to_thread(self.authorize,principal)
        chat=await asyncio.to_thread(self.ledger.chat,principal.user_id,chat_id)
        session=await self.sessions.get_session(app_name=self.config.app_name,
            user_id=session_user(self.config.workspace,principal.user_id),session_id=chat["session_id"])
        if session is None:raise FileNotFoundError("Managed session not found or expired")
        return chat,session

    async def messages(self, principal, chat_id, offset=0):
        chat,session=await self.session(principal,chat_id)
        runs=(await asyncio.to_thread(self.ledger.read))['runs']
        messages=[]
        for event in session.events:
            if event.partial or not event.content:continue
            run=runs.get(event.invocation_id)
            if event.author!='user' and run and run['status']!='completed':continue
            if event.error_code or event.error_message:continue
            # Function calls/responses and model thoughts remain protocol data, never UI messages.
            parts=[p.text for p in event.content.parts or [] if p.text and not p.thought]
            if not parts:continue
            messages.append({"message_id":event.id,"role":"user" if event.author=="user" else "assistant",
                "text":"\n".join(parts),"invocation_id":event.invocation_id,"timestamp":event.timestamp})
        await asyncio.to_thread(self.authorize,principal)
        await asyncio.to_thread(self.ledger.chat,principal.user_id,chat_id)
        return {"messages":messages[offset:offset+30],"next_offset":offset+30 if offset+30<len(messages) else None,
                "chat":self.public_chat(chat,session_title(session))}

    async def send(self, principal, chat_id, request_id, text):
        if not text.strip():raise ValueError("Enter a question")
        chat,session=await self.session(principal,chat_id)
        from google.adk.events import Event, EventActions
        from google.genai import types
        run=await asyncio.to_thread(self.ledger.admit,principal.user_id,chat_id,request_id,text,
            identity_expires_at=principal.expires_at)
        if run["claimed"] or run["status"] in TERMINAL:
            return self.public_run(run)
        try:
            run=await asyncio.to_thread(self.ledger.claim,principal.user_id,run["run_id"])
        except Conflict:
            return self.public_run(await asyncio.to_thread(self.ledger.run,principal.user_id,run["run_id"]))
        # Stable event ID and a read-before-append reconcile timeout/unknown append outcomes.
        event_id=run["message_id"]
        if not any(e.id==event_id for e in session.events):
            title_delta={}
            if not normalize_title(session.state.get(TITLE_KEY)):
                title_delta[TITLE_KEY]=first_question_title(session) or normalize_title(text)
            event=Event(id=event_id,invocation_id=run["run_id"],author="user",
                content=types.Content(role="user",parts=[types.Part(text=text)]),timestamp=time.time(),
                actions=EventActions(state_delta=title_delta))
            try:
                await self.sessions.append_event(session,event)
            except Exception:
                _,fresh=await self.session(principal,chat_id)
                if not any(e.id==event_id for e in fresh.events):
                    await asyncio.to_thread(self.ledger.finish,principal.user_id,run['run_id'],run['fence'],'interrupted')
                    raise RuntimeError('Message append could not be verified') from None
        scope=RunScope(principal,self.config.workspace,run["run_id"],chat_id,run["fence"],
            session.user_id,"evidence-"+run["run_id"][4:],run['request_contract'],run['deadline'])
        token=current_run.set(scope)
        try:
            async with asyncio.timeout(max(.1,run["deadline"]-time.time())):
                await self.execute_until_stopped(scope,session)
            await asyncio.to_thread(self.authorize,principal)
            await asyncio.to_thread(self.ledger.check,principal.user_id,run["run_id"],run["fence"])
            status="completed"
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.to_thread(self.ledger.finish,principal.user_id,run["run_id"],run["fence"],"interrupted"))
            raise
        except Exception:
            status="failed"
        finally:
            scope.stop.set()
            current_run.reset(token)
        await asyncio.to_thread(self.ledger.finish,principal.user_id,run["run_id"],run["fence"],status)
        await self.purge_tasks(run['run_id'])
        return self.public_run(await asyncio.to_thread(self.ledger.run,principal.user_id,run["run_id"]))

    async def execute_until_stopped(self, scope, session):
        """Observe the shared cancellation fence even while a provider call is pending.

        Both tasks belong to this HTTP request. No detached execution/retry is
        introduced; remote provider computation may still finish and be billed.
        """
        async def monitor():
            while True:
                await asyncio.sleep(1)
                await asyncio.to_thread(self.ledger.check,scope.principal.user_id,scope.run_id,scope.fence)
        execution=asyncio.create_task(self.executor(scope,session))
        watcher=asyncio.create_task(monitor())
        try:
            done,_=await asyncio.wait({execution,watcher},return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                scope.stop.set()
                await watcher  # Propagate cancelled/expired durable fence as a normal failure.
            await execution
        finally:
            for task in (execution,watcher):
                if not task.done():task.cancel()
            await asyncio.gather(execution,watcher,return_exceptions=True)

    async def purge_tasks(self, run_id):
        from ..persistence.control_store import mutate
        await asyncio.to_thread(mutate,self.ledger.store,'tasks/'+run_id+'.json',
            lambda old:{'tasks':{},'purged':True})

    @staticmethod
    def public_run(run):
        return {k:run[k] for k in ("run_id","message_id","chat_id","status","cancel_requested","events")}

    async def get_run(self, principal, run_id):
        await asyncio.to_thread(self.authorize,principal)
        run=await asyncio.to_thread(self.ledger.run,principal.user_id,run_id)
        if run["status"] not in TERMINAL and run["deadline"]<=time.time():
            await asyncio.to_thread(self.ledger.finish,principal.user_id,run_id,run["fence"],"interrupted")
            await self.purge_tasks(run_id)
            run=await asyncio.to_thread(self.ledger.run,principal.user_id,run_id)
        return run

    async def cancel(self, principal, run_id):
        await self.get_run(principal,run_id)
        await asyncio.to_thread(self.ledger.cancel,principal.user_id,run_id)
        return self.public_run(await self.get_run(principal,run_id))

    async def delete(self, principal, chat_id):
        await asyncio.to_thread(self.authorize,principal)
        chat=await asyncio.to_thread(self.ledger.chat,principal.user_id,chat_id,include_deleted=True)
        await asyncio.to_thread(self.ledger.mark_deleted,principal.user_id,chat_id)
        await self.sessions.delete_session(app_name=self.config.app_name,
            user_id=session_user(self.config.workspace,principal.user_id),session_id=chat["session_id"])
        from ..persistence.control_store import mutate
        runs=(await asyncio.to_thread(self.ledger.read))['runs']
        for run in runs.values():
            if run['chat_id']!=chat_id:continue
            await self.sessions.delete_session(app_name=self.config.app_name+'_evidence',
                user_id=session_user(self.config.workspace,principal.user_id),session_id='evidence-'+run['run_id'][4:])
            await self.purge_tasks(run['run_id'])
        return {"deleted":True,"chat_id":chat_id}
