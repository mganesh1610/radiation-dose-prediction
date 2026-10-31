"""Opt-in cloud inference/read-only retrieval; isolated Mac control and A2A services.

No production entry point imports this module. No cloud control writes, session
creation, quota expansion, model training or hosted deployment occur here.
"""
import asyncio
import json
import logging
import os
import secrets
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .comparison import ARMS, digest, save_new, parse_direct, validate_questions, check_preflight
from .references import sha


def verify_inputs(frozen, questions, admission, protocol, source):
    validate_questions(frozen,questions)
    for path, field in [(admission,'admission_sha256'),(protocol,'protocol_sha256')]:
        if sha(Path(path).read_bytes()) != frozen[field]: raise ValueError('Frozen input changed: '+field)
    actual = {str(p.relative_to(source)):sha(p.read_bytes()) for p in sorted(Path(source).rglob('*.py'))}
    if actual != frozen['runtime_sha256']: raise ValueError('Runtime changed after freeze')


async def collect(frozen, questions, preflight, admission_path, protocol_path, output, *, continue_after_failure=False):
    """Only question strings enter answering code; references are never supplied to agents."""
    from google.adk import Runner
    from google.adk.agents.run_config import RunConfig
    from google.adk.events import Event
    from google.adk.models import Gemini
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    import uvicorn
    from ..agents.evidence_specialist.agent import create_agent
    from ..services.coordinator_runner import CoordinatorExecutor
    from ..services.evidence_a2a import create_app, ServiceAuthentication
    from ..services.chat_service import ChatService
    from ..persistence.control_store import MemoryControlStore
    from ..persistence.managed_sessions import cloud_credentials
    from ..persistence.ledger import Ledger
    from ..policies.access import AccessPolicy, Grant, Principal
    from ..policies.runtime import ChatConfig
    from ..policies.agent_guard import AgentGuard
    from ..tools.sample_queries import SampleQueries
    from ..tools.bigquery_reader import BigQueryReader
    from ..tools.expression_reader import ExpressionReader, CloudMatrixSource, restore_admission
    from ..tools.matrix_cache import MatrixCache
    from ..tools.chat_tools import ChatTools
    from ..tools.protocol_search import ProtocolEmbeddings, ProtocolSearch, ProtocolChatTool
    from ..audit_cloud import gcloud_token

    check_preflight(preflight,digest(frozen))
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    # Exclusive process lock: fail closed after a crash until operator reconciliation.
    lock = output/'RUNNING.lock'
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600);os.close(fd)
    server = task = None
    try:
        admission = restore_admission(json.loads(Path(admission_path).read_text()))
        protocol = json.loads(Path(protocol_path).read_text())
        owner, workspace = 'benchmark_operator', 'architecture_comparison'
        principal = Principal(owner,time.time()+7200)
        access = AccessPolicy(lambda u,w: Grant(u,w,frozenset({admission.core.release_id}),'benchmark-v1')
                              if (u,w)==(owner,workspace) else None)

        class LocalSessions(InMemorySessionService):
            async def create_session(self, *, expire_time=None, ttl=None, **kwargs):
                return await super().create_session(**kwargs)

        class JournalStore(MemoryControlStore):
            def __init__(self,path):
                super().__init__(); self.path=path
                if path.exists(): self.records=json.loads(path.read_text())
            def write(self,key,value,generation):
                super().write(key,value,generation)
                with self.lock:
                    tmp=self.path.with_suffix('.tmp')
                    with tmp.open('w') as f:
                        json.dump(self.records,f,allow_nan=False);f.flush();os.fsync(f.fileno())
                    tmp.replace(self.path)

        store=JournalStore(output/'control.json'); ledger=Ledger(store,workspace,max_runs=54,max_chats=54)
        sessions=LocalSessions()
        guard=AgentGuard(ledger,access,admission.core.release_id,allowed_studies={m[1] for m in admission.core.members})
        credentials=cloud_credentials()
        # One cached token, same read-only backends for all arms. Never logged/persisted.
        import threading
        from google.auth.transport.requests import Request
        token_lock=threading.Lock()
        def access_token():
            with token_lock:
                if not credentials.valid: credentials.refresh(Request())
                return credentials.token
        def model():
            return Gemini(model=frozen['design']['model'],client_kwargs={'enterprise':True,
                'project':admission.scope.project,'location':'global','credentials':credentials,
                'http_options':types.HttpOptions(timeout=45000,retry_options=types.HttpRetryOptions(attempts=1))})
        lookup=lambda w,r,e=None: admission if w==workspace and r==admission.core.release_id and (e is None or e==admission.expression_set_id) else None
        queries=lambda:SampleQueries(BigQueryReader(admission.scope,token_provider=access_token,max_queries=1),
            access,lambda w,r:lookup(w,r).core if lookup(w,r) else None)
        # Separate initially cold matrix caches prevent one arm warming another.
        toolsets={}
        for arm in ARMS:
            reader=ExpressionReader(MatrixCache(),access,lookup,lambda a:CloudMatrixSource(a.scope,token_provider=access_token))
            toolsets[arm]=ChatTools(queries,reader,guard,admission)
        search=ProtocolSearch(protocol,f'{admission.scope.project}.radiation_protocols_v1.protocol_passages',
            lambda:BigQueryReader(admission.scope,token_provider=access_token,max_queries=1),
            ProtocolEmbeddings(admission.scope.project,credentials))
        protocol_tool=ProtocolChatTool(search,guard)
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        secret=secrets.token_urlsafe(32)
        config=ChatConfig(workspace,admission.core.release_id,'architecture_comparison','https://benchmark.invalid',
            f'http://127.0.0.1:{port}',max_runs=54,max_chats=54)
        evidence=create_agent(model(),toolsets['a2a'],guard,admission.core.release_id,admission.expression_set_id,protocol_tool=protocol_tool)
        app=create_app(evidence,config,sessions,ledger,store,access,
            ServiceAuthentication(config.evidence_url,'unused',local_token=secret))
        server=uvicorn.Server(uvicorn.Config(app,log_level='critical',access_log=False))
        task=asyncio.create_task(server.serve(sockets=[sock]))
        for _ in range(500):
            if server.started:break
            if task.done():raise RuntimeError('Loopback Evidence server failed to start')
            await asyncio.sleep(.01)
        if not server.started:raise TimeoutError('Loopback startup timeout')
        multi=CoordinatorExecutor(config,sessions,model,guard,lambda:secret,protocol_enabled=True)
        current={'arm':None,'coverage':None,'error_type':None}

        async def execute(scope,session):
            arm=current['arm']
            try:
                if arm=='a2a':return await multi(scope,session)
                if arm=='single_adk':
                    agent=create_agent(model(),toolsets[arm],guard,admission.core.release_id,admission.expression_set_id,protocol_tool=protocol_tool)
                    runner=Runner(app_name=config.app_name,agent=agent,session_service=sessions)
                    final=False
                    async for event in runner.run_async(user_id=session.user_id,session_id=session.id,
                            new_message=None,invocation_id=scope.run_id,run_config=RunConfig(max_llm_calls=6)):
                        await guard.authorize()
                        if event.error_code or event.error_message:raise RuntimeError('Single agent failed')
                        if event.is_final_response() and event.content and any(p.text and not p.thought for p in event.content.parts or []):final=True
                    if not final:raise RuntimeError('Single agent produced no final answer')
                    return
                question=next(e.content.parts[0].text for e in reversed(session.events) if e.author=='user')
                parsed=parse_direct(question,admission.core.release_id,admission.expression_set_id)
                if parsed:
                    kind,plan=parsed
                    value=await toolsets[arm].execute(kind,plan)
                    text=json.dumps(value,indent=2,allow_nan=False)
                    current['coverage']='supported'
                else:
                    text='Unsupported by the deterministic query grammar. No data was retrieved or changed. Use an explicit supported lookup or the ADK assistant.'
                    current['coverage']='unsupported'
                await sessions.append_event(session,Event(author='deterministic_baseline',invocation_id=scope.run_id,
                    content=types.Content(role='model',parts=[types.Part(text=text)]),timestamp=time.time()))
            except Exception as exc:
                current['error_type']=type(exc).__name__
                raise

        chat=ChatService(config,ledger,sessions,access,execute)
        prompt_by_id={c['id']:c['question'] for c in questions['cases']}
        for item in frozen['schedule']:
            case_id,arm=item['case_id'],item['arm']; key=case_id+'-'+arm
            destination=output/(key+'.json'); journal_path=output/(key+'.journal.json')
            if destination.exists():
                old=json.loads(destination.read_text())
                if old['question']!=prompt_by_id[case_id] or old['reference_sha256']!=digest(frozen):
                    raise ValueError('Existing result changed')
                if old['status']!='completed' and not continue_after_failure:
                    raise RuntimeError('Prior failure retained; explicit continuation required')
                continue
            if journal_path.exists():raise RuntimeError('Unresolved prior attempt: reconcile without a new model call')
            check_preflight(preflight,digest(frozen))
            journal={'case_id':case_id,'arm':arm,'reference_sha256':digest(frozen),
                'create_request_id':uuid.uuid4().hex,'message_request_id':uuid.uuid4().hex,
                'started_at_utc':datetime.now(timezone.utc).isoformat()}
            save_new(journal_path,journal)  # Durable intent before all paid calls.
            current.update(arm=arm,coverage='supported',error_type=None)
            created=await chat.create(principal,journal['create_request_id'])
            start=time.perf_counter()
            result=await chat.send(principal,created['chat_id'],journal['message_request_id'],prompt_by_id[case_id])
            elapsed=time.perf_counter()-start
            receipt=await chat.get_run(principal,result['run_id'])
            history=await chat.messages(principal,created['chat_id'])
            answer='\n\n'.join(m['text'] for m in history['messages'] if m['role']=='assistant' and m['invocation_id']==result['run_id'])
            record={'case_id':case_id,'arm':arm,'question':prompt_by_id[case_id],
                'reference_sha256':digest(frozen),'status':result['status'],'coverage':current['coverage'],
                'answer':answer,'elapsed_seconds':elapsed,'receipt':receipt,'messages':history['messages'],
                'error_type':current['error_type'],'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                'timing_scope':'chat.send only: local admission/session/control, inference, read-only tools and loopback transport; excludes process startup, create and history read'}
            save_new(destination,record)
            print(json.dumps({'case':case_id,'arm':arm,'status':result['status'],'seconds':round(elapsed,3),
                'model_calls':receipt['model_calls'],'tool_calls':receipt['tool_calls'],'error_type':current['error_type']}),flush=True)
            if result['status']!='completed':
                raise RuntimeError('Failed attempt preserved; inspect before explicitly continuing other cases')
        save_new(output/'complete.json',{'attempts':54,'reference_sha256':digest(frozen),
            'finished_at_utc':datetime.now(timezone.utc).isoformat(),'production_writes':0})
    finally:
        if server is not None:
            server.should_exit = True
            if task is not None: await task
        lock.unlink()
