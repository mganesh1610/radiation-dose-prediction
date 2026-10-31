"""Explicit production factories. Policy/admission files are protected deployment inputs."""
import json
import os
import threading
import time
from pathlib import Path
from .startup_timing import phase
_imports_started=time.perf_counter()
from google.auth.transport.requests import Request
from google.cloud import storage
from google.adk.models import Gemini
from google.genai import types
from ..persistence.managed_sessions import ManagedSessions,cloud_credentials
from ..persistence.control_store import GcsControlStore
from ..persistence.ledger import Ledger
from ..policies.runtime import PolicyFile,Memberships,ChatConfig
from ..policies.access import IapAuthenticator,AccessPolicy
from ..policies.agent_guard import AgentGuard
from ..tools.bigquery_reader import BigQueryReader
from ..tools.sample_queries import SampleQueries
from ..tools.expression_reader import ExpressionReader,CloudMatrixSource,restore_admission
from ..tools.matrix_cache import MatrixCache
from ..tools.chat_tools import ChatTools
from .chat_service import ChatService
from .coordinator_runner import CoordinatorExecutor
from .execution_loop import AgentExecutionLoop
from . import chat_api
phase('bootstrap_imports',_imports_started)


class Components:
    def __init__(self,path):
        started=time.perf_counter()
        self.settings=json.loads(Path(path).read_text());p=self.settings
        self.config=ChatConfig(**p['chat'])
        self.policy=PolicyFile(p['policy_file']);policy=self.policy.read()
        if policy['workspace']!=self.config.workspace or self.config.release_id not in policy['releases']:
            raise ValueError('Protected policy does not admit this deployment')
        raw=Path(p['admission_file']).read_bytes()
        if len(raw)>2_000_000:raise ValueError('Admission too large')
        import hashlib
        if hashlib.sha256(raw).hexdigest()!=p['admission_sha256']:raise ValueError('Admission hash mismatch')
        self.admission=restore_admission(json.loads(raw))
        if self.admission.core.release_id!=self.config.release_id:raise ValueError('Release mismatch')
        phase('protected_configuration',started)
        started=time.perf_counter()
        self.credentials=cloud_credentials();self.lock=threading.Lock()
        self.store=GcsControlStore(p['control_bucket'],'copilot/v1/'+self.config.workspace+'/',
            client=storage.Client(project=p['project'],credentials=self.credentials))
        self.ledger=Ledger(self.store,self.config.workspace,
            max_runs=self.config.max_runs,max_chats=self.config.max_chats)
        self.memberships=Memberships(self.policy,self.store)
        self.access=AccessPolicy(self.memberships.grant)
        self.guard=AgentGuard(self.ledger,self.access,self.config.release_id,
            allowed_studies={member[1] for member in self.admission.core.members})
        self.sessions=ManagedSessions(project=p['project'],location=p['session_location'],
            engine_id=p['engine_id'],credentials=self.credentials,ledger=self.ledger)
        self.auth=IapAuthenticator(p['iap_audience'],self.policy.emails,on_authenticated=self.memberships.bind,
            assertion_header='X-Research-IAP-Assertion')
        phase('credentials_and_control_clients',started)
        started=time.perf_counter()
        self.cache=MatrixCache()
        def lookup(w,r,e=None):
            self.policy.read()  # fail closed if mounted policy is unavailable
            return self.admission if (w,r,e or self.admission.expression_set_id)==(
                self.config.workspace,self.config.release_id,self.admission.expression_set_id) else None
        reader=ExpressionReader(self.cache,self.access,lookup,
            lambda a:CloudMatrixSource(a.scope,token_provider=self.access_token))
        queries=lambda:SampleQueries(BigQueryReader(self.admission.scope,token_provider=self.access_token,max_queries=1),
            self.access,lambda w,r:lookup(w,r).core if lookup(w,r) else None)
        self.tools=ChatTools(queries,reader,self.guard,self.admission)
        from ..persistence.analysis_artifacts import AnalysisArtifacts
        from ..tools.scientific_analysis import AnalysisTool
        self.artifacts=AnalysisArtifacts(self.store)
        self.analysis_enabled=p.get('analysis_enabled',False)
        if type(self.analysis_enabled) is not bool:raise ValueError('Analysis feature flag must be boolean')
        self.analysis_tool=AnalysisTool(queries,reader,self.admission,self.guard,self.artifacts) if self.analysis_enabled else None
        self.prediction_enabled=p.get('prediction_enabled',False)
        if type(self.prediction_enabled) is not bool:raise ValueError('Prediction feature flag must be boolean')
        self.prediction_tool=None
        if self.prediction_enabled and os.environ.get('COPILOT_ROLE')=='evidence':
            from ..tools.prediction_tool import create_prediction_tool
            self.prediction_tool=create_prediction_tool(p,self.guard,self.memberships,self.store)
        self.training_enabled=p.get('training_enabled',False)
        if type(self.training_enabled) is not bool:raise ValueError('Training feature flag must be boolean')
        self.training_tool=None
        if self.training_enabled and os.environ.get('COPILOT_ROLE')=='evidence':
            if not self.prediction_tool:raise ValueError('Bridge requires the verified runtime source directory')
            from ..tools.training_tool import create_training_tool
            self.training_tool=create_training_tool(p,self.guard,self.memberships,self.store)
        self.preferences = None
        self.protocol_tool = None
        for feature in ('memory_enabled', 'protocol_rag_enabled'):
            if type(p.get(feature, False)) is not bool: raise ValueError('Feature flags must be boolean')
        self.feature_users=p.get('phase5_user_ids',[])
        if p.get('memory_enabled') or p.get('protocol_rag_enabled'):
            import re
            if (not isinstance(self.feature_users,list) or not 1<=len(self.feature_users)<=3
                    or not all(isinstance(u,str) and re.fullmatch(r'usr_[a-f0-9]{64}',u) for u in self.feature_users)):
                raise ValueError('Phase 5 requires a protected pilot-user allowlist')
        if p.get('memory_enabled', False):
            from ..persistence.memory_bank import PreferenceMemoryBank
            from ..persistence.preferences import Preferences
            provider = PreferenceMemoryBank(project=p['project'], location=p['session_location'],
                engine_id=p['engine_id'], app_name=self.config.app_name, credentials=self.credentials)
            self.preferences = Preferences(self.config, self.access, self.store, provider, allowed_users=self.feature_users)
        if p.get('protocol_rag_enabled', False):
            from ..tools.protocol_search import ProtocolEmbeddings, ProtocolSearch, ProtocolChatTool
            raw = Path(p['protocol_manifest_file']).read_bytes()
            if len(raw)>500_000 or hashlib.sha256(raw).hexdigest()!=p['protocol_manifest_sha256']:
                raise ValueError('Protocol manifest integrity failure')
            search = ProtocolSearch(json.loads(raw), p['protocol_table'],
                lambda:BigQueryReader(self.admission.scope, token_provider=self.access_token, max_queries=1),
                ProtocolEmbeddings(p['project'],self.credentials))
            if not search.studies.issubset({m[1] for m in self.admission.core.members}):
                raise ValueError('Protocol pilot exceeds admitted core study scope')
            self.protocol_tool = ProtocolChatTool(search, self.guard, allowed_users=self.feature_users)
        phase('tool_configuration',started)

    def access_token(self):
        with self.lock:
            if not self.credentials.valid:self.credentials.refresh(Request())
            return self.credentials.token

    def model(self):
        return Gemini(model=self.config.model,client_kwargs={'enterprise':True,'project':self.settings['project'],
            'location':'global','credentials':self.credentials,'http_options':types.HttpOptions(timeout=45000,
                retry_options=types.HttpRetryOptions(attempts=1))})

    def service_token(self):
        local=os.environ.get('COPILOT_LOOPBACK_TOKEN')
        if local:
            from . import evidence_a2a
            evidence_a2a.ServiceAuthentication(self.config.evidence_url,'unused',local_token=local)
            return local
        from google.oauth2.id_token import fetch_id_token
        return fetch_id_token(Request(),self.config.evidence_url)

    def chat_service(self):
        return ChatService(self.config,self.ledger,self.sessions,self.access,
            AgentExecutionLoop(CoordinatorExecutor(self.config,self.sessions,self.model,self.guard,self.service_token,
                preferences=self.preferences,protocol_enabled=self.protocol_tool is not None,feature_users=self.feature_users,
                analysis_enabled=self.analysis_enabled,prediction_enabled=self.prediction_enabled,training_enabled=self.training_enabled)))

    def evidence_app(self):
        from ..agents.evidence_specialist.agent import create_agent
        from . import evidence_a2a
        agent=create_agent(self.model(),self.tools,self.guard,self.config.release_id,self.admission.expression_set_id,
            protocol_tool=self.protocol_tool,preferences=self.preferences,analysis_tool=self.analysis_tool,prediction_tool=self.prediction_tool,training_tool=self.training_tool)
        return evidence_a2a.create_app(agent,self.config,self.sessions,self.ledger,self.store,self.access,
            evidence_a2a.ServiceAuthentication(self.config.evidence_url,self.settings['coordinator_service_email'],
                local_token=os.environ.get('COPILOT_LOOPBACK_TOKEN')))


def coordinator_app():
    started=time.perf_counter()
    components=Components(os.environ['COPILOT_CONFIG'])
    app=chat_api.create_app(components.chat_service(),components.auth,components.preferences,
        components.protocol_tool is not None,components.feature_users,
        artifacts=components.artifacts if components.analysis_enabled else None,prediction_enabled=components.prediction_enabled,training_enabled=components.training_enabled)
    phase('coordinator_factory',started)
    return app


def evidence_app():
    return Components(os.environ['COPILOT_CONFIG']).evidence_app()
