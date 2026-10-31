"""Regression tests for observed cloud failures, using hostile model/storage doubles."""
import asyncio
import json
import time
from types import SimpleNamespace
import pytest
from google.api_core.exceptions import TooManyRequests, ServiceUnavailable, PreconditionFailed, Forbidden
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from research_agents.persistence import control_store as cs
from research_agents.persistence.ledger import Ledger
from research_agents.policies.request_contract import compile_contract, bind_plan
from research_agents.policies.run_scope import RunScope, current_run
from research_agents.policies.access import Principal
from research_agents.policies.agent_guard import AgentGuard
from research_agents.tools.chat_tools import ChatTools
from research_agents.tools.sample_queries import IntegrityError
from test_chat import setup, key


class ObjectStore:
    """Atomic fake with failure modes before and after provider commit."""
    def __init__(self, errors):
        self.value={'counter':0};self.generation=1;self.errors=list(errors);self.writes=[]
        self.store=object.__new__(cs.GcsControlStore);self.store.blob=lambda key:self
    @property
    def size(self):return len(json.dumps(self.value))
    def reload(self, **kwargs):pass
    def download_as_bytes(self, **kwargs):return json.dumps(self.value).encode()
    def upload_from_string(self, data, **kwargs):
        self.writes.append(kwargs['if_generation_match'])
        if kwargs['if_generation_match']!=self.generation:raise PreconditionFailed('raced')
        mode=self.errors.pop(0) if self.errors else None
        if isinstance(mode, Exception):raise mode
        self.value=json.loads(data);self.generation+=1
        if mode=='committed_then_other_writer':
            self.value['unrelated']=True;self.generation+=1
            raise ServiceUnavailable('response lost')
        if mode=='committed':raise TimeoutError('response lost')
        if mode=='uncertain_no_marker':
            self.value.pop(cs.CONTROL_OPS);self.generation+=1
            raise TimeoutError('unverifiable')


@pytest.mark.parametrize('failures',[[TooManyRequests('rate'),TooManyRequests('rate')],['committed'],['committed_then_other_writer']])
def test_storage_retry_applies_increment_exactly_once(failures,monkeypatch):
    waits=[];monkeypatch.setattr(cs,'backoff',lambda attempt,deadline:waits.append(attempt))
    storage=ObjectStore(failures);calls=[]
    def increment(old):calls.append(1);old['counter']+=1;return old
    cs.mutate(storage.store,'ledger.json',increment)
    assert storage.value['counter']==1 and len(calls)==1
    assert set(storage.writes)=={1}  # Every retry uses the original CAS generation.
    if len(failures)==2:assert waits==[0,1]


def test_exhausted_storage_retry_is_bounded_and_not_a_logical_retry(monkeypatch):
    monkeypatch.setattr(cs,'backoff',lambda *args:None)
    storage=ObjectStore([TooManyRequests('rate')]*10);calls=[]
    def increment(old):calls.append(1);old['counter']+=1;return old
    with pytest.raises(cs.StorageUnavailable):cs.mutate(storage.store,'ledger.json',increment)
    assert len(storage.writes)==4 and len(calls)==1 and storage.value['counter']==0


def test_unknown_commit_without_retained_marker_fails_closed(monkeypatch):
    monkeypatch.setattr(cs,'backoff',lambda *args:None)
    storage=ObjectStore(['uncertain_no_marker']);calls=[]
    def increment(old):calls.append(1);old['counter']+=1;return old
    with pytest.raises(cs.StorageUnavailable):cs.mutate(storage.store,'ledger.json',increment)
    assert len(calls)==1 and len(storage.writes)==1


def test_storage_permission_errors_are_not_retried():
    storage=ObjectStore([Forbidden('denied')])
    with pytest.raises(Forbidden):cs.mutate(storage.store,'ledger.json',lambda old:old)
    assert len(storage.writes)==1


def test_conflict_before_commit_recomputes_against_fresh_state(monkeypatch):
    monkeypatch.setattr(cs,'backoff',lambda *args:None)
    storage=ObjectStore([PreconditionFailed('concurrent writer')]);calls=[]
    def increment(old):calls.append(1);old['counter']+=1;return old
    cs.mutate(storage.store,'ledger.json',increment)
    assert storage.value['counter']==1 and len(calls)==2


def test_rate_limited_read_recovers_without_treating_ledger_as_missing(monkeypatch):
    waits=[];monkeypatch.setattr(cs,'backoff',lambda *args:waits.append(1))
    storage=ObjectStore([]);attempts=[]
    def reload(**kwargs):
        attempts.append(1)
        if len(attempts)==1:raise TooManyRequests('rate limited')
    storage.reload=reload
    assert storage.store.read('ledger.json')==({'counter':0},'1')
    assert waits==[1]


def test_identity_binding_accepts_control_metadata_but_rejects_different_email():
    from research_agents.policies.runtime import Memberships
    policy=SimpleNamespace(emails=lambda:['one@example.test','two@example.test'])
    store=cs.MemoryControlStore();members=Memberships(policy,store);p=Principal('alice',time.time()+60)
    members.bind(p,'one@example.test');generation=store.read('identities/alice.json')[1]
    assert store.read('identities/alice.json')[0]=={'user_id':'alice','email':'one@example.test'}
    members.bind(p,'one@example.test')
    assert store.read('identities/alice.json')[1]==generation
    with pytest.raises(PermissionError):members.bind(p,'two@example.test')


@pytest.mark.parametrize('cancelled',[False,True])
def test_storage_does_not_write_after_deadline_or_cancellation(cancelled):
    scope=RunScope(Principal('alice',time.time()+100),'w','r','c','f','s','e',deadline=time.time()+(-1 if not cancelled else 100))
    if cancelled:scope.stop.set()
    token=current_run.set(scope);storage=ObjectStore([])
    try:
        with pytest.raises(cs.StorageUnavailable):cs.mutate(storage.store,'ledger.json',lambda old:old)
        assert storage.writes==[]
    finally:current_run.reset(token)


def test_marker_history_is_bounded_and_retained_by_replacement_operations():
    store=cs.MemoryControlStore()
    for i in range(140):cs.mutate(store,'ledger.json',lambda old:{'n':i})
    state,_=store.read('ledger.json')
    assert state['n']==139 and len(state[cs.CONTROL_OPS])==128


def count_plan(metric='study_group_count',study='GSE102971'):
    return {'metric':metric,'sample_id':None,'filters':{'study_id':study,'dose_gy':0.0},'release_id':'core_test'}


@pytest.mark.parametrize('unit,metric',[('profiles','sample_count'),('samples','sample_count'),('study-local donor groups','study_group_count')])
def test_original_request_binds_count_unit_even_when_model_changes_it(unit,metric):
    contract=compile_contract(f'How many {unit} belong to GSE102971 at 0 Gy?')
    bound,error=bind_plan('query',count_plan(),contract,{'GSE102971'})
    assert error is None and bound['metric']==metric


def test_ambiguous_count_and_changed_study_require_clarification():
    for question in ['How many profiles and donors are in GSE102971?', 'How many profiles in GSE8917?']:
        plan,error=bind_plan('query',count_plan(),compile_contract(question),{'GSE102971','GSE8917'})
        assert plan is None and error['status']=='clarification_required'


def test_unsupported_study_does_not_become_zero_count_or_backend_exception():
    plan,error=bind_plan('query',count_plan(study='GSE85570'),compile_contract('How many GSE85570 profiles?'),{'GSE102971'})
    assert plan is None and error['status']=='unsupported' and 'zero' in error['reason']


@pytest.mark.parametrize('question,allowed',[
    ('Give TP53 expression for GSM2751832.',False),
    ('TP53 at 7157 Gy for GSM2751832.',False),
    ('For GSM2751832 retrieve NCBI Gene ID 7157.',True),
    ('For GSM2751832 retrieve NCBI Gene IDs 7157, 9100 and 400451.',True),
])
def test_numeric_gene_ids_must_come_from_explicit_gene_id_selector(question,allowed):
    bound,error=bind_plan('expression',{'sample_ids':['GSM2751832'],'gene_ids':['7157']},compile_contract(question),set())
    assert (bound is not None)==allowed
    if not allowed:assert error['status']=='clarification_required'


def test_admission_contract_is_not_changed_on_idempotent_retry():
    _,chat,_,_=setup();c=chat.ledger.create_chat('alice',key(),'core_test');chat.ledger.ready('alice',c['chat_id'])
    request=key();r=chat.ledger.admit('alice',c['chat_id'],request,'How many profiles in GSE102971?')
    assert r['request_contract']['count_metric']=='sample_count'
    assert chat.ledger.admit('alice',c['chat_id'],request,'How many profiles in GSE102971?')['request_contract']==r['request_contract']


def test_tools_block_symbol_translation_and_reject_wrong_returned_units():
    async def run():
        _,chat,_,_=setup();p=Principal('alice',time.time()+600);c=await chat.create(p,key())
        r=chat.ledger.admit('alice',c['chat_id'],key(),'How many profiles in GSE102971?');chat.ledger.claim('alice',r['run_id'])
        scope=RunScope(p,'workspace',r['run_id'],c['chat_id'],r['fence'],'s','e',r['request_contract'],r['deadline'])
        guard=AgentGuard(chat.ledger,chat.access,'core_test');calls=[]
        class Queries:
            def run(self,*args):calls.append(1);return {'answer':{'count':20,'unit':'study_local_donor_groups'}}
        tools=ChatTools(Queries,None,guard,SimpleNamespace(core=SimpleNamespace(members=(('GSM1','GSE102971','d'),)),features=()))
        token=current_run.set(scope)
        try:
            result=await tools.expression({'gene_ids':['7157'],'sample_ids':['GSM1']})
            assert result['status']=='clarification_required' and not calls
            with pytest.raises(IntegrityError):await tools.query(count_plan())
        finally:current_run.reset(token)
    asyncio.run(run())


@pytest.mark.parametrize('text',['Sample not found.','The result is not recorded.'])
def test_final_answer_has_durable_verified_evidence_id(text):
    async def run():
        _,chat,_,_=setup();p=Principal('alice',time.time()+600);c=await chat.create(p,key())
        r=chat.ledger.admit('alice',c['chat_id'],key(),'Look up GSM999.');chat.ledger.claim('alice',r['run_id'])
        chat.ledger.progress('alice',r['run_id'],r['fence'],'verified_query',{'evidence_id':'ev-abc123'})
        scope=RunScope(p,'workspace',r['run_id'],c['chat_id'],r['fence'],'s','e')
        token=current_run.set(scope);guard=AgentGuard(chat.ledger,chat.access,'core_test')
        try:
            response=LlmResponse(content=types.Content(role='model',parts=[types.Part(text=text)]))
            corrected=await guard.after_model(None,response)
            assert 'ev-abc123' in corrected.content.parts[-1].text
            assert len(response.content.parts)==1  # Original SDK response preserved.
            assert await guard.after_model(None,corrected) is None
            bad=LlmResponse(content=types.Content(role='model',parts=[types.Part(text='Evidence: ev-invented')]))
            with pytest.raises(RuntimeError):await guard.after_model(None,bad)
        finally:current_run.reset(token)
    asyncio.run(run())


@pytest.mark.parametrize('label',['Evidence ID: not_found.','**Evidence ID:** N/A','Evidence: unsupported'])
def test_scope_refusal_never_presents_status_placeholder_as_evidence(label):
    async def run():
        _,chat,_,_=setup();p=Principal('alice',time.time()+600);c=await chat.create(p,key())
        r=chat.ledger.admit('alice',c['chat_id'],key(),'How many profiles in GSE85570?');chat.ledger.claim('alice',r['run_id'])
        scope=RunScope(p,'workspace',r['run_id'],c['chat_id'],r['fence'],'s','e')
        token=current_run.set(scope);guard=AgentGuard(chat.ledger,chat.access,'core_test')
        try:
            response=LlmResponse(content=types.Content(role='model',parts=[types.Part(text='Study is outside the connected release. '+label)]))
            corrected=await guard.after_model(None,response)
            assert corrected.content.parts[0].text=='Study is outside the connected release. No source receipt was generated for this request.'
            assert label in response.content.parts[0].text
        finally:current_run.reset(token)
    asyncio.run(run())
