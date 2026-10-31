"""Bounded regressions for hosted control starvation and evidence follow-ups."""
import asyncio
import time
from types import SimpleNamespace
import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from research_agents.services.chat_service import ChatService
from research_agents.persistence.ledger import Ledger
from research_agents.persistence.control_store import QuotaExceeded
from research_agents.policies.access import Principal
from research_agents.policies.agent_guard import AgentGuard
from research_agents.policies.run_scope import RunScope,current_run
from research_agents.policies.protocol_disclosure import verified_protocol_disclosure
from test_chat import setup,key


def test_two_active_requests_allow_other_replica_cancel_and_reject_third():
    async def go():
        entered=[];stopped=[]
        async def execute(scope,session):
            entered.append(scope.run_id)
            try:await asyncio.Event().wait()
            finally:stopped.append(scope.run_id)
        _,service,_,_=setup(execute)
        replica=ChatService(service.config,Ledger(service.ledger.store,'workspace'),service.sessions,service.access,execute)
        a,b=Principal('alice',time.time()+60),Principal('bob',time.time()+60)
        chats=[await service.create(p,key()) for p in (a,b,a)]
        keys=[key(),key()]
        tasks=[asyncio.create_task(service.send(p,c['chat_id'],k,'fixture')) for p,c,k in zip((a,b),chats,keys)]
        for _ in range(500):
            if len(entered)==2:break
            await asyncio.sleep(.01)
        assert len(entered)==2, [(t.result() if t.done() and not t.exception() else str(t.exception()) if t.done() else 'pending') for t in tasks]
        with pytest.raises(QuotaExceeded):await replica.send(a,chats[2]['chat_id'],key(),'overflow')
        own=[service.ledger.chat(p.user_id,c['chat_id'])['active_run'] for p,c in zip((a,b),chats)]
        with pytest.raises(FileNotFoundError):await replica.cancel(a,own[1])
        started=time.monotonic();ack=await replica.cancel(a,own[0])
        assert ack['cancel_requested'] and time.monotonic()-started<3
        result=await asyncio.wait_for(tasks[0],3)
        assert result['status']=='cancelled' and own[0] in stopped and not tasks[1].done()
        assert (await replica.send(a,chats[0]['chat_id'],keys[0],'fixture'))['run_id']==own[0]
        assert len(entered)==2
        assert not [m for m in (await service.messages(a,chats[0]['chat_id']))['messages'] if m['role']=='assistant']
        await replica.cancel(b,own[1]);assert (await asyncio.wait_for(tasks[1],3))['status']=='cancelled'
    asyncio.run(go())


@pytest.mark.parametrize('operation',['service','guard','preferences'])
def test_cloud_membership_read_does_not_block_event_loop(operation):
    async def go():
        _,service,_,_=setup();p=Principal('alice',time.time()+60)
        c=await service.create(p,key());r=service.ledger.admit('alice',c['chat_id'],key(),'fixture');service.ledger.claim('alice',r['run_id'])
        scope=RunScope(p,'workspace',r['run_id'],c['chat_id'],r['fence'],'s','e')
        original=service.access.lookup
        def slow(*args):time.sleep(.25);return original(*args)
        service.access.lookup=slow
        token=current_run.set(scope)
        try:
            if operation=='service':call=service.session(p,c['chat_id'])
            elif operation=='guard':call=AgentGuard(service.ledger,service.access,'core_test').authorize()
            else:
                from test_preferences import setup as pref_setup
                pref,*_=pref_setup();pref.access=service.access
                call=pref._read(p,'display_format')
            task=asyncio.create_task(call);start=time.monotonic()
            await asyncio.sleep(.02)
            assert time.monotonic()-start<.15
            assert not task.done()
            await task
        finally:current_run.reset(token)
    asyncio.run(go())


@pytest.mark.parametrize('with_passage',[False,True])
@pytest.mark.parametrize('prior_owner,prior_chat,model_calls,recover',[
    ('alice','same',1,True),('bob','same',1,False),('alice','other',1,False),('alice','same',2,False)])
def test_known_old_citation_gets_one_fresh_delegation_not_a_trusted_answer(prior_owner,prior_chat,model_calls,recover,with_passage):
    async def go():
        _,service,_,_=setup();p=Principal('alice',time.time()+60)
        c=await service.create(p,key());old=service.ledger.admit('alice',c['chat_id'],key(),'first');service.ledger.claim('alice',old['run_id'])
        service.ledger.progress('alice',old['run_id'],old['fence'],'verified_query',{'evidence_id':'ev-old',**({'tool':'protocol','verified_result':{'passages':[{'chunk_id':'pc-old'}]}} if with_passage else {})})
        service.ledger.finish('alice',old['run_id'],old['fence'],'completed')
        service.ledger.change(lambda s:s['runs'][old['run_id']].update(owner=prior_owner,chat_id=c['chat_id'] if prior_chat=='same' else 'other'))
        run=service.ledger.admit('alice',c['chat_id'],key(),'repeat');service.ledger.claim('alice',run['run_id'])
        service.ledger.change(lambda s:s['runs'][run['run_id']].update(model_calls=model_calls))
        token=current_run.set(RunScope(p,'workspace',run['run_id'],c['chat_id'],run['fence'],'s','e'))
        guard=AgentGuard(service.ledger,service.access,'core_test')
        response=LlmResponse(content=types.Content(role='model',parts=[types.Part(text='Old answer ev-old'+(' pc-old' if with_passage else ''))]))
        try:
            if recover:
                result=await guard.after_model(SimpleNamespace(agent_name='research_coordinator'),response)
                call=result.content.parts[0].function_call
                assert call.name=='transfer_to_agent' and call.args=={'agent_name':'evidence_service'}
                assert not service.ledger.run('alice',run['run_id'])['receipts']
            else:
                with pytest.raises(RuntimeError):await guard.after_model(SimpleNamespace(agent_name='research_coordinator'),response)
            with pytest.raises(RuntimeError):await guard.after_model(SimpleNamespace(agent_name='evidence_specialist'),response)
        finally:current_run.reset(token)
    asyncio.run(go())


@pytest.mark.parametrize('wrong',[
    'Three sections were withheld from retrieval.',
    '*Limitations: GEO removed three sections from retrieval.*',
    'The protocol has 3 unavailable sections.',
    'There are three quarantined sections.',
])
def test_protocol_coverage_count_and_attribution_are_structured(wrong):
    receipt={'tool':'protocol','evidence_id':'ev-protocol','verified_result':{'study_id':'GSE123',
        'withheld_sections':['Sample_hyb_protocol'],'passages':[{'text':'Features excluded by the source investigators.'}]}}
    text='Scanner resolution is 10μm.\n'+wrong
    result=verified_protocol_disclosure(text,[receipt])
    assert 'Scanner resolution is 10μm.' in result and wrong not in result
    assert 'GSE123: 1 section withheld' in result and 'Sample_hyb_protocol' in result
    assert 'remain in the original GEO source' in result
    assert verified_protocol_disclosure(result,[receipt])==result
    assert receipt['verified_result']['withheld_sections']==['Sample_hyb_protocol']


def test_disclosure_distinguishes_zero_withheld_from_omitted_context_and_preserves_source():
    quote='The original study excluded sections from its laboratory protocol.'
    r={'tool':'protocol','verified_result':{'study_id':'GSE123','withheld_sections':[],
        'omitted_passage_count':3,'passages':[{'text':quote}]}}
    answer=verified_protocol_disclosure('"'+quote+'"\nThree sections were withheld.',[r])
    assert quote in answer and '0 sections withheld' in answer and '3 sections' not in answer
    assert verified_protocol_disclosure('Unchanged sample answer.',[])=='Unchanged sample answer.'
