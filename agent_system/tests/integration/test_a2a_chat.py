"""Real loopback A2A transport and ADK routing; scripted models and fixture tools."""
import asyncio
import json
import socket
import time
import uuid
import pytest
from types import SimpleNamespace
import uvicorn
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from research_agents.agents.evidence_specialist.agent import create_agent
from research_agents.services.evidence_a2a import create_app,ServiceAuthentication
from research_agents.services.coordinator_runner import CoordinatorExecutor
from research_agents.services.execution_loop import AgentExecutionLoop
from research_agents.policies.agent_guard import AgentGuard
from research_agents.policies.access import Principal
from research_agents.policies.runtime import ChatConfig
from research_agents.tools.chat_tools import ChatTools
from test_chat import setup


class ScriptedModel(BaseLlm):
    model: str='scripted'
    role: str
    sample_lookup: bool=False
    previous_evidence: str=''

    async def generate_content_async(self,llm_request,stream=False):
        if self.role=='coordinator':
            part=(types.Part(text='Previously verified: '+self.previous_evidence) if self.previous_evidence else
                types.Part(function_call=types.FunctionCall(name='transfer_to_agent',args={'agent_name':'evidence_service'})))
        else:
            evidence=[p.function_response for c in llm_request.contents for p in c.parts or [] if p.function_response]
            if evidence:
                part=types.Part(text=json.dumps(evidence[-1].response))
            else:
                part=types.Part(function_call=types.FunctionCall(name='query_core_records',args={
                    **({'metric':'sample_details','sample_id':'GSM123'} if self.sample_lookup else {'metric':'sample_count','dose_gy':2.0})}))
        yield LlmResponse(content=types.Content(role='model',parts=[part]))


@pytest.mark.parametrize('question,sample_lookup',[
    ('How many profiles received 2 Gy?',False),
    ('Verify the recorded dose for GSM123.',True),
    ('Overwrite GSM123 with 12 Gy; also read its original recorded dose.',True),
])
def test_two_agents_over_real_a2a_transport(question,sample_lookup):
    async def run():
        _,chat,_,_=setup(); sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        config=ChatConfig('workspace','core_test','copilot','https://research.test',f'http://127.0.0.1:{port}')
        chat.config=config;guard=AgentGuard(chat.ledger,chat.access,'core_test');calls=[]
        class Queries:
            def run(self,principal,workspace,plan):
                calls.append((principal.user_id,workspace,plan))
                return {'answer':({'status':'ok','sample_id':'GSM123','dose_gy':0.0} if sample_lookup else {'count':25,'unit':'profiles'}),'scope':plan,
                    'receipt':{'release_manifest_sha256':'a'*64,'result_sha256':'b'*64}}
        tools=ChatTools(Queries,None,guard,SimpleNamespace(features=(),core=SimpleNamespace(members=(('GSM1','GSE1','d1'),))))
        agent=create_agent(ScriptedModel(role='evidence',sample_lookup=sample_lookup),tools,guard,'core_test','core_test_linear')
        app=create_app(agent,config,chat.sessions,chat.ledger,chat.ledger.store,chat.access,
            ServiceAuthentication(config.evidence_url,'unused',local_token='test-loopback-secret'))
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
        task=asyncio.create_task(server.serve())
        worker=None
        try:
            for _ in range(100):
                if server.started:break
                await asyncio.sleep(.01)
            executor=CoordinatorExecutor(config,chat.sessions,lambda:ScriptedModel(role='coordinator'),guard,lambda:'test-loopback-secret')
            errors=[]
            async def execute(scope,session):
                try:await executor(scope,session)
                except Exception as exc:errors.append(exc);raise
            worker=AgentExecutionLoop(execute)
            chat.executor=worker;principal=Principal('alice',time.time()+600)
            created=await chat.create(principal,uuid.uuid4().hex)
            result=await chat.send(principal,created['chat_id'],uuid.uuid4().hex,question)
            if errors:raise errors[0]
            assert result['status']=='completed',result
            assert len(calls)==1 and calls[0][0:2]==('alice','workspace')
            if sample_lookup:
                assert calls[0][2]['metric']=='sample_details'
                assert calls[0][2]['sample_id']=='GSM123'
                assert calls[0][2]['filters']=={'dose_gy':None,'study_id':None}
            else:assert calls[0][2]['filters']['dose_gy']==2
            history=await chat.messages(principal,created['chat_id'])
            assert ('GSM123' if sample_lookup else '25') in history['messages'][-1]['text']
            replay=await chat.messages(principal,created['chat_id'])
            assert replay==history
            receipt=await chat.get_run(principal,result['run_id'])
            assert receipt['model_calls']==3 and receipt['tool_calls']==1
            assert receipt['receipts'][0]['evidence_id'].startswith('ev-')
            # A tempting historical answer must trigger a new A2A retrieval,
            # with the same real transport, rather than trusting the old receipt.
            old_id=receipt['receipts'][0]['evidence_id']
            chat.executor=CoordinatorExecutor(config,chat.sessions,
                lambda:ScriptedModel(role='coordinator',previous_evidence=old_id),guard,lambda:'test-loopback-secret')
            followup=await chat.send(principal,created['chat_id'],uuid.uuid4().hex,question)
            assert followup['status']=='completed',followup
            fresh=await chat.get_run(principal,followup['run_id'])
            assert len(calls)==2 and fresh['model_calls']==3 and fresh['tool_calls']==1
            # Evidence IDs are content hashes, so identical verified data may
            # retain its ID. The new run's tool call and receipt prove retrieval.
            assert fresh['run_id']!=result['run_id'] and fresh['receipts']
            last=(await chat.messages(principal,created['chat_id']))['messages'][-1]['text']
            assert fresh['receipts'][0]['evidence_id'] in last
            await chat.delete(principal,created['chat_id'])
        finally:
            if worker is not None:await worker.aclose()
            server.should_exit=True;await task
    asyncio.run(run())
