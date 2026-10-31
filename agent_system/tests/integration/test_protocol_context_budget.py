"""Exercise full enabled tool schemas and protocol evidence through the real ADK request.

Fixture text is synthetic. This checks context accounting/persistence, not LLM
interpretation; live source-backed acceptance separately checks answer wording.
"""
import asyncio
import json
import time
import socket
import uvicorn
from types import SimpleNamespace
from google.adk import Runner
from research_agents.services.evidence_a2a import create_app, ServiceAuthentication
from research_agents.services.coordinator_runner import CoordinatorExecutor
from research_agents.policies.runtime import ChatConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from research_agents.agents.evidence_specialist.agent import create_agent
from research_agents.policies.agent_guard import AgentGuard
from research_agents.policies.access import Principal
from research_agents.tools.protocol_search import ProtocolChatTool, SOURCE_APPLICABILITY
from test_chat import setup, key
from test_preferences import setup as preference_setup, change


class ProtocolModel(BaseLlm):
    model: str='scripted'
    coordinator: bool=False
    async def generate_content_async(self,llm_request,stream=False):
        responses=[p.function_response for c in llm_request.contents for p in c.parts or [] if p.function_response]
        if self.coordinator:
            part=types.Part(function_call=types.FunctionCall(name='transfer_to_agent',args={'agent_name':'evidence_service'}))
        elif responses:
            evidence=responses[-1].response
            assert evidence['answer']['source_scope']=='original_study'
            assert evidence['answer']['omitted_passage_count']>0
            part=types.Part(text='Original-study evidence only. '+evidence['evidence_id'])
        else:
            part=types.Part(function_call=types.FunctionCall(name='search_study_protocols',args={'study_id':'GSE123','question':'Which original-study processing differs from the current project?'}))
        yield LlmResponse(content=types.Content(role='model',parts=[part]))


def test_full_enabled_evidence_agent_fits_protocol_context_and_keeps_complete_receipt():
    async def run():
        _,chat,_,_=setup();sizes=[]
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        chat.config=ChatConfig('workspace','core_test','copilot','https://research.test',f'http://127.0.0.1:{port}')
        class Guard(AgentGuard):
            async def before_model(self,callback_context,llm_request):
                result=await super().before_model(callback_context,llm_request)
                sizes.append(len(llm_request.model_dump_json(exclude_none=True,exclude={'tools_dict'}).encode()))
                return result
        guard=Guard(chat.ledger,chat.access,'core_test')
        rows=[{'chunk_id':'pc-'+str(i),'study_id':'GSE123','section':'Sample_data_processing',
               'text':'The original investigators processed measured intensities. '*18,
               'source_url':'https://example.org/study/GSE123','document_version':'a'*64} for i in range(4)]
        class Search:
            studies={'GSE123'}
            def search(self,study,question,check):
                check()
                return {'status':'evidence_found','source_scope':'original_study','applicability':SOURCE_APPLICABILITY,
                        'passages':rows,'study_id':study,'withheld_sections':[],
                        'coverage_note':'Synthetic original-study fixture, not current project methods.'},{}
        preferences,_,_,_,pref_principal,_=preference_setup()
        await preferences.save(pref_principal,'display_format',change('table'))
        await preferences.save(pref_principal,'response_style',change('detailed'))
        unused=SimpleNamespace(run=None)
        agent=create_agent(ProtocolModel(),unused,guard,'core_test','core_test_linear',
            protocol_tool=ProtocolChatTool(Search(),guard),preferences=preferences,
            analysis_tool=unused,prediction_tool=unused,training_tool=unused)
        app=create_app(agent,chat.config,chat.sessions,chat.ledger,chat.ledger.store,chat.access,
            ServiceAuthentication(chat.config.evidence_url,'unused',local_token='offline-fixture-token'))
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
        task=asyncio.create_task(server.serve())
        for _ in range(100):
            if server.started:break
            await asyncio.sleep(.01)
        execute=CoordinatorExecutor(chat.config,chat.sessions,lambda:ProtocolModel(coordinator=True),guard,
            lambda:'offline-fixture-token',preferences=preferences,protocol_enabled=True,analysis_enabled=True,prediction_enabled=True,training_enabled=True)
        chat.executor=execute;p=Principal('alice',time.time()+600)
        created=await chat.create(p,key());request=key()
        result=await chat.send(p,created['chat_id'],request,'For GSE123, compare the original processing with the current project implementation.')
        server.should_exit=True;await task
        assert result['status']=='completed',sizes
        assert len(sizes)>=3 and max(sizes)<=16000
        stored=chat.ledger.run('alice',result['run_id'])['receipts'][0]
        assert stored['verified_result']['passages']==rows
        assert 0<len(stored['model_passage_ids'])<len(rows)
        history=await chat.messages(p,created['chat_id'])
        assert 'Original-study evidence only.' in history['messages'][-1]['text']
        assert 'Retrieved protocol excerpts:' in history['messages'][-1]['text']
        assert all(p in history['messages'][-1]['text'] for p in stored['model_passage_ids'])
        assert (await chat.send(p,created['chat_id'],request,'For GSE123, compare the original processing with the current project implementation.'))['run_id']==result['run_id']
        assert (await chat.messages(p,created['chat_id']))==history
    asyncio.run(run())
