"""Observed I09/I08 defects and neighboring routes, including durable ADK output."""
import asyncio
import time
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from research_agents.policies.request_contract import compile_contract
from research_agents.policies.response_contract import clarify_response
from research_agents.policies.agent_guard import AgentGuard
from research_agents.policies.access import Principal
from research_agents.services.coordinator_runner import CoordinatorExecutor
from test_chat import setup, key

I09 = 'For sample GSM2751839, show TP53 expression. If an exact identifier is needed, tell me which identifier system and format to use.'
BAD_GENE = 'Please provide the numeric NCBI Gene ID (e.g., "7157" for human TP53).'
BAD_UPLOAD = 'For a specific prediction result, provide the relevant job ID.'

@pytest.mark.parametrize('question,missing,present', [
    (I09, 'NCBI Gene ID', 'GSM sample accession'),
    ('Show expression values.', 'GSM sample accession', None),
    ('Retrieve expression for NCBI Gene ID 7157.', 'GSM sample accession', 'numeric NCBI Gene ID'),
])
def test_missing_selectors_are_precise_without_example_mapping(question,missing,present):
    c=compile_contract(question)
    answer=clarify_response(BAD_GENE,c,has_evidence=False)
    assert missing in answer and '7157' not in answer and 'TP53' not in answer
    assert 'No expression value has been retrieved' in answer
    if present:assert present not in answer
    assert clarify_response(answer,c,has_evidence=False)==answer

@pytest.mark.parametrize('question',[
    'For GSM2751839 show expression for NCBI Gene ID 7157.',
    'How many profiles have expression data?',
    'Explain the expression normalization protocol for GSE8917.',
    'Plot NCBI Gene ID 7157 expression by recorded dose in GSE8917.',
    'Why is TP53 expression useful?',
    'Predict dose from expression.',
])
def test_complete_or_other_workflow_not_replaced(question):
    assert clarify_response('Verified result.',compile_contract(question),has_evidence=False)=='Verified result.'


def test_scientific_receipts_are_never_discarded_by_selector_wording_control():
    assert clarify_response('Verified result ev-real.',compile_contract(I09),has_evidence=True)=='Verified result ev-real.'


def test_study_scope_explanation_is_not_overwritten_by_missing_selector_template():
    text='GSE85570 is outside the connected core release; this is not a zero result.'
    c=compile_contract('Show TP53 expression for GSE85570.')
    assert clarify_response(text,c,has_evidence=False)==text


@pytest.mark.parametrize('studies,expected',[
    ({'GSE102971','GSE8917','GSE65292'}, 'outside the connected'),
    (set(), 'outside the connected'),
    ({'GSE85570'}, 'numeric NCBI Gene ID'),
])
def test_study_coverage_comes_from_admission_not_model_guess(studies,expected):
    c=compile_contract('Show TP53 expression for GSE85570.')
    answer=clarify_response(BAD_GENE,c,has_evidence=False,allowed_studies=studies)
    assert expected in answer and '7157' not in answer
    if 'outside' in answer:
        assert 'not a zero result' in answer and 'original study lacks data' in answer

@pytest.mark.parametrize('question',[
    'Show my prediction result.',
    'Does being within 0.5 Gy make a prediction clinically safe?',
    'Estimate dose from my upload.',
])
def test_prediction_requests_use_upload_id_and_explicit_model(question):
    answer=clarify_response(BAD_UPLOAD,compile_contract(question),has_evidence=False)
    assert 'job ID' not in answer
    assert 'upload ID and explicitly selected frozen model' in answer

@pytest.mark.parametrize('question',[
    'What is the status of my training job?',
    'Show training and prediction results.',
    'Status for web_202610_'+'a'*32,
])
def test_training_job_wording_is_preserved(question):
    text='Please provide the training job ID.'
    assert clarify_response(text,compile_contract(question),has_evidence=False)==text

class UnreliableModel(BaseLlm):
    model: str='scripted'
    answer: str
    async def generate_content_async(self,llm_request,stream=False):
        yield LlmResponse(content=types.Content(role='model',parts=[types.Part(text=self.answer)]))

@pytest.mark.parametrize('question,bad,expected,absent',[
    (I09,BAD_GENE,'numeric NCBI Gene ID','7157'),
    ('Show my prediction result.',BAD_UPLOAD,'upload ID','job ID'),
    ('Show TP53 expression for GSE85570.',BAD_GENE,'outside the connected','7157'),
])
def test_corrected_answer_is_persisted_before_user_can_read_history(question,bad,expected,absent):
    async def run():
        _,chat,_,_=setup()
        guard=AgentGuard(chat.ledger,chat.access,'core_test',allowed_studies={'GSE102971'})
        chat.executor=CoordinatorExecutor(chat.config,chat.sessions,
            lambda:UnreliableModel(answer=bad),guard,lambda:'local-test-only')
        p=Principal('alice',time.time()+600)
        c=await chat.create(p,key());request=key()
        result=await chat.send(p,c['chat_id'],request,question)
        assert result['status']=='completed'
        history=await chat.messages(p,c['chat_id'])
        answer=history['messages'][-1]['text']
        assert expected in answer and absent not in answer
        assert chat.ledger.run('alice',result['run_id'])['receipts']==[]
        # Retrying the same request must not call the model or mutate its answer.
        replay=await chat.send(p,c['chat_id'],request,question)
        assert replay['run_id']==result['run_id']
        assert (await chat.messages(p,c['chat_id']))==history
    asyncio.run(run())
