import copy
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from research_agents.policies.model_context import fit_protocol_context, request_bytes


def request(tool='search_study_protocols',evidence_id='ev-approved'):
    payload={'evidence_id':evidence_id,'answer':{'source_scope':'original_study','status':'evidence_found',
       'omitted_passage_count':1,'passages':[{'chunk_id':'pc-first','text':'First exact passage: 10μm.'},
                                         {'chunk_id':'pc-second','text':'Complete second method. '*100}]}}
    return LlmRequest(contents=[types.Content(role='user',parts=[types.Part(text='Original question')]),
        types.Content(role='tool',parts=[types.Part(function_response=types.FunctionResponse(name=tool,response=payload))])])


def test_trim_whole_last_passage_only_in_outbound_copy():
    stored=request();original=stored.model_dump_json();outbound=stored.model_copy(deep=True)
    limit=request_bytes(outbound)-1000
    fit_protocol_context(outbound,{'ev-approved'},limit)
    result=outbound.contents[-1].parts[0].function_response.response['answer']
    assert request_bytes(outbound)<=limit and len(result['passages'])==1
    assert result['passages'][0]==stored.contents[-1].parts[0].function_response.response['answer']['passages'][0]
    assert result['omitted_passage_count']==2 and result['status']=='evidence_found'
    assert stored.model_dump_json()==original
    assert outbound.contents[0]==stored.contents[0]


def test_never_compress_unverified_or_non_protocol_results():
    for r in [request(evidence_id='ev-unverified'),request(tool='get_expression_values')]:
        original=r.model_dump_json();fit_protocol_context(r,{'ev-approved'},100)
        assert r.model_dump_json()==original


def test_empty_passages_explicitly_report_insufficient_context():
    r=request();fit_protocol_context(r,{'ev-approved'},1)
    answer=r.contents[-1].parts[0].function_response.response['answer']
    assert answer['passages']==[] and answer['status']=='insufficient_context_budget'
    assert answer['omitted_passage_count']==3
    assert request_bytes(r)>1  # Caller must still enforce the hard limit, never bypass it.
