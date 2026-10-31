"""Adversarial evaluator/collector tests, not fabricated scientific demo results."""
from copy import deepcopy
from datetime import datetime,timezone
import json

import httpx
import pytest

from research_agents.evaluation.scoring import digest,contains_exact,report,check_record
from research_agents.evaluation.runner import collect,check_allowance,http_client
from research_agents.evaluation.references import load_frozen


def fixtures():
    expected={'status':'ok','count':5,'unit':'profiles'}
    cases=[{'id':f'F{i:02}','question':f'Question {i}','factual':True,'category':'counts','expected_tool':'query',
            'expected':expected,'expected_receipt':{'parameters_sha256':'parameter-hash'},'review_checks':[]} for i in range(1,31)]
    cases += [{'id':f'B{i:02}','question':f'Boundary {i}','factual':False,'category':'boundary','expected_tool':None,'expected':None,'review_checks':[]} for i in range(1,9)]
    bundle={'suite_id':'unit-test','scope':{'cloud_manifest_sha256':'release-hash'},'gates':{'minimum_correct':27,'human_review_required':True},'cases':cases}
    records=[];reviews={}
    for case in cases:
        r={'case_id':case['id'],'question':case['question'],'reference_sha256':'frozen-hash','status':'completed',
           'run_id':'run-test','answer':'5 profiles, ev-'+'a'*24 if case['factual'] else 'This request is unsupported.',
           'elapsed_seconds':1.0,'receipt':{'run_id':'run-test','status':'completed','model_calls':3,'tool_calls':1 if case['factual'] else 0,
            'receipts':[{'tool':'query','release_manifest_sha256':'release-hash','result_sha256':digest(expected),
                         'evidence_id':'ev-'+'a'*24,'parameters_sha256':'parameter-hash','verified_result':deepcopy(expected)}] if case['factual'] else []}}
        records.append(r)
        reviews[case['id']]={'record_sha256':digest(r),'reviewer':'unit-test','reviewed_at_utc':'2026-10-23T07:00:00+00:00',
                            'review_type':'human','human_attestation':{'personally_reviewed':True,'independent_of_implementation':True,'record_sha256':digest(r)},
                            'correct':True,'numeric_claims_supported':True,'no_fabricated_dose':True,'behavior_safe':True,'notes':'synthetic human fixture; no real review'}
    return bundle,records,reviews


def test_no_live_results_do_not_look_passed():
    b,_,_=fixtures();r,_=report(b,'frozen-hash',[])
    assert r['status']=='incomplete' and r['factual_accuracy'] is None
    assert r['factual_denominator']==30 and len(r['cases'])==38


def test_matching_receipts_require_answer_review():
    b,r,_=fixtures();out,_=report(b,'frozen-hash',r)
    assert out['status']=='incomplete' and out['reviewed']==0
    assert all(x['state']=='needs_review' for x in out['cases'])


def test_review_bound_to_exact_record():
    b,r,v=fixtures();assert report(b,'frozen-hash',r,v,human_reviewers=['unit-test'])[0]['status']=='passed'
    r[0]['answer']='A changed claim'
    assert report(b,'frozen-hash',r,v,human_reviewers=['unit-test'])[0]['status']=='incomplete'


@pytest.mark.parametrize('change,expected_error',[
    (lambda r:r.update(status='failed'),'run_not_completed'),
    (lambda r:r.update(reference_sha256='another'),'wrong_frozen_suite'),
    (lambda r:r['receipt']['receipts'][0].update(result_sha256='wrong'),'evidence_checksum_mismatch'),
    (lambda r:r['receipt']['receipts'][0].update(release_manifest_sha256='wrong'),'release_manifest_mismatch'),
    (lambda r:r['receipt']['receipts'][0].update(parameters_sha256='different-filter'),'provenance_mismatch:parameters_sha256'),
    (lambda r:r['receipt']['receipts'][0]['verified_result'].update(count=0),'reference_mismatch'),
    (lambda r:r.update(answer='5 profiles'),'answer_missing_evidence_id'),
    (lambda r:r['receipt'].update(run_id='other'),'receipt_run_mismatch'),
    (lambda r:r['receipt'].update(receipts=[]),'evidence_missing'),
])
def test_rejects_misleading_evidence(change,expected_error):
    b,r,_=fixtures();change(r[0]);assert expected_error in check_record(b['cases'][0],r[0],b,'frozen-hash')


def test_factual_threshold_does_not_override_safety():
    b,r,v=fixtures()
    for i in range(3):v[r[i]['case_id']]['correct']=False
    assert report(b,'frozen-hash',r,v,human_reviewers=['unit-test'])[0]['status']=='passed'
    v['F01']['no_fabricated_dose']=False
    assert report(b,'frozen-hash',r,v,human_reviewers=['unit-test'])[0]['status']=='failed'


def test_deterministic_failure_not_overridden_by_reviewer():
    b,r,v=fixtures();r[0]['receipt']['receipts'][0]['parameters_sha256']='wrong';v['F01']['record_sha256']=v['F01']['human_attestation']['record_sha256']=digest(r[0])
    assert report(b,'frozen-hash',r,v,human_reviewers=['unit-test'])[0]['status']=='failed'


def test_probe_order_missingness_and_boolean_values_are_not_coerced():
    assert contains_exact({'values':[[None,2.0]],'detected':[[False,True]]},{'values':[[None,2]],'detected':[[False,True]]})
    assert not contains_exact([0,2],[None,2])
    assert not contains_exact([2,1],[1,2])
    assert not contains_exact(True,1)
    assert not contains_exact(float('nan'),0.0)


def test_duplicate_and_unknown_results_rejected():
    b,r,_=fixtures()
    with pytest.raises(ValueError):report(b,'frozen-hash',r+[r[0]])
    r[0]['case_id']='F99'
    with pytest.raises(ValueError):report(b,'frozen-hash',r)


def allowance(runs=100,chats=100):
    return {'checked_at_utc':datetime.now(timezone.utc).isoformat(),'max_runs':runs,'max_chats':chats,
            'runs_used':0,'chats_used':0,'billing_review_completed':True}


def questions():
    b,_,_=fixtures();return {'suite_id':'unit-test','reference_sha256':'frozen-hash','cases':[{'id':c['id'],'question':c['question']} for c in b['cases']]}


def test_insufficient_allowance_makes_no_http_calls(tmp_path):
    def no_call(request):raise AssertionError('Must stop before network')
    with httpx.Client(base_url='https://example.invalid',transport=httpx.MockTransport(no_call)) as client:
        with pytest.raises(ValueError,match='insufficient'):collect(questions(),allowance(20),client,tmp_path/'results')
    assert not (tmp_path/'results').exists()


def test_requires_billing_review_and_fresh_preflight():
    p=allowance();p['billing_review_completed']=False
    with pytest.raises(ValueError,match='usage/credits'):check_allowance(p,38)
    p=allowance();p['checked_at_utc']='2000-01-01T00:00:00+00:00'
    with pytest.raises(ValueError,match='Refresh'):check_allowance(p,38)


def test_auth_redirect_never_sends_question(tmp_path):
    requests=[]
    def transport(request):requests.append(request);return httpx.Response(302,headers={'Location':'https://accounts.example/'})
    with httpx.Client(base_url='https://example.invalid',transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(RuntimeError,match='sign-in'):collect(questions(),allowance(),client,tmp_path/'results')
    assert len(requests)==1 and requests[0].method=='GET'


def test_unknown_outcome_resume_reuses_keys_and_never_logs_auth(tmp_path):
    requests=[]; fail=True
    def transport(request):
        nonlocal fail
        body=json.loads(request.content or '{}');requests.append((request.method,request.url.path,body))
        if request.url.path.endswith('/status'):return httpx.Response(200,json={'enabled':True,'mode':'adk_a2a'})
        if request.url.path.endswith('/chats'):return httpx.Response(200,json={'chat_id':'chat-unit'})
        if request.method=='POST' and request.url.path.endswith('/messages'):
            if fail:raise httpx.ReadTimeout('private-token-do-not-log')
            return httpx.Response(200,json={'run_id':'run-unit','status':'failed'})
        if request.url.path.endswith('/receipt'):return httpx.Response(200,json={'run_id':'run-unit','status':'failed'})
        return httpx.Response(200,json={'messages':[]})
    with httpx.Client(base_url='https://example.invalid',transport=httpx.MockTransport(transport)) as client:
        dest=tmp_path/'results';collect(questions(),allowance(),client,dest)
        first=json.loads((dest/'F01.journal.json').read_text());fail=False
        collect(questions(),allowance(),client,dest,resume=True)
        second=json.loads((dest/'F01.journal.json').read_text())
    assert first['message_request_id']==second['message_request_id']
    assert len([r for r in requests if r[1].endswith('/chats')])==1
    assert 'private-token' not in ''.join(p.read_text() for p in dest.glob('*.json'))
    posted=[r[2] for r in requests if r[0]=='POST']
    assert all(set(x)<= {'request_id','text'} for x in posted)


def test_http_origin_and_reference_tampering_rejected(tmp_path):
    with pytest.raises(ValueError):http_client('http://example.invalid',{})
    (tmp_path/'references.json').write_text('{}');(tmp_path/'references.sha256').write_text('wrong')
    with pytest.raises(ValueError,match='checksum'):load_frozen(tmp_path)


def test_resume_uses_remaining_slots_not_the_full_suite(tmp_path):
    from research_agents.evaluation.runner import pending_requirements
    from research_agents.evaluation.references import sha
    q=questions();b,r,_=fixtures();dest=tmp_path/'results';dest.mkdir()
    (dest/'questions.sha256').write_text(sha(json.dumps(q,sort_keys=True).encode()))
    for record in r[:-1]:
        key=record['case_id']
        (dest/(key+'.json')).write_text(json.dumps(record))
        (dest/(key+'.journal.json')).write_text(json.dumps({'run_id':record['run_id'],'chat_id':'chat-unit'}))
    need=pending_requirements(q,dest,resume=True)
    assert need=={'turns':1,'chats':1,'pending_cases':1}
    check_allowance(allowance(1,1),need['turns'],need['chats'])
    (dest/'B08.journal.json').write_text(json.dumps({'run_id':'known','chat_id':'existing'}))
    need=pending_requirements(q,dest,resume=True)
    assert need=={'turns':0,'chats':0,'pending_cases':1}
    check_allowance(allowance(0,0),need['turns'],need['chats'])


@pytest.mark.parametrize('receipt',[None,[],{'receipts':[None]}])
def test_malformed_receipt_cannot_pass(receipt):
    b,r,v=fixtures();r[0]['receipt']=receipt;v['F01']['record_sha256']=v['F01']['human_attestation']['record_sha256']=digest(r[0])
    assert report(b,'frozen-hash',r,v,human_reviewers=['unit-test'])[0]['status']=='failed'


def test_resumed_and_failed_attempts_not_mislabeled_as_normal_latency():
    b,r,_=fixtures();r[0]['elapsed_seconds']=1000;r[0]['attempt']=2
    r[1]['elapsed_seconds']=2000;r[1]['status']='failed'
    result,_=report(b,'frozen-hash',r)
    assert result['measurements']['timed_cases']==36
    assert result['measurements']['latency_max_seconds']==1
