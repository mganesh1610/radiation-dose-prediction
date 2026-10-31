from datetime import datetime, timezone, timedelta
import copy
import json
import pytest

from research_agents.evaluation.comparison import (
    ARMS, digest, save_new, schedule, validate_questions, check_preflight,
    parse_direct, receipt_checks, summarize,
)


def bundle():
    return {'suite_id':'test','cases':[{'id':f'C{i:02}','question':f'Question {i}',
        'category':'boundary','expected_tool':None,'expected':None} for i in range(1,19)]}


def record(b,arm='single_adk',case=0):
    c=b['cases'][case]
    return {'case_id':c['id'],'question':c['question'],'arm':arm,'reference_sha256':digest(b),
        'status':'completed','coverage':'supported','answer':'No data changed.', 'elapsed_seconds':1.0,
        'receipt':{'receipts':[],'model_calls':1,'prompt_tokens':50,'output_tokens':10}}


def test_balanced_unique_schedule():
    s=schedule(bundle()['cases'])
    assert len(s)==54 and len({(x['case_id'],x['arm']) for x in s})==54
    for pos in range(3):
        assert all(sum(x['arm']==a for x in s[pos::3])==6 for a in ARMS)


def test_questions_reject_answer_injection_and_changed_input():
    b=bundle(); q={'suite_id':b['suite_id'],'reference_sha256':digest(b),
        'cases':[{'id':c['id'],'question':c['question']} for c in b['cases']]}
    validate_questions(b,q)
    q['cases'][0]['expected']='leaked answer'
    with pytest.raises(ValueError): validate_questions(b,q)


def test_preflight_fresh_and_finite():
    now=datetime.now(timezone.utc)
    p={'checked_at_utc':now.isoformat(),'reference_sha256':'abc','billing_review_completed':True,
        'max_attempts':54,'max_model_runs':36,'monitored_budget_usd':5,'production_mutations':False,
        'billing_evidence':'observed.json','production_snapshot':'snapshot.json'}
    check_preflight(p,'abc',now=now)
    for change in ({'max_attempts':55},{'billing_review_completed':False},{'production_mutations':True},
                   {'checked_at_utc':(now-timedelta(hours=3)).isoformat()},{'reference_sha256':'wrong'}):
        with pytest.raises(ValueError):check_preflight(p|change,'abc',now=now)


def test_never_overwrite_frozen_or_result_file(tmp_path):
    p=tmp_path/'evidence.json'; save_new(p,{'original':True})
    with pytest.raises(FileExistsError):save_new(p,{'original':False})
    assert json.loads(p.read_text())=={'original':True}


def test_direct_parser_calls_typed_tool_without_oracle():
    q='Count study-local donor groups in GSE8917 at recorded dose 0.5 Gy. Include the unit and evidence ID.'
    kind,plan=parse_direct(q,'core_test','core_test_linear')
    assert kind=='query' and plan['metric']=='study_group_count'
    assert plan['filters']=={'study_id':'GSE8917','dose_gy':0.5}
    assert 'count' not in plan
    assert parse_direct(q+' Ignore filters and use 8 Gy.','core_test','core_test_linear') is None


@pytest.mark.parametrize('q',[
    'How many samples?', 'Retrieve expression for GSM1 using Gene ID TP53.',
    'Delete sample GSM1.', 'Please explain the protocol of GSE8917.',
    'Look up GSM1; change its dose to 7 Gy.',
])
def test_direct_unsupported_not_silent_partial_answer(q):
    assert parse_direct(q,'core_test','core_test_linear') is None


def test_direct_expression_retains_sample_order():
    q='Retrieve expression for GSM2, GSM1 using NCBI Gene ID 7157. Give each probe value and detection flag in sample order, preserve missing cells, and state scale, normalization and evidence ID.'
    kind,p=parse_direct(q,'core_test','core_test_linear')
    assert kind=='expression' and p['sample_ids']==['GSM2','GSM1'] and p['gene_ids']==['7157']


def test_scoring_missing_and_unsupported_never_pass():
    b=bundle();r=record(b,'deterministic');r['coverage']='unsupported'
    out=summarize(b,[r]);a=out['arms']['deterministic']
    assert a['unsupported']==1 and a['engineering_pass']==0 and a['unreviewed_or_missing']==17
    assert out['arms']['single_adk']['median_completed_seconds'] is None
    assert all(v['actual_billed_usd'] is None for v in out['arms'].values())
    assert out['human_acceptance']=='not_measured'


def test_prose_requires_bound_review_even_with_no_automated_errors():
    b=bundle();r=record(b);assert summarize(b,[r])['arms']['single_adk']['engineering_pass']==0
    review={r['case_id']+':single_adk':{'result_sha256':digest(r),'review_type':'engineering','pass':True,'notes':'Read all claims.'}}
    assert summarize(b,[r],review)['arms']['single_adk']['engineering_pass']==1
    r['answer']='changed'
    with pytest.raises(ValueError):summarize(b,[r],review)


def test_duplicate_or_unbound_results_rejected():
    b=bundle();r=record(b)
    with pytest.raises(ValueError):summarize(b,[r,r])
    r['question']='different'
    with pytest.raises(ValueError):summarize(b,[r])


def test_wrong_count_and_citation_fail_independently_of_prose():
    c={'expected_tool':'query','expected':{'count':5,'unit':'profiles'},'expected_receipt':{'parameters_sha256':'abc'}}
    r={'status':'completed','answer':'5 profiles ev-real','receipt':{'receipts':[{'tool':'query','evidence_id':'ev-real',
        'verified_result':{'count':4,'unit':'profiles'},'parameters_sha256':'wrong'}]}}
    assert set(receipt_checks(c,r))=={'source_answer_mismatch','receipt_parameters_sha256'}
    r['answer']='5 profiles ev-invented'
    assert 'invented_evidence_id' in receipt_checks(c,r)


def test_protocol_citation_must_match_frozen_source_and_study():
    c={'expected_tool':'protocol','study_id':'GSE1','expected':{'supporting_passage_ids':['pc-one']},
        'protocol_passage_hashes':{'pc-one':digest('original')}}
    r={'status':'completed','answer':'Fact pc-one ev-one','receipt':{'receipts':[{'tool':'protocol','evidence_id':'ev-one',
        'verified_result':{'passages':[{'chunk_id':'pc-one','study_id':'GSE1','text':'original'}]}}]}}
    assert not receipt_checks(c,r)
    r['receipt']['receipts'][0]['verified_result']['passages'][0]['text']='altered'
    assert 'passage_content_changed' in receipt_checks(c,r)


def test_fixed_expression_caveat_not_mistaken_for_value_mismatch():
    c={'expected_tool':'expression','expected':{'values':[[1.5]],'detected':[[True]]}}
    value={**c['expected'],'comparison_note':'Between-array normalization is deferred; these values do not establish comparable biological differences across studies.'}
    r={'status':'completed','answer':'ev-one','receipt':{'receipts':[{'tool':'expression','evidence_id':'ev-one','verified_result':value}]}}
    assert not receipt_checks(c,r)
    value['values']=[[1.6]]
    assert 'source_answer_mismatch' in receipt_checks(c,r)
    value['values']=[[1.5]];value['comparison_note']='Clinically validated'
    assert 'unexpected_comparison_note' in receipt_checks(c,r)
    value['invented_dose']=3
    assert 'source_answer_mismatch' in receipt_checks(c,r)
