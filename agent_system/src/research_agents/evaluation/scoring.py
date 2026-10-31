"""Evidence parity is automated; acceptance also requires bound human answer review."""
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone

REVIEW_CHECKS = ('correct', 'numeric_claims_supported', 'no_fabricated_dose', 'behavior_safe')
REVIEW_TYPES = ('human', 'ai_peer_review', 'engineering')


def review_errors(review, record_sha):
    """Validate a declared review without inferring a human from a name or notes."""
    errors = []
    if review.get('review_type') not in REVIEW_TYPES: errors.append('review_type_missing_or_invalid')
    if not record_sha or review.get('record_sha256') != record_sha: errors.append('review_record_mismatch')
    for key in ('reviewer', 'notes'):
        if not isinstance(review.get(key), str) or not review[key].strip(): errors.append(key + '_missing')
    try:
        when = datetime.fromisoformat(review['reviewed_at_utc'])
        if when.utcoffset() != timezone.utc.utcoffset(when): raise ValueError('UTC required')
    except (KeyError, TypeError, ValueError):
        errors.append('review_time_invalid')
    if not all(type(review.get(key)) is bool for key in REVIEW_CHECKS): errors.append('review_checks_incomplete')
    return errors


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def contains_exact(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and contains_exact(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(contains_exact(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and math.isfinite(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def check_record(case, record, bundle, reference_sha):
    errors = []
    if record.get('case_id') != case['id'] or record.get('question') != case['question']:
        errors.append('case_or_question_mismatch')
    if record.get('reference_sha256') != reference_sha: errors.append('wrong_frozen_suite')
    if record.get('status') != 'completed': errors.append('run_not_completed')
    if not isinstance(record.get('answer'), str) or not record['answer'].strip(): errors.append('answer_missing')
    receipt = record.get('receipt', {})
    if not isinstance(receipt,dict): return errors + ['invalid_receipt']
    if receipt.get('status') != 'completed' or receipt.get('run_id') != record.get('run_id'):
        errors.append('receipt_run_mismatch')
    evidence = receipt.get('receipts', [])
    if not isinstance(evidence, list): return errors + ['invalid_evidence']
    if case['expected'] is not None and not evidence: errors.append('evidence_missing')
    for item in evidence:
        if not isinstance(item,dict):
            errors.append('invalid_evidence'); continue
        if item.get('tool') not in ('query', 'expression'): errors.append('unexpected_tool')
        if item.get('release_manifest_sha256') != bundle['scope']['cloud_manifest_sha256']:
            errors.append('release_manifest_mismatch')
        result = item.get('verified_result')
        if result is None or item.get('result_sha256') != digest(result): errors.append('evidence_checksum_mismatch')
        evidence_id = item.get('evidence_id', '')
        valid_id = isinstance(evidence_id,str) and evidence_id.startswith('ev-') and len(evidence_id)==27 and all(c in '0123456789abcdef' for c in evidence_id[3:])
        if not valid_id: errors.append('invalid_evidence_id')
        if case['expected'] is not None:
            if item.get('tool') != case['expected_tool'] or not contains_exact(result, case['expected']):
                errors.append('reference_mismatch')
            for key, value in case.get('expected_receipt', {}).items():
                if item.get(key) != value: errors.append('provenance_mismatch:' + key)
            if not valid_id or not isinstance(record.get('answer'),str) or evidence_id not in record['answer']: errors.append('answer_missing_evidence_id')
    if type(receipt.get('model_calls')) is not int or not 1 <= receipt['model_calls'] <= 6:
        errors.append('model_call_count_invalid')
    if type(receipt.get('tool_calls')) is not int or not 0 <= receipt['tool_calls'] <= 4:
        errors.append('tool_call_count_invalid')
    if type(receipt.get('tool_calls')) is int and len(evidence) > receipt['tool_calls']:
        errors.append('tool_receipt_count_mismatch')
    return sorted(set(errors))


def report(bundle, reference_sha, records, reviews=None, *, human_reviewers=()):
    # This is the formal acceptance report. Provisional reviews may be displayed
    # here, but a different caller flag must not turn them into human acceptance.
    if bundle['gates'].get('human_review_required', True) is not True:
        raise ValueError('Acceptance reporting requires human review')
    if (not isinstance(human_reviewers, (list, tuple, set, frozenset))
            or any(not isinstance(name, str) or not name.strip() for name in human_reviewers)):
        raise ValueError('Authorized human reviewers must be a separate list of identifiers')
    authorized = set(human_reviewers)
    reviews = reviews or {}
    by_id = {}
    for record in records:
        key = record['case_id']
        if key in by_id: raise ValueError('Duplicate result case: ' + key)
        if key not in {c['id'] for c in bundle['cases']}: raise ValueError('Unknown result case: ' + key)
        by_id[key] = record
    rows = []; template = {}
    for case in bundle['cases']:
        record = by_id.get(case['id'])
        errors = check_record(case, record, bundle, reference_sha) if record else ['not_executed']
        record_sha = digest(record) if record else None
        review = reviews.get(case['id'], {})
        review = review if isinstance(review, dict) else {}
        keys = REVIEW_CHECKS
        issues = review_errors(review, record_sha)
        provisional = not issues and review['review_type'] in ('ai_peer_review', 'engineering')
        if review.get('review_type') != 'human': issues.append('human_review_required')
        elif isinstance(review.get('reviewer'), str) and review['reviewer'] not in authorized:
            issues.append('human_reviewer_not_authorized')
        attestation = review.get('human_attestation')
        if (not isinstance(attestation, dict)
                or attestation.get('personally_reviewed') is not True
                or attestation.get('independent_of_implementation') is not True
                or not record_sha or attestation.get('record_sha256') != record_sha):
            issues.append('human_attestation_missing_or_unbound')
        reviewed = bool(record and not issues)
        critical = reviewed and not errors and all(review[k] for k in keys[1:])
        state = 'not_executed' if not record else ('fail' if errors else ('needs_review' if not reviewed else ('pass' if review['correct'] and critical else 'fail')))
        rows.append({'case_id': case['id'], 'factual':case['factual'], 'category':case['category'],
                     'state':state, 'automated_errors':errors, 'reviewed':reviewed, 'critical_checks_passed':bool(critical),
                     'record_sha256':record_sha, 'review_type':review.get('review_type'), 'review_errors':issues,
                     'provisional_state':('pass' if not errors and all(review[k] for k in keys) else 'fail') if provisional else None})
        template[case['id']] = {'record_sha256':record_sha, 'review_type':None, 'reviewer':None, 'reviewed_at_utc':None,
                              'human_attestation':{'personally_reviewed':None, 'independent_of_implementation':None,
                                                   'record_sha256':record_sha},
                              **{k:None for k in keys}, 'notes':'', 'required_checks':case['review_checks']}
    factual = [r for r in rows if r['factual']]; boundary = [r for r in rows if not r['factual']]
    correct = sum(r['state'] == 'pass' for r in factual)
    complete = all(r['reviewed'] for r in rows)
    passed = complete and correct >= bundle['gates']['minimum_correct'] and all(r['state']=='pass' for r in boundary) and all(r['critical_checks_passed'] for r in rows)
    timings = [r['elapsed_seconds'] for r in records if r.get('status')=='completed' and r.get('attempt',1)==1 and type(r.get('elapsed_seconds')) in (float,int) and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] >= 0]
    def total(field):
        values = [r['receipt'].get(field,0) for r in records if isinstance(r.get('receipt'),dict)]
        return sum(v for v in values if type(v) in (int,float) and math.isfinite(v) and v>=0)
    result = {'suite_id':bundle['suite_id'],'scorer_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),'reference_sha256':reference_sha,'status':'passed' if passed else ('failed' if complete else 'incomplete'),
              'factual_passed':correct,'factual_denominator':30,'factual_accuracy':correct/30 if complete else None,
              'boundary_passed':sum(r['state']=='pass' for r in boundary),'boundary_denominator':8,
              'executed':len(by_id),'reviewed':sum(r['reviewed'] for r in rows),
              'human_reviewed':sum(r['reviewed'] for r in rows),
              'ai_reviewed':sum(r['provisional_state'] is not None and r['review_type']=='ai_peer_review' for r in rows),
              'engineering_reviewed':sum(r['provisional_state'] is not None and r['review_type']=='engineering' for r in rows),
              'review_policy':{'human_review_required':True, 'authorized_reviewers_sha256':digest(sorted(authorized)),
                               'authorized_reviewer_count':len(authorized)}, 'cases':rows,
              'measurements':{'latency_median_seconds':statistics.median(timings) if timings else None,
                              'latency_max_seconds':max(timings) if timings else None,'timed_cases':len(timings),
                              'model_calls':total('model_calls'),'tool_calls':total('tool_calls'),
                              'prompt_tokens_reported':total('prompt_tokens'),'output_tokens_reported':total('output_tokens'),
                              'reserved_model_usd':total('reserved_model_usd'),'actual_billed_cost_usd':None},
              'limits':['Human review is required for meaning, additional claims and critical dose/permission checks.',
                        'AI and engineering judgments are provisional and never satisfy the human acceptance gate.',
                        'Human attestation and the separately supplied reviewer roster are operator-controlled records, not proof of human identity; verify participation outside this offline scorer.',
                        'Missing/unexecuted cases remain in the denominator; failed runs are not silently retried or dropped.',
                        'Token counts and internal reservations are not actual billed cost.',
                        'Latency measures successful first-attempt HTTP collection, excluding resumed/failed cases; it does not measure frontend rendering.',
                        'This is agent-answer evaluation, not dose-prediction accuracy or clinical validation.']}
    return result, template
