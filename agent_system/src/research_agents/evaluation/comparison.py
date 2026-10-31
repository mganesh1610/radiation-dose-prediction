"""Controlled architecture benchmark: frozen inputs, durable attempts, no retries.

This is an engineering comparison, not a clinical or independent-human evaluation.
Reference generation is separate from every answering arm. No production quota,
chat, session, permission or model is modified by this module.
"""
import hashlib
import json
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

ARMS = ('deterministic', 'single_adk', 'a2a')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save_new(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def schedule(cases):
    """Rotate position deterministically; all arms see each identical question once."""
    return [{'case_id': c['id'], 'arm': arm} for i, c in enumerate(cases)
            for arm in ARMS[i % 3:] + ARMS[:i % 3]]


def validate_questions(bundle, questions):
    expected = {'suite_id': bundle['suite_id'], 'reference_sha256': digest(bundle),
                'cases': [{'id': c['id'], 'question': c['question']} for c in bundle['cases']]}
    if questions != expected:
        raise ValueError('Question file changed or contains reference answers')
    if len(bundle['cases']) != 18 or len({c['id'] for c in bundle['cases']}) != 18:
        raise ValueError('This pilot requires exactly 18 unique cases')
    if len({c['question'] for c in bundle['cases']}) != 18:
        raise ValueError('Duplicate benchmark question')


def check_preflight(preflight, frozen_sha, *, now=None):
    now = now or datetime.now(timezone.utc)
    checked = datetime.fromisoformat(preflight['checked_at_utc'])
    if checked.tzinfo is None or not 0 <= (now - checked).total_seconds() <= 7200:
        raise ValueError('Fresh billing and usage check required (two-hour limit)')
    if preflight.get('reference_sha256') != frozen_sha or preflight.get('billing_review_completed') is not True:
        raise ValueError('Preflight must bind frozen suite and actual billing review')
    if (preflight.get('max_attempts') != 54 or preflight.get('max_model_runs') != 36
            or preflight.get('monitored_budget_usd') != 5 or preflight.get('production_mutations') is not False):
        raise ValueError('Finite approved comparison bounds required')
    if not preflight.get('billing_evidence') or not preflight.get('production_snapshot'):
        raise ValueError('Preflight evidence paths required')


def parse_direct(question, release_id, expression_set_id):
    """New, explicitly limited no-LLM baseline. No expected answers/case IDs here.

    Accept only complete grammar matches. Broader language is unsupported, never
    partially interpreted. All resulting requests still pass production facades.
    """
    from ..contracts.queries import QueryPlan
    from ..contracts.expression import ExpressionPlan
    q = question.strip()
    match = re.fullmatch(r'Count (profiles|study-local donor groups) in (GSE\d+) at recorded dose ([0-9]+(?:\.[0-9]+)?) Gy\. Include the unit and evidence ID\.', q)
    base = {'version': 1, 'release_id': release_id, 'population': 'approved_core_release_v1', 'dose_kind': 'recorded'}
    if match:
        plan = QueryPlan.model_validate({**base, 'metric': 'sample_count' if match[1] == 'profiles' else 'study_group_count',
                'filters': {'study_id': match[2], 'dose_gy': float(match[3])}})
        return 'query', plan.model_dump()
    match = re.fullmatch(r'Look up (GSM\d+)\. Give recorded dose, study, study-local donor group, time after exposure, organism and tissue, with units and evidence ID\.', q)
    if match:
        plan = QueryPlan.model_validate({**base, 'metric': 'sample_details', 'sample_id': match[1]})
        return 'query', plan.model_dump()
    match = re.fullmatch(r'Retrieve expression for (GSM\d+(?:, GSM\d+)*) using NCBI Gene ID ([1-9]\d*)\. Give each probe value and detection flag in sample order, preserve missing cells, and state scale, normalization and evidence ID\.', q)
    if match:
        plan = ExpressionPlan.model_validate({'version': 1, 'release_id': release_id,
                'expression_set_id': expression_set_id, 'sample_ids': match[1].split(', '), 'gene_ids': [match[2]],
                'gene_namespace': 'NCBI_GeneID', 'aggregation': 'none'})
        return 'expression', plan.model_dump()
    return None


def receipt_checks(case, result):
    """Check source values/citations independently of prose; never auto-score prose."""
    errors = []
    if result.get('status') != 'completed': return ['run_not_completed']
    if result.get('coverage') == 'unsupported': return []
    receipts = result.get('receipt', {}).get('receipts', [])
    ids = {r['evidence_id'] for r in receipts}
    cited = set(re.findall(r'\bev-[a-zA-Z0-9_-]+', result.get('answer', '')))
    if cited - ids: errors.append('invented_evidence_id')
    tool = case.get('expected_tool')
    if tool in {'query', 'expression'}:
        relevant = [r for r in receipts if r.get('tool') == tool]
        if len(relevant) != 1: errors.append('expected_one_receipt')
        else:
            r = relevant[0]
            actual = r['verified_result']
            # The reader also emits a fixed interpretation caveat. The independent
            # reference binds scientific values; validate this extra contract field
            # explicitly rather than treating it as a changed expression cell.
            extra = set(actual) - set(case['expected'])
            if tool == 'expression' and 'comparison_note' in extra:
                if actual['comparison_note'] != ('Between-array normalization is deferred; these values do not establish comparable biological differences across studies.'):
                    errors.append('unexpected_comparison_note')
                extra.remove('comparison_note')
            if extra or {k:actual.get(k) for k in case['expected']} != case['expected']:
                errors.append('source_answer_mismatch')
            for key, value in case.get('expected_receipt', {}).items():
                if r.get(key) != value: errors.append('receipt_' + key)
            if r['evidence_id'] not in cited: errors.append('missing_evidence_citation')
    elif tool == 'protocol':
        relevant = [r for r in receipts if r.get('tool') == 'protocol']
        passages = {p['chunk_id']: p for r in relevant for p in r['verified_result']['passages']}
        cited_passages = set(re.findall(r'\bpc-[a-zA-Z0-9_-]+', result.get('answer', '')))
        if not relevant: errors.append('missing_protocol_receipt')
        if not set(case['expected']['supporting_passage_ids']) & cited_passages:
            errors.append('required_passage_not_cited')
        if cited_passages - passages.keys(): errors.append('invented_passage')
        for pid in cited_passages & passages.keys():
            if digest(passages[pid]['text']) != case['protocol_passage_hashes'].get(pid):
                errors.append('passage_content_changed')
        if any(p['study_id'] != case['study_id'] for p in passages.values()): errors.append('wrong_protocol_study')
    return errors


def summarize(bundle, records, reviews=None):
    """Coverage denominator and supported correctness remain distinct; missing != pass."""
    reviews = reviews or {}
    by_key = {(r['case_id'], r['arm']): r for r in records}
    if len(by_key) != len(records): raise ValueError('Duplicate architecture/case result')
    expected_keys = {(c['id'], a) for c in bundle['cases'] for a in ARMS}
    if set(by_key) - expected_keys: raise ValueError('Unknown architecture/case result')
    per_case = []
    for case in bundle['cases']:
        for arm in ARMS:
            r = by_key.get((case['id'], arm))
            row = {'case_id': case['id'], 'arm': arm, 'category': case['category'], 'state': 'missing'}
            if r:
                if r['question'] != case['question'] or r['reference_sha256'] != digest(bundle):
                    raise ValueError('Result does not match frozen question/reference')
                row['automated_errors'] = receipt_checks(case, r)
                review = reviews.get(case['id'] + ':' + arm)
                row['state'] = ('unsupported' if r.get('coverage') == 'unsupported' and r['status'] == 'completed'
                                else 'failed' if row['automated_errors'] else 'needs_review')
                if review:
                    if (review.get('result_sha256') != digest(r) or review.get('review_type') != 'engineering'
                            or type(review.get('pass')) is not bool or not review.get('notes')):
                        raise ValueError('Invalid or stale engineering review')
                    if row['state'] not in {'unsupported', 'failed'}:
                        row['state'] = 'engineering_pass' if review['pass'] else 'failed'
                row['result_sha256'] = digest(r)
            per_case.append(row)
    totals = {}
    for arm in ARMS:
        rows = [r for r in per_case if r['arm'] == arm]
        results = [r for r in records if r['arm'] == arm]
        completed = [r for r in results if r['status'] == 'completed']
        covered = [r for r in completed if r.get('coverage') != 'unsupported']
        totals[arm] = {'scheduled': len(rows), 'executed': len(results), 'completed': len(completed),
            'unsupported': sum(r['state'] == 'unsupported' for r in rows),
            'engineering_pass': sum(r['state'] == 'engineering_pass' for r in rows),
            'failed': sum(r['state'] == 'failed' for r in rows),
            'unreviewed_or_missing': sum(r['state'] in {'missing','needs_review'} for r in rows),
            'model_calls': sum(r.get('receipt', {}).get('model_calls', 0) for r in results),
            'tool_calls': sum(r.get('receipt', {}).get('tool_calls', 0) for r in results),
            'prompt_tokens': sum(r.get('receipt', {}).get('prompt_tokens', 0) for r in results),
            'output_tokens': sum(r.get('receipt', {}).get('output_tokens', 0) for r in results),
            'median_completed_seconds': statistics.median([r['elapsed_seconds'] for r in completed]) if completed else None,
            'median_covered_seconds': statistics.median([r['elapsed_seconds'] for r in covered]) if covered else None,
            'actual_billed_usd': None}
    common_ids = {c['id'] for c in bundle['cases'] if c['category'] in {'counts','sample_details','expression'}}
    paired = {cid for cid in common_ids if all((cid,a) in by_key and by_key[cid,a]['status']=='completed'
              and by_key[cid,a].get('coverage')!='unsupported' for a in ARMS)}
    common = {'planned_cases':len(common_ids),'completed_all_arms':len(paired),
        'median_seconds':{a: statistics.median([by_key[c,a]['elapsed_seconds'] for c in paired]) if paired else None for a in ARMS},
        'median_paired_a2a_minus_single_seconds':statistics.median([
            by_key[c,'a2a']['elapsed_seconds']-by_key[c,'single_adk']['elapsed_seconds'] for c in paired]) if paired else None}
    return {'suite_id': bundle['suite_id'], 'status': 'engineering_review_only', 'arms': totals,
            'common_lookup_comparison':common,'cases': per_case, 'human_acceptance': 'not_measured',
            'limits': ['18 deliberately selected questions, one attempt per arm; no population accuracy or SLA.',
                'Mac orchestration, loopback A2A, local sessions/control; cloud inference and read-only retrieval.',
                'Production identity, managed sessions, cloud A2A latency and resilience are not benchmarked.',
                'Unsupported coverage is separate from supported-answer correctness; no silent reruns.',
                'Token counts and internal reservations are not actual billed cost.']}
