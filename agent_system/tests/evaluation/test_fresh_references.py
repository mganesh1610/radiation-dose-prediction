"""Protect evaluation novelty independently of question wording."""
from copy import deepcopy
import pytest

from research_agents.evaluation.fresh_references import novelty_report
from test_acceptance import fixtures


def fresh_cases():
    bundle, _, _ = fixtures()
    for i, c in enumerate(bundle['cases']):
        c['selectors'] = {'metric': 'sample_count', 'study_id': f'STUDY{i}'}
    return bundle['cases']


def test_paraphrasing_does_not_hide_repeated_query():
    cases = fresh_cases()
    old = deepcopy(cases[0]); old['question'] = 'Completely different wording'
    old['selectors']['dose_gy'] = None
    with pytest.raises(ValueError, match='overlaps'):
        novelty_report(cases, [{'cases': [old]}])


def test_new_gene_query_cannot_reuse_old_probe_cells():
    cases = fresh_cases()
    c = cases[0]
    c.update(expected_tool='expression', selectors={'sample_ids': ['S1','S2'], 'gene_ids': ['10']},
             expected={'features': [{'probe_id': 'P1'}]})
    old = deepcopy(c); old['question'] = 'Old question'
    old['selectors'] = {'sample_ids': ['S2'], 'gene_ids': ['10']}
    with pytest.raises(ValueError, match='overlaps'):
        novelty_report(cases, [{'cases': [old]}])


def test_distinct_references_are_accepted_without_claiming_new_data():
    cases = fresh_cases()
    result = novelty_report(cases, [{'cases': []}])
    assert result['question_count'] == 38 and result['factual_selector_overlap'] == 0
