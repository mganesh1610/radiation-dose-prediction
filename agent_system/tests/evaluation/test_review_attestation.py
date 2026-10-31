"""Review-origin controls: synthetic judgments cannot become human acceptance."""
from copy import deepcopy
import pytest
from research_agents.evaluation.scoring import report
from test_acceptance import fixtures


def human_reviews():
    bundle, records, reviews = fixtures()
    bundle['gates']['human_review_required'] = True
    for review in reviews.values():
        review.update(review_type='human', reviewer='test-person', human_attestation={
            'personally_reviewed': True, 'independent_of_implementation': True,
            'record_sha256': review['record_sha256'],
        })
    return bundle, records, reviews


def test_explicit_ai_reviews_cannot_satisfy_human_gate():
    b, r, v = human_reviews()
    for review in v.values():
        review['review_type'] = 'ai_peer_review'
    result, _ = report(b, 'frozen-hash', r, v)
    assert result['status'] == 'incomplete'
    assert result['reviewed'] == 0
    assert result['factual_accuracy'] is None


def test_human_review_requires_separate_authorized_roster():
    b, r, v = human_reviews()
    assert report(b, 'frozen-hash', r, v)[0]['status'] == 'incomplete'
    result, _ = report(b, 'frozen-hash', r, v, human_reviewers=['test-person'])
    assert result['status'] == 'passed' and result['human_reviewed'] == 38


@pytest.mark.parametrize('change', [
    lambda v: v.pop('review_type'),
    lambda v: v.update(review_type='engineering'),
    lambda v: v.update(review_type='ai_peer_review'),
    lambda v: v.update(review_type='unknown'),
    lambda v: v.update(reviewer='unapproved-person'),
    lambda v: v.update(reviewer='  '),
    lambda v: v.pop('human_attestation'),
    lambda v: v.update(human_attestation=True),
    lambda v: v['human_attestation'].update(personally_reviewed=False),
    lambda v: v['human_attestation'].update(personally_reviewed=1),
    lambda v: v['human_attestation'].update(independent_of_implementation=False),
    lambda v: v['human_attestation'].update(record_sha256='stale'),
    lambda v: v.update(reviewed_at_utc='not-a-time'),
    lambda v: v.update(reviewed_at_utc='2026-10-23T07:00:00'),
    lambda v: v.update(notes=' '),
])
def test_incomplete_or_misclassified_attestation_blocks_acceptance(change):
    b, r, v = human_reviews()
    change(v['F01'])
    result, _ = report(b, 'frozen-hash', r, v, human_reviewers=['test-person'])
    assert result['status'] == 'incomplete' and result['human_reviewed'] == 37
    assert result['cases'][0]['review_errors']


def test_provisional_judgments_remain_separate_even_with_authorized_name():
    b, r, v = human_reviews()
    for review in v.values():
        review['review_type'] = 'ai_peer_review'
    original = deepcopy(v)
    result, template = report(b, 'frozen-hash', r, v, human_reviewers=['test-person'])
    assert result['status'] == 'incomplete' and result['ai_reviewed'] == 38
    assert result['factual_passed'] == result['boundary_passed'] == 0
    assert all(c['provisional_state'] == 'pass' for c in result['cases'])
    assert v == original
    assert template['F01']['review_type'] is None
    assert template['F01']['human_attestation']['personally_reviewed'] is None


def test_result_change_invalidates_both_review_and_attestation():
    from research_agents.evaluation.scoring import digest
    b, r, v = human_reviews()
    r[0]['answer'] += ' Additional unreviewed claim.'
    v['F01']['record_sha256'] = digest(r[0])
    result, _ = report(b, 'frozen-hash', r, v, human_reviewers=['test-person'])
    assert result['status'] == 'incomplete'
    assert 'human_attestation_missing_or_unbound' in result['cases'][0]['review_errors']


def test_review_requirement_cannot_be_disabled_for_acceptance_report():
    b, r, v = human_reviews()
    b['gates']['human_review_required'] = False
    with pytest.raises(ValueError, match='human review'):
        report(b, 'frozen-hash', r, v, human_reviewers=['test-person'])


def test_legacy_reviews_are_not_silently_promoted():
    b, r, v = human_reviews()
    for review in v.values():
        review.pop('review_type')
        review.pop('human_attestation')
    result, _ = report(b, 'frozen-hash', r, v, human_reviewers=['test-person'])
    assert result['status'] == 'incomplete' and result['reviewed'] == 0


@pytest.mark.parametrize('roster', ['test-person', {'test-person': True}, [None], [' ']])
def test_malformed_reviewer_roster_rejected(roster):
    b, r, v = human_reviews()
    with pytest.raises(ValueError, match='separate list'):
        report(b, 'frozen-hash', r, v, human_reviewers=roster)


def test_cli_preserves_templates_and_requires_separate_roster(tmp_path, monkeypatch):
    import json
    import sys
    from research_agents.workflows import evaluate_chat as cli
    b, records, reviews = human_reviews()
    monkeypatch.setattr(cli, 'load_frozen', lambda _: (b, 'frozen-hash'))
    results = tmp_path / 'results'; results.mkdir()
    for record in records:
        (results / (record['case_id'] + '.json')).write_text(json.dumps(record))
    review_path = tmp_path / 'reviews.json'; review_path.write_text(json.dumps(reviews))
    roster_path = tmp_path / 'roster.json'; roster_path.write_text(json.dumps(['test-person']))
    output = tmp_path / 'report'; output.mkdir()
    original_template = b'{"preserve":"original decisions"}\n'
    (output / 'review-template.json').write_bytes(original_template)
    args = ['evaluate_chat', 'score', '--frozen', str(tmp_path), '--results', str(results),
            '--reviews', str(review_path), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', args)
    cli.main()
    assert json.loads((output / 'report.json').read_text())['status'] == 'incomplete'
    monkeypatch.setattr(sys, 'argv', args + ['--human-reviewers', str(roster_path)])
    cli.main()
    result = json.loads((output / 'report.json').read_text())
    assert result['status'] == 'passed'
    assert result['review_policy']['authorized_reviewer_count'] == 1
    assert result['reviews_sha256']
    assert 'human-reviewed: 38/38' in (output / 'report.md').read_text()
    assert (output / 'review-template.json').read_bytes() == original_template
    assert json.loads(review_path.read_text()) == reviews
