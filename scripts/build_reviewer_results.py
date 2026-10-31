"""Export curated aggregate results and model cards; no fitting, cloud or inference.

Only explicit application fields leave the local scientific record. Source files,
provenance ledgers, sample identifiers, serialized models and report memory stay local.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'context/locked-evaluation/v1'
MODELS = ROOT / 'data/models/final-selection-v2'
RESULT_FREEZE = 'a93f0e43f65229d989dbf9763739d75d55f82e5feaf3005db2f2b83a66029277'
MODEL_FREEZE = '3b051050b682601006fc965d01ca871aa175799863a48ee73fee314ddd66e725'
ORDER = ('A', 'B', 'C', 'D', 'elastic_net', 'pls', 'gradient_boosting', 'median_null')
DESCRIPTIONS = {
    'A': ('Core neural regression', 'Neural',
          'Two hidden layers (64 and 16 ReLU units) learn dose from the core whole-blood cohort.'),
    'B': ('Pretrain then fine-tune', 'Neural',
          'The same neural architecture first learns binary exposure from auxiliary profiles, then learns core dose. Auxiliary numeric doses are not used.'),
    'C': ('Multitask neural regression', 'Neural',
          'The shared neural encoder learns core dose and binary exposure on core and auxiliary profiles. Auxiliary numeric doses are not used.'),
    'D': ('Weighted neural regression', 'Neural',
          'The same neural architecture learns numerical dose from core and auxiliary profiles with equal total core and auxiliary loss mass.'),
    'elastic_net': ('Elastic Net', 'Reference',
                    'Linear dose regression with combined L1 and L2 penalties. Core-only reference model.'),
    'pls': ('Partial least squares', 'Reference',
            'Core-only regression using 15 latent components of the selected rank features.'),
    'gradient_boosting': ('Gradient boosting', 'Reference',
                         'Core-only boosted regression trees: 300 estimators with maximum depth 2.'),
    'median_null': ('Training median', 'Null baseline',
                   'Always returns the ordinary core-profile training median, 4.45 Gy. No gene-expression features or auxiliary fitting.')}
LIMITATIONS = [
    'Research only; not clinically validated. No model is automatically selected for deployment.',
    'One test study with 12 donor groups and unequal dose coverage. Donor 7 has no deposited control; there are 11 controls and six 4 Gy profiles.',
    'Public donor identifiers do not independently establish globally unique people or independence across studies. Paired-t inference is conditional on independence and its approximation at n=12.',
    'The final study is now unblinded and consumed. Changes motivated by these results are exploratory and need independent validation.',
    'The test covers human whole blood, laboratory photon/control exposure, approximately 24 hours, GPL13497, and 0–4 Gy. It does not validate other tissues, RNA-seq, neutrons, other timings, in-vivo or clinical use, or accuracy above 4 Gy.',
    'Aggregate bootstrap intervals describe donor-mean error conditional on these trained models; they are not individual dose prediction intervals.',
    'Negative and above-range predictions were preserved. No post-test clipping, recalibration, outlier deletion, new feature selection or best-seed selection was performed.',
    'Development selection included all core studies within donor folds, so its scores are not equivalent to unseen-study transfer. Study bias and seed sensitivity remain.',
    'New uploads require trusted assay, tissue, timing, provenance and missingness review plus saved-pipeline and cloud-runtime checks before any numerical prediction. The authenticated prediction workspace enforces these gates separately from this result snapshot.'
]
MODEL_NOTES = {
    'A': 'Overpredicted all 52 primary-seed test profiles. Its signed bias equals its MAE.',
    'B': 'Lowest primary-seed MAE among A–D. C has lower neural MAE at sensitivity seed 20261003; there is no seed-invariant neural winner.',
    'C': 'Binary classification is a training objective. The final reported endpoint is continuous dose; classification performance is not established by these dose results.',
    'D': 'Uses auxiliary 0/2 Gy numeric labels. This does not expand the supported prediction domain to auxiliary tissues or conditions.',
    'elastic_net': 'Small overall signed bias masks control error and underprediction at 4 Gy.',
    'pls': 'Preserved negative predictions and study-transfer error; latent components are not independent biological mechanisms.',
    'gradient_boosting': 'Lowest observed primary-model MAE here is descriptive. Positive overall bias remains, and no new confirmatory superiority test or deployment promotion follows.',
    'median_null': 'An untuned comparison anchor. The 4.45 Gy training median was never replaced by this test cohort’s median.'}


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_freeze(freeze, expected, root, key):
    if digest(freeze) != expected:
        raise ValueError('Pinned scientific freeze changed')
    entries = read(freeze)[key]
    for name, expected_hash in entries.items():
        p = root / name
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or digest(p) != expected_hash:
            raise ValueError('Scientific source integrity check failed: ' + name)
    return len(entries)


def build_payload(evaluation=EVAL, models=MODELS):
    evaluation, models = Path(evaluation), Path(models)
    count = verify_freeze(evaluation/'result-freeze.json', RESULT_FREEZE, evaluation, 'sha256')
    count += verify_freeze(ROOT/'context/final-training/v1/model-bundle-freeze.json', MODEL_FREEZE, models, 'files')
    complete = read(evaluation/'run-complete.json')
    if (complete['status'], complete['profiles'], complete['donors'], complete['model_versions'], complete['new_fits']) != ('COMPLETE_ONE_TIME_LOCKED_EVALUATION', 52, 12, 18, 0):
        raise ValueError('Unexpected evaluation state')
    index = read(models/'artifact-index.json')['artifacts']
    metrics = read(evaluation/'model-metrics.json')
    comparisons = read(evaluation/'primary-comparisons.json')
    with (evaluation/'stratum-metrics.tsv').open() as f:
        slices = list(csv.DictReader(f, delimiter='\t'))
    metric_keys = ('seed', 'seed_role', 'donor_mae', 'donor_rmse', 'donor_bias')
    cards = []
    for name in ORDER:
        selected = [a for a in index if a['model'] == name and a['seed_role'] != 'sensitivity']
        if len(selected) != 1:
            raise ValueError('Missing or ambiguous primary model')
        artifact = selected[0]
        is_null = name == 'median_null'
        fitted = read(models/artifact['path']/('model.json' if is_null else 'result.json'))
        params = {} if is_null else {k: v for k, v in fitted['params'].items() if k not in ('id', 'model', 'strategy')}
        train_count = len(fitted['training_ids'])
        aux_count = len(fitted.get('auxiliary_training_ids', []))
        if train_count != 145 or aux_count != (422 if name in ('B', 'C', 'D') else 0):
            raise ValueError('Unexpected training membership')
        scores = []
        for m in metrics:
            if m['model'] == name:
                scores.append({**{k:m[k] for k in metric_keys}, 'mae_ci':{
                    k:m['mae_bootstrap_ci'][k] for k in ('low', 'high', 'confidence')}})
        expected_seeds = [0] if is_null else ([20261002] if name in ('elastic_net', 'pls') else [20261002,20261003,20261004])
        if [s['seed'] for s in scores] != expected_seeds:
            raise ValueError('Missing or reordered frozen seeds')
        strata = [{**{k:r[k] for k in ('stratum',)},
                   **{k:int(r[k]) for k in ('profiles','donors')},
                   **{k:float(r[k]) for k in ('donor_mae','donor_rmse','donor_bias')}}
                  for r in slices if r['model'] == name and int(r['seed']) == artifact['seed']]
        label, family, description = DESCRIPTIONS[name]
        cards.append(dict(id=name, name=label, family=family, description=description,
            primary_seed=artifact['seed'], representation='none' if is_null else params['representation'],
            selected_genes=0 if is_null else 500, core_profiles=train_count, auxiliary_profiles=aux_count,
            settings=params, scores=scores, primary_strata=strata, interpretation=MODEL_NOTES[name]))
    tests = []
    for name in 'BCD':
        c = comparisons[name]
        tests.append(dict(comparator=name, mean_difference=c['mean_difference'],
            raw_p=c['raw_p'], holm_adjusted_p=c['holm_adjusted_p'], reject=c['holm_reject'],
            simultaneous_ci={k:c['simultaneous_t_ci'][k] for k in ('low','high','confidence')}))
    payload = dict(schema_version=1, release_id='locked-evaluation-2026-10-03-v1',
        evaluated_date='2026-10-03', status='completed_consumed', purpose='saved_research_results',
        live_prediction_enabled=False, study='GSE90909', profiles=52, donors=12, dose_min_gy=0, dose_max_gy=4,
        controls=11, high_dose_profiles=6, model_versions=18, model_cards=cards,
        primary_finding='Model A superiority was not supported.',
        comparison_method='A minus comparator donor MAE; negative favors A. Three one-sided paired t tests (12 donors, df=11), Holm-adjusted at 0.05. The displayed simultaneous t intervals have 98.333% coverage per comparison. Non-rejection does not establish equivalence.',
        comparisons=tests,
        metric_definition='MAE averages absolute errors within each donor, then gives each donor equal weight. RMSE is the square root of mean donor MSE. Bias is mean donor signed error (predicted minus observed). All values are Gy.',
        interval_definition='95% descriptive donor-percentile bootstrap intervals, 10,000 shared resamples, seed 20261003; conditional on fixed models and this dose design.',
        preprocessing='Raw microarray background correction and technical masks; frozen core-training probe eligibility and quantile reference; one log2 transform and fixed probe-to-gene aggregation. Training-only imputation, dose-informed feature selection (500 of 9,131 eligible genes in the full fit), and feature scaling. Rank models use the saved within-profile rank representation. No test-cohort normalization fit.',
        target_scaling='Dose standardization was fitted on core training profiles and inverted to Gy. Time after exposure is an eligibility condition, not a dose normalization factor.',
        training='145 core profiles across three studies and 30 study-local donor groups. B–D additionally use 422 auxiliary profiles across three studies and 211 groups. Auxiliary label-free references were fitted on their complete training cohorts. These groups are not independently verified globally unique people.',
        selection='Eight candidates per fitted family, matched five donor folds; preprocessing refitted inside each fold. Settings and primary seed were frozen before the final test. Sensitivity seeds do not add independent donors or replace the primary result.',
        measurement_note='Approximately 30 million source expression measurements are not independent training samples. The curated development release contains 567 eligible profiles (145 core + 422 auxiliary).',
        limitations=LIMITATIONS)
    encode(payload)  # reject non-finite JSON
    return payload, count


def markdown(payload):
    text = ['# Defender model cards', '', 'Version: '+payload['release_id'], '',
            '**Saved research results only. Final test completed and consumed. No model is automatically selected or recommended by this result snapshot.**', '',
            payload['primary_finding'], '', '## Evaluation and intended use', '',
            'Academic review of frozen radiation-dose regression experiments, not clinical triage or treatment.', '',
            'GSE90909: 52 human whole-blood photon/control profiles, 12 donor groups, approximately 24 hours, GPL13497, 0–4 Gy.', '',
            payload['training'], '', payload['measurement_note'], '', '## Methods', '',
            payload['preprocessing'], '', payload['target_scaling'], '', payload['selection'], '',
            payload['metric_definition'], '', payload['interval_definition'], '', payload['comparison_method'], '']
    for card in payload['model_cards']:
        text += ['## '+card['id']+' · '+card['name'], '', card['description'], '',
                 f"Features: {card['selected_genes']} genes; representation: {card['representation']}. Core training: {card['core_profiles']} profiles; auxiliary training: {card['auxiliary_profiles']} profiles.", '',
                 'Frozen settings: '+json.dumps(card['settings'], sort_keys=True)+'.', '',
                 '| Seed / role | Donor MAE (95% interval) | Donor RMSE | Signed bias |',
                 '|---|---:|---:|---:|']
        for score in card['scores']:
            ci=score['mae_ci']
            text.append(f"| {score['seed']} / {score['seed_role']} | {score['donor_mae']:.3f} ({ci['low']:.3f}–{ci['high']:.3f}) | {score['donor_rmse']:.3f} | {score['donor_bias']:+.3f} |")
        text += ['', card['interpretation'], '', '| Primary dose slice | Profiles / donors | MAE | Signed bias |', '|---|---:|---:|---:|']
        for r in card['primary_strata']:
            if r['stratum'] in ('control','above_2gy'):
                text.append(f"| {r['stratum']} | {r['profiles']} / {r['donors']} | {r['donor_mae']:.3f} | {r['donor_bias']:+.3f} |")
        text += ['', 'The >2 Gy slice consists entirely of six 4 Gy profiles. All shared limitations below apply to this card.', '']
    text += ['## Shared limitations and release gates', '']+['- '+x for x in payload['limitations']]
    text += ['', 'The app’s existing live engineering training presets are separate from these frozen models. Reading or downloading these cards starts no job. Cloud inference parity and trusted upload compatibility enforcement remain required before model promotion.', '']
    return '\n'.join(text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'data/reviewer_release')
    parser.add_argument('--evidence',type=Path,default=ROOT/'context/cloud/reviewer-results-v1')
    args=parser.parse_args()
    payload, checked=build_payload()
    args.output.mkdir(parents=True,exist_ok=True); args.evidence.mkdir(parents=True,exist_ok=True)
    files={'research-results.json':encode(payload), 'MODEL_CARDS.md':markdown(payload).encode()}
    for name, content in files.items():
        target=args.output/name
        if target.is_symlink():raise ValueError('Refusing symlink output')
        target.write_bytes(content)
    summary=dict(source_files_verified=checked, model_cards=8, model_versions=18, new_fits=0, cloud_changes=0,
        files={name:dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest()) for name,content in files.items()})
    (args.evidence/'export-verification.json').write_bytes(encode(summary))
    print(json.dumps(summary))


if __name__=='__main__':main()
