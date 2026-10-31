"""Curate saved development diagnostics and scores for the app; never fit models.

Source ledgers remain local. Only explicitly selected scientific fields are exported.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'clustering': '5a4f36789194c3a90169f1b8bad28ebb593575c205a32ea2d5d7afb6978f4998',
    'extended-models': 'e2e19e14821e64f08d7eebba6c3c0cdaa340141c8c0f8344a0a8b61c36b8b3f1',
}
NAMES = {
    'A': ('Neural A · core regression', 'Neural'),
    'B': ('Neural B · pretraining', 'Neural'),
    'C': ('Neural C · multitask', 'Neural'),
    'D': ('Neural D · weighted pooling', 'Neural'),
    'elastic_net': ('Elastic Net · L1 + L2', 'Linear'), 'ridge': ('Ridge · L2', 'Linear'),
    'lasso': ('Lasso · L1', 'Linear'), 'pls': ('Partial least squares', 'Linear'),
    'svr_linear': ('Linear SVR', 'SVM'), 'svr_rbf': ('RBF-kernel SVR', 'SVM'),
    'random_forest': ('Random Forest', 'Ensemble'), 'gradient_boosting': ('Gradient boosting', 'Ensemble'),
    'xgboost': ('XGBoost', 'Ensemble'), 'adaboost': ('AdaBoost', 'Ensemble'),
    'polynomial_ridge': ('Polynomial + Ridge', 'Polynomial'),
    'median_null': ('Training median', 'Baseline'),
    'pca_ridge': ('PCA + Ridge', 'Clustering / PCA'),
    'kmeans_ridge': ('K-means + Ridge', 'Clustering / PCA'),
    'gmm_ridge': ('Gaussian mixture + Ridge', 'Clustering / PCA'),
    'cluster_median': ('Cluster-median predictor', 'Clustering / PCA'),
}


def build_payload():
    ledgers = {}
    for experiment, expected in PINS.items():
        raw = (ROOT / f'context/{experiment}/v1/artifact-manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'Changed {experiment} source manifest')
        ledgers[experiment] = json.loads(raw)['files']
    checked = {}

    def read(experiment, name):
        path = name if name.startswith('data/') else f'context/{experiment}/v1/{name}'
        source = ROOT / path
        raw = source.read_bytes()
        expected = ledgers[experiment][path]
        digest = hashlib.sha256(raw).hexdigest()
        if source.is_symlink() or digest != expected['sha256'] or len(raw) != expected['bytes']:
            raise ValueError(f'Changed scientific input: {path}')
        checked[path] = digest
        return raw

    def rows(experiment, name):
        return list(csv.DictReader(io.StringIO(read(experiment, name).decode()), delimiter='\t'))

    assignments = rows('clustering', 'cluster-assignments.tsv')
    metrics = rows('clustering', 'diagnostic-metrics.tsv')
    selected = json.loads(read('clustering', 'descriptive-selected.json'))
    meta = {}
    for row in assignments:
        record = dict(id=row['profile_id'], study=row['study'], dose_gy=float(row['dose_gy']))
        if record['id'] in meta and meta[record['id']] != record:
            raise ValueError('Conflicting profile metadata')
        meta[record['id']] = record
    if len(meta) != 567 or {r['study'] for r in meta.values()} != {
            'GSE102971', 'GSE8917', 'GSE65292', 'GSE85570', 'GSE97000', 'GSE20173'}:
        raise ValueError('Unexpected development membership')

    views = []
    for scope in ('core', 'all'):
        for representation in ('log2', 'rank'):
            path = f'data/models/clustering-v1/diagnostics/{scope}-{representation}/view/panel.npz'
            with np.load(io.BytesIO(read('clustering', path)), allow_pickle=False) as saved:
                ids = saved['ids'].tolist()
                z = saved['z']
            if len(ids) != (145 if scope == 'core' else 567) or z.shape != (len(ids), 10):
                raise ValueError('Unexpected diagnostic panel')
            fits = []
            for method in ('kmeans', 'ward', 'gmm'):
                for controls in (False, True):
                    k = (3 if scope == 'core' else 6) if controls else selected[f'{scope}-{representation}-{method}']['k']
                    def matches(row):
                        return (row['scope'], row['representation'], row['method'], row['controls_only'], row['k'], row['seed_index']) == (scope, representation, method, str(controls), str(k), '0')
                    score, = [r for r in metrics if matches(r)]
                    labels = {r['profile_id']: int(r['cluster']) for r in assignments if matches(r)}
                    expected_ids = {i for i in ids if not controls or meta[i]['dose_gy'] == 0}
                    if set(labels) != expected_ids:
                        raise ValueError('Assignment membership mismatch')
                    if not controls:
                        saved = selected[f'{scope}-{representation}-{method}']
                        if labels != dict(zip(saved['ids'], saved['labels'])):
                            raise ValueError('Selected partition changed')
                    sizes = [sum(c == i for c in labels.values()) for i in range(k)]
                    if sizes != json.loads(score['cluster_sizes']):
                        raise ValueError('Cluster counts differ from saved metrics')
                    fits.append(dict(method=method, controls_only=controls, k=k,
                                     labels=[labels.get(i) for i in ids],
                                     **{key: float(score[key]) if score[key] else None for key in ('silhouette', 'study_ari', 'dose_group_ari', 'exposure_ari')}))
            views.append(dict(scope=scope, representation=representation,
                              points=[dict(meta[i], x=float(z[n, 0]), y=float(z[n, 1])) for n, i in enumerate(ids)], fits=fits))

    regression = json.loads(read('extended-models', 'regression-summary.json'))
    regression += [r for r in json.loads(read('clustering', 'regression-summary.json')) if r['model'] in ('pca_ridge', 'kmeans_ridge', 'gmm_ridge', 'cluster_median')]
    output = []
    for r in regression:
        name, family = NAMES[r['model']]
        output.append(dict(model=r['model'], label=name, family=family,
                           **{k: r[k] for k in ('mae_gy', 'rmse_gy', 'bias_gy', 'worst_study_mae_gy')},
                           equal_study_mae_gy=r.get('macro_study_mae_gy', r.get('equal_study_mae_gy')),
                           training='Core + auxiliary' if r['model'] in ('B', 'C', 'D') else 'Core only'))
    if len(output) != 20 or len({r['model'] for r in output}) != 20:
        raise ValueError('Unexpected model family inventory')
    classifiers = []
    names = {'logistic_l1': 'Logistic regression · L1', 'logistic_l2': 'Logistic regression · L2', 'gaussian_nb': 'Gaussian Naive Bayes'}
    for r in rows('extended-models', 'classification-metrics.tsv'):
        if r['seed_index'] == '0' and r['slice'] == 'all':
            classifiers.append(dict(model=r['model'], label=names[r['model']], study=r['study'], profiles=int(r['profiles']),
                                    **{k: float(r[k]) for k in ('balanced_accuracy', 'specificity', 'sensitivity')}))
    payload = dict(schema_version=1, release_id='development-exploration-v1', purpose='exploratory_development',
                   final_test_included=False, candidate_profiles=607, eligible_profiles=567, core_profiles=145,
                   source_expression_values=30053455, views=views, regression=sorted(output, key=lambda r: r['mae_gy']),
                   classification=classifiers, primary_seed=20261002, diagnostic_seed=20261003)
    json.dumps(payload, allow_nan=False)
    return payload, checked


if __name__ == '__main__':
    payload, checked = build_payload()
    target = ROOT / 'app/static/development-results.json'
    target.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    evidence = ROOT / 'context/cloud/development-explorer-v1'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'source-verification.json').write_text(json.dumps(dict(verified_inputs=checked, output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size), indent=2))
    print(json.dumps(dict(verified_inputs=len(checked), bytes=target.stat().st_size, models=len(payload['regression']), views=len(payload['views']))))
