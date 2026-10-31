"""Execute prespecified nested core-study baseline evaluation locally."""
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from core_baseline import (MODELS, SEED, CorePredictor, GenePreparation, bootstrap_intervals,
                           donor_metrics, fit_estimator, inner_splits, model_grid)
from core_transfer import CoreTransfer

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/baseline-modeling/v1'
OUT = ROOT / 'data/models/core-baselines-v1'
INPUT = ROOT / 'data/processed_pilot/core-reprocessing-v1/common_probe_background_corrected.npz'
LABELS = ROOT / 'context/normalization-validation/v2/core-sample-labels-v2.tsv'
STUDIES = ('GSE102971', 'GSE8917', 'GSE65292')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_tsv(path, rows):
    if not rows: return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader(); w.writerows(rows)


def load_data():
    source = json.loads((ROOT / 'context/core-reprocessing/v1/summary.json').read_text())
    record = next(a for a in source['artifacts'] if a['path'] == str(INPUT.relative_to(ROOT)))
    if sha(INPUT) != record['sha256'] or sha(LABELS) != source['inputs_sha256'][str(LABELS.relative_to(ROOT))]:
        raise ValueError('Core input or label hash mismatch')
    with np.load(INPUT, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    records = list(csv.DictReader(LABELS.open(), delimiter='\t'))
    labels = {r['gsm']: r for r in records}
    samples = data['sample_ids']; studies = data['study_ids']; groups = data['split_group_ids']
    if (len(records) != len(labels) or len(samples) != 145 or len(set(samples)) != 145
            or set(samples) != set(labels) or set(studies) != set(STUDIES)
            or len(set(groups)) != 30):
        raise ValueError('Unexpected core membership')
    for i, s in enumerate(samples):
        if (labels[s]['study'] != studies[i] or labels[s]['split_group_id'] != groups[i]
                or float(labels[s]['time_hours']) != 24):
            raise ValueError('Label identity/time mismatch')
    data['y'] = np.array([float(labels[s]['candidate_dose_gy']) for s in samples])
    if not np.isfinite(data['y']).all() or np.any(data['y'] < 0): raise ValueError('Invalid dose')
    if np.isinf(data['values']).any() or np.any(data['values'][np.isfinite(data['values'])] <= 0):
        raise ValueError('Invalid corrected intensity')
    return data


def fit_preparation(data, train):
    """Only slices assigned to training reach any fitting or ranking method."""
    x = data['values'][:, train]
    samples = data['sample_ids'][train]; studies = data['study_ids'][train]
    q = CoreTransfer.fit(x, np.isfinite(x), data['detected'][:, train], data['probe_ids'],
                         data['gene_keys'], samples, studies, list(samples))
    genes, matrix, _ = q.transform(x, np.isfinite(x), data['probe_ids'])
    prep = {k: GenePreparation.fit(matrix.T, genes, data['y'][train], studies, samples, k)
            for k in (100, 500)}
    return q, genes, matrix.T, prep


def preparation_signature(q, preparations):
    return dict(target=hashlib.sha256(q.reference.target.tobytes()).hexdigest(),
                keep=hashlib.sha256(q.keep.tobytes()).hexdigest(),
                fit_sample_ids=list(q.reference.fit_sample_ids),
                gene_preparation={str(k): p.fingerprint() for k, p in preparations.items()})


def prepare_fold(data, train, validation, name):
    started = time.perf_counter()
    if set(data['split_group_ids'][train]) & set(data['split_group_ids'][validation]):
        raise ValueError('Donor overlap')
    q, genes, xtrain, prep = fit_preparation(data, train)
    held = data['values'][:, validation]
    vg, xval, _ = q.transform(held, np.isfinite(held), data['probe_ids'])
    if not np.array_equal(vg, genes): raise ValueError('Transformed gene identity mismatch')
    # Deliberately corrupt ALL nontraining expression, detection and dose outcomes.
    # Refit through the same public fold function and compare every fitted state.
    changed = dict(data)
    nontraining = np.ones(len(data['sample_ids']), bool); nontraining[train] = False
    changed['values'] = data['values'].copy(); changed['values'][:, nontraining] = np.nan
    changed['detected'] = data['detected'].copy(); changed['detected'][:, nontraining] = False
    changed['y'] = data['y'].copy(); changed['y'][nontraining] = 123456.
    q2, _, _, prep2 = fit_preparation(changed, train)
    signature = preparation_signature(q, prep)
    if signature != preparation_signature(q2, prep2): raise ValueError('Nontraining data affected preparation')
    q.save(OUT / f'{name}_reference.npz')
    for k, p in prep.items(): p.save(OUT / f'{name}_genes{k}.npz')
    panel = {k: {scaled: (p.transform(xtrain, genes, scaled), p.transform(xval.T, genes, scaled))
                  for scaled in (False, True)} for k, p in prep.items()}
    audit = dict(fold=name, training_samples=len(train), validation_samples=len(validation),
                 training_groups=sorted(set(data['split_group_ids'][train])),
                 validation_groups=sorted(set(data['split_group_ids'][validation])),
                 training_sample_ids=list(data['sample_ids'][train]), validation_sample_ids=list(data['sample_ids'][validation]),
                 eligible_probes=int(q.keep.sum()), genes=len(genes),
                 nontraining_expression_mask_label_perturbation_unchanged=True, signature=signature,
                 runtime_seconds=time.perf_counter() - started)
    return dict(transfer=q, preparations=prep, panel=panel, train=train, validation=validation, audit=audit)


def main():
    started = time.perf_counter(); LOG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if (LOG / 'summary.json').exists(): raise FileExistsError('Completed v1 exists; preserve it and use a new version')
    method = LOG / 'METHOD.md'
    save_json(LOG / 'run-start.json', dict(started_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
              method_sha256=sha(method), input_sha256=sha(INPUT), labels_sha256=sha(LABELS),
              python=sys.version, sklearn=sklearn.__version__, scipy=scipy.__version__, numpy=np.__version__,
              joblib=joblib.__version__, platform=platform.platform()))
    data = load_data(); groups = data['split_group_ids']; studies = data['study_ids']; y = data['y']
    all_predictions=[]; inner_predictions=[]; trials=[]; audits=[]; selected=[]; model_checks=[]
    for held in STUDIES:
        outer_train = np.flatnonzero(studies != held); outer_test = np.flatnonzero(studies == held)
        print(f'Preparing outer {held}: {len(outer_train)} train / {len(outer_test)} held', flush=True)
        inner=[]
        for fold, (tr, va) in enumerate(inner_splits(studies[outer_train], groups[outer_train])):
            f = prepare_fold(data, outer_train[tr], outer_train[va], f'holdout_{held}_inner{fold}')
            inner.append(f); audits.append(f['audit'])
        print(f'{held}: five inner preparations and perturbation checks pass', flush=True)
        choices={}
        for name in MODELS[1:]:
            scores=[]
            for trial, params in enumerate(model_grid(name)):
                before=time.perf_counter(); predictions=[]; truths=[]; donor_ids=[]; errors=[]; diagnostics=[]
                for fold, f in enumerate(inner):
                    tr, va = f['train'], f['validation']
                    xt, xv = f['panel'][params['k']][name != 'gradient_boosting']
                    try:
                        model = fit_estimator(name, params, xt, y[tr])
                        pred = np.asarray(model.predict(xv)).reshape(-1)
                        if not np.isfinite(pred).all(): raise ValueError('Nonfinite validation predictions')
                        niter = getattr(model, 'n_iter_', None)
                        diagnostics.append(dict(fold=fold, n_iter=np.asarray(niter).tolist() if niter is not None else None,
                                                dual_gap=float(model.dual_gap_) if hasattr(model, 'dual_gap_') else None))
                        predictions.extend(pred); truths.extend(y[va]); donor_ids.extend(groups[va])
                        for i, p in zip(va, pred):
                            inner_predictions.append(dict(held_study=held, model=name, trial=trial, inner_fold=fold,
                                gsm=str(data['sample_ids'][i]), study=str(studies[i]), group=str(groups[i]), actual_gy=float(y[i]), predicted_gy=float(p)))
                    except ValueError as e:
                        errors.append(f'inner{fold}: {e}')
                score = None if errors else donor_metrics(truths, predictions, donor_ids)[0]['mae']
                row = dict(held_study=held, model=name, trial=trial, parameters=json.dumps(params, sort_keys=True),
                           status='rejected' if errors else 'valid', inner_donor_mae_gy=score,
                           errors=' | '.join(errors), convergence=json.dumps(diagnostics), runtime_seconds=time.perf_counter()-before)
                trials.append(row)
                if score is not None: scores.append((score, trial, params))
            if not scores: raise ValueError(f'No converged candidates for {held}/{name}')
            best = min(scores, key=lambda item: (item[0], item[1]))
            choices[name] = best[2]
            selected.append(dict(held_study=held, model=name, parameters=best[2], inner_donor_mae_gy=best[0], trial=best[1]))
            print(f'{held} {name}: tuned {len(scores)}/{len(model_grid(name))} valid candidates; inner MAE={best[0]:.4f}', flush=True)
        # Holdout outcomes are used only after tuning is complete.
        outer = prepare_fold(data, outer_train, outer_test, f'holdout_{held}_outer'); audits.append(outer['audit'])
        test_x = data['values'][:, outer_test]
        for name in MODELS:
            if name == 'median_null':
                prediction = np.full(len(outer_test), np.median(y[outer_train]))
                save_json(OUT / f'holdout_{held}_median_null.json', dict(training_sample_ids=list(data['sample_ids'][outer_train]),
                          median_dose_gy=float(prediction[0])))
            else:
                params=choices[name]; prep=outer['preparations'][params['k']]
                xt, _ = outer['panel'][params['k']][name != 'gradient_boosting']
                estimator=fit_estimator(name, params, xt, y[outer_train])
                predictor=CorePredictor(outer['transfer'], prep, estimator, name)
                prediction=predictor.predict(test_x, data['probe_ids'])
                path=OUT / f'holdout_{held}_{name}.joblib'; joblib.dump(predictor, path, compress=3)
                loaded=joblib.load(path)
                reload_delta=float(np.max(abs(loaded.predict(test_x, data['probe_ids'])-prediction)))
                reverse_delta=float(np.max(abs(predictor.predict(test_x[:, ::-1], data['probe_ids'])[::-1]-prediction)))
                single=np.array([predictor.predict(test_x[:, j:j+1], data['probe_ids'])[0] for j in range(test_x.shape[1])])
                single_delta=float(np.max(abs(single-prediction)))
                if max(reload_delta, reverse_delta, single_delta) > 1e-8: raise ValueError('Prediction invariance failed')
                model_checks.append(dict(held_study=held, model=name, reload_max_difference_gy=reload_delta,
                    reversed_max_difference_gy=reverse_delta, single_max_difference_gy=single_delta,
                    selected_genes=len(prep.selected), training_min_gy=float(y[outer_train].min()), training_max_gy=float(y[outer_train].max()),
                    below_zero=int((prediction<0).sum()), outside_training_range=int(((prediction<y[outer_train].min())|(prediction>y[outer_train].max())).sum())))
            if prediction.shape != (len(outer_test),) or not np.isfinite(prediction).all(): raise ValueError('Incomplete predictions')
            for i, p in zip(outer_test, prediction):
                all_predictions.append(dict(model=name, held_study=held, gsm=str(data['sample_ids'][i]),
                    group=str(groups[i]), actual_gy=float(y[i]), predicted_gy=float(p), error_gy=float(p-y[i])))
        # Persist progress without publishing aggregate held-out scores during tuning.
        write_tsv(LOG/'outer-predictions.tsv', all_predictions); write_tsv(LOG/'inner-predictions.tsv', inner_predictions)
        write_tsv(LOG/'tuning-trials.tsv', trials); save_json(LOG/'fold-audits.json', audits)
        save_json(LOG/'selected-configurations.json', selected); save_json(LOG/'prediction-checks.json', model_checks)
        print(f'{held}: outer model artifacts, predictions and replay checks saved', flush=True)
    metrics=[]; donor_rows=[]
    for name in MODELS:
        for st in (*STUDIES, 'pooled'):
            rows=[r for r in all_predictions if r['model']==name and (st=='pooled' or r['held_study']==st)]
            actual=np.array([r['actual_gy'] for r in rows]); predicted=np.array([r['predicted_gy'] for r in rows])
            gg=np.array([r['group'] for r in rows]); ss=np.array([r['held_study'] for r in rows])
            m, dd=donor_metrics(actual, predicted, gg)
            ci=bootstrap_intervals(actual, predicted, gg, ss)
            metrics.append(dict(model=name, study=st, **m, **{k+'_ci_low':v[0] for k,v in ci.items()},
                                **{k+'_ci_high':v[1] for k,v in ci.items()}))
            if st!='pooled': donor_rows.extend(dict(model=name, study=st, **r) for r in dd)
    write_tsv(LOG/'metrics.tsv', metrics); write_tsv(LOG/'donor-errors.tsv', donor_rows)
    macro={name:float(np.mean([m['mae'] for m in metrics if m['model']==name and m['study']!='pooled'])) for name in MODELS}
    code=['core_baseline.py','run_core_baselines.py','core_transfer.py','masked_quantile.py','assay_qc.py']
    artifacts=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()]
    summary=dict(completed_utc=datetime.now(timezone.utc).isoformat(), scope='Nested core-study baseline development evaluation',
        samples=145, study_local_donor_groups=30, outer_folds=3, inner_folds_per_outer=5,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (INPUT,LABELS,method)},
        code_sha256={str(ROOT.joinpath('scripts',p).relative_to(ROOT)):sha(ROOT/'scripts'/p) for p in code},
        metrics=metrics, equal_study_macro_mae=macro, selected=selected, prediction_checks=model_checks,
        valid_trials=sum(t['status']=='valid' for t in trials), rejected_trials=sum(t['status']=='rejected' for t in trials),
        total_trials=len(trials), inner_fit_attempts=len(trials)*5, artifacts=artifacts,
        model_artifact_bytes=sum(a['bytes'] for a in artifacts), runtime_seconds=time.perf_counter()-started,
        limitations=['Development CV across only three studies; not locked-test performance.',
                    'Bootstrap intervals are conditional on observed studies and fitted models; five-donor cohorts have limited precision.',
                    'Study-local donor groups are inferred from source evidence, not independently established global identities.',
                    'Normalization and biological portability remain candidates; substantial study effects remain.',
                    'GSE65292 candidate 2.23 Gy retains source 2.2/2.25 Gy discrepancy.',
                    'Equal profile training weights; donor-balanced tuning/evaluation. Top-k/parameter grids are limited.',
                    'Auxiliary learning, A-D neural strategies, locked-test evaluation and cloud serving are not performed.'])
    save_json(LOG/'summary.json', summary)
    print('COMPLETE', json.dumps({m['model']:m['mae'] for m in metrics if m['study']=='pooled'}), flush=True)


if __name__ == '__main__': main()
