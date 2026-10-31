"""Execute the immutable final-selection-v2 plan locally; never open test data."""
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits

import run_abcd_experiments as previous
from abcd_models import balanced_weights, fit_network, predict, save_model
from core_baseline import donor_metrics
from development_preprocessing import load_release
from matched_classical import fit_classical
from run_abcd_experiments import completed, finish, save_json, sha, utc, verify_files, write_tsv
from verify_final_selection_plan import validate

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'context/final-selection/v2'
LOG = ROOT / 'context/final-training/v1'
OUT = ROOT / 'data/models/final-selection-v2'
SOURCE_NAMES = [
    'run_final_selection.py', 'final_model_bundle.py', 'verify_final_selection_plan.py',
    'run_abcd_experiments.py', 'abcd_models.py', 'matched_classical.py',
    'development_preprocessing.py', 'core_baseline.py', 'core_transfer.py',
    'frozen_affymetrix.py', 'masked_quantile.py', 'assay_qc.py',
]
PROTECTED_DIRS = [
    'context/model-abcd', 'context/classical-comparison', 'context/control-bias-investigation',
    'context/transfer-robust', 'context/extended-models', 'context/clustering',
    'context/final-selection', 'context/auxiliary-processing', 'output/final_proposal',
]


def read(path):
    return json.loads(Path(path).read_text())


def check_plan():
    freeze = read(PLAN / 'protocol-freeze.json')
    for name, expected in freeze['sha256'].items():
        if sha(PLAN / name) != expected:
            raise ValueError('Frozen plan changed: ' + name)
    if sha(ROOT / 'scripts/prepare_final_selection.py') != freeze['preparation_source_sha256']:
        raise ValueError('Preparation source changed')
    verify_files(read(PLAN / 'evidence-hashes.json'))
    cfg, splits = read(PLAN / 'config.json'), read(PLAN / 'splits.json')
    validate(splits, cfg, read(PLAN / 'planned-jobs.json'))
    if sha(ROOT / cfg['release'] / 'release.json') != cfg['release_manifest_sha256']:
        raise ValueError('Release identity changed')
    return cfg, splits, freeze


def protected_snapshot():
    paths = set()
    for name in PROTECTED_DIRS:
        paths.update(p for p in (ROOT / name).rglob('*') if p.is_file())
    # Previous fitted artifacts only; reserved raw expression is neither read nor hashed.
    for directory in (ROOT / 'data/models').iterdir():
        if directory != OUT and directory.is_dir():
            paths.update(p for p in directory.rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)
            if '__pycache__' not in p.parts and not p.name.endswith('.partial')}


def read_auxiliary(splits):
    """Reuse only the exact, label-free 422-profile reference with verified provenance."""
    directory = ROOT / 'data/models/abcd-v1/auxiliary'
    start = read(ROOT / 'context/model-abcd/v1/run-start.json')
    verify_files(start['source_sha256'])
    if not completed(directory, start['identity']):
        raise ValueError('Missing original auxiliary reference')
    membership = read(directory / 'fit-membership.json')
    ids = sorted(s for record in membership.values() for s in record['training_profiles'])
    if ids != sorted(splits['auxiliary_profiles']) or len(ids) != 422:
        raise ValueError('Auxiliary reference membership mismatch')
    with np.load(directory / 'gene-views.npz', allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    if list(result['profile_ids']) != splits['auxiliary_profiles']:
        raise ValueError('Auxiliary reference order mismatch')
    return result, dict(directory=str(directory.relative_to(ROOT)),
                        completion_sha256=sha(directory / 'complete.json'),
                        source_identity=start['identity'], profiles=422,
                        refitted_this_run=False, labels_used_in_reference=False)


def candidate_score(results, expected_ids, labels):
    ids = [s for r in results for s in r['prediction_ids']]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('Incomplete or repeated out-of-fold predictions')
    values = [v for r in results for v in r['prediction_gy']]
    return donor_metrics([labels[s]['dose_gy'] for s in ids], values,
                         [labels[s]['donor_group'] for s in ids])[0]


def choose_candidate(candidates):
    if len(candidates) != 8 or sorted(r['params']['id'] for r in candidates) != list(range(8)):
        raise ValueError('All original candidates must complete')
    if not all(np.isfinite(r['metrics']['mae']) for r in candidates):
        raise ValueError('Invalid candidate score')
    return min(candidates, key=lambda r: (r['metrics']['mae'], r['params']['id']))


def one_fit(directory, fit, labels, model_name, params, seed, destination, identity, final=False):
    destination.mkdir(parents=True, exist_ok=True)
    if completed(destination, identity):
        return read(destination / 'result.json')
    if any(destination.iterdir()):
        raise ValueError('Partial fit exists; preserve and diagnose before retrying: ' + str(destination))
    with np.load(directory / f"{params['representation']}-panel.npz", allow_pickle=False) as z:
        panel = {k: z[k] for k in z.files}
    if list(panel['training_ids']) != fit['training'] or list(panel['validation_ids']) != fit['validation']:
        raise ValueError('Panel membership mismatch')
    prediction_ids = fit['training'] if final else fit['validation']
    x = panel['train'] if final else panel['validation']
    started = time.perf_counter()
    if model_name in ('A', 'B', 'C', 'D'):
        batch = previous.make_batch(panel, fit, labels, model_name)
        model, trace = fit_network(batch, params, seed)
        values = predict(model, x, fit['target_mean'], fit['target_scale'])
        save_model(destination / 'model.pt', model, params, seed, fit['target_mean'], fit['target_scale'])
        weights = batch['core_weights'].numpy().tolist()
    else:
        records = [labels[s] for s in fit['training']]
        w = balanced_weights([r['study'] for r in records], [r['donor_group'] for r in records])
        y = (np.array([r['dose_gy'] for r in records]) - fit['target_mean']) / fit['target_scale']
        model, trace = fit_classical(params, panel['train'], y, w, seed)
        values = np.asarray(model.predict(x)).ravel() * fit['target_scale'] + fit['target_mean']
        joblib.dump(model, destination / 'model.joblib', compress=3)
        weights = w.tolist()
    if not np.isfinite(values).all() or len(values) != len(prediction_ids):
        raise ValueError('Invalid predictions')
    metrics, _ = donor_metrics([labels[s]['dose_gy'] for s in prediction_ids], values,
                               [labels[s]['donor_group'] for s in prediction_ids])
    result = dict(model=model_name, params=params, seed=seed, phase='full_refit' if final else 'selection',
                  evaluation='in_training_replay_not_accuracy' if final else 'tuning_oof_not_external_accuracy',
                  fold=directory.name, preprocessing=str(directory.relative_to(OUT)),
                  training_ids=fit['training'], auxiliary_training_ids=list(panel['auxiliary_ids']) if model_name in ('B', 'C', 'D') else [],
                  prediction_ids=prediction_ids, prediction_gy=values.tolist(), metrics=metrics,
                  core_weights=weights, trace=trace, fit_seconds=time.perf_counter() - started,
                  target_mean=fit['target_mean'], target_scale=fit['target_scale'])
    save_json(destination / 'result.json', result)
    finish(destination, identity)
    return result


def prediction_rows(result, labels):
    return [dict(model=result['model'], candidate_id=result['params']['id'], seed=result['seed'],
                 fold=result['fold'], evaluation=result['evaluation'], profile_id=s,
                 study=labels[s]['study'], donor_group=labels[s]['donor_group'],
                 actual_gy=labels[s]['dose_gy'], predicted_gy=p)
            for s, p in zip(result['prediction_ids'], result['prediction_gy'])]


def main():
    started = time.perf_counter()
    cfg, splits, freeze = check_plan()
    LOG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    code = {f'scripts/{name}': sha(ROOT / 'scripts' / name) for name in SOURCE_NAMES}
    runtime = dict(python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__,
                   torch=str(torch.__version__), joblib=joblib.__version__, platform=platform.platform(),
                   device='cpu', threads=4)
    identity = hashlib.sha256(json.dumps(dict(protocol=freeze, code=code, runtime=runtime), sort_keys=True).encode()).hexdigest()
    if (LOG / 'run-start.json').exists():
        if read(LOG / 'run-start.json')['identity'] != identity:
            raise ValueError('Run changed; preserve existing attempt and use a new run version')
        verify_files(read(LOG / 'protected-before.json'))
        if (LOG / 'run-complete.json').exists():
            print('Already complete; use the verifier, not another training run.', flush=True)
            return
    else:
        save_json(LOG / 'protected-before.json', protected_snapshot())
        save_json(LOG / 'run-start.json', dict(utc=utc(), identity=identity, source_sha256=code,
                  runtime=runtime, protocol_freeze_sha256=sha(PLAN / 'protocol-freeze.json'),
                  locked_test_opened=False, cloud_execution=False))
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    data, metadata = load_release(ROOT / cfg['release'])
    labels = {r['profile_id']: r for r in metadata['profiles']}
    if set(labels) != set(splits['core_profiles'] + splits['auxiliary_profiles']):
        raise ValueError('Release membership changed')
    auxiliary, aux_record = read_auxiliary(splits)
    save_json(LOG / 'auxiliary-reuse.json', aux_record)
    # Redirect only this process's helper output root; the previous module is unchanged.
    previous.OUT = OUT
    prepared = {}
    for f in splits['folds']:
        fold = dict(name=f['fold_id'], training=f['training'], validation=f['validation'])
        prepared[f['fold_id']] = previous.prepare_fold(data, metadata, labels, auxiliary, fold, identity)
    rows, chosen, count = [], [], 0
    for name in cfg['models']:
        scores = []
        for params in cfg['candidates'][name]:
            results = []
            for f in splits['folds']:
                directory, fit = prepared[f['fold_id']]
                r = one_fit(directory, fit, labels, name, params, f['seed'],
                            OUT / 'trials' / f['fold_id'] / name / str(params['id']), identity)
                results.append(r); rows.extend(prediction_rows(r, labels)); count += 1
            score = candidate_score(results, splits['core_profiles'], labels)
            scores.append(dict(params=params, metrics=score))
            print(name, params['id'], 'selection donor MAE', round(score['mae'], 6), flush=True)
            save_json(LOG / 'progress.json', dict(status='selection_running', candidate_fits=count, utc=utc()))
        selected = choose_candidate(scores)
        chosen.append(dict(model=name, chosen=selected, candidates=scores))
        save_json(LOG / 'selections.json', chosen)
        write_tsv(LOG / 'candidate-predictions.tsv', rows)
    if count != 280 or len(rows) != 8120:
        raise ValueError('Incomplete candidate budget')
    # Freeze fitted settings before any full refit; these are not final-test findings.
    selected_hash = sha(LOG / 'selections.json')
    save_json(LOG / 'selected-settings-freeze.json', dict(utc=utc(), selections_sha256=selected_hash,
              identity=identity, locked_test_opened=False))
    fold = dict(name='full-development', training=splits['core_profiles'], validation=[])
    directory, fit = previous.prepare_fold(data, metadata, labels, auxiliary, fold, identity)
    by_model = {r['model']: r['chosen']['params'] for r in chosen}
    final_rows, artifacts = [], []
    for planned in read(PLAN / 'expected-artifacts.json'):
        name, seed = planned['model'], planned['seed']
        if name == 'median_null':
            continue
        dest = OUT / 'final' / name / str(seed)
        result = one_fit(directory, fit, labels, name, by_model[name], seed, dest, identity, final=True)
        final_rows.extend(prediction_rows(result, labels))
        artifacts.append(dict(model=name, seed=seed, seed_role=planned['seed_role'],
                         candidate_id=by_model[name]['id'], path=str(dest.relative_to(OUT)),
                         complete_sha256=sha(dest / 'complete.json')))
        print('Full refit saved', name, seed, flush=True)
        save_json(LOG / 'progress.json', dict(status='full_refits_running', candidate_fits=count,
                  full_refits=len(artifacts), utc=utc()))
    median = float(np.median([labels[s]['dose_gy'] for s in splits['core_profiles']]))
    null_dir = OUT / 'final/median_null'; null_dir.mkdir(parents=True, exist_ok=True)
    if not completed(null_dir, identity):
        save_json(null_dir / 'model.json', dict(model='median_null', prediction_gy=median,
                  training_ids=splits['core_profiles'], definition='ordinary_median_of_core_profile_doses',
                  no_auxiliary_fit=True))
        finish(null_dir, identity)
    artifacts.append(dict(model='median_null', seed=0, seed_role='untuned', candidate_id=None,
                     path=str(null_dir.relative_to(OUT)), complete_sha256=sha(null_dir / 'complete.json')))
    if len(artifacts) != 18 or len(final_rows) != 2465:
        raise ValueError('Incomplete final artifacts')
    save_json(OUT / 'artifact-index.json', dict(identity=identity, protocol='final-selection-v2',
              selected_settings_sha256=selected_hash, artifacts=artifacts,
              intended_scope_sha256=sha(PLAN / 'prediction-scope.json'), deployed=False,
              evaluation_status='development_selection_only_locked_test_pending'))
    write_tsv(LOG / 'full-fit-replay-predictions.tsv', final_rows)
    save_json(LOG / 'artifacts.json', artifacts)
    verify_files(read(LOG / 'protected-before.json'))
    save_json(LOG / 'run-complete.json', dict(utc=utc(), identity=identity, candidate_fits=280,
              full_refits=17, median_artifacts=1, candidate_prediction_rows=len(rows),
              full_fit_replay_rows=len(final_rows), wall_seconds=time.perf_counter() - started,
              process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              reused_auxiliary_reference=True, locked_test_opened=False, cloud_changed=False,
              deployed=False, production_model_selected=False))
    save_json(LOG / 'progress.json', dict(status='training_complete_verification_pending', utc=utc(),
              candidate_fits=280, full_refits=17, median_artifacts=1))
    print('COMPLETE: 280 selection fits, 17 full refits, one median artifact.', flush=True)


if __name__ == '__main__':
    try:
        with threadpool_limits(limits=4):
            main()
    except Exception as exc:
        LOG.mkdir(parents=True, exist_ok=True)
        save_json(LOG / f'failure-{time.time_ns()}.json', dict(utc=utc(), error=str(exc), type=type(exc).__name__))
        raise
