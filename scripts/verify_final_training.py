"""Independent membership, ranking, raw replay and artifact audit; no model fitting."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from development_preprocessing import load_release, DevelopmentPreprocessor
from final_model_bundle import load_features, predict_panel
from run_abcd_experiments import sha, save_json, write_tsv, verify_files, completed, utc
from run_final_selection import ROOT, PLAN, LOG, OUT, check_plan, read


def metric(rows):
    if not rows:
        return dict(mae_gy=None, rmse_gy=None, bias_gy=None, donors=0, profiles=0)
    groups = {}
    for r in rows:
        groups.setdefault(r['donor_group'], []).append(float(r['predicted_gy']) - float(r['actual_gy']))
    return dict(mae_gy=float(np.mean([np.mean(np.abs(e)) for e in groups.values()])),
                rmse_gy=float(np.sqrt(np.mean([np.mean(np.square(e)) for e in groups.values()]))),
                bias_gy=float(np.mean([np.mean(e) for e in groups.values()])),
                donors=len(groups), profiles=len(rows))


def rows_for(result, labels):
    assert len(result['prediction_ids']) == len(result['prediction_gy'])
    return [dict(model=result['model'], profile_id=s, study=labels[s]['study'],
                 donor_group=labels[s]['donor_group'], actual_gy=labels[s]['dose_gy'], predicted_gy=v)
            for s, v in zip(result['prediction_ids'], result['prediction_gy'])]


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a - b)))


def main():
    started = time.perf_counter()
    cfg, splits, freeze = check_plan()
    start, end = read(LOG / 'run-start.json'), read(LOG / 'run-complete.json')
    identity = start['identity']
    assert end['identity'] == identity and freeze['created_utc'] < start['utc'] < end['utc']
    verify_files(start['source_sha256'])
    assert start['protocol_freeze_sha256'] == sha(PLAN / 'protocol-freeze.json')
    selected_freeze = read(LOG / 'selected-settings-freeze.json')
    assert selected_freeze['selections_sha256'] == sha(LOG / 'selections.json')
    assert start['utc'] < selected_freeze['utc'] < end['utc']
    assert end['candidate_fits'] == 280 and end['full_refits'] == 17 and end['median_artifacts'] == 1
    data, metadata = load_release(ROOT / cfg['release'])
    labels = {r['profile_id']: r for r in metadata['profiles']}
    selections = {r['model']: r for r in read(LOG / 'selections.json')}
    assert set(selections) == set(cfg['models'])
    # Reconstruct both representations for all five selection folds and the full-development fit.
    raw_panels, fold_records = {}, []
    all_folds = splits['folds'] + [dict(fold_id='full-development', training=splits['core_profiles'], validation=[])]
    for fold in all_folds:
        directory = OUT / 'folds' / fold['fold_id']; f = read(directory / 'fit.json')
        assert completed(directory, identity)
        assert f['training'] == fold['training'] and f['validation'] == fold['validation']
        assert f['nontraining_perturbation_invariant']
        prep = DevelopmentPreprocessor.load(directory / 'preprocessing')
        assert sorted(prep.core_training) == sorted(f['training']) and not prep.aux_training
        ids = f['training'] + f['validation']; view = prep.transform(data, ids)
        target = np.array([labels[s]['dose_gy'] for s in f['training']])
        assert f['target_mean'] == float(target.mean()) and f['target_scale'] == float(target.std())
        for rep in ('log2', 'rank'):
            gp = load_features(directory / f'{rep}-features.npz')
            assert list(gp.training_ids) == f['training'] and len(gp.selected) == 500
            assert gp.fingerprint() == f['representations'][rep]['preparation_fingerprint']
            x = gp.transform(view[f'{rep}_values'].T, view['gene_ids'], True)
            with np.load(directory / f'{rep}-panel.npz', allow_pickle=False) as z:
                assert list(z['training_ids']) == f['training'] and list(z['validation_ids']) == f['validation']
                assert list(z['auxiliary_ids']) == splits['auxiliary_profiles']
                assert difference(x[:len(target)], z['train']) < 1e-12
                if f['validation']:
                    assert difference(x[len(target):], z['validation']) < 1e-12
            raw_panels[(fold['fold_id'], rep)] = x[len(target):] if f['validation'] else x
        fold_records.append(dict(fold=fold['fold_id'], training=len(f['training']), validation=len(f['validation']),
                                 raw_genes=f['raw_gene_count'], holdout_perturbation_passed=True))
    chosen_rows, audits, candidate_scores = [], [], []
    for name in cfg['models']:
        scores = []
        for params in cfg['candidates'][name]:
            rows = []
            for fold in splits['folds']:
                directory = OUT / 'trials' / fold['fold_id'] / name / str(params['id'])
                result = read(directory / 'result.json')
                assert result['params'] == params and result['seed'] == fold['seed'] and result['model'] == name
                assert result['phase'] == 'selection' and result['training_ids'] == fold['training']
                assert result['prediction_ids'] == fold['validation']
                assert result['auxiliary_training_ids'] == (splits['auxiliary_profiles'] if name in 'ABCD'[1:] else [])
                assert result['preprocessing'] == f"folds/{fold['fold_id']}"
                replay = predict_panel(directory, raw_panels[(fold['fold_id'], params['representation'])], identity)
                delta = difference(replay, result['prediction_gy']); assert delta < 1e-6
                audits.append(dict(phase='selection', model=name, candidate=params['id'], fold=fold['fold_id'],
                                   raw_replay_max_difference_gy=delta))
                rows.extend(rows_for(result, labels))
            assert sorted(r['profile_id'] for r in rows) == sorted(splits['core_profiles'])
            score = metric(rows)
            stored = selections[name]['candidates'][params['id']]
            assert stored['params'] == params
            for key in ('mae', 'rmse', 'bias'):
                assert abs(score[key + '_gy'] - stored['metrics'][key]) < 1e-12
            scores.append((score['mae_gy'], params['id']))
            candidate_scores.append(dict(model=name, candidate_id=params['id'], representation=params['representation'], **score))
            if params == selections[name]['chosen']['params']:
                chosen_rows.extend(rows)
        assert min(scores)[1] == selections[name]['chosen']['params']['id']
    # Cross-check exported 8,120 prediction rows against the underlying fit records.
    exported = list(csv.DictReader((LOG / 'candidate-predictions.tsv').open(), delimiter='\t'))
    assert len(exported) == 8120
    key_rows = {(r['fold'], r['model'], int(r['candidate_id']), r['profile_id']): r for r in exported}
    assert len(key_rows) == 8120
    for path in (OUT / 'trials').rglob('result.json'):
        result = read(path)
        for row in rows_for(result, labels):
            r = key_rows[(result['fold'], result['model'], result['params']['id'], row['profile_id'])]
            assert float(r['predicted_gy']) == row['predicted_gy'] and float(r['actual_gy']) == row['actual_gy']
            assert r['study'] == row['study'] and r['donor_group'] == row['donor_group']
    # Untuned median evaluated on the same five validation folds, no extra fitted predictor.
    for fold in splits['folds']:
        value = float(np.median([labels[s]['dose_gy'] for s in fold['training']]))
        chosen_rows.extend(dict(model='median_null', profile_id=s, study=labels[s]['study'],
                           donor_group=labels[s]['donor_group'], actual_gy=labels[s]['dose_gy'], predicted_gy=value)
                           for s in fold['validation'])
    write_tsv(LOG / 'selected-oof-predictions.tsv', chosen_rows)
    summary, slices = [], []
    for name in cfg['models'] + ['median_null']:
        rr = [r for r in chosen_rows if r['model'] == name]
        studies = sorted({r['study'] for r in rr})
        study_errors = [metric([r for r in rr if r['study'] == s])['mae_gy'] for s in studies]
        summary.append(dict(model=name, candidate_id=None if name == 'median_null' else selections[name]['chosen']['params']['id'],
                       evaluation='selection_cv_optimistic_not_external_accuracy', **metric(rr),
                       equal_study_mae_gy=float(np.mean(study_errors)), worst_study_mae_gy=max(study_errors)))
        parts = [('all', rr), ('control', [r for r in rr if r['actual_gy'] == 0]),
                 ('0–2 Gy', [r for r in rr if r['actual_gy'] <= 2]), ('>2 Gy', [r for r in rr if r['actual_gy'] > 2])]
        for study in studies:
            part = [r for r in rr if r['study'] == study]
            parts.extend([(study, part), (study + ':control', [r for r in part if r['actual_gy'] == 0]),
                          (study + ':>2 Gy', [r for r in part if r['actual_gy'] > 2])])
        parts.extend((f'dose={dose:g} Gy', [r for r in rr if r['actual_gy'] == dose]) for dose in sorted({r['actual_gy'] for r in rr}))
        slices.extend(dict(model=name, slice=label, **metric(part)) for label, part in parts)
    save_json(LOG / 'selection-summary.json', summary); write_tsv(LOG / 'selection-slices.tsv', slices)
    write_tsv(LOG / 'candidate-scores.tsv', candidate_scores)
    index = read(OUT / 'artifact-index.json'); assert index['identity'] == identity
    assert index['selected_settings_sha256'] == sha(LOG / 'selections.json') and not index['deployed']
    artifacts = read(LOG / 'artifacts.json'); assert artifacts == index['artifacts'] and len(artifacts) == 18
    expected = {(a['model'], a['seed']) for a in read(PLAN / 'expected-artifacts.json') if a['model'] != 'median_null'}
    assert {(a['model'], a['seed']) for a in artifacts if a['model'] != 'median_null'} == expected
    assert len(list((OUT / 'trials').rglob('result.json'))) == 280
    assert len(list((OUT / 'final').rglob('result.json'))) == 17
    # Raw single-input and reversed-order checks, including one control and high doses from each study.
    full = OUT / 'folds/full-development'; prep = DevelopmentPreprocessor.load(full / 'preprocessing')
    core = splits['core_profiles']; probes = []
    for study in sorted({labels[s]['study'] for s in core}):
        ordered = sorted((s for s in core if labels[s]['study'] == study), key=lambda s: (labels[s]['dose_gy'], s))
        probes.extend([ordered[0], ordered[len(ordered)//2], ordered[-1]])
    singles, reversed_panels = {}, {}
    reversed_view = prep.transform(data, list(reversed(core)))
    for rep in ('log2', 'rank'):
        gp = load_features(full / f'{rep}-features.npz')
        reversed_panels[rep] = gp.transform(reversed_view[f'{rep}_values'].T, reversed_view['gene_ids'], True)[::-1].copy()
        values = []
        for s in probes:
            view = prep.transform(data, [s]); values.append(gp.transform(view[f'{rep}_values'].T, view['gene_ids'], True)[0])
        singles[rep] = np.asarray(values)
        assert difference(reversed_panels[rep], raw_panels[('full-development', rep)]) < 1e-12
        assert difference(singles[rep], raw_panels[('full-development', rep)][[core.index(s) for s in probes]]) < 1e-12
    seed_rows = []
    for artifact in artifacts:
        directory = OUT / artifact['path']; assert sha(directory / 'complete.json') == artifact['complete_sha256']
        assert completed(directory, identity)
        name = artifact['model']
        if name == 'median_null':
            null = read(directory / 'model.json')
            assert null['training_ids'] == core and null['prediction_gy'] == float(np.median([labels[s]['dose_gy'] for s in core]))
            continue
        r = read(directory / 'result.json'); rep = r['params']['representation']
        assert r['phase'] == 'full_refit' and r['evaluation'] == 'in_training_replay_not_accuracy'
        assert r['seed'] == artifact['seed'] and r['model'] == name and r['params'] == selections[name]['chosen']['params']
        assert r['training_ids'] == core and r['prediction_ids'] == core and r['preprocessing'] == 'folds/full-development'
        assert r['auxiliary_training_ids'] == (splits['auxiliary_profiles'] if name in ('B', 'C', 'D') else [])
        values = predict_panel(directory, raw_panels[('full-development', rep)], identity)
        delta = difference(values, r['prediction_gy']); assert delta < 1e-6
        individual = np.array([predict_panel(directory, x[None], identity)[0] for x in singles[rep]])
        single_delta = difference(individual, values[[core.index(s) for s in probes]]); assert single_delta < 1e-4
        order_delta = difference(values, predict_panel(directory, reversed_panels[rep], identity)); assert order_delta < 1e-6
        audits.append(dict(phase='full_refit', model=name, seed=r['seed'], raw_replay_max_difference_gy=delta,
                           individual_max_difference_gy=single_delta, reversed_order_difference_gy=order_delta))
        seed_rows.append(dict(model=name, seed=r['seed'], seed_role=artifact['seed_role'],
                         evaluation='in_training_replay_not_accuracy', **metric(rows_for(r, labels))))
    write_tsv(LOG / 'full-fit-seed-replay-metrics.tsv', seed_rows)
    manifests = 0
    for path in OUT.rglob('complete.json'):
        assert completed(path.parent, identity); manifests += 1
    assert manifests == 304  # 6 preparations + 280 candidates + 18 final artifacts
    verify_files(read(LOG / 'protected-before.json'))
    verify_files(start['source_sha256'])
    save_json(LOG / 'replays.json', audits)
    save_json(LOG / 'verification.json', dict(status='PASS_LOCAL_SELECTION_AND_FULL_REFITS', utc=utc(),
              candidate_fits=280, selected_family_rankings=56, final_trained_artifacts=17, median_artifacts=1,
              fold_preprocessing_checks=fold_records, raw_model_replays=len(audits),
              max_raw_replay_difference_gy=max(r['raw_replay_max_difference_gy'] for r in audits),
              max_individual_difference_gy=max(r.get('individual_max_difference_gy', 0) for r in audits),
              raw_single_inputs_checked=len(probes), completed_manifests=manifests,
              protected_files_unchanged=len(read(LOG / 'protected-before.json')),
              verifier_sha256=sha(Path(__file__)), verification_seconds=time.perf_counter() - started,
              locked_test_opened=False, hypothesis_tests_performed=False, prediction_accuracy_established=False,
              cloud_changed=False, production_model_promoted=False, app_enforcement_implemented=False))
    save_json(LOG / 'model-bundle-freeze.json', dict(utc=utc(), identity=identity,
              files={str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              final_models=18, purpose='research_local_verified_locked_evaluation_pending', cloud_runtime_verified=False))
    save_json(LOG / 'progress.json', dict(status='complete_verified_local_only', utc=utc(), candidate_fits=280,
              full_refits=17, median_artifacts=1))
    print(json.dumps(summary, indent=2)); print('VERIFIED: 56 rankings, 297 raw model replays, 304 manifests.', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with threadpool_limits(limits=4):
        main()
