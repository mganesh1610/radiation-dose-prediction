"""Frozen, limited joint comparison; old development artifacts are read-only."""
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

from abcd_models import balanced_weights, fit_network, predict, save_model, load_model
from core_baseline import GenePreparation
from development_preprocessing import AUX, DevelopmentPreprocessor, sample_ranks
from matched_classical import fit_classical
from run_abcd_experiments import (sha, save_json, write_tsv, verify_files, utc,
                                  completed, finish, fit_core, signature, make_batch)
from transfer_robust import fit_stable, panel_transform, candidate_score

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/transfer-robust/v1'
OUT = ROOT / 'data/models/transfer-robust-v1'
PARENT = ROOT / 'data/models/abcd-v1'
MODELS = ['A', 'B', 'C', 'D', 'elastic_net', 'pls', 'gradient_boosting']


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fitted(data, meta, train, labels):
    prep, base, mean, scale = fit_core(data, meta, train, labels)
    view = prep.transform(data, train)
    y = np.array([labels[s]['dose_gy'] for s in train])
    studies = np.array([labels[s]['study'] for s in train]); donors = np.array([labels[s]['donor_group'] for s in train])
    panels, stats, arrays = {}, {}, {}
    for rep in ('log2', 'rank'):
        panels[rep+'-original'] = base[rep]
        stats[rep+'-original'] = dict(selected_genes=len(base[rep].selected), zero_padding=500-len(base[rep].selected))
        gp, detail, evidence = fit_stable(view[rep+'_values'].T, view['gene_ids'], y, studies, donors, train)
        panels[rep+'-stable'] = gp; stats[rep+'-stable'] = detail; arrays[rep+'-stable'] = evidence
    return prep, panels, mean, scale, stats, arrays


def prepare_fold(data, meta, labels, aux, fold, identity):
    path = OUT / 'folds' / fold['name']; path.mkdir(parents=True, exist_ok=True)
    if not completed(path, identity):
        train, val = fold['training'], fold['validation']
        assert set(train).isdisjoint(val)
        assert {labels[s]['donor_group'] for s in train}.isdisjoint({labels[s]['donor_group'] for s in val})
        p, g, mean, scale, stats, arrays = fitted(data, meta, train, labels)
        # Replace every nontraining core signal, detection flag and target.
        altered = dict(data); a = dict(data['core']); nontrain = ~np.isin(a['sample_ids'], train)
        a['values'] = a['values'].copy(); a['values'][:, nontrain] = 987654.
        a['detected'] = a['detected'].copy(); a['detected'][:, nontrain] = False
        altered['core'] = a
        fake_labels = {s: (r if s in train else dict(r, dose_gy=9876.)) for s, r in labels.items()}
        p2, g2, m2, s2, _, _ = fitted(altered, meta, train, fake_labels)
        assert signature(p, g, mean, scale) == signature(p2, g2, m2, s2)
        del p2, g2, altered, a
        parent_match = None
        old = PARENT / 'folds' / fold['name']
        if (old / 'fit.json').exists():
            oldfit = json.loads((old/'fit.json').read_text())
            assert oldfit['training'] == train and oldfit['validation'] == val
            pp = DevelopmentPreprocessor.load(old/'preprocessing')
            assert np.array_equal(p.core.reference.target, pp.core.reference.target) and np.array_equal(p.core.keep, pp.core.keep)
            for rep in ('log2', 'rank'):
                assert g[rep+'-original'].fingerprint() == oldfit['representations'][rep]['preparation_fingerprint']
            parent_match = True
        p.save(path/'preprocessing')
        view = p.transform(data, train+val)
        lookup = {str(gene): i for i, gene in enumerate(aux['gene_ids'])}
        aux_log = aux['log2_values'][[lookup[gene] for gene in p.gene_ids]]
        aux_views = {'log2': aux_log, 'rank': sample_ranks(aux_log)}
        for name, gp in g.items():
            rep = name.split('-')[0]
            gp.save(path/f'{name}-features.npz')
            x = panel_transform(gp, view[rep+'_values'].T, view['gene_ids'])
            ax = panel_transform(gp, aux_views[rep].T, view['gene_ids'])
            np.savez_compressed(path/f'{name}-panel.npz', train=x[:len(train)], validation=x[len(train):],
                aux=ax, training_ids=train, validation_ids=val, auxiliary_ids=aux['profile_ids'])
            if name in arrays:
                np.savez_compressed(path/f'{name}-selection-evidence.npz', **arrays[name])
            stats[name].update(fingerprint=gp.fingerprint(), genes=[str(gp.gene_ids[i]) for i in gp.selected])
        save_json(path/'fit.json', dict(training=train, validation=val, target_mean=mean, target_scale=scale,
            raw_genes=len(p.gene_ids), preparations=stats, nontraining_perturbation_invariant=True,
            parent_original_preparation_exact=parent_match, auxiliary_training_ids=list(aux['profile_ids'])))
        finish(path, identity)
        print('Prepared', fold['name'], 'stable log2/rank genes', stats['log2-stable']['selected_genes'], stats['rank-stable']['selected_genes'], flush=True)
    return path, json.loads((path/'fit.json').read_text())


def one_fit(fold, params, seed, labels, path, identity, selected=False):
    path.mkdir(parents=True, exist_ok=True)
    if not completed(path, identity):
        fd = OUT/'folds'/fold['name']; fit = json.loads((fd/'fit.json').read_text())
        name = params['representation']+'-'+params['selector']
        panel = read_npz(fd/f'{name}-panel.npz')
        assert fit['training'] == fold['training'] and fit['validation'] == fold['validation']
        assert list(panel['training_ids']) == fit['training'] and list(panel['validation_ids']) == fit['validation']
        started = time.perf_counter(); model_name = params['model']
        if model_name in ('A', 'B', 'C', 'D'):
            batch = make_batch(panel, fit, labels, model_name)
            estimator, trace = fit_network(batch, params, seed)
            pred = predict(estimator, panel['validation'], fit['target_mean'], fit['target_scale'])
        else:
            rr = [labels[s] for s in fit['training']]
            weights = balanced_weights([r['study'] for r in rr], [r['donor_group'] for r in rr])
            y = (np.array([r['dose_gy'] for r in rr])-fit['target_mean'])/fit['target_scale']
            estimator, trace = fit_classical(params, panel['train'], y, weights, seed)
            pred = np.asarray(estimator.predict(panel['validation'])).ravel()*fit['target_scale']+fit['target_mean']
        elapsed = time.perf_counter()-started; replay = None; single = None
        if selected:
            if model_name in ('A', 'B', 'C', 'D'):
                save_model(path/'model.pt', estimator, params, seed, fit['target_mean'], fit['target_scale'])
                loaded, ck = load_model(path/'model.pt')
                again = predict(loaded, panel['validation'], ck['target_mean'], ck['target_scale'])
                individual = np.array([predict(loaded, row[None], ck['target_mean'], ck['target_scale'])[0] for row in panel['validation']])
            else:
                joblib.dump(estimator, path/'model.joblib', compress=3)
                loaded = joblib.load(path/'model.joblib')
                again = np.asarray(loaded.predict(panel['validation'])).ravel()*fit['target_scale']+fit['target_mean']
                individual = np.array([np.asarray(loaded.predict(row[None])).ravel()[0] for row in panel['validation']])*fit['target_scale']+fit['target_mean']
            replay = float(max(abs(pred-again))); single = float(max(abs(pred-individual)))
            assert replay < 1e-6 and single < 1e-4
        save_json(path/'result.json', dict(params=params, seed=seed, fold=fold['name'],
            training_ids=fit['training'], validation_ids=fit['validation'], predicted_gy=pred.tolist(),
            target_mean=fit['target_mean'], target_scale=fit['target_scale'], active_genes=fit['preparations'][name]['selected_genes'],
            fit_seconds=elapsed, trace=trace, reload_max_difference_gy=replay, single_max_difference_gy=single))
        finish(path, identity)
    return json.loads((path/'result.json').read_text())


def prediction_rows(result, labels):
    return [dict(profile_id=s, study=labels[s]['study'], donor_group=labels[s]['donor_group'],
        actual_gy=labels[s]['dose_gy'], predicted_gy=v) for s, v in zip(result['validation_ids'], result['predicted_gy'])]


def main():
    start = time.perf_counter(); cfg = json.loads((LOG/'config.json').read_text()); splits = json.loads((LOG/'splits.json').read_text())
    freeze = json.loads((LOG/'protocol-freeze.json').read_text())
    for name, h in freeze['sha256'].items():
        assert sha(LOG/name) == h
    sources = ['scripts/run_transfer_robust.py','scripts/transfer_robust.py','scripts/abcd_models.py','scripts/run_abcd_experiments.py',
               'scripts/core_baseline.py','scripts/matched_classical.py','scripts/development_preprocessing.py','scripts/core_transfer.py',
               'scripts/masked_quantile.py','scripts/assay_qc.py']
    code = {s: sha(ROOT/s) for s in sources}
    identity = hashlib.sha256(json.dumps(dict(protocol=freeze['sha256'], code=code), sort_keys=True).encode()).hexdigest()
    if (LOG/'run-start.json').exists():
        assert json.loads((LOG/'run-start.json').read_text())['identity'] == identity
    else:
        save_json(LOG/'run-start.json', dict(utc=utc(), identity=identity, code_sha256=code, python=sys.version,
            torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__, platform=platform.platform(), threads=4))
    verify_files(json.loads((LOG/'protected-before.json').read_text()))
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    release = ROOT/cfg['release']; assert sha(release/'release.json') == cfg['release_sha256']
    meta = json.loads((release/'metadata.json').read_text()); labels = {r['profile_id']: r for r in meta['profiles']}
    data = {'core': read_npz(release/'core.npz'), **{s: dict(profile_ids=np.array([], dtype=str)) for s in AUX}}
    assert list(data['core']['sample_ids']) == splits['core_profiles']
    am = json.loads((PARENT/'auxiliary/complete.json').read_text())
    for name, h in am['files'].items():
        assert sha(PARENT/'auxiliary'/name) == h
    aux = read_npz(PARENT/'auxiliary/gene-views.npz')
    assert list(aux['profile_ids']) == splits['auxiliary_training'] and len(aux['profile_ids']) == 422
    # All learned-preprocessing isolation checks complete before any predictor fit.
    for outer in splits['outer']:
        for fold in outer['inner']+[outer['outer_fold']]:
            prepare_fold(data, meta, labels, aux, fold, identity)
    del data, aux
    output, selections, count = [], [], 0
    with threadpool_limits(limits=4):
        for outer in splits['outer']:
            study = outer['study']
            for model in MODELS:
                candidates = []
                for params in cfg['candidates'][model]:
                    validation = {'donor': [], 'source': []}
                    for fold in outer['inner']:
                        r = one_fit(fold, params, fold['seed'], labels,
                            OUT/'trials'/fold['name']/model/str(params['id']), identity)
                        count += 1; validation[fold['design']] += prediction_rows(r, labels)
                    candidates.append(dict(params=params, **candidate_score(validation['donor'], validation['source'])))
                chosen = min(candidates, key=lambda r: (r['score'], r['params']['id']))
                arms = {arm: min([r for r in candidates if r['params']['selector']==arm], key=lambda r: (r['score'], r['params']['id'])) for arm in ['original','stable']}
                selections.append(dict(study=study, model=model, chosen=chosen, arms=arms, candidates=candidates))
                save_json(LOG/'selections.json', selections)
                for arm, selected in arms.items():
                    seeds = outer['seeds'] if model in ['A','B','C','D','gradient_boosting'] else outer['seeds'][:1]
                    for si, seed in enumerate(seeds):
                        r = one_fit(outer['outer_fold'], selected['params'], seed, labels,
                            OUT/'selected'/study/model/arm/str(seed), identity, True)
                        count += 1
                        for row in prediction_rows(r, labels):
                            output.append(dict(model=model, selector_arm=arm, joint_selected=(arm==chosen['params']['selector']),
                                seed_index=si, seed=seed, candidate_id=selected['params']['id'], representation=selected['params']['representation'],
                                active_genes=r['active_genes'], **row))
                write_tsv(LOG/'predictions.tsv', output)
                save_json(LOG/'progress.json', dict(utc=utc(), completed_fits=count, expected_fits=1278, last_study=study, last_model=model))
                print('Selected', study, model, chosen['params'], 'inner score', round(chosen['score'],4), 'fits', count, flush=True)
    assert count == 1278
    verify_files(json.loads((LOG/'protected-before.json').read_text()))
    save_json(LOG/'run-complete.json', dict(utc=utc(), identity=identity, total_fits=count, inner_fits=1176, outer_fits=102,
        wall_seconds=time.perf_counter()-start, maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        prediction_rows=len(output), locked_test_opened=False, cloud_changed=False))
    print('COMPLETE', count, 'fits;', len(output), 'prediction rows', flush=True)


if __name__ == '__main__':
    main()
