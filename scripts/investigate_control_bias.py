"""Read-only diagnostics of frozen development pipelines; no predictor fitting.

Only the explicitly named core-development release is opened. Donor-control
contrasts explain errors; they are never substituted for validation predictions.
"""
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from abcd_models import load_model, predict
from assay_qc import aggregate_median, technical_valid
from core_baseline import GenePreparation
from development_preprocessing import DevelopmentPreprocessor
from prepare_core_background import parse, VALUES, FLAGS
from check_core_compatibility import read_platform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'context/control-bias-investigation/v1'
OLD = ROOT / 'context/classical-comparison/v1'
NEURAL = ROOT / 'context/model-abcd/v1'
FOLDS = ROOT / 'data/models/abcd-v1/folds'
RELEASE = ROOT / 'data/model_ready/development_raw_f6022cfaaca6d1d9'
RAW = ROOT / 'data/processed_pilot/core-reprocessing-v1'
STUDIES = ('GSE102971', 'GSE65292', 'GSE8917')
INPUTS = {}


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def use(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return path


def js(path):
    return json.loads(use(path).read_text())


def npz(path):
    with np.load(use(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rows(path):
    with use(path).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tsv(name, data):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), delimiter='\t')
        w.writeheader()
        w.writerows(data)


def donor_contrasts(records):
    """Require exactly one zero-dose observation per model/seed/donor."""
    groups = defaultdict(list)
    for r in records:
        groups[r['model'], int(r['seed_index']), r['study'], r['donor_group']].append(r)
    detail, slopes = [], []
    for (model, seed, study, donor), rr in sorted(groups.items()):
        rr = sorted(rr, key=lambda r: float(r['actual_gy']))
        controls = [r for r in rr if float(r['actual_gy']) == 0]
        doses = np.array([float(r['actual_gy']) for r in rr])
        pred = np.array([float(r['predicted_gy']) for r in rr])
        if len(controls) != 1 or len(set(doses)) != len(doses) or len(rr) < 3:
            raise ValueError('Missing/duplicate donor control or dose')
        if not np.isfinite(doses).all() or not np.isfinite(pred).all():
            raise ValueError('Nonfinite dose/prediction')
        control = float(controls[0]['predicted_gy'])
        for r, y, p in zip(rr, doses, pred):
            if y == 0:
                continue
            increment = p - control
            detail.append(dict(model=model, seed_index=seed, study=study, donor_group=donor,
                profile_id=r['profile_id'], dose_gy=float(y), predicted_gy=float(p),
                control_prediction_gy=control, predicted_increment_gy=float(increment),
                increment_error_gy=float(increment-y), residual_gy=float(p-y)))
        centered = doses - doses.mean()
        slope = float(centered @ (pred-pred.mean()) / (centered @ centered))
        slopes.append(dict(model=model, seed_index=seed, study=study, donor_group=donor,
            profiles=len(rr), control_prediction_gy=control, slope_gy_per_gy=slope,
            spearman=float(spearmanr(doses, pred).statistic) if np.ptp(pred) else None,
            top_actual_increment_gy=float(doses[-1]-doses[-2]),
            top_predicted_increment_gy=float(pred[-1]-pred[-2])))
    return detail, slopes


def linear_contributions(coef, control_train, control_held, target_scale):
    """Exact linear score displacement, not a causal explanation."""
    coef, a, b = map(np.asarray, (coef, control_train, control_held))
    if coef.ndim != 1 or coef.shape != a.shape or a.shape != b.shape:
        raise ValueError('Contribution axes disagree')
    return coef * (b-a) * target_scale


def source_audit(labels, core):
    soft = use(ROOT / 'data/public_radiation/core/GSE8917/GSE8917_family.soft.gz')
    samples, current = {}, None
    with gzip.open(soft, 'rt') as f:
        for line in f:
            if line.startswith('^SAMPLE = '):
                current = line.strip().split(' = ')[1]
                samples[current] = defaultdict(list)
            elif current and line.startswith('!Sample_') and ' = ' in line:
                k, v = line.strip().split(' = ', 1)
                samples[current][k[8:]].append(v)
    _, platform, _, platform_path = read_platform('GSE8917')
    use(platform_path)
    c = npz(RAW / 'GSE8917_canonical_source.npz')
    selected = [r for r in labels.values() if r['study'] == 'GSE8917']
    downloads = js(ROOT / 'context/core-reprocessing/v1/downloads.json')['files']
    acquired = {r['gsm']: r for r in downloads if r['study'] == 'GSE8917'}
    audit = []
    for r in selected:
        s = r['profile_id']; m = samples[s]
        match = re.fullmatch(r'ExVivoBlood_24hr_([\d.]+)Gy_(rep[1-5])', m['title'][0])
        assert match and float(match[1]) == r['dose_gy'] and r['donor_group'] == 'GSE8917:' + match[2]
        assert m['channel_count'] == ['1'] and m['organism_ch1'] == ['Homo sapiens'] and m['platform_id'] == ['GPL1708']
        assert r['time_hours'] == 24 and r['source_arrays'] == [s]
        download = acquired[s]; path = use(ROOT / download['path'])
        assert sha(path) == download['sha256']
        assert any(u.split('/')[-1] == path.name for u in m['supplementary_file'])
        fi, pr, x, q, seq, controls = parse(path, platform, 'GPL1708')
        j = list(c['sample_ids']).index(s)
        assert np.array_equal(fi, c['feature_ids']) and np.array_equal(pr, c['probe_names'])
        assert list(c['value_columns']) == VALUES and list(c['flag_columns']) == FLAGS
        assert np.array_equal(x, c['values'][:, j]) and np.array_equal(q, c['flags'][:, j])
        audit.append(dict(profile_id=s, title=m['title'][0], donor_group=r['donor_group'], dose_gy=r['dose_gy'],
            time_hours=24, channel=1, platform='GPL1708', raw_sha256=sha(path),
            raw_cells_reconciled=int(x.size+q.size), physical_features=len(fi),
            raw_sequence_available=seq, source_metadata_and_canonical_match=True))
    a = npz(RAW / 'GSE8917_axes_flags.npz')
    y = np.fromfile(use(RAW / 'GSE8917_normexp.bin'), dtype='<f8').reshape(len(a['sample_ids']), -1).T
    assert np.array_equal(np.isnan(y), ~a['technical_valid']) and np.all(y[a['technical_valid']] > 0)
    probes, signal, _ = aggregate_median(y, a['probe_names'])
    ix = {p: i for i, p in enumerate(probes)}
    selected_signal = signal[[ix[p] for p in core['probe_ids']]]
    cj = [list(core['sample_ids']).index(s) for s in a['sample_ids']]
    assert np.array_equal(selected_signal, core['values'][:, cj], equal_nan=True)
    assert set(c['sample_ids']) == {r['profile_id'] for r in selected} == set(acquired)
    tsv('source-audit.tsv', audit)
    return dict(profiles=len(audit), donor_groups=len({r['donor_group'] for r in audit}),
        source_and_canonical_match=True, background_to_release_exact=True,
        raw_cells_reconciled=sum(r['raw_cells_reconciled'] for r in audit),
        raw_sequence_limitation='Sequence column absent; external platform sequence concordance is available')


def technical_audit(labels):
    output = []
    params = rows(ROOT / 'context/core-reprocessing/v1/background-parameters.tsv')
    for study in STUDIES:
        cp = ROOT / 'data/processed_pilot/GSE102971/v1/canonical_source.npz' if study == 'GSE102971' else RAW / f'{study}_canonical_source.npz'
        c = npz(cp)
        if study == 'GSE102971':
            q = npz(ROOT / 'data/processed_pilot/GSE102971/audit-v1/quality_flags.npz')
            assert np.array_equal(c['sample_ids'], q['sample_ids'])
        else:
            q = c
        valid = technical_valid(q['flags'], list(q['flag_columns']))
        names = list(c['value_columns']); flags = list(q['flag_columns'])
        for j, s in enumerate(c['sample_ids']):
            fg = c['values'][:, j, names.index('gMedianSignal')]
            bg = c['values'][:, j, names.index('gBGMedianSignal')]
            good = valid[:, j]; p = next(r for r in params if r['study'] == study and int(r['column']) == j+1)
            assert p['convergence'] == '0' and not p['warnings']
            output.append(dict(study=study, profile_id=str(s), donor_group=labels[s]['donor_group'], dose_gy=labels[s]['dose_gy'],
                technical_mask_fraction=float((~good).mean()),
                detected_fraction=float(q['flags'][good, j, flags.index('gIsWellAboveBG')].mean()),
                foreground_median=float(np.median(fg[good])), background_median=float(np.median(bg[good])),
                nonpositive_fg_minus_bg_fraction=float(np.mean((fg-bg)[good] <= 0)),
                normexp_mu=float(p['mu']), normexp_sigma=float(p['sigma']), normexp_alpha=float(p['alpha'])))
    tsv('technical-qc.tsv', output)


def main():
    torch.set_num_threads(4)
    started = datetime.now(timezone.utc).isoformat()
    protected = js(OUT / 'protected-before.json')
    for p, h in protected.items():
        assert sha(ROOT / p) == h, p
    freeze = js(OUT / 'scope-freeze.json')
    assert sha(OUT / 'METHOD.md') == freeze['method_sha256'] and freeze['utc'] < started
    meta = js(RELEASE / 'metadata.json')
    labels = {r['profile_id']: r for r in meta['profiles'] if r['role'] == 'core_development'}
    assert len(labels) == 145 and {r['study'] for r in labels.values()} == set(STUDIES)
    core = npz(RELEASE / 'core.npz'); data = {'core': core}
    assert set(core['sample_ids']) == set(labels)
    audit = source_audit(labels, core)
    print('Source audit passed for 25 GSE8917 arrays', flush=True)
    technical_audit(labels)
    saved = rows(OLD / 'combined-predictions.tsv')
    detail, slopes = donor_contrasts(saved)
    tsv('paired-dose-decomposition.tsv', detail); tsv('donor-slopes.tsv', slopes)
    assert max(abs(r['residual_gy']-r['control_prediction_gy']-r['increment_error_gy']) for r in detail) < 1e-12
    splits = js(NEURAL / 'splits.json')['outer']
    cs = {(r['study'], r['model']): r['chosen']['params'] for r in js(OLD / 'selections.json')}
    ns = {(r['outer'], r['strategy']): r['chosen']['params'] for r in js(NEURAL / 'selections.json')}
    annot = rows(ROOT / 'context/core-reprocessing/v1/shared-probe-annotation.tsv')
    symbols = {r['GSE8917_gene_id']: r['current_symbol'] for r in annot if r['status'] == 'accepted'}
    profiles, distances, pca_rows, contributions, checks, concentration = [], [], [], [], [], []
    for split in splits:
        study = split['study']; d = FOLDS / study / 'outer'; f = js(d / 'fit.json')
        ids = f['training'] + f['validation']; n = len(f['training']); lab = [labels[s] for s in ids]
        assert f['training'] == split['training'] and f['validation'] == split['validation']
        use(d / 'preprocessing/core.npz'); use(d / 'preprocessing/processing.json')
        processor = DevelopmentPreprocessor.load(d / 'preprocessing')
        assert not processor.aux_training and set(processor.core_training) == set(f['training'])
        view = processor.transform(data, ids)
        panels, features = {}, {}
        for rep in ('log2', 'rank'):
            z = npz(d / f'{rep}-features.npz'); g = GenePreparation(tuple(z['gene_ids']), z['medians'], z['selected'], z['mean'], z['scale'], z['selection_scores'], tuple(z['training_ids']))
            assert set(g.training_ids) == set(f['training'])
            x = g.transform(view[f'{rep}_values'].T, view['gene_ids'], True)
            cached = npz(d / f'{rep}-panel.npz')
            assert np.allclose(x[:n], cached['train'], atol=1e-12, rtol=0) and np.allclose(x[n:], cached['validation'], atol=1e-12, rtol=0)
            panels[rep], features[rep] = x, g
            if study == 'GSE8917':
                pca = PCA(n_components=2, svd_solver='full').fit(x[:n])
                pcs = pca.transform(x)
                for i, (s, r) in enumerate(zip(ids, lab)):
                    outside = (x[i] < x[:n].min(0)) | (x[i] > x[:n].max(0))
                    # Training comparisons exclude all observations from that donor.
                    candidates = [j for j in range(n) if lab[j]['donor_group'] != r['donor_group']]
                    dist = np.sqrt(np.mean((x[candidates] - x[i])**2, axis=1))
                    nearest = candidates[int(np.argmin(dist))]
                    row = dict(representation=rep, profile_id=s, study=r['study'], donor_group=r['donor_group'], dose_gy=r['dose_gy'],
                        partition='training' if i < n else 'held_out', outside_training_range_fraction=float(outside.mean()),
                        nearest_other_donor_rms_z=float(dist.min()), nearest_profile_id=ids[nearest], nearest_dose_gy=lab[nearest]['dose_gy'])
                    distances.append(row)
                    pca_rows.append(dict(**row, pc1=float(pcs[i, 0]), pc2=float(pcs[i, 1]),
                        explained_pc1=float(pca.explained_variance_ratio_[0]), explained_pc2=float(pca.explained_variance_ratio_[1])))
        for model in ['A', 'B', 'C', 'D', 'elastic_net', 'pls', 'gradient_boosting']:
            params = ns[study, model] if model in 'ABCD' else cs[study, model]
            x = panels[params['representation']]; seed = split['seeds'][0]
            if model in 'ABCD':
                path = ROOT / 'data/models/abcd-v1/selected' / study / model / str(seed)
                use(path / 'model.pt'); estimator, ck = load_model(path / 'model.pt')
                pred = np.r_[predict(estimator, x[:n], ck['target_mean'], ck['target_scale']), predict(estimator, x[n:], ck['target_mean'], ck['target_scale'])]
            else:
                path = ROOT / 'data/models/classical-matched-v1/selected' / study / model / str(seed)
                r = js(path / 'result.json'); estimator = joblib.load(use(path / 'model.joblib'))
                pred = np.asarray(estimator.predict(x)).ravel() * r['target_scale'] + r['target_mean']
            expected = {r['profile_id']: float(r['predicted_gy']) for r in saved if r['model'] == model and r['study'] == study and int(r['seed_index']) == 0}
            delta = float(max(abs(pred[n+j] - expected[s]) for j, s in enumerate(f['validation'])))
            assert delta < 1e-5
            checks.append(dict(study=study, model=model, raw_replay_max_difference_gy=delta))
            for i, s in enumerate(ids):
                profiles.append(dict(held_out_study=study, model=model, representation=params['representation'], partition='training' if i < n else 'held_out',
                    profile_id=s, study=lab[i]['study'], donor_group=lab[i]['donor_group'], actual_gy=lab[i]['dose_gy'], predicted_gy=float(pred[i])))
            if study == 'GSE8917' and model == 'elastic_net':
                g = features[params['representation']]
                tr = [i for i in range(n) if lab[i]['dose_gy'] == 0]
                va = [i for i in range(n, len(ids)) if lab[i]['dose_gy'] == 0]
                training_studies = sorted({lab[i]['study'] for i in tr})
                means = [x[[i for i in tr if lab[i]['study'] == st]].mean(0) for st in training_studies]
                reference = np.mean(means, axis=0); held = x[va].mean(0)
                coeff = np.asarray(estimator.coef_).ravel(); gycoef = coeff * f['target_scale']
                c = linear_contributions(coeff, reference, held, f['target_scale'])
                reference_pred = float(estimator.predict(reference[None])[0] * f['target_scale'] + f['target_mean'])
                assert abs(c.sum() - (pred[va].mean()-reference_pred)) < 1e-12
                genes = np.asarray(g.gene_ids)[g.selected]
                values = view['log2_values'].T[:, g.selected]
                common_ids = list(core['sample_ids']); selected_probes = np.asarray(processor.core.probe_ids)[processor.core.keep]
                for j, gene in enumerate(genes):
                    mask = (core['gene_keys'] == gene) & np.isin(core['probe_ids'], selected_probes)
                    probes = core['probe_ids'][mask]
                    cols = [common_ids.index(ids[i]) for i in va]
                    contributions.append(dict(gene_id=str(gene), symbol=symbols.get(str(gene), ''), probes=';'.join(probes),
                        contributing_probes=len(probes), coef_gy_per_training_sd=float(gycoef[j]),
                        source_balanced_control_z=float(reference[j]), held_control_mean_z=float(held[j]), shift_training_sd=float(held[j]-reference[j]),
                        control_displacement_contribution_gy=float(c[j]),
                        held_control_missing_fraction=float(np.isnan(values[va, j]).mean()),
                        held_control_detected_probe_fraction=float(core['detected'][mask][:, cols].mean()),
                        training_control_missing_fraction=float(np.isnan(values[tr, j]).mean()),
                        training_scale_log2=float(g.scale[j])))
                order = np.argsort(-abs(c))
                concentration = dict(reference_control_prediction_gy=reference_pred,
                    held_control_prediction_mean_gy=float(pred[va].mean()), held_control_prediction_min_gy=float(pred[va].min()),
                    held_control_prediction_max_gy=float(pred[va].max()), total_displacement_gy=float(c.sum()),
                    sum_absolute_contributions_gy=float(abs(c).sum()), nonzero_coefficients=int(np.count_nonzero(coeff)),
                    top10_absolute_share=float(abs(c[order[:10]]).sum()/abs(c).sum()),
                    top20_absolute_share=float(abs(c[order[:20]]).sum()/abs(c).sum()),
                    missing_held_control_cells=int(np.isnan(values[va]).sum()), held_control_cells=int(values[va].size),
                    abs_shift_over_2_training_sd=int((abs(held-reference)>2).sum()),
                    abs_shift_over_1_training_sd=int((abs(held-reference)>1).sum()),
                    negative_contribution_gy=float(c[c<0].sum()), positive_contribution_gy=float(c[c>0].sum()))
                # The five leave-one-control-out means describe donor robustness,
                # not separately trained models or an uncertainty interval.
                concentration['leave_one_control_out_mean_range_gy'] = [float(min(np.delete(pred[va], j).mean() for j in range(len(va)))), float(max(np.delete(pred[va], j).mean() for j in range(len(va))))]
                response_rows = []
                for source in STUDIES:
                    donors = sorted({r['donor_group'] for r in lab if r['study'] == source})
                    doses = sorted({r['dose_gy'] for r in lab if r['study'] == source and r['dose_gy'] > 0})
                    for dose in doses:
                        contrasts = []
                        prediction_increments = []
                        for donor in donors:
                            lo = [i for i, r in enumerate(lab) if r['donor_group'] == donor and r['dose_gy'] == 0]
                            hi = [i for i, r in enumerate(lab) if r['donor_group'] == donor and r['dose_gy'] == dose]
                            assert len(lo) == len(hi) == 1
                            contrasts.append(x[hi[0]]-x[lo[0]])
                            prediction_increments.append(pred[hi[0]]-pred[lo[0]])
                        response = np.mean(contrasts, axis=0)
                        effect = response * gycoef
                        assert abs(effect.sum()-np.mean(prediction_increments)) < 1e-12
                        for j, gene in enumerate(genes):
                            response_rows.append(dict(study=source, partition='held_out' if source == study else 'training', dose_gy=dose,
                                gene_id=str(gene), symbol=symbols.get(str(gene), ''), donor_groups=len(donors),
                                donor_paired_mean_change_training_sd=float(response[j]),
                                predicted_increment_contribution_gy=float(effect[j])))
                tsv('elastic-net-dose-gene-contributions.tsv', response_rows)
        print('Replayed primary pipelines:', study, flush=True)
    tsv('frozen-predictions-training-and-held.tsv', profiles)
    tsv('feature-domain-distances.tsv', distances); tsv('training-pca-projection.tsv', pca_rows)
    tsv('elastic-net-control-gene-contributions.tsv', sorted(contributions, key=lambda r: -abs(r['control_displacement_contribution_gy'])))
    save('elastic-net-control-decomposition.json', concentration); save('replays.json', checks)
    for p, h in protected.items():
        assert sha(ROOT / p) == h, p
    for p in ['scripts/investigate_control_bias.py', 'scripts/assay_qc.py', 'scripts/prepare_core_background.py', 'scripts/core_transfer.py', 'scripts/masked_quantile.py', 'scripts/development_preprocessing.py', 'scripts/core_baseline.py', 'scripts/abcd_models.py']:
        use(ROOT / p)
    save('inputs-sha256.json', INPUTS)
    save('verification.json', dict(status='PASS_TARGETED_DIAGNOSTICS', started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
        source_audit=audit, frozen_primary_pipelines_replayed=len(checks), paired_dose_rows=len(detail), donor_slopes=len(slopes),
        raw_replay_max_difference_gy=max(r['raw_replay_max_difference_gy'] for r in checks), protected_files_unchanged=len(protected),
        new_predictor_fits=0, training_predictions_are_resubstitution=True, held_controls_used_only_for_diagnostics=True,
        no_samples_removed=True, locked_test_opened=False, cloud_changed=False, final_model_selected=False))
    print(json.dumps(concentration, indent=2), flush=True)


if __name__ == '__main__':
    main()
