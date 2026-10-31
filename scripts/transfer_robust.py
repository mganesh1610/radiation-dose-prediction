"""Training-only stable feature selection and study-aware development scoring."""
import numpy as np
from core_baseline import GenePreparation, donor_metrics


def fit_stable(x, genes, y, studies, donors, ids, width=500):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    studies = np.asarray(studies); donors = np.asarray(donors)
    if (x.shape != (len(ids), len(genes)) or y.shape != (len(ids),)
            or studies.shape != y.shape or donors.shape != y.shape
            or len(set(ids)) != len(ids) or len(set(genes)) != len(genes)
            or not np.isfinite(y).all() or np.isinf(x).any() or width < 4):
        raise ValueError('Invalid stable-selection axes')
    if any(len(set(studies[donors == d])) != 1 for d in set(donors)):
        raise ValueError('Donor spans sources')
    present = np.isfinite(x).any(0); medians = np.zeros(len(genes))
    medians[present] = np.nanmedian(x[:, present], axis=0)
    filled = np.where(np.isfinite(x), x, medians)
    scale = filled.std(0); valid = present & (scale > 1e-6)
    correlations, controls = [], []
    for study in sorted(set(studies)):
        take = studies == study; xx = filled[take].copy(); yy = y[take].copy(); dd = donors[take]
        if not np.any(yy == 0):
            raise ValueError('Training source lacks controls')
        controls.append(xx[yy == 0].mean(0))
        for donor in set(dd):
            part = dd == donor
            if np.sum(y[take][part] == 0) != 1 or len(set(y[take][part])) < 2:
                raise ValueError('Donor needs one control and multiple doses')
            xx[part] -= xx[part].mean(0); yy[part] -= yy[part].mean()
        denom = np.sqrt((xx**2).sum(0) * (yy**2).sum())
        correlations.append(np.divide(xx.T @ yy, denom, out=np.zeros(len(genes)), where=denom > 1e-12))
    corr = np.stack(correlations)
    agreement = np.all(corr > 0, axis=0) | np.all(corr < 0, axis=0)
    minimum = abs(corr).min(0)
    shift = np.divide(np.ptp(np.stack(controls), axis=0), scale, out=np.full(len(genes), np.inf), where=valid)
    eligible = valid & agreement & (minimum >= .10) & (shift <= 1.)
    score = np.divide(minimum, 1+shift, out=np.zeros_like(minimum), where=np.isfinite(shift))
    order = sorted(np.flatnonzero(eligible), key=lambda i: (-score[i], str(genes[i])))[:width]
    if len(order) < 4:
        raise ValueError(f'Only {len(order)} stable genes; frozen minimum is four')
    selected = np.array(order, dtype=int)
    gp = GenePreparation(tuple(genes), medians, selected, filled[:, selected].mean(0),
                         scale[selected], score, tuple(ids))
    detail = dict(training_studies=sorted(set(studies)), eligible_genes=int(eligible.sum()),
        selected_genes=len(selected), zero_padding=width-len(selected),
        single_source_fallback=len(correlations)==1, consistent_direction_genes=int((valid & agreement).sum()),
        correlation_qualified_genes=int((valid & agreement & (minimum >= .10)).sum()))
    arrays = dict(correlations=corr, control_shift_sd=shift, eligible=eligible, scores=score)
    return gp, detail, arrays


def load_features(path):
    with np.load(path, allow_pickle=False) as z:
        return GenePreparation(tuple(z['gene_ids']), z['medians'], z['selected'], z['mean'],
                               z['scale'], z['selection_scores'], tuple(z['training_ids']))


def panel_transform(prep, values, genes, width=500):
    z = prep.transform(values, genes, True)
    if not np.isfinite(z).all() or z.shape[1] > width:
        raise ValueError('Invalid feature panel')
    return np.pad(z, ((0, 0), (0, width-z.shape[1])))


def balanced_metrics(records):
    metric, _ = donor_metrics([r['actual_gy'] for r in records], [r['predicted_gy'] for r in records],
                             [r['donor_group'] for r in records])
    return metric


def design_score(records):
    if len({r['profile_id'] for r in records}) != len(records):
        raise ValueError('Each profile must appear once per validation design')
    by_study = []
    for study in sorted({r['study'] for r in records}):
        rr = [r for r in records if r['study'] == study]
        score = {}
        for name, subset in [('all', rr), ('control', [r for r in rr if r['actual_gy'] == 0]),
                             ('high', [r for r in rr if r['actual_gy'] > 2])]:
            if not subset:
                raise ValueError('Required validation stratum absent')
            score[name] = balanced_metrics(subset)['mae']
        score['study'] = study
        score['loss'] = .5*score['all'] + .25*score['control'] + .25*score['high']
        by_study.append(score)
    return float(np.mean([r['loss'] for r in by_study])), by_study


def candidate_score(donor_records, source_records):
    if {r['profile_id'] for r in donor_records} != {r['profile_id'] for r in source_records}:
        raise ValueError('Validation designs must cover the same profiles')
    d, dd = design_score(donor_records); s, ss = design_score(source_records)
    return dict(score=.5*(d+s), donor_score=d, source_score=s, donor_studies=dd, source_studies=ss)
