"""Fold-local feature preparation, estimators and donor-balanced metrics."""
from dataclasses import dataclass
import hashlib
import itertools
import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

SEED = 20260930
MODELS = ('median_null', 'elastic_net', 'pls', 'gradient_boosting')


def inner_splits(studies, groups, n_splits=5):
    studies, groups = np.asarray(studies), np.asarray(groups)
    if studies.shape != groups.shape or studies.ndim != 1:
        raise ValueError('Invalid split axes')
    if any(len(set(studies[groups == g])) != 1 for g in set(groups)):
        raise ValueError('A group spans studies')
    if any(len(set(groups[studies == st])) < n_splits for st in set(studies)):
        raise ValueError('Insufficient donor groups for stratified folds')
    for fold in range(n_splits):
        val_groups = []
        for st in sorted(set(studies)):
            val_groups.extend(sorted(set(groups[studies == st]))[fold::n_splits])
        val = np.isin(groups, val_groups)
        yield np.flatnonzero(~val), np.flatnonzero(val)


def donor_metrics(actual, predicted, groups):
    actual, predicted, groups = np.asarray(actual), np.asarray(predicted), np.asarray(groups)
    if (actual.ndim != 1 or actual.shape != predicted.shape or actual.shape != groups.shape
            or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError('Invalid metric input')
    error = predicted - actual
    rows = []
    for g in sorted(set(groups)):
        e = error[groups == g]
        rows.append(dict(group=str(g), samples=len(e), mae=float(np.mean(abs(e))),
                         mse=float(np.mean(e**2)), bias=float(np.mean(e))))
    return dict(mae=float(np.mean([r['mae'] for r in rows])),
                rmse=float(np.sqrt(np.mean([r['mse'] for r in rows]))),
                bias=float(np.mean([r['bias'] for r in rows])), donors=len(rows), samples=len(actual)), rows


def bootstrap_intervals(actual, predicted, groups, studies, repeats=2000):
    """Conditional on the observed studies, without refitting estimators."""
    studies, groups = np.asarray(studies), np.asarray(groups)
    _, rows = donor_metrics(actual, predicted, groups)
    if studies.shape != groups.shape or any(len(set(studies[groups == r['group']])) != 1 for r in rows):
        raise ValueError('Invalid study/group axes')
    by_study = []
    for st in sorted(set(studies)):
        by_study.append(np.array([[r['mae'], r['mse'], r['bias']] for r in rows
                                 if r['group'] in set(groups[studies == st])]))
    rng = np.random.default_rng(SEED)
    samples = np.concatenate([a[rng.integers(len(a), size=(repeats, len(a)))] for a in by_study], axis=1)
    stats = samples.mean(axis=1)
    stats[:, 1] = np.sqrt(stats[:, 1])
    return {name: np.quantile(stats[:, j], [.025, .975]).tolist()
            for j, name in enumerate(('mae', 'rmse', 'bias'))}


@dataclass
class GenePreparation:
    gene_ids: tuple
    medians: np.ndarray
    selected: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    selection_scores: np.ndarray
    training_ids: tuple

    @classmethod
    def fit(cls, x, gene_ids, y, studies, sample_ids, k):
        x, y, studies = np.asarray(x), np.asarray(y), np.asarray(studies)
        if (x.ndim != 2 or x.shape != (len(sample_ids), len(gene_ids)) or y.shape != (len(sample_ids),)
                or studies.shape != y.shape or len(set(sample_ids)) != len(sample_ids)
                or len(set(gene_ids)) != len(gene_ids) or np.isinf(x).any()
                or not np.isfinite(y).all() or k < 1):
            raise ValueError('Invalid gene preparation inputs')
        present = np.isfinite(x).any(axis=0)
        medians = np.zeros(x.shape[1])
        medians[present] = np.nanmedian(x[:, present], axis=0)
        filled = np.where(np.isfinite(x), x, medians)
        eligible = present & (filled.var(axis=0) > 1e-12)
        centered, yc = filled.copy(), y.copy()
        for st in set(studies):
            take = studies == st
            centered[take] -= filled[take].mean(axis=0)
            yc[take] -= y[take].mean()
        denom = np.sqrt(np.sum(centered**2, axis=0) * np.sum(yc**2))
        scores = np.divide(abs(centered.T @ yc), denom, out=np.zeros_like(denom), where=denom > 1e-12)
        eligible &= denom > 1e-12
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            raise ValueError('No eligible variable genes')
        # Explicit GeneID tie-break independent of supplied feature order.
        order = sorted(candidates, key=lambda i: (-scores[i], str(gene_ids[i])))
        selected = np.array(order[:k], dtype=int)
        panel = filled[:, selected]
        return cls(tuple(gene_ids), medians, selected, panel.mean(axis=0), panel.std(axis=0), scores, tuple(sample_ids))

    def transform(self, x, gene_ids, standardize):
        x = np.asarray(x)
        if tuple(gene_ids) != self.gene_ids or x.ndim != 2 or x.shape[1] != len(gene_ids) or np.isinf(x).any():
            raise ValueError('Gene order/dimension mismatch or infinite values')
        panel = x[:, self.selected]
        panel = np.where(np.isfinite(panel), panel, self.medians[self.selected])
        return (panel - self.mean) / self.scale if standardize else panel

    def save(self, path):
        np.savez_compressed(path, gene_ids=self.gene_ids, medians=self.medians, selected=self.selected,
                            mean=self.mean, scale=self.scale, selection_scores=self.selection_scores,
                            training_ids=self.training_ids)

    def fingerprint(self):
        h = hashlib.sha256()
        for a in (np.asarray(self.gene_ids), self.medians, self.selected, self.mean, self.scale,
                  self.selection_scores, np.asarray(self.training_ids)):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def model_grid(name):
    if name == 'elastic_net':
        return [dict(k=k, alpha=a, l1_ratio=l) for k, a, l in itertools.product((100, 500), (.01, .1, 1.), (.1, .5, .9))]
    if name == 'pls':
        return [dict(k=k, n_components=n) for k, n in itertools.product((100, 500), (2, 5, 10))]
    if name == 'gradient_boosting':
        return [dict(k=k, max_depth=d, n_estimators=n) for k, d, n in itertools.product((100, 500), (1, 2), (100, 300))]
    raise ValueError('Unknown tuned model')


def fit_estimator(name, params, x, y):
    p = {k: v for k, v in params.items() if k != 'k'}
    if name == 'elastic_net':
        model = ElasticNet(**p, max_iter=100000, tol=1e-5, selection='cyclic')
    elif name == 'pls':
        if p['n_components'] >= len(x) or p['n_components'] > x.shape[1]:
            raise ValueError('PLS components exceed training dimensions')
        model = PLSRegression(**p, scale=False, max_iter=1000, tol=1e-6)
    elif name == 'gradient_boosting':
        model = GradientBoostingRegressor(**p, learning_rate=.05, min_samples_leaf=3,
                                          subsample=1., random_state=SEED, loss='squared_error')
    else:
        raise ValueError('Unknown estimator')
    # No silently accepted convergence failure. The runner records rejected trials.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x, y)
    if caught:
        raise ValueError('Fit warning: ' + '; '.join(str(w.message) for w in caught))
    if not np.isfinite(np.asarray(model.predict(x))).all():
        raise ValueError('Nonfinite fitted predictions')
    return model


@dataclass
class CorePredictor:
    transfer: object
    preparation: GenePreparation
    estimator: object
    name: str

    def predict(self, values, detected_probe_order):
        genes, x, _ = self.transfer.transform(values, np.isfinite(values), detected_probe_order)
        panel = self.preparation.transform(x.T, genes, self.name != 'gradient_boosting')
        return np.asarray(self.estimator.predict(panel)).reshape(-1)
