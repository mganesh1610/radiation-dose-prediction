"""Small complete-data, fixed-platform quantile-reference prototype.

Features are rows, samples columns. Ties receive the mean of the target ranks
they occupy. Missing/nonpositive intensities require an explicit separate policy.
This is not a general microarray reprocessing package or a gene-mapping method.
"""
from dataclasses import dataclass
import numpy as np


def validate(matrix, features):
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(features) or x.shape[1] == 0:
        raise ValueError('Expected nonempty features-by-samples matrix')
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError('This prototype requires finite positive linear intensities')
    if len(features) != len(set(features)):
        raise ValueError('Source feature identifiers must be unique')
    return x


@dataclass(frozen=True)
class FrozenQuantile:
    target: np.ndarray
    feature_ids: tuple
    fit_sample_ids: tuple

    @classmethod
    def fit(cls, matrix, feature_ids, sample_ids, training_ids):
        if len(sample_ids) != len(set(sample_ids)) or len(training_ids) != len(set(training_ids)):
            raise ValueError('Duplicate sample identifiers')
        # Inspect training values only. Validation values cannot define a target,
        # feature filter, failure threshold, or other fitted parameter.
        locations = {s: i for i, s in enumerate(sample_ids)}
        ordered = tuple(sorted(training_ids))
        if not ordered or not set(ordered).issubset(locations):
            raise ValueError('Invalid training membership')
        source = np.asarray(matrix)
        if source.ndim != 2 or source.shape[1] != len(sample_ids):
            raise ValueError('Sample dimension mismatch')
        x = validate(source[:, [locations[s] for s in ordered]], feature_ids)
        target = np.sort(x, axis=0).mean(axis=1)
        target.setflags(write=False)
        return cls(target, tuple(feature_ids), ordered)

    def transform(self, matrix, feature_ids):
        if tuple(feature_ids) != self.feature_ids:
            raise ValueError('Feature identities/order differ from fitted contract')
        x = validate(matrix, feature_ids)
        result = np.empty_like(x)
        for j in range(x.shape[1]):
            order = np.argsort(x[:, j], kind='stable')
            sorted_values = x[order, j]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1]
            lengths = np.diff(np.r_[starts, len(order)])
            values = np.add.reduceat(self.target, starts) / lengths
            result[order, j] = np.repeat(values, lengths)
        return result

    def save(self, path):
        np.savez_compressed(path, target=self.target,
                            feature_ids=np.array(self.feature_ids),
                            fit_sample_ids=np.array(self.fit_sample_ids))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            target = z['target'].copy()
            ids = tuple(z['feature_ids'].tolist())
            fitted = tuple(z['fit_sample_ids'].tolist())
        if target.shape != (len(ids),) or not np.isfinite(target).all() or np.any(target <= 0) or np.any(np.diff(target) < 0):
            raise ValueError('Invalid frozen target')
        if len(ids) != len(set(ids)) or not fitted or len(fitted) != len(set(fitted)):
            raise ValueError('Invalid artifact membership')
        target.setflags(write=False)
        return cls(target, ids, fitted)
