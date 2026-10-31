"""Frozen quantile reference fitted only on technically usable training spots.

Interpolate each sorted usable column over an endpoint-inclusive [0,1] grid,
average training quantile curves, and map usable ranks to the frozen curve.
Ties get the mean of their occupied target ranks. This explicit convention
reduces to the prior complete-data implementation when no spots are masked.
Missingness may be intensity-dependent; engineering tests do not validate the
biological assumptions of quantile normalization or cross-platform transfer.
"""
from dataclasses import dataclass

import numpy as np


def validate(matrix, usable, feature_ids):
    x, mask = np.asarray(matrix, dtype=float), np.asarray(usable)
    if (x.ndim != 2 or x.shape[0] != len(feature_ids) or x.shape[0] < 2
            or x.shape[1] < 1 or mask.shape != x.shape or mask.dtype != bool
            or len(set(feature_ids)) != len(feature_ids)):
        raise ValueError('Invalid matrix, mask, or feature identity contract')
    if not np.isfinite(x[mask]).all() or np.any(x[mask] <= 0):
        raise ValueError('Usable linear intensities must be finite and positive')
    if np.any(mask.sum(axis=0) < 2):
        raise ValueError('At least two usable features required per sample')
    return x, mask


@dataclass(frozen=True)
class MaskedFrozenQuantile:
    target: np.ndarray
    feature_ids: tuple
    fit_sample_ids: tuple

    @classmethod
    def fit(cls, matrix, usable, feature_ids, sample_ids, training_ids):
        x, mask = np.asarray(matrix), np.asarray(usable)
        if (x.ndim != 2 or x.shape[1] != len(sample_ids) or mask.shape != x.shape
                or mask.dtype != bool or len(set(sample_ids)) != len(sample_ids)
                or not training_ids or len(set(training_ids)) != len(training_ids)
                or not set(training_ids).issubset(sample_ids)):
            raise ValueError('Invalid training/sample membership or dimensions')
        ordered = tuple(sorted(training_ids))
        index = {s:i for i,s in enumerate(sample_ids)}
        take = [index[s] for s in ordered]
        # Validate training values only, never consult validation data or masks.
        x, mask = validate(x[:, take], mask[:, take], feature_ids)
        grid = np.linspace(0., 1., len(feature_ids))
        curves = []
        for j in range(x.shape[1]):
            values = np.sort(x[mask[:, j], j])
            curves.append(np.interp(grid, np.linspace(0., 1., len(values)), values))
        target = np.column_stack(curves).mean(axis=1)
        target.setflags(write=False)
        return cls(target, tuple(feature_ids), ordered)

    def transform(self, matrix, usable, feature_ids):
        if tuple(feature_ids) != self.feature_ids:
            raise ValueError('Feature identity/order mismatch')
        x, mask = validate(matrix, usable, feature_ids)
        result = np.full_like(x, np.nan)
        grid = np.linspace(0., 1., len(self.target))
        for j in range(x.shape[1]):
            indices = np.flatnonzero(mask[:, j])
            order = np.argsort(x[indices, j], kind='stable')
            indices = indices[order]
            values = x[indices, j]
            target_ranks = np.interp(np.linspace(0., 1., len(indices)), grid, self.target)
            starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
            lengths = np.diff(np.r_[starts, len(indices)])
            averages = np.add.reduceat(target_ranks, starts) / lengths
            result[indices, j] = np.repeat(averages, lengths)
        return result

    def save(self, path):
        np.savez_compressed(path, target=self.target, feature_ids=np.array(self.feature_ids),
            fit_sample_ids=np.array(self.fit_sample_ids), method=np.array('masked-endpoint-quantile-v1'))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if str(z['method']) != 'masked-endpoint-quantile-v1':
                raise ValueError('Unsupported method')
            target = z['target'].copy(); ids = tuple(z['feature_ids']); samples = tuple(z['fit_sample_ids'])
        if (len(ids) < 2 or len(set(ids)) != len(ids) or target.shape != (len(ids),)
                or not np.isfinite(target).all() or np.any(target <= 0) or np.any(np.diff(target) < 0)
                or not samples or len(set(samples)) != len(samples)):
            raise ValueError('Invalid frozen artifact')
        target.setflags(write=False)
        return cls(target, ids, samples)
