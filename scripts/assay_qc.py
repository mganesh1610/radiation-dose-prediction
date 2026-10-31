"""Auditable, sample-independent Agilent aggregation and training-only filters.

Policy thresholds are development candidates, not validated assay cutoffs.
"""
import warnings

import numpy as np


def platform_probe_id(platform_id, row):
    """Use explicit GPL semantics; NAME is a transcript accession on GPL1708."""
    columns = {'GPL1708': 'SPOT_ID', 'GPL10332': 'NAME', 'GPL13497': 'ID'}
    if platform_id not in columns or not row.get(columns[platform_id]):
        raise ValueError('Unknown platform or missing probe identifier')
    return row[columns[platform_id]]


def technical_valid(flags, columns):
    columns = list(columns)
    required = ['gIsSaturated', 'IsManualFlag', 'gIsFeatNonUnifOL', 'gIsFeatPopnOL']
    if len(set(columns)) != len(columns) or not set(required).issubset(columns):
        raise ValueError('Missing or duplicate quality flags')
    flags = np.asarray(flags)
    if flags.ndim != 3 or flags.shape[2] != len(columns) or not np.isin(flags, [0, 1]).all():
        raise ValueError('Expected binary feature-by-sample-by-flag array')
    # IsFound includes signal-strength/spot-finding criteria, not just damage.
    valid = np.ones(flags.shape[:2], dtype=bool)
    for key in required:
        valid &= flags[:, :, columns.index(key)] == 0
    return valid


def aggregate_median(values, keys, valid=None):
    """Median within each key separately in each sample; missing stays missing."""
    values = np.asarray(values, dtype=float)
    keys = np.asarray(keys, dtype=str)
    if values.ndim != 2 or keys.shape != (values.shape[0],):
        raise ValueError('Key/value shape mismatch')
    if np.isinf(values).any() or np.any(keys == ''):
        raise ValueError('Infinite value or empty key')
    mask = np.isfinite(values)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != values.shape or valid.dtype != bool:
            raise ValueError('Expected boolean validity mask')
        mask &= valid
    names, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    result = np.full((len(names), values.shape[1]), np.nan)
    counts = np.zeros(result.shape, dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i, group in enumerate(groups):
            result[i] = np.nanmedian(np.where(mask[group], values[group], np.nan), axis=0)
            counts[i] = mask[group].sum(axis=0)
    return names, result, counts


def fit_probe_filter(values, detected, training_mask, min_valid=0.95, min_detected=0.20):
    """Return fixed probe eligibility; validation values never enter the fit."""
    values, detected = np.asarray(values), np.asarray(detected)
    training_mask = np.asarray(training_mask)
    if (values.ndim != 2 or detected.shape != values.shape or detected.dtype != bool
            or training_mask.dtype != bool or training_mask.shape != (values.shape[1],)
            or not training_mask.any()):
        raise ValueError('Invalid filter input')
    if not 0 <= min_valid <= 1 or not 0 <= min_detected <= 1:
        raise ValueError('Invalid filter thresholds')
    train = values[:, training_mask]
    if np.isinf(train).any():
        raise ValueError('Infinite training value')
    valid_fraction = np.isfinite(train).mean(axis=1)
    detected_fraction = (detected[:, training_mask] & np.isfinite(train)).mean(axis=1)
    return (valid_fraction >= min_valid) & (detected_fraction >= min_detected), valid_fraction, detected_fraction


def resolve_gene(source_ids, current_genes):
    """Accept one consistent numeric source GeneID; never guess from a symbol."""
    ids = {str(x).strip() for x in source_ids}
    if ids == {''}:
        return '', 'unmapped'
    if len(ids) != 1 or not next(iter(ids)).isdigit():
        return '', 'ambiguous_or_inconsistent'
    gene = next(iter(ids))
    if gene not in current_genes:
        return '', 'absent_from_current_human_snapshot'
    return gene, 'current_exact_gene_id'
