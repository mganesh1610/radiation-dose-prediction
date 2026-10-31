"""Frozen-contract helpers for GPL13497; no training or prediction entry point."""
import gzip
import csv
import hashlib
import numpy as np
from assay_qc import aggregate_median

REQUIRED = ['FeatureNum','ProbeName','ControlType','gMedianSignal','gBGMedianSignal',
            'gProcessedSignal','gBGSubSignal','gIsSaturated','IsManualFlag',
            'gIsFeatNonUnifOL','gIsFeatPopnOL','gIsWellAboveBG','gIsBGNonUnifOL',
            'gIsBGPopnOL','gIsFound','gIsPosAndSignif']


def inspect_header(path):
    # Deliberately stop before the FEATURES data rows; no expression values or QC statistics returned.
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('FEATURES\t'):
                columns = line.rstrip('\r\n').split('\t')
                if len(columns) != len(set(columns)) or not set(REQUIRED).issubset(columns):
                    raise ValueError('Missing/duplicate Agilent fields')
                return dict(columns=columns, required_present=True, sequence_column='Sequence' in columns,
                            expression_rows_parsed=0)
    raise ValueError('No Agilent feature header')


def inspect_design(path, platform, required_probe_order):
    """Read identifier/sequence columns only; ignore signal and quality-flag values."""
    seen = set(); probes = set(); count = 0; design_hash = hashlib.sha256()
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('FEATURES\t'):
                header = line.rstrip('\r\n').split('\t'); break
        else: raise ValueError('No feature header')
        for name in ['FeatureNum', 'ProbeName', 'ControlType', 'Sequence']:
            if name not in header: raise ValueError('Missing design column: ' + name)
        ix = {name: header.index(name) for name in ['FeatureNum', 'ProbeName', 'ControlType', 'Sequence']}
        for row in csv.reader(f, delimiter='\t'):
            if not row or row[0] != 'DATA': continue
            if len(row) != len(header): raise ValueError('Ragged feature record')
            if row[ix['ControlType']] != '0': continue
            fid, probe, sequence = row[ix['FeatureNum']], row[ix['ProbeName']], row[ix['Sequence']].upper()
            if fid in seen: raise ValueError('Duplicate physical feature')
            if probe not in platform or platform[probe]['SEQUENCE'].upper() != sequence:
                raise ValueError('Probe identity/sequence conflicts with GPL13497')
            seen.add(fid); probes.add(probe); count += 1
            design_hash.update(f'{fid}\t{probe}\t{sequence}\n'.encode())
    if not count or not set(required_probe_order).issubset(probes):
        raise ValueError('Incomplete frozen probe universe')
    return dict(noncontrol_physical_features=count, distinct_noncontrol_probes=len(probes),
                design_sha256=design_hash.hexdigest(), required_probes_present=len(required_probe_order),
                intensity_values_parsed=0, quality_flag_values_parsed=0, sequence_identity_passed=True)


def assemble_corrected_probes(corrected, probe_names, valid, detected, required_probe_order):
    """Same per-array spot collapse/order as frozen core preprocessing. No fit."""
    y, mask, detection = np.asarray(corrected), np.asarray(valid), np.asarray(detected)
    if y.ndim != 2 or mask.shape != y.shape or detection.shape != y.shape or mask.dtype != bool or detection.dtype != bool:
        raise ValueError('Invalid physical spot axes/masks')
    if not np.array_equal(np.isnan(y), ~mask) or not np.isfinite(y[mask]).all() or np.any(y[mask] <= 0):
        raise ValueError('Invalid corrected values/technical mask')
    probes, signal, _ = aggregate_median(y, probe_names)
    names, d, _ = aggregate_median(detection.astype(float), probe_names, mask)
    if not np.array_equal(probes, names) or len(set(required_probe_order)) != len(required_probe_order):
        raise ValueError('Probe identity/order mismatch')
    index = {str(p): i for i,p in enumerate(probes)}
    if not set(required_probe_order).issubset(index):
        raise ValueError('Array lacks frozen input probe identities')
    take = [index[str(p)] for p in required_probe_order]
    return dict(values=signal[take], detected=(np.isfinite(d) & (d >= .5))[take],
                probe_ids=np.array(required_probe_order))
