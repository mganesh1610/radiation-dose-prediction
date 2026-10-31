"""Local GSE102971 grouping/QC/mapping audit; no cloud or locked-test access.

Consumes the verified engineering pilot and a pinned NCBI human gene_info file.
Produces a candidate derivative, never overwrites the v1 source/pilot artifacts.
"""
import csv
import gzip
import hashlib
import html
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from assay_qc import aggregate_median, fit_probe_filter, resolve_gene, technical_valid

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'context/raw-pilot/GSE102971'
LOG = PILOT / 'audit-v1'
OUT = ROOT / 'data/processed_pilot/GSE102971/audit-v1'
FLAGS = ['gIsSaturated', 'IsManualFlag', 'gIsFeatNonUnifOL', 'gIsFeatPopnOL',
         'gIsFound', 'gIsBGNonUnifOL', 'gIsBGPopnOL', 'gIsPosAndSignif', 'gIsWellAboveBG']


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_tsv(p, rows):
    if not rows:
        p.write_text(''); return
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader(); w.writerows(rows)


def platform_rows():
    p = ROOT / 'data/public_radiation/core/GSE102971/GSE102971_family.soft.gz'
    metadata, lines = [], []
    with gzip.open(p, 'rt') as f:
        for line in f:
            if line.startswith(('!Series_title', '!Series_overall_design', '!Platform_title', '!Platform_last_update_date')):
                metadata.append(line.strip())
            if line.startswith('!platform_table_begin'): break
        for line in f:
            if line.startswith('!platform_table_end'): break
            lines.append(line)
    (LOG / 'sources/platform-metadata.txt').write_text('\n'.join(metadata) + '\n')
    return list(csv.DictReader(lines, delimiter='\t')), p


def raw_flags(samples, feature_ids, probe_names, platform):
    matrices, sequence_conflicts = [], Counter()
    for i, r in enumerate(samples):
        p = (ROOT / r['path']).resolve()
        if not p.is_relative_to((ROOT / 'data/raw_radiation/GSE102971').resolve()) or sha(p) != r['sha256']:
            raise ValueError('Source path or checksum mismatch')
        rows, ids, probes = [], [], []
        with gzip.open(p, 'rt') as f:
            for line in f:
                if line.startswith('FEATURES\t'):
                    header = line.rstrip('\r\n').split('\t'); break
            if len(header) != len(set(header)):
                raise ValueError('Duplicate raw column')
            idx = {k: header.index(k) for k in FLAGS + ['FeatureNum', 'ProbeName', 'ControlType', 'Sequence']}
            for row in csv.reader(f, delimiter='\t'):
                if not row or row[0] != 'DATA': continue
                if len(row) != len(header): raise ValueError('Malformed raw row')
                if row[idx['ControlType']] != '0': continue
                fid, probe = row[idx['FeatureNum']], row[idx['ProbeName']]
                ids.append(fid); probes.append(probe)
                rows.append([int(row[idx[k]]) for k in FLAGS])
                seq = row[idx['Sequence']].strip().upper()
                if not seq or seq != (platform[fid].get('SEQUENCE') or '').strip().upper():
                    sequence_conflicts[probe] += 1
        if ids != feature_ids.tolist() or probes != probe_names.tolist():
            raise ValueError('Feature order or identity mismatch')
        a = np.asarray(rows, dtype=np.uint8)
        if not np.isin(a, [0, 1]).all(): raise ValueError('Nonbinary flag')
        matrices.append(a)
        if (i + 1) % 20 == 0: print(f'Checked raw checksums, feature identities, sequences and flags: {i+1}/100', flush=True)
    return np.stack(matrices, axis=1), sequence_conflicts


def main():
    started = time.perf_counter()
    LOG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((PILOT / 'summary.json').read_text())
    for a in prior['artifact_files']:
        if sha(ROOT / a['path']) != a['sha256']: raise ValueError('Pilot artifact changed')
    sources = json.loads((PILOT / 'downloads.json').read_text())
    samples = sources['files']
    if len(samples) != 100 or any(r['organism'] != 'Homo sapiens' for r in samples):
        raise ValueError('Incorrect study subset')
    canonical = np.load(ROOT / 'data/processed_pilot/GSE102971/v1/canonical_source.npz', allow_pickle=False)
    normalized = np.load(ROOT / 'data/processed_pilot/GSE102971/v1/engineering_log2_qn.npz', allow_pickle=False)
    sample_ids, feature_ids, probe_names = (canonical[k] for k in ['sample_ids', 'feature_ids', 'probe_names'])
    if sample_ids.tolist() != [r['gsm'] for r in samples]: raise ValueError('Sample order mismatch')
    for k in ['sample_ids', 'feature_ids']:
        if not np.array_equal(canonical[k], normalized[k]): raise ValueError('Normalized axis mismatch')
    train, logged = normalized['training_mask'], normalized['values']
    metadata = {r['gsm']: r for r in csv.DictReader((ROOT / 'context/dataset-acquisition/sample-metadata.tsv').open(), delimiter='\t')
                if r['study'] == 'GSE102971' and r['organism_ch1'] == 'Homo sapiens'}
    group_samples = defaultdict(list)
    group_rows = []
    for r in samples:
        m = metadata[r['gsm']]
        traits = dict(x.split(': ', 1) for x in m['characteristics_ch1'].split(' | '))
        match = re.fullmatch(r'ExVivo_Blood_(\d+)Gy_([A-Z])', m['title'])
        if (not match or float(match[1]) != r['dose_gy'] or match[2] != r['source_donor_group']
                or float(traits['dose'].removesuffix('Gy')) != r['dose_gy']
                or traits['gender'] != r['gender'] or m['platform_id'] != 'GPL10332'
                or not re.search(rf"_{match[2]}_{match[1]}Gy\.txt\.gz$", r['path'])):
            raise ValueError('Label inconsistency')
        group_samples[r['source_donor_group']].append(r)
        group_rows.append(dict(gsm=r['gsm'], study='GSE102971', split_group_id='GSE102971:' + r['source_donor_group'],
                               source_group=r['source_donor_group'], source_gender=r['gender'], dose_gy=r['dose_gy'],
                               time_after_exposure_hours=24, platform_id='GPL10332',
                               evidence_status='metadata_supported_inferred_donor_group',
                               independent_donor_key_available=False,
                               source_url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + r['gsm']))
    if len(group_samples) != 20: raise ValueError('Unexpected group count')
    for group, rows in group_samples.items():
        if len(rows) != 5 or {r['dose_gy'] for r in rows} != {0., 2., 5., 6., 7.} or len({r['gender'] for r in rows}) != 1:
            raise ValueError('Incomplete/inconsistent donor group')
        roles = {bool(train[list(sample_ids).index(r['gsm'])]) for r in rows}
        if len(roles) != 1: raise ValueError('Group split across training and validation')
    write_tsv(LOG / 'sample-groups.tsv', group_rows)
    (LOG / 'sources/human-treatment-protocol.txt').write_text(metadata[sample_ids[0]]['treatment_protocol_ch1'] + '\n')
    platform_data, platform_path = platform_rows()
    byid = {r['ID']: r for r in platform_data}
    if len(byid) != len(platform_data) or any(byid.get(f, {}).get('NAME') != p for f, p in zip(feature_ids, probe_names)):
        raise ValueError('GPL feature/probe mismatch')
    gene_path = LOG / 'sources/Homo_sapiens.gene_info.gz'
    gene_provenance = json.loads((LOG / 'sources/downloads.json').read_text())
    if sha(gene_path) != gene_provenance['sha256']: raise ValueError('Gene annotation checksum mismatch')
    with gzip.open(gene_path, 'rt') as f:
        genes = {r['GeneID']: r for r in csv.DictReader(f, delimiter='\t') if r['#tax_id'] == '9606'}
    flags, sequence_conflicts = raw_flags(samples, feature_ids, probe_names, byid)
    # Ensure old flag fields agree with the independently reparsed originals.
    for j, key in enumerate(canonical['flag_columns']):
        if not np.array_equal(flags[:, :, FLAGS.index(key)], canonical['flags'][:, :, j]):
            raise ValueError('Flag extraction disagrees with pilot')
    valid = technical_valid(flags, FLAGS)
    bgwarn = (flags[:, :, FLAGS.index('gIsBGNonUnifOL')] | flags[:, :, FLAGS.index('gIsBGPopnOL')]).astype(bool)
    detected = flags[:, :, FLAGS.index('gIsWellAboveBG')].astype(bool)
    np.savez_compressed(OUT / 'quality_flags.npz', flags=flags, flag_columns=np.array(FLAGS),
                        technical_valid=valid, background_warning=bgwarn, sample_ids=sample_ids, feature_ids=feature_ids)
    probe_ids, probe_values, valid_spots = aggregate_median(logged, probe_names, valid)
    # At least half of the technically valid repeated spots must be detected.
    _, detection_fraction, _ = aggregate_median(detected.astype(float), probe_names, valid)
    probe_detected = np.isfinite(detection_fraction) & (detection_fraction >= 0.5)
    keep_qc, valid_frac, det_frac = fit_probe_filter(probe_values, probe_detected, train)
    byprobe = defaultdict(list)
    for fid, probe in zip(feature_ids, probe_names): byprobe[probe].append(byid[fid])
    mapping = []
    for i, probe in enumerate(probe_ids):
        entries = byprobe[probe]
        gid, status = resolve_gene([r['GENE'] or '' for r in entries], genes)
        if probe in sequence_conflicts:
            gid, status = '', 'raw_platform_sequence_conflict'
        source_symbols = sorted({r['GENE_SYMBOL'] or '' for r in entries})
        source_ids = sorted({r['GENE'] or '' for r in entries})
        mapping.append(dict(probe_id=probe, physical_spots=len(entries), source_gene_ids='|'.join(source_ids),
                            source_symbols='|'.join(source_symbols), current_gene_id=gid,
                            current_symbol=genes[gid]['Symbol'] if gid else '', mapping_status=status,
                            source_feature_ids='|'.join(r['ID'] for r in entries),
                            raw_platform_sequence_conflict_cells=sequence_conflicts[probe],
                            training_valid_fraction=float(valid_frac[i]), training_detected_fraction=float(det_frac[i]),
                            qc_eligible=bool(keep_qc[i]), gene_eligible=bool(keep_qc[i] and gid)))
    selected = np.array([r['gene_eligible'] for r in mapping])
    gene_ids, gene_values, contributing_probes = aggregate_median(probe_values[selected],
        [r['current_gene_id'] for r in mapping if r['gene_eligible']])
    gene_keep = np.isfinite(gene_values[:, train]).mean(axis=1) >= .95
    gene_ids, gene_values, contributing_probes = gene_ids[gene_keep], gene_values[gene_keep], contributing_probes[gene_keep]
    np.savez_compressed(OUT / 'gene_candidate.npz', values=gene_values, gene_ids=gene_ids,
        gene_symbols=np.array([genes[g]['Symbol'] for g in gene_ids]), sample_ids=sample_ids,
        training_mask=train, contributing_probe_counts=contributing_probes)
    write_tsv(LOG / 'probe-map.tsv', mapping)
    gene_probe_counts = Counter(r['current_gene_id'] for r in mapping if r['gene_eligible'])
    write_tsv(LOG / 'gene-catalog.tsv', [dict(gene_id=g, current_symbol=genes[g]['Symbol'],
        gene_type=genes[g]['type_of_gene'], eligible_probe_count=gene_probe_counts[g],
        missing_cells=int(np.isnan(gene_values[i]).sum())) for i, g in enumerate(gene_ids)])
    # Sample flags are review prompts, never deletions. MAD/IQR are descriptive only.
    medians = np.median(logged, axis=0)
    raw_medians = np.median(np.log2(canonical['values'][:, :, 0]), axis=0)
    center = np.median(raw_medians[train]); mad = np.median(abs(raw_medians[train] - center))
    q1, q3 = np.quantile(raw_medians[train], [.25, .75]); iqr = q3 - q1
    sample_qc = []
    for i, r in enumerate(samples):
        z = float(abs(raw_medians[i] - center) / (1.4826 * mad)) if mad > 0 else None
        review = []
        if (~valid[:, i]).mean() > .05: review.append('technical_defects_gt_5_percent')
        if bgwarn[:, i].mean() > .05: review.append('background_warnings_gt_5_percent')
        if detected[:, i].mean() < .50: review.append('above_background_lt_50_percent')
        if z is not None and z > 5: review.append('raw_median_MAD_gt_5')
        if iqr > 0 and (raw_medians[i] < q1 - 3 * iqr or raw_medians[i] > q3 + 3 * iqr): review.append('raw_median_outer_IQR')
        sample_qc.append(dict(gsm=r['gsm'], group_id='GSE102971:' + r['source_donor_group'], dose_gy=r['dose_gy'],
            role='reference_fit' if train[i] else 'engineering_validation',
            technical_defect_fraction=float((~valid[:, i]).mean()), background_warning_fraction=float(bgwarn[:, i].mean()),
            not_found_fraction=float((flags[:, i, FLAGS.index('gIsFound')] == 0).mean()),
            above_background_fraction=float(detected[:, i].mean()), raw_log2_median=float(raw_medians[i]),
            raw_median_robust_z=z, candidate_gene_missing_fraction=float(np.isnan(gene_values[:, i]).mean()),
            decision='retain_review' if review else 'retain', review_reasons='|'.join(review)))
    write_tsv(LOG / 'sample-qc-decisions.tsv', sample_qc)
    # Checks catch validation-dependent feature selection and aggregation.
    perturbed, det_changed = probe_values.copy(), probe_detected.copy()
    perturbed[:, ~train] = np.nan; det_changed[:, ~train] = False
    filter_invariant = np.array_equal(fit_probe_filter(perturbed, det_changed, train)[0], keep_qc)
    val_index = int(np.flatnonzero(~train)[0])
    _, one, _ = aggregate_median(logged[:, [val_index]], probe_names, valid[:, [val_index]])
    single_equal = np.array_equal(one, probe_values[:, [val_index]], equal_nan=True)
    _, onegene, _ = aggregate_median(one[selected], [r['current_gene_id'] for r in mapping if r['gene_eligible']])
    gene_single_equal = np.array_equal(onegene[gene_keep], gene_values[:, [val_index]], equal_nan=True)
    # Frozen trained probe/gene catalog is reused, not refitted on the validation sample.
    signatures = [hashlib.sha256(canonical['values'][:, i, 0].tobytes()).hexdigest() for i in range(100)]
    status_counts = dict(Counter(r['mapping_status'] for r in mapping))
    sensitivity = []
    for detection_min in [.10, .20, .50]:
        keep, _, _ = fit_probe_filter(probe_values, probe_detected, train, min_detected=detection_min)
        gids = {r['current_gene_id'] for i, r in enumerate(mapping) if keep[i] and r['current_gene_id']}
        sensitivity.append(dict(min_training_detected_fraction=detection_min, qc_probes=int(keep.sum()), mapped_gene_candidates=len(gids)))
    write_tsv(LOG / 'threshold-sensitivity.tsv', sensitivity)
    strict_ids, strict_values, _ = aggregate_median(logged, probe_names, valid & ~bgwarn)
    _, strict_detection_fraction, _ = aggregate_median(detected.astype(float), probe_names, valid & ~bgwarn)
    strict_detection = np.isfinite(strict_detection_fraction) & (strict_detection_fraction >= .5)
    strict_filter = fit_probe_filter(strict_values, strict_detection, train)[0]
    both = np.isfinite(strict_values) & np.isfinite(probe_values)
    strict_max_delta = float(np.max(abs(strict_values[both] - probe_values[both])))
    summary = dict(study='GSE102971', status='completed local grouping/QC/mapping audit; candidate derivative only',
        completed_utc=datetime.now(timezone.utc).isoformat(), samples=100, groups=20,
        grouping_status='metadata-supported inferred donor groups; no independent donor key; namespace and keep all doses together',
        source_gender_groups=dict(Counter(rows[0]['gender'] for rows in group_samples.values())),
        dose_grid_gy=[0, 2, 5, 6, 7], physical_features=len(feature_ids), distinct_probes=len(probe_ids),
        repeated_probe_names=sum(r['physical_spots'] > 1 for r in mapping),
        extra_repeated_spots=len(feature_ids)-len(probe_ids), mapping_status_counts=status_counts,
        raw_platform_sequence_conflict_probes=len(sequence_conflicts),
        mapped_current_genes_before_qc=len({r['current_gene_id'] for r in mapping if r['current_gene_id']}),
        probes_with_updated_symbol=sum(bool(r['current_gene_id']) and r['source_symbols'] != r['current_symbol'] for r in mapping),
        qc_eligible_probes=int(keep_qc.sum()), mapped_qc_eligible_probes=int(selected.sum()),
        candidate_genes=len(gene_ids), candidate_gene_cells=int(gene_values.size),
        candidate_gene_missing_cells=int(np.isnan(gene_values).sum()),
        technical_defect_cells=int((~valid).sum()), background_warning_cells=int(bgwarn.sum()),
        not_found_cells=int((flags[:, :, FLAGS.index('gIsFound')] == 0).sum()),
        source_flag_totals={k: int(flags[:, :, j].sum()) for j,k in enumerate(FLAGS)},
        retained_samples=100, samples_flagged_for_review=[r['gsm'] for r in sample_qc if r['review_reasons']],
        duplicate_full_signal_vectors=100-len(set(signatures)),
        checks=dict(validation_filter_invariance=bool(filter_invariant), single_sample_probe_aggregation=bool(single_equal),
                    single_sample_gene_aggregation=bool(gene_single_equal), donor_groups_disjoint=True,
                    all_raw_sha256_verified=True, all_platform_feature_probe_ids_match=True),
        sensitivity=dict(threshold_grid=sensitivity, background_strict_qc_eligible_probes=int(strict_filter.sum()),
                         background_strict_additional_missing_probe_cells=int(np.isnan(strict_values).sum()-np.isnan(probe_values).sum()),
                         background_strict_max_log2_probe_change=strict_max_delta,
                         note='Background sensitivity recomputes detection and valid fractions on strict usable spots; diagnostic, not a selected pipeline'),
        artifact_files=[], runtime_seconds=time.perf_counter()-started)
    if not all(summary['checks'].values()): raise ValueError('Audit verification failed')
    for p in sorted(OUT.glob('*.npz')):
        summary['artifact_files'].append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)))
    summary['input_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in [platform_path, gene_path, PILOT/'downloads.json', PILOT/'split.json', PILOT/'summary.json']}
    summary['code_sha256'] = {p.name: sha(p) for p in [Path(__file__), ROOT/'scripts/assay_qc.py']}
    (LOG / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    figures(summary, mapping, sample_qc, sensitivity)
    print(json.dumps(summary, indent=2))


def figures(summary, mapping, samples, sensitivity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    colors = ['#258a83', '#b1802f', '#a65353', '#747a87']
    counts = summary['mapping_status_counts']
    labels = [k.replace('_', '\n') for k in counts]
    axes[0].bar(range(len(counts)), list(counts.values()), color=colors[:len(counts)])
    axes[0].set_xticks(range(len(counts)), labels, fontsize=8)
    axes[0].set(title='Probe mapping audit', ylabel='Distinct probes')
    for i, n in enumerate(counts.values()): axes[0].text(i, n, f'{n:,}', ha='center', va='bottom')
    for key, label in [('technical_defect_fraction', 'Technical defect'), ('background_warning_fraction', 'Background warning')]:
        axes[1].plot([100*r[key] for r in samples], label=label)
    axes[1].set(title='Source quality flags', xlabel='Sample index', ylabel='Physical features (%)'); axes[1].legend()
    axes[2].bar([str(int(r['min_training_detected_fraction']*100))+'%' for r in sensitivity],
                [r['mapped_gene_candidates'] for r in sensitivity], color='#258a83')
    axes[2].set(title='Training detection threshold sensitivity', xlabel='Minimum detected training fraction', ylabel='Mapped genes before gene-level missingness filter')
    fig.suptitle('GSE102971 | grouping, quality and gene mapping audit', fontsize=17, fontweight='bold')
    fig.savefig(LOG / 'audit-overview.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
