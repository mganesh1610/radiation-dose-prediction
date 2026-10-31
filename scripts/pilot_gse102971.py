"""Local raw-source/QC and frozen-reference feasibility pilot, not a dose model.

Run after acquire_gse102971_raw.py. Only its human-source manifest is accepted;
locked-test and auxiliary data are not read. All provenance/reports stay local.
"""
import csv
import gzip
import hashlib
import html
import json
import platform
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from frozen_quantile import FrozenQuantile

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/raw-pilot/GSE102971'
OUT = ROOT / 'data/processed_pilot/GSE102971/v1'
VALUES = ('gProcessedSignal', 'gMedianSignal', 'gBGMedianSignal', 'gBGSubSignal')
FLAGS = ('gIsSaturated', 'gIsFeatNonUnifOL', 'gIsBGNonUnifOL', 'gIsPosAndSignif', 'IsManualFlag', 'gIsWellAboveBG')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_tsv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)


def read_array(path):
    features, values, flags = [], [], []
    controls = Counter()
    with gzip.open(path, 'rt', encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('FEATURES\t'):
                header = line.rstrip('\r\n').split('\t'); break
        else:
            raise ValueError(f'No FEATURES header: {path.name}')
        needed = ('FeatureNum', 'ProbeName', 'ControlType') + VALUES + FLAGS
        if len(set(header)) != len(header) or not set(needed).issubset(header):
            raise ValueError('Missing/duplicate Agilent fields')
        idx = {key: header.index(key) for key in needed}
        for row in csv.reader(f, delimiter='\t'):
            if not row or row[0] != 'DATA': continue
            if len(row) != len(header): raise ValueError('Malformed feature record')
            controls[row[idx['ControlType']]] += 1
            if row[idx['ControlType']] != '0': continue
            features.append((row[idx['FeatureNum']], row[idx['ProbeName']]))
            values.append([float(row[idx[k]]) for k in VALUES])
            flag_values = [int(row[idx[k]]) for k in FLAGS]
            if any(v not in (0, 1) for v in flag_values): raise ValueError('Non-binary source flag')
            flags.append(flag_values)
    if len(features) != len(set(x[0] for x in features)):
        raise ValueError('Duplicate physical feature ID')
    values = np.array(values, dtype=np.float64)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite source measurement')
    return features, values, np.array(flags, dtype=np.uint8), dict(controls)


def figures(raw, logged, rows, metrics, fit_mask, feature_ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    assets = LOG / 'assets'; assets.mkdir(exist_ok=True)
    indices = np.arange(len(rows))
    colours = np.where(fit_mask, '#187f86', '#d87932')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout='constrained')
    for ax, x, title in [(axes[0, 0], np.log2(raw), 'Before between-array normalization'),
                         (axes[0, 1], logged, 'After frozen quantile reference + log2')]:
        q = np.quantile(x, [.25, .5, .75], axis=0)
        ax.vlines(indices, q[0], q[2], color=colours, alpha=.5, linewidth=2)
        ax.scatter(indices, q[1], c=colours, s=10)
        ax.set(title=title, xlabel='Human source sample index', ylabel='log2 per-array signal (median and IQR)')
    axis = axes[1, 0]
    for key, name, colour in [('not_positive_significant_fraction', 'Below detection flag', '#6977ad'),
                               ('nonuniform_fraction', 'Feature nonuniformity flag', '#c85151'),
                               ('saturated_fraction', 'Saturation flag', '#d7a528')]:
        axis.plot(indices, [100 * r[key] for r in metrics], label=name, color=colour)
    axis.set(title='Preserved source quality flags', xlabel='Human source sample index', ylabel='Non-control features (%)')
    axis.legend(fontsize=8)
    # Feature selection, centering and PCA are fitted on the 80 training samples.
    variance = logged[:, fit_mask].var(axis=1)
    panel = np.argsort(-variance, kind='stable')[:1000]
    train_mean = logged[panel][:, fit_mask].mean(axis=1)
    train = logged[panel][:, fit_mask].T - train_mean
    _, singular, vt = np.linalg.svd(train, full_matrices=False)
    coordinates = (logged[panel].T - train_mean) @ vt[:2].T
    explained = singular[:2] ** 2 / np.square(singular).sum()
    np.savez_compressed(OUT / 'pca_reference.npz', feature_ids=feature_ids[panel], mean=train_mean,
                        components=vt[:2], explained_variance_ratio=explained)
    write_tsv(LOG / 'pca-coordinates.tsv', [dict(gsm=r['gsm'], source_donor_group=r['source_donor_group'],
               role='reference_fit' if fit_mask[i] else 'engineering_validation', dose_gy=r['dose_gy'],
               pc1=float(coordinates[i, 0]), pc2=float(coordinates[i, 1])) for i, r in enumerate(rows)])
    dose = np.array([r['dose_gy'] for r in rows])
    ax = axes[1, 1]
    for mask, marker, label in [(fit_mask, 'o', 'Reference fit'), (~fit_mask, '^', 'Engineering validation')]:
        points = ax.scatter(coordinates[mask, 0], coordinates[mask, 1], c=dose[mask], cmap='viridis',
                            vmin=0, vmax=7, marker=marker, s=35, label=label, edgecolors='white', linewidth=.4)
    fig.colorbar(points, ax=ax, label='Source dose (Gy)')
    ax.set(title='PCA fitted on training samples', xlabel=f'PC1 ({explained[0]:.1%})', ylabel=f'PC2 ({explained[1]:.1%})')
    ax.legend(fontsize=8)
    fig.suptitle('GSE102971 | 100 human profiles | engineering QC pilot', fontsize=17, fontweight='bold')
    fig.savefig(assets / 'qc-overview.png', dpi=160); plt.close(fig)

    # Keep training-fitted display parameters distinct from the source matrix.
    top = panel[:40]
    mean, sd = logged[top][:, fit_mask].mean(axis=1), logged[top][:, fit_mask].std(axis=1)
    if np.any(sd == 0): raise ValueError('Constant selected display feature')
    z = (logged[top] - mean[:, None]) / sd[:, None]
    corr = np.corrcoef(logged.T)
    np.savez_compressed(OUT / 'display_cache.npz', heatmap_feature_ids=feature_ids[top],
                        heatmap_train_mean=mean, heatmap_train_sd=sd, heatmap_z=z, sample_correlation=corr,
                        sample_ids=np.array([r['gsm'] for r in rows]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout='constrained')
    im = axes[0].imshow(z, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest')
    fig.colorbar(im, ax=axes[0], shrink=.7, label='Deviation from training mean (SD); colour clipped at ±3')
    axes[0].set(title='40 highest-variance training features', xlabel='Human source sample index', ylabel='Physical array features, not unique genes')
    im = axes[1].imshow(corr, cmap='viridis', vmin=max(-1, float(corr.min())), vmax=1)
    fig.colorbar(im, ax=axes[1], shrink=.7, label='Pearson correlation of log2 normalized signals')
    axes[1].set(title='Descriptive sample–sample correlation', xlabel='Human source sample index', ylabel='Human source sample index')
    fig.suptitle('Exploratory displays | source samples retained | no biomarker claim', fontsize=15)
    fig.savefig(assets / 'heatmap-correlation.png', dpi=160); plt.close(fig)
    return dict(pca_training_features=len(panel), heatmap_features=len(top),
                pca_explained_variance_ratio=explained.tolist(),
                off_diagonal_correlation_range=[float(corr[np.triu_indices(len(rows), 1)].min()), float(corr[np.triu_indices(len(rows), 1)].max())])


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((LOG / 'downloads.json').read_text())
    if not manifest['complete'] or manifest['errors']: raise ValueError('Incomplete acquisition')
    rows = manifest['files']
    assert len(rows) == 100 and all(r['organism'] == 'Homo sapiens' for r in rows)
    matrix, flag_matrix, metrics, reference_features = [], [], [], None
    for i, r in enumerate(rows):
        path = (ROOT / r['path']).resolve()
        if not path.is_relative_to((ROOT / 'data/raw_radiation/GSE102971').resolve()): raise ValueError('Unexpected source path')
        if sha(path) != r['sha256']: raise ValueError('Raw source checksum mismatch')
        features, values, flags, control_counts = read_array(path)
        if reference_features is None: reference_features = features
        elif features != reference_features: raise ValueError('Source feature identity/order mismatch')
        matrix.append(values); flag_matrix.append(flags)
        x = values[:, 0]
        flag_index = {k: j for j, k in enumerate(FLAGS)}
        metrics.append(dict(gsm=r['gsm'], source_donor_group=r['source_donor_group'], dose_gy=r['dose_gy'],
                            source_rows=sum(control_counts.values()), non_control_features=len(features),
                            signal_min=float(x.min()), signal_median=float(np.median(x)), signal_max=float(x.max()),
                            nonpositive_signal=int(np.sum(x <= 0)),
                            nonpositive_background_subtracted=int(np.sum(values[:, 3] <= 0)),
                            saturated_fraction=float(flags[:, flag_index['gIsSaturated']].mean()),
                            nonuniform_fraction=float(flags[:, flag_index['gIsFeatNonUnifOL']].mean()),
                            background_nonuniform_fraction=float(flags[:, flag_index['gIsBGNonUnifOL']].mean()),
                            not_positive_significant_fraction=float(1 - flags[:, flag_index['gIsPosAndSignif']].mean()),
                            manual_flag_fraction=float(flags[:, flag_index['IsManualFlag']].mean()),
                            sample_decision='retained; exploratory QC; not approved for production modeling'))
        if (i + 1) % 10 == 0: print(f'Parsed {i+1}/100 source arrays', flush=True)
    # Separate source fields and original binary quality flags, no clipping/imputation.
    source = np.stack(matrix, axis=1)
    source_flags = np.stack(flag_matrix, axis=1)
    del matrix, flag_matrix
    sample_ids = tuple(r['gsm'] for r in rows)
    feature_ids = np.array([x[0] for x in reference_features])
    probe_names = np.array([x[1] for x in reference_features])
    np.savez_compressed(OUT / 'canonical_source.npz', values=source, flags=source_flags,
                        value_columns=np.array(VALUES), flag_columns=np.array(FLAGS),
                        feature_ids=feature_ids, probe_names=probe_names, sample_ids=np.array(sample_ids))
    write_tsv(OUT / 'features.tsv', [dict(source_feature_id=f, source_probe_name=p) for f, p in reference_features])
    write_tsv(LOG / 'sample-qc.tsv', metrics)
    raw = source[:, :, 0]
    groups = sorted({r['source_donor_group'] for r in rows})
    assert len(groups) == 20
    held_groups = groups[::5]
    train_ids = tuple(r['gsm'] for r in rows if r['source_donor_group'] not in held_groups)
    fit_mask = np.array([s in train_ids for s in sample_ids])
    assert fit_mask.sum() == 80 and (~fit_mask).sum() == 20
    split = dict(purpose='engineering preprocessing check; not model evaluation or final CV',
                 rule='alphabetically sorted source donor groups; every fifth group held out',
                 reference_fit_groups=[g for g in groups if g not in held_groups], validation_groups=held_groups,
                 reference_fit_samples=list(train_ids), validation_samples=[s for s in sample_ids if s not in train_ids],
                 donor_status='source title/filename groups consistent; identity linkage not independently confirmed')
    (LOG / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
    fitted = FrozenQuantile.fit(raw, feature_ids, sample_ids, train_ids)
    normalized = fitted.transform(raw, feature_ids)
    logged = np.log2(normalized)
    fitted.save(OUT / 'frozen_quantile_reference.npz')
    np.savez_compressed(OUT / 'engineering_log2_qn.npz', values=logged, feature_ids=feature_ids,
                        sample_ids=np.array(sample_ids), training_mask=fit_mask)
    # Meaningful independence checks on actual source data, with an intentionally
    # cohort-fitted positive control demonstrating that the test detects dependence.
    altered = raw.copy(); altered[:, ~fit_mask] *= 1.7
    ref_after_change = FrozenQuantile.fit(altered, feature_ids, sample_ids, train_ids)
    alone_max = 0.
    for j in np.flatnonzero(~fit_mask):
        alone = fitted.transform(raw[:, j:j+1], feature_ids)[:, 0]
        alone_max = max(alone_max, float(np.abs(alone - normalized[:, j]).max()))
    all_fit = FrozenQuantile.fit(raw, feature_ids, sample_ids, sample_ids)
    all_changed = FrozenQuantile.fit(altered, feature_ids, sample_ids, sample_ids)
    checks = dict(validation_perturbation_target_max_abs_change=float(np.abs(fitted.target-ref_after_change.target).max()),
                  validation_single_vs_batch_max_abs_difference=alone_max,
                  reversed_batch_max_abs_difference=float(np.abs(fitted.transform(raw[:, ::-1], feature_ids)-normalized[:, ::-1]).max()),
                  artifact_roundtrip_max_abs_difference=float(np.abs(FrozenQuantile.load(OUT / 'frozen_quantile_reference.npz').transform(raw[:, ~fit_mask], feature_ids)-normalized[:, ~fit_mask]).max()),
                  deliberately_cohort_fitted_control_target_change=float(np.abs(all_fit.target-all_changed.target).max()),
                  training_groups_disjoint=not bool(set(split['reference_fit_groups']) & set(held_groups)))
    checks['passed'] = all(checks[k] == 0. for k in list(checks)[:4]) and checks['deliberately_cohort_fitted_control_target_change'] > 0 and checks['training_groups_disjoint']
    if not checks['passed']: raise AssertionError(checks)
    display = figures(raw, logged, rows, metrics, fit_mask, feature_ids)
    import matplotlib
    summary = dict(study='GSE102971', status='completed engineering feasibility pilot; not production-ready',
                   completed_utc=datetime.now(timezone.utc).isoformat(), report_date='2026-09-30',
                   human_samples=100, source_donor_groups=20, training_samples=80, validation_samples=20,
                   donor_identity_status=split['donor_status'], non_control_features=len(feature_ids),
                   distinct_probe_names=len(set(probe_names.tolist())), expression_cells=int(raw.size),
                   raw_compressed_bytes=sum(r['bytes'] for r in rows), raw_expanded_bytes=sum(r['expanded_bytes'] for r in rows),
                   source_signal='Agilent gProcessedSignal, source per-array background/feature-extraction processing; before our between-array normalization',
                   representation='linear train-reference quantile normalization followed by log2; no pseudocount',
                   source_feature_policy='ControlType==0; physical FeatureNum retained; repeated ProbeName not collapsed',
                   quality_policy='source flags retained; no sample deletion, per-cell masking, imputation or gene aggregation in this prototype',
                   signal_range=[float(raw.min()), float(raw.max())],
                   source_below_detection_fraction_range=[min(m['not_positive_significant_fraction'] for m in metrics), max(m['not_positive_significant_fraction'] for m in metrics)],
                   independence_checks=checks, display=display,
                   runtime_seconds=time.perf_counter()-start, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   runtime=dict(python=sys.version.split()[0], numpy=np.__version__, matplotlib=matplotlib.__version__, platform=platform.platform()),
                   code_sha256={p.name: sha(p) for p in [Path(__file__), ROOT/'scripts/frozen_quantile.py', ROOT/'scripts/acquire_gse102971_raw.py']},
                   input_manifest_sha256=sha(LOG/'downloads.json'),
                   artifact_files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.glob('*')) if p.is_file()],
                   limitations=['Donor group labels are source-derived, not independently proven identities.',
                                'Source Feature Extraction background correction was retained, not reimplemented from scanner images.',
                                'Biological appropriateness and cross-platform transfer of quantile normalization are not established by invariance tests.',
                                'Repeated probe names, ambiguous gene mappings and quality-flag policies require review before modeling.',
                                'No comparison against published model performance, dose prediction, differential expression or external-test evaluation.',
                                'No locked-test sources read and no cloud/API operations performed.'])
    summary['derived_data_bytes'] = sum(r['bytes'] for r in summary['artifact_files'])
    (LOG / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_report(summary, checks)
    print(json.dumps({k:summary[k] for k in ['human_samples','source_donor_groups','non_control_features','raw_compressed_bytes','raw_expanded_bytes','derived_data_bytes','runtime_seconds','peak_process_rss_bytes']}), flush=True)
    print('INDEPENDENCE', json.dumps(checks), flush=True)


def write_report(s, checks):
    bullets = '\n'.join(f'<li>{html.escape(x)}</li>' for x in s['limitations'])
    checks_html = ''.join(f'<tr><td>{html.escape(k.replace("_", " "))}</td><td>{v}</td></tr>' for k,v in checks.items())
    text = f'''<!doctype html><html><head><meta charset="utf-8"><title>GSE102971 raw-source pilot</title>
<style>body{{margin:0;background:#f2f5f7;color:#172c3a;font:16px/1.55 system-ui}}main{{max-width:1150px;margin:auto;padding:36px}}h1{{font-size:32px;margin-bottom:4px}}.note{{color:#576c7b}}.cards{{display:flex;gap:16px;margin:26px 0}}.card{{background:white;border:1px solid #d8e2e7;border-radius:12px;padding:17px;flex:1}}b.big{{font-size:29px;display:block;color:#13747b}}img{{width:100%;background:white;border-radius:12px;border:1px solid #d8e2e7}}table{{border-collapse:collapse;width:100%;background:white}}td{{border-bottom:1px solid #ddd;padding:9px}}.status{{background:#d7ece8;padding:10px 16px;border-radius:8px}}a{{color:#086879}}</style></head><body><main>
<p class="note">HEALTH PROJECT · SEPTEMBER 30, 2026 · LOCAL RESEARCH EVIDENCE</p><h1>GSE102971 raw-source pilot</h1>
<p class="status">Preprocessing independence checks passed. Engineering feasibility only; no dose model trained.</p>
<div class="cards"><div class="card"><b class="big">100</b>human source profiles</div><div class="card"><b class="big">20</b>source-coded donor groups</div><div class="card"><b class="big">80 / 20</b>reference fit / validation</div><div class="card"><b class="big">43,118</b>physical array features</div></div>
<p>Downloaded {s['raw_compressed_bytes']/1e6:,.1f} MB; expanded content {s['raw_expanded_bytes']/1e9:.2f} GB. Files were streamed from gzip. Derived data: {s['derived_data_bytes']/1e6:.1f} MB.</p>
<p>The prototype uses source per-array processed intensity, fits a quantile target only on the training groups, then applies that fixed target and log2 transformation to each sample. Controls are kept in the original files; their rows are excluded from this analytical representation. Source quality flags are retained and no samples are deleted.</p>
<img src="assets/qc-overview.png" alt="Raw and normalized intensity distributions, source quality flags and training-fitted PCA">
<p class="note">Matching post-normalization distributions are expected by construction and do not prove biological suitability. PCA is exploratory; group separation does not authorize exclusion.</p>
<img src="assets/heatmap-correlation.png" alt="Training-selected feature heatmap and descriptive sample correlations">
<p class="note">Heatmap uses training mean/SD; colour clipping changes display only. Probe names are not unique genes. Correlation is descriptive and does not establish dose-prediction accuracy.</p>
<h2>Verified independence checks</h2><table>{checks_html}</table>
<h2>Remaining review requirements</h2><ul>{bullets}</ul>
<p>Evidence: <a href="summary.json">machine-readable summary</a> · <a href="sample-qc.tsv">sample QC</a> · <a href="split.json">split membership</a> · <a href="downloads.json">download integrity</a></p>
</main></body></html>'''
    (LOG / 'report.html').write_text(text)
    (LOG / 'REPORT.md').write_text(f'''# GSE102971 raw-source pilot

September 30, 2026. **Completed local engineering feasibility pilot; no dose model or cloud deployment.**

- 100 human profiles in 20 source-coded groups; 80 reference-fit samples and 20 engineering validation samples. Group letters match GEO titles and raw filenames; donor identities are not independently proven.
- {s['non_control_features']:,} non-control physical features, {s['distinct_probe_names']:,} distinct probe names and {s['expression_cells']:,} measurement cells. These are not unique-gene counts.
- Original raw files: {s['raw_compressed_bytes']:,} bytes compressed; {s['raw_expanded_bytes']:,} bytes expanded content. Expanded text was not retained. Derived scientific artifacts: {s['derived_data_bytes']:,} bytes, excluding the local runtime, report and provenance.
- Input fields include foreground/background measurements, source per-array processed signal, negative background-subtracted values and source quality flags. The prototype uses positive `gProcessedSignal` before its own between-array normalization; negative `gBGSubSignal` values remain unchanged in the canonical artifact.
- Frozen training-reference quantile normalization followed by log2; no extra background correction, pseudocount, masking, imputation, gene aggregation or sample deletion. This bypasses the published cohort-normalized matrix but retains source Feature Extraction processing.
- Independence checks passed: changing validation values did not change the fitted target; single-sample and batch processing, batch order and artifact reload returned identical results. An intentionally cohort-fitted control changed when validation values changed, demonstrating sensitivity of the check.
- Parsing, transformations, checks and figures took {s['runtime_seconds']:.1f} seconds; peak process RSS {s['peak_process_rss_bytes']/1e6:.1f} MB on macOS. This is a local processor benchmark, excluding download/environment setup; it is not a Cloud Run cost estimate.

See [interactive browser report](report.html), [summary](summary.json), [sample QC](sample-qc.tsv), [split](split.json), [PCA coordinates](pca-coordinates.tsv) and [download manifest](downloads.json). The NPZ/TSV artifact paths and SHA-256 hashes appear in the summary. Gzip CRC and publisher-listed sizes were checked; SHA-256 values are local reproducibility records, not publisher-provided checksums.

## Limits and next work

''' + '\n'.join('- ' + x for x in s['limitations']) + '''

Before model use, review source flags and replicate-probe/gene aggregation, establish donor linkage confidence, compare scientifically defensible preprocessing choices within nested development folds, and validate transfer to other platforms. Preserve the existing processed-data track for a prespecified comparison. The current engineering holdout is not a replacement for that evaluation. No GSE90909 data were read.

Reproduction: run `scripts/acquire_gse102971_raw.py`, `data/local_runtime/gse102971/bin/python -m unittest discover -s tests`, then `data/local_runtime/gse102971/bin/python scripts/pilot_gse102971.py`. Acquisition is idempotent against local checksums; this v1 pilot rerun regenerates its local derivative/report files. Original source files are preserved. Context, reports and provenance remain local and ignored by Git.
''')


if __name__ == '__main__':
    main()
