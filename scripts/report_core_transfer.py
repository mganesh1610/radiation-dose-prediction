"""Render the local core-reprocessing evidence after successful verification runs."""
import csv
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'context/core-reprocessing/v1'
s=json.loads((LOG/'summary.json').read_text());acq=json.loads((LOG/'downloads.json').read_text())
t=json.loads((LOG/'unit-test-result.json').read_text())
qc=list(csv.DictReader((LOG/'sample-raw-qc.tsv').open(),delimiter='\t'))
fits=list(csv.DictReader((LOG/'background-parameters.tsv').open(),delimiter='\t'))
rows=[]
for r in s['source_studies']:
    total=sum(int(q['nonpositive_published_processed_usable']) for q in qc if q['study']==r['study'])
    rows.append(f"| {r['study']} | {r['samples']} | {total:,} | {r['missing_physical_cells']:,} | {r['unfiltered_current_genes']:,} |")
folds=[]
for f in s['folds']:
    folds.append(f"| {f['held_study']} | {f['training_samples']} / {f['held_samples']} | {f['eligible_probes']:,} | {f['genes']:,} | {100*f['held_missing_gene_fraction']:.3f}% | {f['absolute_control_median_shift_log2']['median']:.3f} |")
amount=sum(r['bytes'] for r in acq['files']);expanded=sum(r['expanded_bytes'] for r in acq['files'])
artifacts=sum(r['bytes'] for r in s['artifacts'])
report=f'''# Core raw reprocessing and study-transfer audit

September 30, 2026 (America/Phoenix). Completed local acquisition and a new preprocessing candidate for all **145 core profiles**. This run tests preprocessing transfer, not dose-model accuracy. GSE90909 remains sealed.

## Acquisition and source validation

All 25 selected 24-hour GSE8917 and 20 acute/control GSE65292 raw files are now local: **{amount:,} compressed bytes**, **{expanded:,} expanded bytes** streamed for validation. Two of these 45 were already downloaded for the preceding format audit. Publisher-listed sizes, gzip CRC and local SHA-256 hashes were checked; no publisher SHA-256 was available. The existing 100 GSE102971 raw files and earlier scientific artifacts were hash-checked and preserved. The new source parser checked feature/probe identities throughout both additional cohorts and sequences throughout GSE65292; GSE8917 has no raw Sequence column, so its raw-sequence check remains unavailable. Platform sequence comparisons are a separate check.

## Background correction and QC

Used **R 4.6.1, limma 3.68.5**, installed locally, with the R packages isolated under `data/local_runtime/R-library`. Started from each array's median foreground and median background, excluding explicit technical defects before parameter estimation. Applied limma's **normexp saddle** estimator independently to each array, with offset **0**, followed by spot-to-probe medians. This does not logarithmically transform negative deposited processed signals or add an arbitrary shift to them. Published matrices remain unchanged.

All **145 final fits converged**, with positive finite usable corrected signals and no recorded warnings. Results matched `backgroundCorrect` exactly, and sample-alone versus batch and reversed-batch checks gave zero differences for each study. Defects remain missing; no samples were deleted or values imputed in the scientific expression artifacts.

| Study | Profiles | Non-positive deposited processed values passing technical mask | Masked physical cells | Current genes in unfiltered corrected view |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

**Recorded estimator change:** The original MLE plan stopped on a convergence failure. The complete diagnostic found five GSE102971 MLE fits with iteration-limit/singular-convergence messages. The documented default saddle estimator converged on all 145 and was adopted uniformly before cross-study projections. The [original plan and amendment](METHOD.md), [diagnostic](normexp-convergence-diagnostic.tsv) and [final parameters](background-parameters.tsv) preserve this decision. Some parameter estimates are extreme; convergence and positive output establish numerical behavior, not the biological truth of the normal/exponential model. Source image extraction and detection flags remain inherited. Control spots were excluded from our parameter fits, an explicit adaptation of the reference workflow.

The approach follows limma's [Agilent median-intensity and normexp example](https://bioconductor.org/packages/3.23/bioc/vignettes/limma/inst/doc/usersguide.pdf) and [background-correction documentation](https://bioconductor.org/packages/release/bioc/manuals/limma/man/limma.pdf), with our separately tested training-reference step replacing cohort-wide normalization.

## Common features and outer study holdouts

The external platform annotations share {s['shared_probe_names']:,} probe names with matching sequences. Requiring consistent exact current human GeneIDs leaves **{s['consistent_sequence_and_gene_probes']:,} probes / {s['common_annotation_gene_count']:,} genes** before training-only eligibility filters; {s['shared_probe_names']-s['consistent_sequence_and_gene_probes']:,} shared probe mappings remain inconsistent/unresolved and are excluded from this common candidate, not erased from source data. This externally defined platform universe does not use held-out expression availability.

In each outer check, one entire study is held out. Probe eligibility requires ≥95% usable and ≥20% detected profiles **within each training study**. Training studies contribute equal weight to the frozen quantile target. Apply it independently to each sample, log2 once, then use equal-probe medians per gene. Each fold saves its own reference, feature order and expression matrix. No joint held-out normalization, ComBat fitting or target-cohort filter is performed.

| Held-out study | Train / held profiles | Eligible probes | Genes | Held gene-cell missingness | Median absolute control-median shift (log2) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(folds)}

The control-shift column compares the held study's control medians with equally weighted training-study control medians on observed gene values. It is descriptive and may combine technical and biological differences. It is not a correction factor, equivalence test or model metric. The GSE8917 holdout has {next(f['genes']-f['absolute_control_median_shift_log2']['n'] for f in s['folds'] if f['held_study']=='GSE8917')} genes without a comparable observed control median; those genes remain missing and are excluded only from this descriptive shift calculation. The recorded all-NaN warning reflects these absent control measurements, not an imputation or a failed background fit.

## Executed transfer checks

- **{t['tests']} Python tests passed**, including study weighting, minority-study eligibility, held-out-value/mask/detection isolation, exact probe order and artifact reload.
- All three outer checks: held-out perturbation left training features/reference unchanged; all held profiles gave identical results alone/in batches, in reverse order and after reference reload.
- **15 inner donor-grouped refits** also preserved feature selection/reference under validation-data perturbation. Group IDs are study-local inferred/source-supported repeats; globally unique or independently confirmed donor identities are not claimed.
- PCA uses training-only gene selection (up to 2,000 variable genes), median imputation and centering. Missing values are filled only in the separate display calculation, not the saved scientific matrices. The plot shows held-out projections, not prediction performance.

![Training-fitted projections](cross-study-projection.png)

**Substantial study separation remains visible.** Quantile normalization constrains marginal distributions and does not remove probe-specific or biological study effects. Any separation or overlap in these two PCs is insufficient to establish harmonization. Keep study labels, inspect control differences and evaluate donor-balanced dose-model errors in the future outer study validation.

## Prespecified offset sensitivity

Offset 50 was evaluated only as a diagnostic, without changing feature catalogs or choosing an outcome-based winner. Median absolute gene changes from offset 0 are {', '.join(f["held_study"]+': '+format(f["offset50_absolute_change_log2"]["median"],'.3f') for f in s['folds'])} log2. This reflects low-intensity shrinkage and supports retaining the parameter in method provenance. The primary candidate stays offset 0; predictive performance has not selected either setting.

## Deliverables, limits and next step

New files under `data/processed_pilot/core-reprocessing-v1/` include canonical arrays for the two additional studies, masked background inputs, fitted background parameters, per-study corrected probe/gene views, and outer-fold references/matrices/PCA displays. Their total is **{artifacts:,} bytes** including reproducible intermediate binary inputs/outputs. This is additional local storage, not an assertion that cloud free-tier allowances cover all retained versions.

This completes the requested core raw-processing and preprocessing-transfer checks. The next modeling step can consume the fold-specific candidate while refitting all learned steps inside inner tuning. Residual study effects, background-model assumptions, inferred grouping and unresolved annotation coverage remain reportable limitations. No claim of predictive generalization is available until dose models are evaluated. GSE65292's five middle-dose candidates remain publication-supported 2.23 Gy with the original GEO conflicts preserved. Auxiliary cohorts require their own processing before strategies B–D.

No dose predictor, database publication, cloud upload, paid API call or locked-test expression analysis occurred. Final proposal/presentation and earlier data versions remain unchanged. Figures are generated scientific artifacts; no new UI screenshot was captured. The previous browser local-file restriction was not bypassed or retried.

Evidence: [summary](summary.json), [downloads](downloads.json), [background checks](background-checks.tsv), [sample QC](sample-raw-qc.tsv), [fold QC](fold-sample-qc.tsv), [PCA coordinates](pca-projections.tsv), [annotation decisions](shared-probe-annotation.tsv), [tests](unit-test-result.json), [verification](verification.json), [R environment](R-sessionInfo.txt).

Reproduce in order: `acquire_core_raw.py`, `prepare_core_background.py`, `Rscript --vanilla scripts/background_core.R`, `validate_core_transfer.py`, then this renderer. Run Python scripts with `data/local_runtime/gse102971/bin/python`. Core manifests are allowlisted and all provenance remains local under ignored `context/`.
'''
(LOG/'REPORT.md').write_text(report)
htmlrows=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row.strip('|').split('|'))+'</tr>' for row in folds)
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Core raw reprocessing and transfer checks</title><style>body{{font:16px system-ui;background:#edf3f5;color:#193744;margin:0}}main{{max-width:1400px;margin:auto;padding:30px}}.cards{{display:flex;gap:18px}}.card,section{{background:white;padding:20px;border-radius:12px}}.card{{flex:1}}strong{{display:block;font-size:30px}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #dce5e8;text-align:left}}.note{{border-left:5px solid #ca873d;background:#fff8e9;padding:18px;line-height:1.5}}</style><main><p>RADIATION PROJECT · LOCAL DEVELOPMENT · 30 SEPTEMBER 2026</p><h1>Core raw processing completed; transfer checked</h1><div class="cards"><div class="card">Profiles<strong>145</strong></div><div class="card">Background fits converged<strong>145 / 145</strong></div><div class="card">Python tests<strong>{t['tests']} passed</strong></div><div class="card">Common mapped probes<strong>{s['consistent_sequence_and_gene_probes']:,}</strong></div></div><img src="cross-study-projection.png" alt="Three study-held-out training-fitted PCA projections"><section><h2>Study holdouts</h2><table><tr><th>Held study</th><th>Train / held</th><th>Probes</th><th>Genes</th><th>Missingness</th><th>Control shift, log2</th></tr>{htmlrows}</table></section><p class="note">Computational independence passed. These results do not demonstrate biological harmonization or prediction accuracy. Study differences, inferred donor grouping and background-model assumptions remain under evaluation. GSE90909 stays sealed.</p><p><a href="REPORT.md">Full report</a> · <a href="summary.json">Machine-readable evidence</a> · <a href="METHOD.md">Prespecified method and estimator amendment</a></p></main></html>'''
(LOG/'report.html').write_text(page)
print(LOG/'REPORT.md')
