"""Render local normalization and core-compatibility evidence."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'context/normalization-validation/v2'
n=json.loads((LOG/'normalization-summary.json').read_text());c=json.loads((LOG/'compatibility-summary.json').read_text())
t=json.loads((LOG/'unit-test-result.json').read_text());d=n['sensitivity']['gene_absolute_log2_change_vs_postmask'];s=n['sensitivity']
raw8917=next(r['raw_example'] for r in c['studies'] if r['study']=='GSE8917')
rows=[]
for r in c['studies']:
    rows.append(f"| {r['study']} | {r['samples']} | {r['platform']} | {r['gene_rows']:,} | {100*r['gene_missing_fraction']:.2f}% |")
table='\n'.join(rows)
report=f'''# Normalization validation and core-study compatibility

September 30, 2026. Local development work completed. The known influence of technically excluded values on the GSE102971 between-array reference/ranks is removed in a separate v2 implementation. This is verified computational behavior, not biological validation or dose-model performance.

## GSE102971: executed change and checks

The explicit technical mask is applied before reference fitting and before sample ranking. For each training array, sort usable positive source `gProcessedSignal` values and interpolate its quantile curve to the fixed feature-count grid over [0,1]. Average only the 80 training-array curves. Apply this frozen target to usable ranks of each sample independently, then log2-transform. Ties receive the mean of their occupied target ranks. Excluded cells remain missing. The 20 engineering validation profiles never fit the target; original per-array Feature Extraction processing remains inherited.

This extends a documented quantile-interpolation approach, with an explicit tie convention and frozen fit/apply separation. R is not installed, so no bitwise equivalence to limma/preprocessCore is claimed. A hand-calculated missing-data fixture, complete-data equivalence (including ties) to the previous implementation, and independent invariants check the algorithm.

| Verification | Result |
|---|---:|
| Unit tests | {t['tests']} passed |
| Replace masked values with 1e12: target change | 0 |
| Replace masked values with 1e12: usable output change | 0 |
| Change all validation values and masks: target change | 0 |
| Each of 20 validation samples alone vs batch | 0 maximum difference |
| Reverse batch order / reload artifact | 0 maximum difference |
| Five donor-group holdouts, validation perturbation | All zero target changes |
| Deliberately unmasked positive control | Target changes by ~{n['checks']['flawed_unmasked_control_target_change']:.3g} |
| Candidate matrix | {n['candidate_genes']:,} genes × 100 profiles |
| Missing gene cells | {n['missing_gene_cells']:,}; no imputation |

The gene and sample catalogs are unchanged. Median absolute difference from the earlier post-normalization mask is **{d['median']:.4f} log2**, 95th percentile **{d['p95']:.4f}**, maximum **{d['maximum']:.4f}** across jointly observed gene cells. Within the 16 training groups, median paired 7-vs-0 Gy changes correlate at {s['training_paired_7_vs_0_change_correlation']:.6f} between versions. Among {s['training_paired_change_genes_above_half_log2']:,} genes with an old absolute paired change ≥0.5 log2, {s['training_paired_change_sign_flips_of_old_abs_change_at_least_half_log2']} reverse sign. These exploratory stability measures are not accuracy estimates or biomarker selection.

## Sensitivity and remaining assumptions

- A stricter background-warning mask yields {s['background_strict_missing_gene_cells']:,} missing gene cells on the fixed catalog, versus 253 under the primary technical mask. Among jointly observed cells, {s['background_strict_cells_changed_more_than_one_log2']:,} change by >1 log2; maximum change is {s['gene_absolute_log2_change_background_strict']['maximum']:.2f}. The [largest discrepancies](background-sensitive-cells.tsv) are recorded for review. This is a sensitivity diagnostic, not a rule to automatically remove all background-warning values.
- Frozen median scaling was also evaluated as a diagnostic, with median absolute gene difference {s['gene_absolute_log2_change_frozen_median']['median']:.4f} log2 from the masked quantile method. No method was selected by validation prediction performance.
- Technical flag rates vary across training intensity deciles. Missing-at-random and common-distribution assumptions are not established merely because numerical checks pass. Same-platform reference compatibility does not establish transfer to a different array design.
- Technical flags are source-derived. Earlier background/feature-extraction calculations inside the source files cannot be undone by our later masking. Do not claim independence from those upstream calculations.
- The current ≥95% valid/≥20% detected probe filter and probe→gene median policy are unchanged. Future model folds must refit the reference, eligibility, imputation and scaling on their own training data. The five engineering holdouts are not the final study-held-out evaluation.

![Normalization checks](normalization-checks.png)

## Core-study compatibility

| Study | Selected profiles | Platform | Genes in current view | Gene-cell missingness |
|---|---:|---|---:|---:|
{table}

GSE102971 uses the new raw-derived candidate. GSE8917 and GSE65292 views preserve their **published log2, median-normalized** values and source missingness; no second logarithm or shared normalization was applied. Their views are descriptive audit artifacts, not approved replacements for raw reprocessing.

The platforms share **{c['common_current_annotation_genes']:,}** exact current human GeneIDs. **{c['common_genes_in_current_candidate_and_published_views']:,}** are represented in all three current views with at least one observed value per study. This is not a complete-case model feature list. It reflects platform content and prior processing/filtering. Future model feature eligibility must be determined within training folds, not from this three-study expression overlap.

GSE102971 and GSE65292 platforms share all 34,127 non-control probe names with identical platform sequences. GSE8917 shares 18,561 probe names with the later design, also with matching platform sequences. GPL1708 uses **SPOT_ID** for probe names; its NAME field contains transcript accessions. Explicit per-platform keys and a regression test prevent silently merging these identifier types. The initial incorrect NAME-based audit was superseded and retained separately for traceability.

One raw-format example from each other study was downloaded (about 16.03 MB total), with local SHA-256 and gzip CRC verification. This is not full raw acquisition or validation of the other 45 profiles:

- **GSE8917 / GSM226009:** all 41,675 non-control feature/probe IDs agree with the platform. Its raw file lacks a Sequence column, so raw-sequence verification is unavailable. It contains **12,728 non-positive gProcessedSignal values**, minimum −120.2108; **{raw8917['nonpositive_technically_usable_signals']:,}** remain among the {raw8917['technically_usable_features']:,} measurements passing the current technical mask. The positive-input normalizer must reject it unchanged even after technical masking. Foreground/background fields are available for a validated study-specific background-correction approach. Do not clip, take absolute values or add an arbitrary offset.
- **GSE65292 / GSM1591727:** all 43,118 non-control features match their probe IDs and sequences; processed signals are positive and required flag columns exist. This establishes adapter feasibility for one sample, not a full-cohort QC result.

## Dose and scale reconciliation

The GSE65292 publication's irradiation methods specify 0.56, **2.23** and 4.45 Gy, 1.03 Gy/min acute exposures, log2 median normalization, and five microarray donors. GEO characteristics report 2.2 for the middle group and the series summary says 2.25. The new [candidate labels](core-sample-labels-v2.tsv) adopt the publication-supported 2.23 for the five acute middle-dose profiles, retaining all original source texts and the conflict status. Original GEO files and the earlier selection manifest are unchanged. This is an explicit evidence-based canonicalization, not an inferred dose from gene expression. [Published methods](https://link.springer.com/article/10.1186/s12920-015-0097-x).

## Decision and next implementation

Use pre-reference technical masking as the GSE102971 development implementation; preserve the earlier version for comparison. **Do not publish a pooled, harmonized training dataset yet.** Complete raw acquisition/reprocessing for the other core subsets, address GSE8917's non-positive background-corrected signals with a validated method, retain platform-specific references, and evaluate common-gene transfer through outer study-held-out and inner donor-grouped development. Matching gene IDs and log2 units alone do not remove batch, probe-specific or biological differences. Do not use joint test-cohort ComBat/quantile fitting to make a plot look mixed.

No dose model, cloud upload, database publication, paid API call or locked-test expression access occurred. Original datasets, previous scientific artifacts and supplied final proposal/presentation are preserved.

## Evidence and reproduction

- [Normalization result](normalization-summary.json), [compatibility result](compatibility-summary.json), [unit tests](unit-test-result.json).
- [Per-gene normalization sensitivity](gene-normalization-sensitivity.tsv), [background-sensitive cells](background-sensitive-cells.tsv), [annotation availability](common-current-gene-availability.tsv).
- [Source/method evidence and implementation corrections](sources/method-evidence.json), [raw example downloads](sources/raw-format-probes.json).
- [preprocessCore target normalization documentation](https://bioconductor.org/packages/release/bioc/manuals/preprocessCore/man/preprocessCore.pdf) and [limma algorithm source](https://raw.githubusercontent.com/bioc/limma/devel/R/norm.R).

Run `scripts/validate_masked_normalization.py`, then `scripts/check_core_compatibility.py`, then this report renderer with the local `data/local_runtime/gse102971/bin/python`. Artifacts are separate NPZ files; exact source/code/output hashes are in the JSON summaries. Context and provenance remain local and ignored.
'''
(LOG/'REPORT.md').write_text(report)
html_rows=''.join(f"<tr><td>{r['study']}</td><td>{r['samples']}</td><td>{r['platform']}</td><td>{r['gene_rows']:,}</td><td>{100*r['gene_missing_fraction']:.2f}%</td></tr>" for r in c['studies'])
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Normalization validation — radiation project</title>
<style>body{{font:16px system-ui;background:#eef3f5;color:#173443;margin:0}}main{{max-width:1300px;margin:auto;padding:32px}}h1{{font-size:32px}}.eyebrow{{color:#237b78;font-weight:700}}.cards{{display:flex;gap:16px}}.card,section{{background:white;border-radius:12px;padding:20px}}.card{{flex:1}}strong{{display:block;font-size:30px}}img{{width:100%;margin:24px 0}}section{{margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #dae2e7;text-align:left}}.note{{padding:16px;border-left:5px solid #c28b37;background:#fff9eb;line-height:1.6}}a{{color:#176967}}</style>
<main><div class="eyebrow">RADIATION PROJECT · LOCAL DEVELOPMENT EVIDENCE · 30 SEPTEMBER 2026</div><h1>QC before normalization: verified behavior</h1>
<div class="cards"><div class="card">Automated tests<strong>{t['tests']} passed</strong></div><div class="card">Masked-value influence<strong>0</strong></div><div class="card">Candidate genes<strong>16,792</strong></div><div class="card">Median change<strong>0.0092 log2</strong></div></div>
<img src="normalization-checks.png" alt="References, gene-value changes and intensity-dependent flag rates">
<div class="note">Verified: excluded values cannot change the target or usable outputs. Validation data cannot fit the reference. These checks establish computational behavior; biological transfer and prediction accuracy remain untested.</div>
<section><h2>Core-study compatibility</h2><table><tr><th>Study</th><th>Profiles</th><th>Platform</th><th>Genes in current view</th><th>Missing gene cells</th></tr>{html_rows}</table><p>17,698 current GeneIDs shared by platform annotations; 7,761 represented in all current views. These are descriptive counts, not a selected model panel.</p></section>
<div class="note"><b>Pooling gate:</b> GSE8917 raw example contains 12,728 non-positive processed signals and needs different background treatment. Other-study views are published processed values. GSE65292 middle-dose candidate is 2.23 Gy, with conflicting GEO labels preserved.</div>
<section><a href="REPORT.md">Full report</a> · <a href="normalization-summary.json">Normalization evidence</a> · <a href="compatibility-summary.json">Compatibility evidence</a> · <a href="background-sensitive-cells.tsv">Background-mask sensitivity review</a></section><p>No dose-model result, cloud publication or paid API use. Locked test data remain untouched.</p></main></html>'''
(LOG/'report.html').write_text(page)
print(LOG/'report.html')
