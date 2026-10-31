"""Render the local audit evidence without recomputing or selecting results."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/raw-pilot/GSE102971/audit-v1'
s = json.loads((LOG / 'summary.json').read_text())
tests = json.loads((LOG / 'unit-test-result.json').read_text())
mapping = s['mapping_status_counts']
missing_pct = 100 * s['candidate_gene_missing_cells'] / s['candidate_gene_cells']
report = f'''# GSE102971 grouping, QC and gene mapping audit

September 30, 2026. **Completed local audit and candidate gene matrix; no production release or dose-model result.**

## Verified findings

| Item | Result |
|---|---:|
| Human samples retained | {s['retained_samples']} |
| Study-local inferred donor groups | {s['groups']} (10 male, 10 female) |
| Doses per group | 0, 2, 5, 6, 7 Gy; 24-hour response |
| Original non-control physical features | {s['physical_features']:,} |
| Distinct probe names | {s['distinct_probes']:,} |
| Probe names with repeated physical spots | {s['repeated_probe_names']:,} |
| Additional repeated spots | {s['extra_repeated_spots']:,} |
| Probes with an exact current human GeneID | {mapping['current_exact_gene_id']:,} |
| Probes without a source GeneID | {mapping['unmapped']:,} |
| Probes whose source ID is absent from current snapshot | {mapping['absent_from_current_human_snapshot']:,} |
| Mapped current genes before QC | {s['mapped_current_genes_before_qc']:,} |
| Probe rows with updated display symbols | {s['probes_with_updated_symbol']:,} |
| Training-QC-eligible probes | {s['qc_eligible_probes']:,} |
| Mapped, QC-eligible probes | {s['mapped_qc_eligible_probes']:,} |
| Candidate gene rows | {s['candidate_genes']:,} |
| Candidate gene-by-sample cells | {s['candidate_gene_cells']:,} |
| Explicit missing cells | {s['candidate_gene_missing_cells']:,} ({missing_pct:.3f}%) |
| Compressed candidate gene artifact | {s['artifact_files'][0]['bytes']/1e6:.2f} MB |

All raw feature/probe identities and probe sequences match GPL10332 for the retained features. All 100 raw-file hashes were rechecked. No exact duplicate full signal vectors were found; this does not independently verify biological identities. The study-local group linkage is inferred from consistent titles/filenames/doses/gender and the reported study design. A separate donor-key file was not found or obtained. Use the group IDs conservatively for splitting; do not link them to another study by letter.

## Processing decisions actually executed

The prior training-fitted quantile + log2 representation remains the parent. A separate candidate masks explicit technical flags, summarizes repeated spots by median within each sample, applies training-only probe eligibility, resolves exact current human GeneIDs, and takes a second median across distinct probes per gene. Each probe therefore has equal weight regardless of printed spot count. Missing values remain missing; original source and v1 artifacts are unchanged.

The fixed candidate thresholds are ≥95% usable training profiles and ≥20% detected training profiles; gene availability must also be ≥95% in training. These rules were applied to the existing 80-profile training split only, then reused for the 20-profile engineering validation split. They must be refitted inside actual future training folds. They are development choices, not validated universal thresholds. No dose prediction or outcome-based threshold selection occurred.

Explicit technical flags mask {s['technical_defect_cells']:,} physical measurement cells. Background warnings affect {s['background_warning_cells']:,} cells and remain separate diagnostics; the strict background-mask sensitivity is recorded in the summary. There are {s['not_found_cells']:,} `gIsFound=0` cells. A preliminary rule incorrectly grouped this signal-dependent flag with technical damage; inspection of the manufacturer definition led to its removal from the hard mask before this final candidate. Preliminary summary/policy/output are retained under `preliminary-found-mask/` as superseded development evidence, not active results.

All samples remain present; {len(s['samples_flagged_for_review'])} meet the final candidate's sample review prompts. No PCA- or dose-based sample deletion was performed. The absence of a review prompt is not assay certification.

## Verification

- {tests['tests']} automated tests passed, covering flag semantics, missingness, ambiguity rejection, equal probe weighting, training-only fitting and frozen normalization.
- Changing validation measurements/detection flags leaves the trained probe selection unchanged.
- Probe and gene aggregation of the first engineering validation sample alone exactly matches its batch result; synthetic tests also exercise masked/missing values. The pure aggregation functions do not use other samples.
- Study-local groups remain disjoint across the engineering split. This split tests mechanics and does not replace outer study-held-out/inner donor-grouped evaluation.
- Source and artifact sizes/hashes, mapping decisions and all sample flags are recorded in machine-readable files.

## Limits before modeling or loading a published version

1. Technical masking follows the existing quantile transform: flagged values still influenced its reference/ranks. Compare a scientifically appropriate QC-aware normalization strategy before freezing a model pipeline; do not claim this derivative removes all normalization concerns.
2. The {mapping['absent_from_current_human_snapshot']:,} probe mappings absent from current NCBI human records need a retirement/merge-history review if they are to be rescued. The {mapping['unmapped']:,} unmapped probes need authoritative reannotation if included later. Neither category is silently assigned a gene. Current annotation is a snapshot, not a genome realignment or proof of unique binding.
3. Different probes can measure different transcripts; median gene aggregation is a candidate summary. Preserve probe-level evidence for method comparison and discordance review.
4. Remaining gene missingness requires training-fitted model handling; no imputation/scaling/model training has run. Other core-study adapters and cross-platform compatibility remain outstanding. GSE90909 expression was not accessed.

See [policy and loader contract](POLICY.md), [summary](summary.json), [sample grouping](sample-groups.tsv), [sample QC decisions](sample-qc-decisions.tsv), [probe mapping](probe-map.tsv), [gene catalog](gene-catalog.tsv), [threshold sensitivity](threshold-sensitivity.tsv), and [tests](unit-test-result.json).

![Audit overview](audit-overview.png)

## Primary source references

- [GEO human protocol and sample labels](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2751912)
- [GPL10332 feature annotation](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL10332)
- [NCBI human GeneID snapshot](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz), retrieval metadata/hash in `sources/downloads.json`
- [Agilent flag definitions](https://www.agilent.com/cs/library/usermanuals/public/GEN-MAN-G1662-90059.pdf)
- [limma expression-filtering guidance](https://bioconductor.org/packages/devel/bioc/vignettes/limma/inst/doc/usersguide.pdf)

Reproduce with `data/local_runtime/gse102971/bin/python scripts/audit_gse102971.py` using the pinned source snapshots, then render with `scripts/report_gse102971_audit.py`. No cloud resources, paid API calls, database changes or model results are part of this audit.
'''
(LOG / 'REPORT.md').write_text(report)
cards = [('Human profiles', '100'), ('Source groups', '20'), ('Candidate genes', f"{s['candidate_genes']:,}"), ('Missing values', f'{missing_pct:.3f}%')]
card_html = ''.join(f'<div class="card"><span>{k}</span><strong>{v}</strong></div>' for k,v in cards)
mapping_rows = ''.join(f'<tr><td>{html.escape(k.replace("_", " "))}</td><td>{v:,}</td></tr>' for k,v in mapping.items())
page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>GSE102971 — grouping and gene mapping audit</title>
<style>body{{font:16px system-ui;background:#eef3f5;color:#173443;margin:0}}main{{max-width:1350px;margin:auto;padding:36px}}h1{{font-size:32px;margin:8px 0}}.eyebrow{{color:#237b78;font-weight:700}}.cards{{display:flex;gap:16px;margin:24px 0}}.card,section{{background:white;border-radius:12px;padding:20px}}.card{{flex:1}}.card span{{display:block;color:#536c79}}strong{{display:block;font-size:32px;margin-top:8px}}img{{width:100%;margin-top:20px}}section{{margin:20px 0}}table{{border-collapse:collapse;width:100%}}td{{padding:9px;border-bottom:1px solid #dae2e7}}.note{{padding:16px;border-left:5px solid #c28b37;background:#fff9eb;line-height:1.6}}a{{color:#176967}}li{{margin:8px 0}}footer{{color:#5c737e;font-size:13px}}</style>
<main><div class="eyebrow">RADIATION DOSE PREDICTION · LOCAL DEVELOPMENT EVIDENCE · 30 SEPTEMBER 2026</div>
<h1>Grouping, quality flags and gene mapping</h1><p>GSE102971 human subset · Engineering candidate, before model training or cloud publication</p>
<div class="cards">{card_html}</div><div class="note">All 100 profiles retained. Groups are supported by source labels and study design; no separate donor key is available. Unresolved gene IDs remain excluded from this candidate and preserved in the audit.</div>
<img src="audit-overview.png" alt="Gene mapping counts, technical/background flag fractions and training-only threshold sensitivity">
<section><h2>What the checks established</h2><ul><li>All 43,118 physical features match the platform's probe IDs and sequences.</li><li>Training-only selection and single-sample aggregation checks passed; {tests['tests']} unit tests passed.</li><li>Repeated spots collapse to probes first, then probes to genes. Missing values remain explicit.</li><li>Technical flags and signal-dependent detection flags are handled separately; {len(s['samples_flagged_for_review'])} sample review prompts.</li></ul></section>
<section><h2>Mapping accountability</h2><table>{mapping_rows}</table><p>Gene IDs, source symbols and current display symbols are stored separately. The snapshot is pinned by hash.</p></section>
<div class="note">Normalization limitation: technical masks follow the existing frozen-reference transformation. Flagged values still influenced its reference/ranks. QC-aware normalization, unresolved mappings and cross-study compatibility need review before a final model pipeline.</div>
<section><a href="REPORT.md">Full report</a> · <a href="POLICY.md">Policy and loader contract</a> · <a href="summary.json">Machine-readable results</a> · <a href="probe-map.tsv">Probe mapping</a> · <a href="sample-qc-decisions.tsv">Sample QC</a></section>
<footer>Local report only · No paid API calls, cloud deployment, database load or dose-model result</footer></main></html>'''
(LOG / 'report.html').write_text(page)
print(LOG / 'report.html')
