# Data workflow

Repository data references point to public research sources. Original downloads,
processed matrices, reviewed label manifests, models and local provenance are
excluded from Git. Downloading public data does not make every sample eligible.

| Role | Public accessions |
| --- | --- |
| Core development | [GSE102971](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE102971), [GSE8917](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8917), [GSE65292](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65292) |
| Audited auxiliary development | GSE85570, GSE97000, normal-gravity GSE20173 subset |
| Consumed final evaluation | GSE90909 photon/control subset |
| Conditional testing | GSE15341, GSE274788, GSE40640, Zenodo record 22013472 |

1. `acquire_geo.py`, `audit_geo.py`, `select_core.py` download and inspect source
   metadata. They write local ignored records and must be invoked deliberately.
2. Raw acquisition and preprocessing scripts operate on reviewed sample labels,
   source file inventories and platform annotations. Review organism, tissue,
   time, dose, donor grouping and duplicates before constructing these manifests.
   Some historical manual-audit inputs are not generated automatically by this
   repository; do not treat acquisition alone as a complete reproducible release.
3. Technical flags, per-array background correction and probe-to-gene mapping
   feed the fold-local transfer pipeline. `background_core.R` uses Bioconductor
   limma. The downstream code keeps original and transformed representations
   separate and verifies expected dimensions and identities.
4. `run_core_baselines.py` performs held-out-study development evaluation with
   preprocessing refitted within inner training folds. `training_worker.py`
   provides bounded fixed-preset engineering runs for the live app.
5. `export_training_release.py` checks and packages versioned core metadata;
   `cloud_io.py` uses object generation/hash checks and BigQuery load ledgers.

Full biological reproduction requires independently preparing or securely
provisioning the verified input artifacts. Share approved research datasets
through the project data channel, not by adding excluded local history or raw
files to this repository. For grading the implemented live workflow, use the
reviewer application access described in the README.

The research implementation now includes `run_abcd_experiments.py`,
`run_matched_classical.py`, `run_extended_models.py`, `run_clustering.py`,
`run_final_selection.py` and `run_locked_evaluation.py`. These deliberately check
frozen preparation records and stop on mismatches; do not bypass their integrity
or one-time evaluation guards. New changes informed by the final evaluation are
exploratory and require independent validation.

`export_frozen_inference.py` converts the approved release to numeric serving
artifacts; `frozen_inference.py` applies saved transformations without fitting.
`prediction_service.py` enforces input compatibility, ownership and review state.
The four static result/record JSON exports are generated data, excluded here.
