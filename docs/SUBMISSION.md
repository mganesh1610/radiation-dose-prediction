# Scientific and application scope

The research question is whether eligible auxiliary lymphocyte data help radiation
dose estimation in human whole blood after laboratory photon exposure around
24 hours. Study effects, assays and biological conditions must remain explicit.

| Strategy | Implemented training design |
| --- | --- |
| A | Core whole-blood dose regression |
| B | Auxiliary exposure pretraining, followed by core dose regression |
| C | Exposure and dose heads, with dose loss restricted to eligible core labels |
| D | Dose regression with verified auxiliary labels and source/sample weighting |

The development release contains 145 core profiles and 422 eligible auxiliary
profiles. Approximately 30 million initial expression values are measurements,
not independent training examples. Repeated profiles remain grouped by donor;
study-local donor IDs do not prove globally unique individuals.

Preprocessing and selection are fitted within training partitions. Matched frozen
A–D and classical reference implementations are included, with broader model and
clustering comparisons retained as development exploration. Exploratory chart
normalization is separate from predictive evaluation. Technical flags mask affected
measurements; PCA separation does not automatically delete samples.

GSE90909 photon/control evaluation used 52 profiles from 12 donors after model and
analysis settings were frozen. That test is consumed. The planned superiority
claim for A was not supported; do not use these results to retune models or claim
clinical validity. Study bias, limited dose coverage and uncertainty remain.

The application supports scientific exploration and restricted research prediction.
Fresh uploads require assay, normalization, gene mapping, tissue/timing and QC
compatibility review. No calibrated individual prediction interval is supplied.
Real demonstration profiles that overlap training are labelled accordingly.

The deployed training button runs a bounded fixed-preset engineering job on the
Mac worker. It does not rerun the final experiment or replace frozen predictors.
The public structured assistant returns approved records through deterministic
matching. The authenticated research copilot separately uses an ADK Coordinator
and remote A2A Evidence Specialist with controlled tools and cited retrieval.
Source tests and live human answer review are distinct forms of verification.

Repository tests establish code behavior, not end-to-end clinical performance.
See the hosted experiment view for published metrics and model cards.
