# Testing and reproduction

Run the README commands for source-level checks with Python 3.12. Tests cover
normalization, donor grouping, weighting, neural objectives, estimator reloads,
statistics, queue races, authentication, input contracts and routing. Small fixtures
are numerical unit tests, not fabricated research datasets or app demos.

## Artifact-dependent tests

Core analysis checks need the reviewed core release. Prediction workflow checks
need the audited frozen bundle. Publication/development checks need their verified
release inputs and generated exports; locked metadata checks need the original
public metadata source. The assistant regression cases need its approved records.
These files are intentionally excluded from Git. Missing artifacts produce explicit
skips, while available artifacts still undergo their normal integrity checks.

```sh
DEFENDER_REQUIRE_DATA_TESTS=1 python scripts/run_source_tests.py
DEFENDER_REQUIRE_DATA_TESTS=1 node tests/test_public_data.cjs
DEFENDER_REQUIRE_DATA_TESTS=1 node tests/test_assistant_records.cjs
```

Strict mode fails when artifacts are missing. Do not relax checksums, fabricate
measurements or add private provenance to Git to make checks pass. The public
adapter and assistant checks are also artifact-dependent, so their default skip
is not a passing scientific validation.

Typical local artifact locations:

- `data/cloud_exports/<release-id>/core.npz`: verified core matrix.
- `data/cloud_prediction/frozen-v1/`: numeric models, fixed preprocessing, manifest
  and audited real demonstration profiles.
- `data/public_web/`, `data/public-build-manifest.json`: reviewed public build.
- `data/reviewer_release/`: curated frozen-result export.
- Generated `app/static/*-results.json` and `assistant-records.json`.

Historical research runners and release builders also expect separately reviewed
label manifests and frozen local audit records. They are provided as implementation
source, not a one-command reconstruction of all private records. A reproducibility
handoff needs a separately reviewed artifact package through the authorized data
channel. Do not rerun or retune the consumed final test during ordinary testing.

R preprocessing additionally uses the Bioconductor packages imported in the `.R`
scripts; Python dependencies alone do not reproduce raw-array processing. Cloud
permissions, reviewer identity, worker availability and quotas require separate
live checks. Source-only test success does not establish those conditions.

Install `requirements-test.txt` in a separate test environment. The runner uses
pytest so both unittest classes and pytest functions execute, in a fresh Python
process per module because PyTorch and
XGBoost can load incompatible OpenMP runtimes into a shared macOS process. This
keeps the native runtimes isolated without disabling their safeguards. The runner
reports any failing module and preserves explicit artifact skip messages.
