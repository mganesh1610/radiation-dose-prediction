# Team Defender · Radiation dose prediction from blood gene expression

**[Production research workspace · Google Cloud Run](https://defender-research-460004627518.us-west1.run.app/)** · Approved researchers can analyze studies, review uploads, run frozen models and work with the research copilot.

**[Public research dashboard · Cloudflare](https://prod.defender-asu.workers.dev/)** · Open access to the study explorer, model comparisons, published predictions and guest workspace.

Team Defender studies how accurately changes in gene activity in human blood can estimate a known laboratory radiation dose. The project combines six development studies, a separate evaluation study and a web application that lets researchers trace an estimate back to its model and study.

## Research result

On the GSE90909 evaluation cohort, gradient boosting trained on **145 whole-blood profiles and 422 white blood cell (WBC) profiles** had a **donor-balanced mean absolute error of 0.777 Gy**. Each of the 12 donors counted equally across the 52 profiles. The 95% donor-bootstrap interval was **0.550–1.052 Gy**. The whole-blood-only boosting model scored 0.795 Gy and Model B, the best of the four neural approaches on this run, scored 1.160 Gy.

The comparison with added WBC data was retrospective. It used the same previously evaluated profiles, so the small difference between boosting models still needs validation in new donors. These estimates describe research performance at approximately 24 hours after laboratory exposure. Clinical accuracy has not been established.

## Inside the application

| Section | What it shows |
| --- | --- |
| Overview | The six development studies, their roles and the separate final evaluation cohort |
| Datasets | Searchable study profiles, recorded doses and a CSV export of the filtered results |
| Analysis | Principal component plots, heatmaps, correlations, quality checks and clustering. Height in the spatial PCA view represents recorded dose. |
| Experiments | Final evaluation with donor-level error intervals, model cards and a full JSON export. Separate tabs cover development comparisons and training jobs. |
| Predictions | Estimated and recorded dose for curated study profiles. The team workspace also supports reviewed uploads and frozen-model inference. |
| Assistant | Published answers with sources and a guest question workflow. The signed-in workspace adds a Google ADK coordinator, an evidence specialist and controlled research tools. |

The final evaluation table displays eight models. Its download includes all nine model cards, including the whole-blood-only boosting reference. The development table has its own display filters and exports results for all 20 predictors.

## Local setup

### Guest dashboard preview

A local preview uses the interface from this repository and downloads the release-matched records already available on the public dashboard. It requires Python 3.12, Node.js 22 or newer and a network connection during setup. The prediction examples use saved estimates. The assistant’s published answers, charts and model comparisons work with the downloaded release.

```sh
git clone https://github.com/mganesh1610/radiation-dose-prediction.git
cd radiation-dose-prediction
python3.12 scripts/prepare_public_preview.py
npx --yes wrangler@4.147.0 d1 execute GUEST_DB --local --config app/entry-router/wrangler.jsonc --file app/entry-router/guest-schema.sql
npx --yes wrangler@4.147.0 dev --local --config app/entry-router/wrangler.jsonc
```

Wrangler prints the local address. Training jobs and live AI answers require a separately configured worker. The preview needs no cloud account or model weights. [Guest workspace setup](app/entry-router/README.md)

### Scientific source and tests

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pip check
python scripts/run_source_tests.py
node --test tests/*.mjs tests/*.cjs
```

The Python runner isolates modules that use native scientific libraries. Artifact-dependent tests report skips when their required inputs are absent. Windows activation uses `.venv\Scripts\activate`. On macOS, XGBoost can require the OpenMP runtime (`brew install libomp`). [Testing and reproduction](docs/TESTING.md)

The two-agent service has a separate Python environment and an offline test suite.

```sh
python3.12 -m venv .venv-agents
source .venv-agents/bin/activate
python -m pip install -e 'agent_system[dev]'
python -m pytest agent_system/tests
```

## Google Cloud and repository guide

Google Cloud Run hosts the team website, research coordinator and evidence specialist. Google Identity-Aware Proxy checks team access. BigQuery holds study metadata and protocol records. Cloud Storage holds versioned scientific inputs and job artifacts. Cloudflare serves the guest dashboard, and D1 records guest requests. A separate worker processes approved jobs. Deployment requires operator-managed identities, data releases and configuration. [Google Cloud setup](docs/GOOGLE_CLOUD_SETUP.md) · [Configuration reference](docs/CLOUD_CONFIGURATION.md)

| Directory | Contents |
| --- | --- |
| `app/` | Team web API, both dashboard interfaces and the Cloudflare guest router |
| `scripts/` | Data preparation, model training, evaluation, inference and website packaging |
| `agent_system/` | Research agents, evidence tools, access controls and agent tests |
| `schemas/` | BigQuery table definitions |
| `tests/` | Scientific, API, publication and browser checks |
| `docs/` | Scientific scope, data workflow and deployment guides |

This public source release includes the application, tests and setup guides. Raw expression datasets, trained weights, account policies and credentials are provisioned separately. [Data workflow](docs/DATA_WORKFLOW.md) · [Scientific scope](docs/SUBMISSION.md)
