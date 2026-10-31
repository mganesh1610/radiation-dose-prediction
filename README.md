# Radiation Dose Prediction from Human Gene Expression

## Open the live application

**[Launch the research explorer →](https://prod.defender-asu.workers.dev/)**

Explore the datasets, PCA, clustered heatmaps, model comparisons and published
findings in your browser. No local installation is needed for the public explorer.

- [Direct public site](https://defender-research-public.vercel.app/)
- [Backup public site](https://radiation-prediction-4ffe5.web.app/)
- [Alternate entry links](https://prod.defender-asu.workers.dev/links)
- [Private research workspace](https://defender-research-460004627518.us-west1.run.app/#assistant)

Cloud-tool chat, uploads, predictions and Mac training require a separately
approved Google sign-in. Repository access does not grant application access.
The public site is a published research snapshot; its structured assistant and the
authenticated multi-agent chatbot are different interfaces. Cloudflare selects a
public origin when you open the entry link; it does not proxy private API traffic.

**Prefer to run the code?** Start with [local setup](#local-setup). To provision
your own infrastructure, follow [Google Cloud setup](docs/GOOGLE_CLOUD_SETUP.md).

## About the project

Research software for radiation-dose prediction from human blood gene-expression
profiles, combining donor-aware machine learning, a scientific dashboard and an
authenticated Google ADK research copilot.

- **ML pipeline:** raw-array processing, assay QC, frozen normalization,
  probe-to-gene mapping, training-only feature selection and model export.
- **Model comparison:** matched A–D neural strategies, classical regressors and
  exploratory clustering, with study holdouts and donor grouping.
- **Scientific interface:** dataset filters, automatic PCA, hierarchical heatmaps,
  correlations, dose responses, model cards and separate development/final views.
- **Agent system:** Research Coordinator and remote Evidence Specialist over A2A,
  controlled SQL, verified matrix retrieval/cache, managed Sessions, explicit
  preference memory and cited protocol retrieval.
- **Application engineering:** Cloud Run APIs, Cloud Storage coordination,
  BigQuery metadata, compatibility-gated prediction and bounded Mac training.

The primary frozen A–D comparison did not support Model A superiority. The
GSE90909 final evaluation is consumed and must not be reused for model tuning.
These are research estimates, not clinically validated dose measurements.

## Local setup

### 1. Clone and create an isolated source-test environment

Use Python **3.12** and Node.js **20 or newer**. Keep any existing scientific or
training environment unchanged.

```sh
git clone https://github.com/mganesh1610/radiation-dose-prediction.git
cd radiation-dose-prediction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pip check
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. On macOS, XGBoost may
need OpenMP (`brew install libomp`). The test dependencies include PyTorch and
XGBoost; the hosted application is the quickest way to review research results.

### 2. Run the source checks

```sh
python scripts/run_source_tests.py
node --test tests/test_entry_router.mjs
node tests/test_public_data.cjs
node tests/test_assistant_records.cjs
node tests/test_heatmap_clustering.cjs
node tests/test_analysis_autoload.cjs
```

Python modules run separately to avoid native PyTorch/XGBoost runtime collisions.
Tests requiring excluded scientific artifacts explicitly skip when those artifacts
are absent. This is not scientific validation; use `DEFENDER_REQUIRE_DATA_TESTS=1`
only after provisioning the audited release. These checks do not deploy services,
train new research models or call the paid model service. See [testing](docs/TESTING.md).

### 3. Run the offline agent checks in a separate environment

```sh
deactivate
python3.12 -m venv .venv-agents
source .venv-agents/bin/activate
python -m pip install -e 'agent_system[dev]'
python -m pip check
python -m pytest agent_system/tests
```

The offline agent suite uses controlled fixtures and loopback services. It verifies
software behavior, not live answer accuracy or capacity. Platform lock files and
runtime entry points are in [agent_system/README.md](agent_system/README.md).

### 4. Understand what a local clone can run

This repository contains implementation and tests. Datasets, trained models,
generated record/result exports, real membership policies and credentials are
provisioned separately. A clone can run the source checks; it is not a complete
local copy of the hosted scientific release.

For a complete application, an authorized operator must supply the audited data,
frozen inference bundle, generated frontend exports and protected configuration.
The API deliberately rejects requests without valid Google IAP identity and
allowlist membership. Do not disable those checks to make a preview work.

Read [the data workflow](docs/DATA_WORKFLOW.md), [configuration reference](docs/CLOUD_CONFIGURATION.md)
and [Google Cloud setup](docs/GOOGLE_CLOUD_SETUP.md) before starting a cloud-backed
application. Copy `.env.example` locally for your own values; the application does
not load it automatically.

## Google Cloud deployment

Three Cloud Run services host the website, Coordinator and Evidence Specialist.
Google IAP protects browser access; service identities protect internal hops.
BigQuery stores metadata and protocol passages; Cloud Storage stores scientific
objects, control records and private artifacts. Managed Sessions and Memory Bank
provide chat history and explicit preferences.

The [Google Cloud guide](docs/GOOGLE_CLOUD_SETUP.md) covers project/API setup,
storage, warehouse schemas, service identities, protected configuration, container
packaging, IAP and verification. It uses your own resource placeholders and grants
no access to the hosted project's data. Cloud provisioning can incur charges;
quotas and application allowances are not provider billing caps.

## Repository map

| Path | Purpose |
| --- | --- |
| `app/` | Authenticated web API, frontend, static adapters and entry router |
| `scripts/` | Acquisition/QC, preprocessing, models, inference, workers and exports |
| `agent_system/` | Two-agent runtime, retrieval tools, access controls and tests |
| `schemas/` | Versioned BigQuery table definitions |
| `tests/` | Source, authentication, queue, prediction and publication checks |
| `docs/` | Scientific scope, data workflow, cloud setup and reproduction limits |

## Publication and reproducibility

This is an independent public source release. Ongoing development is maintained
separately. No old pull requests or private development history were imported.
Publish selected, reviewed releases here rather than automatically mirroring
private development history.

Credentials, access lists, datasets, model artifacts, conversation memory, local
reports, provenance and `AGENTS.md` files are excluded. Original proposal and
presentation submissions are distributed separately. Public service URLs are
navigation links, not access credentials.

Root `vercel.json` disables automatic Git deployments. Publish the reviewed static
export from its own prepared directory so a source push cannot replace the live
dashboard with an incomplete checkout. Do not deploy the repository root.
See [Cloudflare routing](app/entry-router/README.md), [scientific scope](docs/SUBMISSION.md)
and [reviewer walkthrough](app/REVIEWER_DEMO.md).
