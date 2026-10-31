# Google Cloud setup

Use the [live application](https://prod.defender-asu.workers.dev/) to review the
existing release. This guide describes provisioning a separate deployment in your
own project. It grants no access to the hosted research data.

Source tests run without cloud access. A complete scientific deployment also needs
audited datasets, frozen models, generated frontend exports and protected policies,
which are not distributed in this repository. These are operator instructions,
not a claim that every Google organization/account policy has been deployment-tested.

## 1. Prepare a project

Create/select your project, arrange billing or credits and install the Google
Cloud CLI. An administrator must grant API-enablement, identity-management, build
and deployment permissions. Use a separate project from the hosted application.

```sh
export DEFENDER_GCP_PROJECT="your-project-id"
export DEFENDER_REGION="us-west1"
export DEFENDER_GCS_BUCKET="your-globally-unique-private-bucket"
gcloud auth login
gcloud auth application-default login
gcloud config set project "$DEFENDER_GCP_PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com storage.googleapis.com bigquery.googleapis.com \
  iap.googleapis.com iam.googleapis.com aiplatform.googleapis.com \
  secretmanager.googleapis.com
gcloud projects describe "$DEFENDER_GCP_PROJECT" --format='value(projectNumber)'
```

Record the returned number as `DEFENDER_GCP_PROJECT_NUMBER` in your protected
environment. Use attached service accounts in Cloud Run; do not commit JSON keys.
Configure billing alerts and explicit service limits. Alerts, maximum instances
and application allowances are not hard spend caps. No custom domain or separate
load balancer is required by this architecture.

## 2. Provision storage, metadata and scientific artifacts

```sh
gcloud storage buckets create "gs://$DEFENDER_GCS_BUCKET" \
  --project="$DEFENDER_GCP_PROJECT" --location="$DEFENDER_REGION" \
  --uniform-bucket-level-access --public-access-prevention
bq --location=US mk --dataset "$DEFENDER_GCP_PROJECT:radiation_research_v1"
gcloud artifacts repositories create research-images --repository-format=docker \
  --location="$DEFENDER_REGION"
```

Keep buckets private and choose compatible BigQuery/GCS locations for load jobs.
`schemas/bigquery-v1.json` maps table names to field definitions. Extract one table's
field list before using it with `bq mk --table`; for example:

```sh
python - <<'PY'
import json
from pathlib import Path
schema = json.loads(Path('schemas/bigquery-v1.json').read_text())
Path('/tmp/research-samples-schema.json').write_text(json.dumps(schema['tables']['samples']))
PY
bq mk --table "$DEFENDER_GCP_PROJECT:radiation_research_v1.samples" \
  /tmp/research-samples-schema.json
```

Repeat for each required table using its own field list. Populate them only from
an independently reviewed release. Follow `DATA_WORKFLOW.md` for acquisition,
label audit, donor grouping, preprocessing and release preparation.

Upload the verified matrix and metadata through your approved data channel. Record
object path, generation, SHA-256 and byte size in `DEFENDER_DATASET_REF`. A source
reference does not automatically identify an object in your new bucket or grant
access. Changing the current 145-profile core contract requires coordinated
scientific, schema and code review.

The `expression_sets` table records matrix representation/provenance; it does not
turn an object-store matrix into millions of SQL rows. The Evidence tool reads
the verified matrix, validates sample/gene identifiers and caches bounded reads.

## 3. Create runtime identities

Create separate website, Coordinator and Evidence service accounts. Review the
resource-scoped permissions required for their actual operations:

| Identity | Access |
| --- | --- |
| Website | Dataset/job/control bucket operations; Coordinator invocation |
| Coordinator | Control/identity records, managed Sessions, permitted model usage; Evidence invocation |
| Evidence | Approved matrices, control/artifact records, admitted BigQuery tables plus query-job creation, Sessions and permitted model/embedding usage |
| Optional Mac worker | Approved inputs and engineering job/result updates |

Use dataset-level read grants for metadata and bucket-level grants for objects;
do not use project-owner credentials in serving containers. Optional prediction,
training, preference memory and protocol RAG add their own permission requirements.

Cloud Run distributes requests and scales instances. This implementation does not
require Kafka, Redis, Firestore or an additional load balancer. Instance-local
matrix caches are not the authoritative copy of chat/session data.

## 4. Configure Sessions, preferences and protocol RAG

Create a managed Agent Engine/Agent Platform resource for Sessions in a supported
location. Record its project/location/engine ID and grant the runtime identities
access. The adapters are `persistence/managed_sessions.py` and `memory_bank.py`
under `agent_system/src/research_agents`. Review [managed agent resources](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale)
for current service/regional prerequisites. No chat history or user memory is
included in this public package.

Begin with optional flags disabled. Preferences require Memory Bank access and
an explicit admitted-user list. Protocol RAG additionally requires a reviewed
corpus/manifest and hash, a BigQuery passage table and the configured embedding
model. See `workflows/prepare_protocols.py` and `tools/protocol_corpus.py` and
`protocol_search.py`. Review ambiguous passages before admitting them. Original
study methods are not evidence of this project's preprocessing.

## 5. Prepare protected configuration

Use `.env.example` and `CLOUD_CONFIGURATION.md` for website environment variables.
Agents read JSON from `COPILOT_CONFIG`; a structural template is provided in
[examples/copilot-config.example.json](examples/copilot-config.example.json).
Its placeholders are deliberately non-deployable.

Provision outside Git:

- Coordinator/Evidence config and a membership policy admitting the same workspace
  and release, with your authorized accounts.
- A verified expression-admission document: objects, identities, mappings and
  checksums from your own reviewed release. Pin its hash in configuration.
- Your exact IAP audience, website origin, Evidence URL and Coordinator service
  identity; optional feature manifests only when their feature is enabled.
- Generated `research-results.json`, `development-results.json`,
  `statistical-results.json`, `assistant-records.json` and the audited frozen
  inference bundle required by the web packager.

Contracts are implemented by `policies/runtime.py`, `services/startup_settings.py`
and `services/bootstrap.py`. Identity and release admission must not come from
browser arguments or prompts. Filled configuration, memberships and credentials
must stay out of source control.

## 6. Build explicit private bundles

Do not send the repository root to a cloud builder. Once all required scientific
artifacts are provisioned, run the website allowlist packager:

```sh
python scripts/package_web.py --evidence-dir data/build-receipts
```

The generated `data/web_build` contains the listed app/proxy/frontend files and
validated frozen-model files. Build from that directory using `app/Dockerfile`.
The packaging command intentionally fails if required release files are absent.

For each agent role, prepare a separate private build directory containing only
`agent_system/`, its Dockerfile and your protected `config/` inputs. The supplied
agent Dockerfile expects `config/config.json`. Keep the image and configuration
private. Optional prediction/bridge features also require their audited runtime
source and bundle at the configured paths; setting a feature flag is insufficient.

Build/tag/push images to your Artifact Registry repository. See Google's
[container build guide](https://docs.cloud.google.com/run/docs/building/containers).
This public repository has no automatic deploy-on-push workflow.

## 7. Deploy internal agents and the protected website

Deploy Evidence and Coordinator as IAM-authenticated Cloud Run services. After
building a role-specific image and creating the corresponding service account:

```sh
gcloud run deploy research-evidence --region="$DEFENDER_REGION" \
  --image="$DEFENDER_REGION-docker.pkg.dev/$DEFENDER_GCP_PROJECT/research-images/evidence:v1" \
  --service-account="research-evidence@$DEFENDER_GCP_PROJECT.iam.gserviceaccount.com" \
  --set-env-vars=COPILOT_ROLE=evidence --no-allow-unauthenticated \
  --min-instances=0 --max-instances=1 --timeout=240
```

After obtaining service URLs, configure/rebuild both agent images with the exact
Evidence URL, Coordinator caller identity, website origin and IAP audience.
Deploy Coordinator similarly with `COPILOT_ROLE=coordinator`. Grant website
identity invocation access to Coordinator and Coordinator identity access to
Evidence. For example:

```sh
gcloud run services add-iam-policy-binding research-evidence \
  --region="$DEFENDER_REGION" \
  --member="serviceAccount:research-coordinator@$DEFENDER_GCP_PROJECT.iam.gserviceaccount.com" \
  --role=roles/run.invoker
```

The proxy uses a signed service ID token internally and carries the original
signed IAP assertion separately. The backend rechecks audience, issuer, current
membership and release grants. See [service-to-service authentication](https://docs.cloud.google.com/run/docs/authenticating/service-to-service).

Create the website with **Require authentication → Identity-Aware Proxy** in the
Cloud Run Console. CLI deployments can use `--iap` and
`--no-allow-unauthenticated` after initial OAuth/IAP setup. Grant the IAP service
agent invocation permission, and approved reviewers IAP access to the website.
External accounts may need a custom OAuth client or organization approval. Obtain
the exact assertion audience for your service; set `DEFENDER_IAP_AUDIENCE` and the
same audience in agent config. Follow [Google's IAP setup instructions](https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run).

Set website `COPILOT_BACKEND_URL` to your Coordinator URL. Keep
`TRAINING_ENABLED=false` until the worker, input contract and bounded allowances
are ready. App membership checks apply in addition to Google IAP.

## 8. Verify before sharing

1. Anonymous website access requires sign-in; unauthenticated internal-agent
   invocation is rejected.
2. Approved users can access only their own chats, preferences, uploads and
   artifacts; revoked/unauthorized users are denied.
3. Sample/count/expression answers match the admitted release; citations support
   the answer and missing data stays missing.
4. Request IDs, retries, cancellation, quotas and startup recovery work as scoped.
   A fast status response is not the same as a fast model answer.
5. The consumed final study stays excluded from retraining/tuning. A deployment
   smoke test is not independent clinical or scientific validation.

See `app/REVIEWER_DEMO.md` and `docs/TESTING.md`. Changes to this separate deployment
do not update the shared live app. A separate public static explorer additionally
needs a reviewed generated export; see `app/public/vercel.json` and
`app/entry-router/README.md`. Publishing source alone does not supply that data.
