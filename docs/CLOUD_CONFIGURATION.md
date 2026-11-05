# Cloud configuration

No credentials are embedded in source. Public project/service identifiers are
not passwords; access is enforced independently by IAM, IAP and the app allowlist.
Use environment variables from `.env.example`. The file is a template, not an
automatically loaded secret store. Keep filled files and tokens outside Git.

| Setting | Purpose |
| --- | --- |
| `DEFENDER_GCP_PROJECT` | Google Cloud project ID |
| `DEFENDER_GCP_PROJECT_NUMBER` | Numeric owner checked by bucket preflight |
| `DEFENDER_GCS_BUCKET` | Private dataset/job bucket |
| `DEFENDER_IAP_AUDIENCE` | Exact expected Cloud Run IAP audience |
| `ALLOWED_EMAILS` | Comma-separated approved sign-in accounts |
| `REVIEWER_EMAILS` | Approved accounts receiving reviewer limits |
| `TRAINING_ENABLED` | New job admission switch |
| `DEFENDER_DATASET_REF` | Optional verified core object/generation/SHA/size JSON |
| `DEFENDER_PUBLIC_RESULT` | Local reviewed engineering-result JSON for public export |
| `DEFENDER_FIREBASE_PROJECT` | Separate static Hosting project for packaging |
| `FROZEN_MODEL_DIR` | Local path to the verified frozen inference bundle |
| `CLOUDFLARE_ACCOUNT_ID` | Account selected for authorized Wrangler deployment |
| `COPILOT_BACKEND_URL` | Website proxy destination: private Coordinator service URL |
| `COPILOT_CONFIG` | Protected agent configuration file; never commit filled values |
| `COPILOT_ROLE` | Agent container role: `coordinator` or `evidence` |
| `DEFENDER_REVIEWER_MONTHLY_API_LIMIT` | Website reviewer API guard (default 150; maximum 750) |

For project provisioning, IAM/IAP, resource admission and deployment order, use
the [Google Cloud setup guide](GOOGLE_CLOUD_SETUP.md).

The current code pins a core release reference, which is metadata and grants no
access. A separate deployment must use its own verified uploaded generation and
matching hash/size, supplied through `DEFENDER_DATASET_REF`. Do not change the
145-profile contract without a scientific and schema review.

The Google Cloud adapter uses an existing local gcloud login or the Cloud Run
service identity. Do not commit service-account JSON keys. Both the Mac worker
and Cloud Run process need the configured project variables. Missing cloud
configuration stops the adapter before it retrieves a token or makes a request.

Build Cloud Run's explicit bundle with `scripts/package_web.py`, never the whole
repository. Use `app.server:app` as the production entrypoint and configure IAP
and the application allowlist. Reviewer access is granted separately; source
access does not grant IAM rights. Train on the Mac, not inside request handlers.

Firebase packaging accepts `--project` (or the environment setting) and deploys
only the reviewed static allowlist. Vercel and Firebase copies do not expose the
private API. The Cloudflare entry redirects directly to the Google Cloud app,
including its research copilot. Google sign-in and the existing application
allowlist remain required. Vercel and Firebase are separate static snapshots,
not Cloudflare routing destinations. The Worker is not a backend proxy or an
outage guarantee. Existing application limits are operational guards, not
cloud-provider billing caps.

The prediction API lazily loads the approved numeric bundle from `FROZEN_MODEL_DIR`
(default `runtime/frozen`). `scripts/package_web.py` validates the source bundle
under `data/cloud_prediction/frozen-v1` and adds only its manifest-listed files to
the deployment package. Missing bundles fail closed. This package contains protected
model/input artifacts and belongs only in the authenticated cloud deployment,
not the static viewer or source repository.
