# Reviewer demonstration

Open the [production research workspace](https://defender-research-460004627518.us-west1.run.app/) with an approved Google account.
The [Analysis link](https://defender-research-460004627518.us-west1.run.app/#analysis) opens the analysis section directly.

Visitors without project access can explore the [guest dashboard](https://prod.defender-asu.workers.dev/).
The [Vercel snapshot](https://defender-research-public.vercel.app/) and
[Firebase snapshot](https://radiation-prediction-4ffe5.web.app/) provide
read-only copies of published results. See the [entry-router guide](entry-router/README.md)
for routing details.

Invited reviewers sign in with their approved Google account, open Experiments,
select a baseline and held-out development study, and choose **Train model**.
**Refresh status** checks progress; **View results** shows the metrics, plot,
predictions and downloadable provenance after publication.

Reviewer admission allows two engineering runs and 150 API requests per UTC
month, subject to the shared project caps of 12 runs and 600 API requests.
Cancelled and failed runs consume an admission. Retries reuse the request ID.
These application limits are not a Google Cloud billing cap. Only the run owner
can read or cancel that run. IAM access and the application allowlist must both
contain each invited account; no project-administrator access is needed.

## Prediction and analysis review

Open **Predictions** in the authenticated workspace. Choose a supported adapter
JSON profile or expand the real-study example list. Select **Upload and check**,
review compatibility/QC, choose a frozen model, then **Run research prediction**.
The export includes the full estimate, model version and input provenance. A new
unreviewed profile stays pending until a trusted researcher reviews compatibility;
reviewers cannot approve their own uploads. Demonstration profiles overlapping
training are labelled and are not independent accuracy tests.

Use public **Analysis** for saved PCA, heatmap, correlation, QC, clustering and
within-study dose-response/contrast views. **Experiments** separates development
comparisons from the consumed final test and provides model cards. The source-linked
assistant answers from approved records and identifies missing information.

## Before a scheduled demonstration

Keep the project Mac awake, connected, and logged in. Ensure the existing gcloud
session is valid. Start a bounded fast pickup window from the project directory:

```sh
python3 scripts/install_mac_worker.py demo --minutes 120
```

The worker checks locally once per minute, polls cloud state every ten minutes by
default, and every minute during the selected window. Fast mode expires after
at most two hours. Actual cloud polls remain limited to 4,500/month, and execution
attempts to 36/month. An expired login, sleeping Mac, exhausted allowance, or
network failure can delay work. The app displays the last check-in rather than
promising the Mac is available now. Detailed worker logs stay local under
`data/worker_service/`.

Deployment environment variable `TRAINING_ENABLED=false` pauses new submissions
while retaining access to existing results. Re-enable it after the reason for a
pause is resolved. Do not bypass the shared caps for a demonstration.

## Scope

Median, Elastic Net, PLS and gradient boosting use fixed engineering settings on
the 145-profile core development release. GSE90909 remains excluded. This checks
the cloud-to-Mac workflow; it does not establish final model selection or clinical
accuracy. A professor or TA needs their own invitation before live training.

## Refreshing the public backup

Build a reviewed snapshot with `scripts/build_public.py`, then use
`scripts/package_firebase.py --manifest <build-allowlist.json> --project <hosting-project>` to prepare only the
approved static files in `data/firebase_deploy`. Deploy **hosting only** to
`radiation-prediction-4ffe5`. Keep the Spark plan; do not add backend rewrites or
deploy the repository root. Verify file hashes on both public origins after each
release. Cloud Run, the worker and stored data are independent of static Hosting.
