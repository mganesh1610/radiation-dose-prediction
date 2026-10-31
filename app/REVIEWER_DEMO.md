# Reviewer demonstration

Share https://prod.defender-asu.workers.dev/ as the public entry link.
It checks Vercel first and temporarily redirects to the Firebase backup if the
Vercel check fails. Direct URLs remain https://defender-research-public.vercel.app/
and https://radiation-prediction-4ffe5.web.app/. Keep both as submission backup
links, also available at https://prod.defender-asu.workers.dev/links.
Each public site links to the private Cloud Run training workspace. Both show
reviewed snapshots, not newly submitted jobs.

The router runs on Cloudflare Workers Free. It checks the homepage and publication
manifest, with a two-second deadline per origin. Successful checks may be cached
for 30 seconds in a Worker isolate; failed checks for five seconds. It does not
continuously monitor already-open pages, proxy assets, or route training requests.
The original Vercel URL does not gain failover; visitors must use the entry link.
Use `/analysis` or `/#analysis` on the entry link to open that section.
See [entry-router/README.md](entry-router/README.md) for validation and maintenance.

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
