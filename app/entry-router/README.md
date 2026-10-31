# Public entry router

Share https://prod.defender-asu.workers.dev/. The optional `/links` page
always serves the two direct site links without making origin requests.

The account namespace was renamed to `defender-asu` and the Worker to `prod` on
October 2, 2026. Earlier entry URLs no longer route to this Worker; replace
previously shared links with this address. The Worker code and origins are unchanged.

This is a Cloudflare Workers Free navigation redirect, not a reverse proxy.
Vercel is preferred; Firebase Hosting is the backup. Neither backend requests
nor training jobs pass through it. No domain purchase, paid plan, Load Balancing,
storage binding, secret, Git integration or scheduled polling is required.

On GET/HEAD navigation, the Worker checks an origin's homepage and small public
publication manifest in parallel. It requires HTTP 200, expected content types,
the Defender page markers, and a valid public-release manifest. It does not follow
origin redirects. An origin gets two seconds including response-body reading.
Bodies are limited to 128 KiB for HTML and 4 KiB for the manifest. Failures cause a
check of the second origin. Both failing produces a 503 page with retry and direct
links. Successful navigation uses a noncacheable 302. No caller cookies, tokens,
query parameters or destination URLs are forwarded. Unknown paths are 404; write
methods are 405. Known section paths map to the public app's hash routes. Browsers
also preserve a fragment on the root entry URL when following the redirect.

In-memory health records last 30 seconds when healthy, five seconds when failing.
Records are per isolate, so this is not a shared or guaranteed global cache. An
outage can remain undetected for those 30 seconds. A cold successful entry uses
two origin requests; testing both origins uses at most four. No origin is polled
when there are no visitors. The checks do not prove every asset or chart works.

Workers Free currently allows 100,000 account-wide requests/day and 10 ms CPU per
invocation; it can reject requests at its limits. Origin quotas are separate.
Cloudflare itself remains a dependency, so keep the two direct links. This router
does not switch a page that is already open or rescue visits to the old Vercel URL.
Some automated clients may be rejected by Cloudflare before the Worker: the
default Python urllib client received 403/error 1010 during verification, whereas
curl and public browser navigation without app sign-in succeeded. No security controls
were disabled to change that behavior.

## Test and deploy

From the repository root:

```sh
node --test tests/test_entry_router.mjs
npx --yes wrangler@4.147.0 deploy --dry-run --config app/entry-router/wrangler.jsonc --outdir ../../data/entry-router-build
npx --yes wrangler@4.147.0 dev --config app/entry-router/wrangler.jsonc --local --port 8789
```

`wrangler.jsonc` describes only this Worker. CLI deployment requires a separately
authorized Cloudflare login; the initial deployment used the already-signed-in
dashboard, pasted the exact tested `worker.mjs`, verified editor contents, and
deployed it. No CLI OAuth or API token was created. Never deploy the repository
root or upload `context/`, `AGENTS.md`, datasets, or credentials as static assets.

Dashboard Worker: `prod`. Select your own account through Wrangler's authorized login
or the `CLOUDFLARE_ACCOUNT_ID` environment variable before deploying.
Workers Logs remain at the dashboard default (enabled); tracing is disabled.
The code logs only probe exceptions, with the fixed origin name and error.

After a public-site update, verify both homepage and publication manifest remain
compatible and that both public copies are refreshed. If publication schema,
origin hostname or page markers change, update `worker.mjs`, run tests, and publish
the reviewed version. Keep the routing limits and the Free plan unchanged.
Do not roll back to the initial Hello World scaffold; redeploy this source when
recovering the first release. Once later releases exist, use a verified earlier
router version from Cloudflare's deployment history for rollback.

References: [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/),
[Workers limits](https://developers.cloudflare.com/workers/platform/limits/).
