# Google Cloud entry router

Share **https://prod.defender-asu.workers.dev/**. The Worker redirects directly to
**https://defender-research-460004627518.us-west1.run.app/**, where Google Cloud
handles sign-in and the existing authorized-user checks. The dashboard, Analysis,
predictions and research copilot use that single application.

This replaces the earlier Vercel/Firebase public-viewer failover. Neither static
hosting service is a routing destination. The optional `/links` page has only the
Google Cloud application link. The older Vercel and Firebase projects are not
deleted by this routing change.

## Behavior

- GET/HEAD navigation returns a noncacheable 302 to the fixed Google Cloud origin.
- `/overview`, `/datasets`, `/analysis`, `/experiments`, `/predictions` and
  `/assistant` map to the application's corresponding hash routes, with optional
  trailing slashes. Browsers retain a hash when the root redirect omits one.
- `/links` explains access and provides the direct application URL; `/robots.txt`
  discourages indexing. Unknown paths return 404 and write methods return 405.
- The Worker makes no upstream requests and does not proxy APIs, forward caller
  cookies/authorization/query parameters, store sessions, or bypass Google sign-in.
  Direct application links can still include the application's own query options.

The existing Workers configuration, account namespace and `prod` Worker name are
unchanged. No paid plan, storage, secret, cron, domain purchase or new binding is
needed. Once redirected, the browser address displays the Google Cloud hostname.

## Verify and deploy

From the repository root:

```sh
node --test tests/test_entry_router.mjs
```

Publish only the tested `worker.mjs` to the existing Cloudflare **prod** Worker,
using an authorized dashboard session or separately authorized Wrangler login.
Never upload the repository root, `context/`, `AGENTS.md`, datasets or credentials.
Retain the prior deployed Worker version for rollback. Confirm the live root and
all section redirects, and verify that signed-out access still requires Google
sign-in. No Google Cloud access settings need to change.
