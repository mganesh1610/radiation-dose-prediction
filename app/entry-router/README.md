# Cloudflare guest dashboard

Cloudflare serves the public dashboard and its versioned scientific records. The team sign-in route opens the Google Cloud Run workspace. The [project README](../../README.md) lists both hosted sites and shows the local preview commands.

The guest dashboard supports all six workspace sections, published profile estimates and source-backed answers. A D1 database stores each guest's queued training and AI requests. An HTTP-only session cookie limits access to that browser's records. The team site handles authenticated uploads, private conversations and project administration.

## Local development

The preview builder checks the publication, all 30 scientific records and the saved profile estimates against the release expected by this source. It composes the website from the repository templates, scripts, styles and media. Generated files stay in `data/shared_site/`. Each build uses a new directory, and existing builds can be retained by selecting another output with `--output`.

The checked-in Wrangler configuration targets a local D1 binding named `GUEST_DB`. The local database uses `guest-schema.sql`. Cloud deployment requires a separate configuration with the operator's database ID. Without a connected worker, submitted training and AI requests remain queued. Published answers and saved predictions remain available.

## Scientific release packaging

Operators with the audited inputs can generate the reviewed research export with `scripts/build_public.py` and package it with `scripts/package_shared_web.py --output data/shared_site`. The packager verifies the public manifest and uses frozen models to calculate the curated profile estimates. The static asset directory contains the published results. Model weights and raw model inputs stay in the protected environment.

## Guest job processing

The worker consumer uses the release-pinned training pipeline and the agent budget ledger. The Cloudflare binding `GUEST_KEY_SHA256` holds the fingerprint of its bearer credential. The consumer loads the credential and operator settings separately. Completed jobs remain scoped to the originating guest session under the configured retention policy. [Cloud and access configuration](../../docs/CLOUD_CONFIGURATION.md)
