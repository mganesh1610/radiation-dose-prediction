import test from "node:test";
import assert from "node:assert/strict";
import { createRouter, ORIGINS } from "../app/entry-router/worker.mjs";

const manifest = { schema_version: 1, locked_test_included: false, profiles: 145,
  analysis_combinations: 24, release_id: "core_fdf0d10c09a29915", data_path: "/data/c1f04d5523054f567e09" };
const good = url => url.endsWith(".json")
  ? Response.json(manifest)
  : new Response('<title>Defender · Overview</title><a data-route="analysis">Analysis</a>', { headers: { "Content-Type": "text/html" } });
const request = (path = "/", options) => new Request(`https://entry.example${path}`, options);

test("healthy primary redirects without forwarding query, cookies or authorization", async () => {
  const calls = [];
  const route = createRouter({ fetcher: async (url, opts) => { calls.push({ url, opts }); return good(url); } });
  const res = await route(request("/?next=https://evil.example&token=private", { headers: { Authorization: "secret", Cookie: "secret" } }));
  assert.equal(res.status, 302);
  assert.equal(res.headers.get("location"), `${ORIGINS[0].url}/`);
  assert.equal(res.headers.get("cache-control"), "no-store");
  assert.equal(res.headers.get("referrer-policy"), "no-referrer");
  assert.equal(calls.length, 2);
  for (const { url, opts } of calls) {
    assert.ok(url.startsWith(ORIGINS[0].url));
    assert.equal(opts.redirect, "manual");
    assert.equal(opts.headers.Authorization, undefined);
    assert.equal(opts.headers.Cookie, undefined);
  }
});

for (const [name, failure] of [
  ["HTTP 503", () => new Response("unavailable", { status: 503 })],
  ["HTTP 429 quota", () => new Response("limited", { status: 429 })],
  ["login redirect", () => new Response(null, { status: 302, headers: { Location: "https://login.example" } })],
  ["network exception", () => { throw new Error("connection closed"); }],
  ["HTML challenge", () => new Response("challenge", { headers: { "Content-Type": "text/html" } })],
  ["bad JSON", url => url.endsWith(".json") ? new Response("{", { headers: { "Content-Type": "application/json" } }) : good(url)],
  ["wrong publication", url => url.endsWith(".json") ? Response.json({ ...manifest, locked_test_included: true }) : good(url)],
  ["wrong homepage", url => url.endsWith(".json") ? good(url) : new Response("other app", { headers: { "Content-Type": "text/html" } })],
  ["oversized streamed body", url => url.endsWith(".json") ? new Response("x".repeat(4097), { headers: { "Content-Type": "application/json" } }) : good(url)],
]) {
  test(`falls back for primary ${name}`, async () => {
    const route = createRouter({ fetcher: async url => url.startsWith(ORIGINS[0].url) ? failure(url) : good(url) });
    const res = await route(request("/analysis"));
    assert.equal(res.status, 302);
    assert.equal(res.headers.get("location"), `${ORIGINS[1].url}/#analysis`);
    assert.equal(res.headers.get("x-defender-destination"), "firebase");
  });
}

test("timeout aborts primary and uses backup", async () => {
  const signals = [];
  const route = createRouter({ timeoutMs: 15, fetcher: async (url, opts) => {
    if (url.startsWith(ORIGINS[0].url)) { signals.push(opts.signal); return new Promise(() => {}); }
    return good(url);
  } });
  assert.equal((await route(request())).headers.get("x-defender-destination"), "firebase");
  assert.ok(signals.every(signal => signal.aborted));
});

test("both unavailable returns accessible 503 with direct links and no automatic loop", async () => {
  const route = createRouter({ fetcher: async () => new Response(null, { status: 503 }) });
  const res = await route(request());
  assert.equal(res.status, 503);
  assert.equal(res.headers.get("location"), null);
  assert.equal(res.headers.get("retry-after"), "30");
  const html = await res.text();
  for (const origin of ORIGINS) assert.ok(html.includes(origin.url));
  assert.ok(!html.includes("<script"));
});

test("cache expires and recovers primary without sticking to backup", async () => {
  let time = 0, broken = false, count = 0;
  const route = createRouter({ now: () => time, fetcher: async url => {
    count++; return broken && url.startsWith(ORIGINS[0].url) ? new Response(null, { status: 503 }) : good(url);
  } });
  assert.equal((await route(request())).headers.get("x-defender-destination"), "vercel");
  broken = true; time = 29999;
  assert.equal((await route(request())).headers.get("x-defender-destination"), "vercel");
  assert.equal(count, 2);
  time = 30000;
  assert.equal((await route(request())).headers.get("x-defender-destination"), "firebase");
  broken = false; time = 35000;
  assert.equal((await route(request())).headers.get("x-defender-destination"), "vercel");
});

test("unknown paths, API requests and write methods cannot reach origins", async () => {
  const route = createRouter({ fetcher: async () => { throw new Error("must not fetch"); } });
  for (const path of ["/api/jobs", "//evil.example", "/https://evil.example", "/favicon.ico", "/%2fanalysis"]) {
    assert.equal((await route(request(path))).status, 404);
  }
  for (const method of ["POST", "PUT", "DELETE", "OPTIONS"]) assert.equal((await route(request("/", { method }))).status, 405);
  const res = await route(request("/links"));
  assert.equal(res.status, 200);
  assert.ok((await res.text()).includes("One entry link"));
});

test("HEAD has matching redirect with no body and approved sections map to fragments", async () => {
  const route = createRouter({ fetcher: async url => good(url) });
  for (const section of ["overview", "datasets", "analysis", "experiments", "predictions", "assistant"]) {
    const res = await route(request(`/${section}/`, { method: "HEAD" }));
    assert.equal(res.status, 302);
    assert.equal(res.headers.get("location"), `${ORIGINS[0].url}/#${section}`);
    assert.equal(await res.text(), "");
  }
  assert.equal(await (await route(request("/links", { method: "HEAD" }))).text(), "");
});
