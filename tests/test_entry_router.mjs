import test from "node:test";
import assert from "node:assert/strict";
import worker, { APP_ORIGIN, createRouter } from "../app/entry-router/worker.mjs";

const request = (path = "/", options) => new Request("https://entry.example" + path, options);

test("root goes directly to the Google Cloud app with no-store navigation headers", async () => {
  const res = await worker.fetch(request());
  assert.equal(res.status, 302);
  assert.equal(res.headers.get("location"), APP_ORIGIN + "/");
  assert.equal(res.headers.get("x-defender-destination"), "google-cloud");
  assert.equal(res.headers.get("cache-control"), "no-store");
  assert.equal(res.headers.get("referrer-policy"), "no-referrer");
  assert.equal(await res.text(), "");
});

test("all six navigation sections map to the same authenticated application", async () => {
  for (const section of ["overview", "datasets", "analysis", "experiments", "predictions", "assistant"]) {
    for (const ending of ["", "/"]) {
      const res = await worker.fetch(request("/" + section + ending));
      assert.equal(res.status, 302);
      assert.equal(res.headers.get("location"), APP_ORIGIN + "/#" + section);
    }
  }
});

test("caller destinations, query tokens, cookies and authorization are not forwarded", async t => {
  const fetchSpy = t.mock.method(globalThis, "fetch", () => { throw new Error("must not fetch"); });
  const res = await createRouter()(request("/assistant?next=https://evil.example&token=private", {
    headers: { Authorization: "private", Cookie: "private" },
  }));
  assert.equal(res.headers.get("location"), APP_ORIGIN + "/#assistant");
  assert.equal(res.headers.get("set-cookie"), null);
  assert.equal(fetchSpy.mock.callCount(), 0);
});

test("the root leaves hash navigation to the browser instead of replacing it", async () => {
  const res = await worker.fetch(request("/#assistant"));
  assert.equal(res.headers.get("location"), APP_ORIGIN + "/");
  assert.ok(!res.headers.get("location").includes("#"));
});

test("links page contains one Google Cloud destination and explains sign-in", async () => {
  const res = await worker.fetch(request("/links"));
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.ok(html.includes(APP_ORIGIN + "/"));
  assert.ok(html.includes("invited Google account"));
  assert.doesNotMatch(html, /vercel|firebase|<script/i);
});

test("unknown paths and API calls are not proxied", async () => {
  for (const path of ["/api/jobs", "//evil.example", "/https://evil.example", "/favicon.ico", "/%2fanalysis"]) {
    const res = await worker.fetch(request(path));
    assert.equal(res.status, 404);
    assert.equal(res.headers.get("location"), null);
  }
});

test("write methods cannot forward a request or its body", async () => {
  for (const method of ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"]) {
    const res = await worker.fetch(request("/assistant", { method }));
    assert.equal(res.status, 405);
    assert.equal(res.headers.get("allow"), "GET, HEAD");
    assert.equal(res.headers.get("location"), null);
  }
});

test("HEAD preserves redirect/status headers and never returns a body", async () => {
  for (const [path, status] of [["/", 302], ["/analysis", 302], ["/links", 200], ["/robots.txt", 200], ["/api/jobs", 404]]) {
    const res = await worker.fetch(request(path, { method: "HEAD" }));
    assert.equal(res.status, status);
    assert.equal(await res.text(), "");
  }
});
