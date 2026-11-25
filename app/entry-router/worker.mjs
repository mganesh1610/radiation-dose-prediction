import { guestRoute } from "./guest-work.mjs";
// The visitor website serves reviewed public assets. Team operations use their protected origin.
const TEAM_ORIGIN = "https://defender-research-460004627518.us-west1.run.app";
const SECTIONS = new Set(["overview", "datasets", "analysis", "experiments", "predictions", "assistant"]);
const PUBLIC_VIEWS = {
  model: new Set(["A", "B", "C", "D", "elastic_net", "pls", "gradient_boosting_wb_wbc", "median_null"]),
  "model-view": new Set(["details"]),
  analysis: new Set(["pca", "heatmap", "clustering", "correlation", "qc"]),
  pool: new Set(["all", "core"]),
  study: new Set(["GSE102971", "GSE8917", "GSE65292", "GSE85570", "GSE97000", "GSE20173"]),
  comparison: new Set(["final", "development"]),
  workspace: new Set(["training"]),
};
const HEADERS = {
  "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "X-Robots-Tag": "noindex, nofollow",
};
function result(body, status, extra = {}) {
  return new Response(body, { status, headers: { ...HEADERS, "Cache-Control": "no-store", ...extra } });
}
function json(value, status = 200) {
  return result(JSON.stringify(value), status, { "Content-Type": "application/json; charset=utf-8" });
}
function dashboard(url, section) {
  const destination = new URL("/", url.origin);
  for (const [name, options] of Object.entries(PUBLIC_VIEWS)) {
    const value = url.searchParams.get(name);
    if (options.has(value)) destination.searchParams.set(name, value);
  }
  if (section) destination.hash = section;
  return destination;
}
async function predictionRequest(request) {
  if (!/^application\/json(?:\s*;|$)/i.test(request.headers.get("Content-Type") || "")) return null;
  const reader = request.body?.getReader();
  if (!reader) return null;
  const bytes = new Uint8Array(1024);
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (size + value.length > bytes.length) { await reader.cancel(); return null; }
    bytes.set(value, size);
    size += value.length;
  }
  try {
    const body = JSON.parse(new TextDecoder().decode(bytes.subarray(0, size)));
    return body && Object.keys(body).length === 2 && typeof body.sample_id === "string" && typeof body.model === "string" ? body : null;
  } catch { return null; }
}
// Workers Static Assets can return the whole MP4 even when a browser requests a byte range.
async function movieResponse(asset, request) {
  if (asset.status !== 200) return asset;
  const headers = new Headers(asset.headers);
  headers.set("Accept-Ranges", "bytes");
  const range = /^bytes=(\d*)-(\d*)$/i.exec(request.headers.get("Range") ?? "");
  const validator = request.headers.get("If-Range");
  const matches = !validator || (validator === headers.get("ETag") && !validator.startsWith("W/")) ||
    validator === headers.get("Last-Modified");
  if (request.method !== "GET" || !range || !(range[1] || range[2]) || !matches) {
    return new Response(asset.body, {status: 200, headers});
  }
  // Only the two small, versioned Overview movies use this fallback.
  const bytes = new Uint8Array(await asset.arrayBuffer());
  const length = bytes.byteLength;
  const start = range[1] ? Number(range[1]) : Math.max(0, length - Number(range[2]));
  const end = range[1] && range[2] ? Math.min(Number(range[2]), length - 1) : length - 1;
  if (start > end || start >= length) {
    headers.set("Content-Range", "bytes */" + length);
    headers.set("Content-Length", "0");
    return new Response(null, {status: 416, headers});
  }
  headers.set("Content-Range", `bytes ${start}-${end}/${length}`);
  headers.set("Content-Length", String(end - start + 1));
  return new Response(bytes.subarray(start, end + 1), {status: 206, headers});
}
export function createRouter() {
  let archive;
  function records(assets, origin) {
    if (!archive) {
      archive = assets.fetch(new Request(new URL("/published/estimates.json", origin))).then(response => {
        if (!response.ok) throw Error("Published estimates unavailable");
        return response.json();
      }).catch(error => { archive = undefined; throw error; });
    }
    return archive;
  }
  return async function route(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    if (path.startsWith("/api/guest/") || path.startsWith("/internal/guest-work/")) return guestRoute(request, env);
    const head = request.method === "HEAD";
    if (path === "/api/predict" && request.method === "POST") {
      const input = await predictionRequest(request);
      if (!input) return json({ error: "Choose a published study profile and model." }, 400);
      const { profiles, catalog } = await records(env.ASSETS, url.origin);
      if (!Object.hasOwn(profiles, input.sample_id) || !catalog.methods.includes(input.model)) {
        return json({ error: "Choose a published study profile and model." }, 400);
      }
      return json(profiles[input.sample_id][input.model]);
    }
    if (request.method !== "GET" && !head) return result(null, 405, { Allow: "GET, HEAD" });
    if (path === "/api/prediction-catalog") {
      const { catalog } = await records(env.ASSETS, url.origin);
      return head ? result(null, 200, { "Content-Type": "application/json" }) : json(catalog);
    }
    if (path === "/team" || path === "/team/") return result(null, 302, { Location: TEAM_ORIGIN + "/" });
    if (path === "/links") {
      const links = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Defender research</title><main><h1>Defender research</h1><p>Anyone with this link can explore the published research.</p><p><a href="/">Open the dashboard</a></p><p><a href="/team">Team sign-in</a></p></main></html>';
      return result(head ? null : links, 200, { "Content-Type": "text/html; charset=utf-8" });
    }
    const section = path.slice(1).replace(/\/$/, "");
    if (SECTIONS.has(section) || path === "/" && dashboard(url).search !== url.search) {
      return result(null, 302, { Location: dashboard(url, section).href });
    }
    if (!(path === "/" || path === "/index.html" || path === "/robots.txt" || path.startsWith("/data/") || path.startsWith("/static/"))) {
      return result(null, 404);
    }
    // The assets binding serves only this version's packaged public files. No requests are proxied.
    const headers = new Headers();
    const movie = path === "/static/assets/overview-dna.mp4" || path === "/static/assets/overview-dna-wide.mp4";
    if (movie) {
      for (const name of ["Range", "If-Range"]) {
        if (request.headers.has(name)) headers.set(name, request.headers.get(name));
      }
    }
    const asset = await env.ASSETS.fetch(new Request(new URL(path, url.origin), { method: request.method, headers }));
    const delivery = movie ? await movieResponse(asset, request) : asset;
    const response = new Response(head ? null : delivery.body, delivery);
    for (const [key, value] of Object.entries(HEADERS)) response.headers.set(key, value);
    response.headers.set("Cache-Control", /^\/data\/[a-f0-9]{20}\//.test(path) && asset.ok ? "public, max-age=31536000, immutable" : "no-cache");
    return response;
  };
}
export default { fetch: createRouter() };
