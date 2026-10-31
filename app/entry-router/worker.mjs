// Public navigation only. No credentials, backend calls, paid bindings, or jobs.
export const ORIGINS = Object.freeze([
  { name: "vercel", url: "https://defender-research-public.vercel.app" },
  { name: "firebase", url: "https://radiation-prediction-4ffe5.web.app" },
]);
const SECTIONS = new Set(["overview", "datasets", "analysis", "experiments", "predictions", "assistant"]);
const HEADERS = {
  "Cache-Control": "no-store",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
};

async function boundedText(response, maximum) {
  if (Number(response.headers.get("content-length")) > maximum) throw new Error("Oversize response");
  if (!response.body) throw new Error("Empty response");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let size = 0;
  let text = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maximum) throw new Error("Oversize response");
      text += decoder.decode(value, { stream: true });
    }
    return text + decoder.decode();
  } finally {
    await reader.cancel().catch(() => {});
  }
}

function linksPage(unavailable = false) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Defender · Site access</title><style>body{margin:0;background:#f5f7f9;color:#17272b;font:18px/1.6 system-ui,sans-serif}main{max-width:640px;margin:10vh auto;padding:32px}h1{font-size:36px;line-height:1.15}p{color:#42545b}a{color:#006c69;text-underline-offset:4px}li{margin:16px 0}a:focus-visible{outline:3px solid #b95118;outline-offset:5px}.tag{font-size:13px;letter-spacing:.14em;font-weight:700;color:#006c69}</style></head><body><main><p class="tag">TEAM DEFENDER · RESEARCH</p><h1>${unavailable ? "The public sites are not responding" : "One entry link. Two public sites."}</h1><p>${unavailable ? "The availability checks did not pass. Try again, or open either site directly; it may still work from your connection." : "The entry link checks Vercel first and uses the Firebase backup when that check fails."}</p><ul><li><a href="/">${unavailable ? "Try the entry link again" : "Open the research explorer"}</a></li><li><a href="${ORIGINS[0].url}/">Open Vercel directly</a></li><li><a href="${ORIGINS[1].url}/">Open Firebase directly</a></li></ul><p>Both sites show the published research snapshot. Live training opens the separate workspace and requires an invited Google account.</p><p>Routing happens when you open the entry link. An already-open page does not switch sites automatically.</p></main></body></html>`;
}

// Dependency injection is local test support, never controlled by a URL or binding.
export function createRouter({ fetcher = fetch, now = Date.now, timeoutMs = 2000 } = {}) {
  const health = new Map(); // Two records per isolate; no KV, database, cron or paid service.
  async function healthy(origin) {
    const cached = health.get(origin.url);
    if (cached && cached.expires > now()) return cached.ok;
    const controller = new AbortController();
    let timer;
    const deadline = new Promise(resolve => {
      timer = setTimeout(() => { controller.abort(); resolve(false); }, timeoutMs);
    });
    const check = async () => {
      const options = {
        method: "GET", redirect: "manual", signal: controller.signal,
        headers: { "Accept": "application/json, text/html", "Cache-Control": "no-cache" },
        cf: { cacheTtl: 0, cacheEverything: false },
      };
      const [home, manifest] = await Promise.all([
        fetcher(`${origin.url}/`, options),
        fetcher(`${origin.url}/data/publication.json`, options),
      ]);
      if (home.status !== 200 || manifest.status !== 200) return false;
      if (!home.headers.get("content-type")?.includes("text/html") ||
          !manifest.headers.get("content-type")?.includes("application/json")) return false;
      const [html, json] = await Promise.all([boundedText(home, 131072), boundedText(manifest, 4096)]);
      const data = JSON.parse(json);
      return html.includes("Defender · Overview") && html.includes('data-route="analysis"') &&
        data.schema_version === 1 && data.locked_test_included === false &&
        Number.isSafeInteger(data.profiles) && data.profiles > 0 &&
        Number.isSafeInteger(data.analysis_combinations) && data.analysis_combinations > 0 &&
        /^core_[a-f0-9]+$/.test(data.release_id) && /^\/data\/[a-f0-9]+$/.test(data.data_path);
    };
    let ok = false;
    try { ok = await Promise.race([check().catch(error => {
      console.warn("Public-site probe failed", origin.name, error.name, error.message);
      return false;
    }), deadline]); }
    finally { clearTimeout(timer); controller.abort(); }
    // Positive checks can be 30 seconds old; failures are retried after 5 seconds.
    health.set(origin.url, { ok, expires: now() + (ok ? 30000 : 5000) });
    return ok;
  }

  return async function route(request) {
    const url = new URL(request.url);
    const head = request.method === "HEAD";
    const reply = (body, status, extra = {}) => new Response(head ? null : body, {
      status, headers: { ...HEADERS, ...extra },
    });
    if (request.method !== "GET" && !head) return reply("Use GET or HEAD.\n", 405, { Allow: "GET, HEAD" });
    if (url.pathname === "/links") return reply(linksPage(), 200, { "Content-Type": "text/html; charset=utf-8" });
    if (url.pathname === "/robots.txt") return reply("User-agent: *\nDisallow: /\n", 200, { "Content-Type": "text/plain" });
    const section = url.pathname.slice(1).replace(/\/$/, "");
    if (url.pathname !== "/" && !SECTIONS.has(section)) return reply("Not found. Open / for the research explorer.\n", 404);
    for (const origin of ORIGINS) {
      if (await healthy(origin)) {
        // No user-supplied destination, cookies, credentials, path or query forwarding.
        // Root redirects omit a fragment so browsers retain an incoming #analysis etc.
        return reply(null, 302, {
          Location: `${origin.url}/${section ? `#${section}` : ""}`,
          "X-Defender-Destination": origin.name,
        });
      }
    }
    return reply(linksPage(true), 503, { "Content-Type": "text/html; charset=utf-8", "Retry-After": "30" });
  };
}

const route = createRouter();
export default { fetch: route };
