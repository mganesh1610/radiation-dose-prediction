// Navigation only. Google Cloud handles sign-in, access checks, and the application.
export const APP_ORIGIN = "https://defender-research-460004627518.us-west1.run.app";
const SECTIONS = new Set(["overview", "datasets", "analysis", "experiments", "predictions", "assistant"]);
const HEADERS = {
  "Cache-Control": "no-store",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
};

function linksPage() {
  return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Defender · Workspace access</title><style>body{margin:0;background:#f5f7f9;color:#17272b;font:18px/1.6 system-ui,sans-serif}main{max-width:640px;margin:10vh auto;padding:32px}h1{font-size:36px;line-height:1.15}p{color:#42545b}a{color:#006c69;text-underline-offset:4px}a:focus-visible{outline:3px solid #b95118;outline-offset:5px}</style></head><body><main><h1>One research workspace.</h1><p>The dashboard, analysis, predictions and research copilot are hosted together on Google Cloud.</p><p><a href="' + APP_ORIGIN + '/">Open the research workspace</a></p><p>Sign in with an invited Google account. Access is checked by the Google Cloud application.</p></main></body></html>';
}

export function createRouter() {
  return function route(request) {
    const url = new URL(request.url);
    const head = request.method === "HEAD";
    const reply = (body, status, extra = {}) => new Response(head ? null : body, {
      status, headers: { ...HEADERS, ...extra },
    });
    if (request.method !== "GET" && !head) return reply("Use GET or HEAD.\n", 405, { Allow: "GET, HEAD" });
    if (url.pathname === "/links") return reply(linksPage(), 200, { "Content-Type": "text/html; charset=utf-8" });
    if (url.pathname === "/robots.txt") return reply("User-agent: *\nDisallow: /\n", 200, { "Content-Type": "text/plain" });
    const section = url.pathname.slice(1).replace(/\/$/, "");
    if (url.pathname !== "/" && !SECTIONS.has(section)) return reply("Not found. Open / for the research workspace.\n", 404);
    // Fixed destination: no upstream fetches, proxying, tokens, cookies, or query forwarding.
    // Omitting a root fragment lets browsers retain incoming #assistant, #analysis, etc.
    return reply(null, 302, {
      Location: APP_ORIGIN + "/" + (section ? "#" + section : ""),
      "X-Defender-Destination": "google-cloud",
    });
  };
}

export default { fetch: createRouter() };
