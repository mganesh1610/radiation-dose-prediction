import demo from "../../agent_system/src/research_agents/policies/demo_quota.json" with { type: "json" };

const COOKIE = "__Host-defender-guest";
const TOKEN = /^[a-f0-9]{64}$/;
const ID = /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/;
const METHODS = new Set(["median_null", "elastic_net", "pls", "gradient_boosting"]);
const STUDIES = new Set(["GSE102971", "GSE8917", "GSE65292"]);
const COLUMNS = "id,kind,input,state,created,updated,result";
const DAY = 86400;
const demoStart = Date.parse(demo.starts_at) / 1000;
const demoEnd = Date.parse(demo.ends_at) / 1000;
function aiWindow(t) {
  if (t >= demoStart && t < demoEnd) {
    const start = demoStart + Math.floor((t - demoStart) / DAY) * DAY;
    return {demo:true, start, end:start + DAY, personal:demo.messages_per_visitor,
      total:demo.public_visitors_per_day * demo.messages_per_visitor};
  }
  return {demo:false, start:t >= demoEnd ? Math.max(demoEnd,t - DAY) : t - DAY, end:t + 1,
    personal:10, total:60};
}
async function aiQuota(db, owner, t) {
  const w = aiWindow(t);
  const own = await db.prepare("SELECT COUNT(*) AS used,MIN(created) AS first FROM guest_work WHERE owner=? AND kind='ai' AND created>=? AND created<?")
    .bind(owner,w.start,w.end).first();
  const left = Math.max(0,w.personal - own.used);
  let reason = "", message = "";
  if (!left) {
    reason = "visitor_quota";
    message = w.demo ? "You’ve used all " + w.personal + " Gemini messages for today. Your daily quota resets at midnight Phoenix time."
      : "You’ve used all " + w.personal + " Gemini messages in the past 24 hours.";
  } else if (w.demo && !own.used) {
    const row = await db.prepare("SELECT COUNT(DISTINCT owner) AS visitors FROM guest_work WHERE kind='ai' AND created>=? AND created<?")
      .bind(w.start,w.end).first();
    if (row.visitors >= demo.public_visitors_per_day) {
      reason = "visitor_capacity";
      message = "All " + demo.public_visitors_per_day + " public Gemini visitor places are in use today. New places open at midnight Phoenix time.";
    }
  }
  const next = w.demo ? w.end : Math.min((own.first ?? t) + DAY + 1, t < demoStart ? demoStart : Infinity);
  const upcoming = t < demoStart ? {starts_at:demo.starts_at, ends_at:demo.ends_at,
    daily_visitors:demo.public_visitors_per_day, daily_messages:demo.messages_per_visitor} : undefined;
  return {demo:w.demo, limit:w.personal, used:own.used, remaining:reason ? 0 : left, reason, message, reset_at:next, upcoming};
}
const hex = bytes => Array.from(bytes, n => n.toString(16).padStart(2, "0")).join("");
const hash = async text => hex(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))));
const now = () => Math.floor(Date.now() / 1000);
const reply = (value, status = 200, extra = {}) => new Response(JSON.stringify(value), {status, headers: {
  "Content-Type":"application/json; charset=utf-8","Cache-Control":"private, no-store","Vary":"Cookie",
  "X-Content-Type-Options":"nosniff","Referrer-Policy":"no-referrer",...extra,
}});
async function body(request, limit) {
  if (!/^application\/json(?:\s*;|$)/i.test(request.headers.get("content-type") || "") || !request.body) return null;
  const reader = request.body.getReader(), parts = [];
  let size = 0;
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    size += chunk.value.length;
    if (size > limit) { await reader.cancel(); return null; }
    parts.push(chunk.value);
  }
  try { return JSON.parse(await new Blob(parts).text()); } catch { return null; }
}
function view(row) {
  return {id:row.id,kind:row.kind,input:JSON.parse(row.input),state:row.state,created:row.created,
    updated:row.updated,result:row.result && JSON.parse(row.result)};
}
function inputFor(kind, data) {
  if (kind === "training" && METHODS.has(data?.method) && STUDIES.has(data?.holdout))
    return {method:data.method,holdout:data.holdout};
  const question = typeof data?.question === "string" && data.question.trim();
  if (kind === "ai" && question && question.length <= 1200) {
    if (data.chat === undefined) return {question};
    if (typeof data.chat === "string" && ID.test(data.chat)) return {question,chat:data.chat};
  }
  return null;
}
async function submit(db, owner, data) {
  const input = inputFor(data?.kind, data?.input);
  if (!input || !ID.test(data?.key)) return reply({error:"Choose a training preset or enter a question up to 1,200 characters."},400);
  const encoded = JSON.stringify(input), kind = data.kind, t = now();
  const month = new Date(t * 1000);
  const window = aiWindow(t), inDemo = kind === "ai" && window.demo;
  const since = kind === "training" ? Date.UTC(month.getUTCFullYear(),month.getUTCMonth(),1) / 1000 : window.start;
  const until = kind === "training" ? Date.UTC(month.getUTCFullYear(),month.getUTCMonth()+1,1) / 1000 : window.end;
  // This single conditional INSERT reserves the visitor's daily place and turn
  // together. Cancelled, failed and deleted requests continue to count.
  const seat = inDemo ? ` AND (EXISTS (SELECT 1 FROM guest_work WHERE owner=? AND kind='ai' AND created>=? AND created<?)
    OR (SELECT COUNT(DISTINCT owner) FROM guest_work WHERE kind='ai' AND created>=? AND created<?) < ?)` : "";
  const queue = inDemo ? "kind='ai' AND " : "";
  await db.prepare(`INSERT OR IGNORE INTO guest_work(id,owner,kind,request_key,input,created,updated)
    SELECT ?,?,?,?,?,?,? WHERE
    (SELECT COUNT(*) FROM guest_work WHERE owner=? AND kind=? AND created>=? AND created<?) < ?
    AND NOT EXISTS (SELECT 1 FROM guest_work WHERE owner=? AND kind=? AND state IN ('queued','running'))
    AND (SELECT COUNT(*) FROM guest_work WHERE ${queue}state IN ('queued','running')) < ?
    AND (SELECT COUNT(*) FROM guest_work WHERE kind=? AND created>=? AND created<?) < ?
    AND NOT EXISTS (SELECT 1 FROM guest_work WHERE owner=? AND kind='ai' AND state='deleted' AND json_extract(input,'$.chat')=?)${seat}`).bind(
      crypto.randomUUID(),owner,kind,data.key,encoded,t,t,owner,kind,since,until,kind === "training" ? 2 : window.personal,
      owner,kind,inDemo ? demo.public_visitors_per_day : 20,kind,since,until,kind === "training" ? 12 : window.total,owner,input.chat || null,
      ...(inDemo ? [owner,since,window.end,since,window.end,demo.public_visitors_per_day] : [])).run();
  const row = await db.prepare("SELECT "+COLUMNS+" FROM guest_work WHERE owner=? AND request_key=?").bind(owner,data.key).first();
  const quota = kind === "ai" ? await aiQuota(db,owner,t) : undefined;
  if (!row) {
    if (quota?.reason) return reply({error:quota.message,quota},429,{"Retry-After":String(Math.max(1,quota.reset_at-t))});
    return reply({error:"The guest queue or request allowance is full. Check your recent requests before trying again."},429);
  }
  if (row.kind !== kind || row.input !== encoded) return reply({error:"This request key is already in use."},409);
  return reply({...view(row),quota},202);
}
async function dispatch(request, env, path) {
  const supplied = request.headers.get("Authorization") || "";
  if (request.method !== "POST" || !env.GUEST_KEY_SHA256 || !supplied.startsWith("Bearer ") || supplied.length > 128 ||
      (await hash(supplied.slice(7))) !== env.GUEST_KEY_SHA256) return reply({error:"Unavailable."},404);
  const data = await body(request,16000), db = env.GUEST_DB, t = now();
  if (!data) return reply({error:"Invalid request."},400);
  if (path === "/internal/guest-work/claim" && ["training","ai"].includes(data.kind)) {
    // An interrupted paid turn is never dispatched twice. Training is recoverable by its fixed job ID.
    const failed = JSON.stringify({message:"The AI worker was interrupted. Check the shared allowance before starting another turn."});
    const queries = [
      db.prepare("UPDATE guest_work SET state='failed',result=?,updated=? WHERE state='running' AND kind='ai' AND lease<?").bind(failed,t,t),
      db.prepare(`UPDATE guest_work SET state='running',fence=?,lease=?,updated=? WHERE id=(
        SELECT id FROM guest_work WHERE kind=? AND ((state='queued' AND available<=?) OR
        (kind='training' AND state='running' AND lease<?)) ORDER BY created,id LIMIT 1) RETURNING *`)
        .bind(crypto.randomUUID(),t+300,t,data.kind,t,t),
    ];
    if (Math.random() < 0.01) queries.push(db.prepare("DELETE FROM guest_work WHERE created<?").bind(t-30*DAY));
    const [, claim] = await db.batch(queries);
    if (!claim.results.length) return reply({work:null});
    const row = claim.results[0], work = {id:row.id,kind:row.kind,created:row.created,owner:row.owner,
      input:JSON.parse(row.input),fence:row.fence};
    if (row.kind === "ai") {
      const past = await db.prepare(`SELECT input,result FROM guest_work WHERE owner=? AND kind='ai' AND state='completed'
        AND json_extract(input,'$.chat') IS ? AND created>=? AND created<=? AND id<>?
        ORDER BY created DESC,rowid DESC LIMIT 3`).bind(row.owner,work.input.chat ?? null,t-30*DAY,row.created,row.id).all();
      work.history = past.results.reverse().map(x=>({question:JSON.parse(x.input).question,answer:JSON.parse(x.result).answer}));
    }
    return reply({work});
  }
  if (path === "/internal/guest-work/finish" && ID.test(data.id) && ID.test(data.fence) && ["completed","failed","queued"].includes(data.state)) {
    const updated = await db.prepare(`UPDATE guest_work SET state=?,result=?,updated=?,available=?,lease=0,fence=NULL
      WHERE id=? AND fence=? AND state='running' AND (kind='training' OR ?<>'queued')`).bind(
      data.state,JSON.stringify(data.result || null),t,t+45,data.id,data.fence,data.state).run();
    return reply({updated:updated.meta.changes===1});
  }
  return reply({error:"Unknown operation."},404);
}
const active = state => state === "queued" || state === "running";
const chatScope = "owner=? AND kind='ai' AND state<>'deleted' AND created>=?";
const threadScope = chatScope+" AND COALESCE(json_extract(input,'$.chat'),'previous')=?";
function chatSummary(row, input) {
  return {chat_id:input.chat || "previous",created_at:row.created,title:input.question.slice(0,72),
    active_run:active(row.state) ? "run-"+row.id : null};
}
async function copilot(request, db, owner, path) {
  const parts = path.slice("/api/guest/copilot/".length).split("/");
  const [resource,id,action] = parts, method = request.method, since = now()-30*DAY;
  if (method !== "GET" && request.headers.get("Origin") !== new URL(request.url).origin)
    return reply({error:"Use this site's chat to make changes."},403);
  if (resource === "quota" && parts.length === 1 && method === "GET")
    return reply({quota:await aiQuota(db,owner,now())});
  if (resource === "chats" && parts.length === 1 && method === "GET") {
    const rows = await db.prepare("SELECT id,input,state,created FROM guest_work WHERE "+chatScope+" ORDER BY created,rowid")
      .bind(owner,since).all();
    const chats = new Map();
    for (const row of rows.results) {
      const chat = chatSummary(row,JSON.parse(row.input)), prior = chats.get(chat.chat_id);
      if (!prior) chats.set(chat.chat_id,chat);
      else if (chat.active_run) prior.active_run = chat.active_run;
    }
    return reply({chats:[...chats.values()]});
  }
  if (resource === "chats" && (id === "previous" || ID.test(id))) {
    if (parts.length === 2 && method === "DELETE") {
      // Keep admission rows for the quotas. Clear the visitor's stored text and fence any late delivery.
      await db.prepare("UPDATE guest_work SET input=?,result=NULL,state='deleted',fence=NULL,lease=0,updated=? WHERE "+threadScope)
        .bind(JSON.stringify({chat:id}),now(),owner,since,id).run();
      return reply({deleted:true});
    }
    if (parts.length === 3 && action === "messages" && method === "POST") {
      const data = await body(request,6000);
      const response = await submit(db,owner,{key:data?.request_id,kind:"ai",
        input:{question:data?.text,...(id === "previous" ? {} : {chat:id})}});
      if (!response.ok) return response;
      const work = await response.json();
      return reply({run_id:"run-"+work.id,status:work.state,quota:work.quota},response.status);
    }
    if (parts.length === 3 && action === "messages" && method === "GET") {
      const rows = (await db.prepare("SELECT "+COLUMNS+" FROM guest_work WHERE "+threadScope+" ORDER BY created,rowid")
        .bind(owner,since,id).all()).results.map(view);
      if (!rows.length) return reply({error:"Conversation not found."},404);
      const chat = chatSummary(rows[0],rows[0].input), messages = [];
      for (const row of rows) {
        messages.push({role:"user",text:row.input.question});
        if (active(row.state)) chat.active_run = "run-"+row.id;
        else if (row.state === "completed") messages.push({role:"assistant",text:row.result.answer,invocation_id:"run-"+row.id});
        else messages.push({role:"assistant",text:row.result?.message || "This response was stopped."});
      }
      return reply({chat,messages,next_offset:null});
    }
  }
  if (resource === "runs" && parts.length === 3 && id?.startsWith("run-") && ID.test(id.slice(4))) {
    const key = id.slice(4);
    const scope = chatScope+" AND id=?";
    if (action === "cancel" && method === "POST") {
      const message = {message:"Response stopped. An AI call already in progress can still use a turn."};
      await db.prepare("UPDATE guest_work SET state='cancelled',result=?,updated=?,fence=NULL,lease=0 WHERE "+scope+" AND state IN ('queued','running')")
        .bind(JSON.stringify(message),now(),owner,since,key).run();
    } else if (action !== "receipt" || method !== "GET") return reply({error:"Unknown chat operation."},404);
    const row = await db.prepare("SELECT state,result FROM guest_work WHERE "+scope).bind(owner,since,key).first();
    if (!row) return reply({error:"Response not found."},404);
    if (action === "cancel") return reply({status:row.state});
    const result = JSON.parse(row.result || "{}") || {};
    const sources = row.state === "completed" ? [
      {label:"Explore the published studies",url:"#datasets"},
      {label:"View the published model comparisons",url:"#experiments"},
    ] : [];
    return reply({run_id:id,status:row.state,model:result.model,source_version:result.source_version,sources});
  }
  return reply({error:"Unknown chat operation."},404);
}
export async function guestRoute(request, env) {
  const url = new URL(request.url), path = url.pathname;
  if (!env.GUEST_DB) return reply({error:"Guest workspace temporarily unavailable."},503);
  if (path.startsWith("/internal/guest-work/")) return dispatch(request,env,path);
  const cookie = (request.headers.get("Cookie") || "").split(/;\s*/).find(x=>x.startsWith(COOKIE+"="))?.slice(COOKIE.length+1);
  if (request.method === "GET" && path === "/api/guest/session") {
    if (!TOKEN.test(cookie)) {
      const token = hex(crypto.getRandomValues(new Uint8Array(32)));
      return reply({requests:[]},200,{"Set-Cookie":COOKIE+"="+token+"; Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000"});
    }
    const rows = await env.GUEST_DB.prepare("SELECT "+COLUMNS+" FROM guest_work WHERE owner=? AND state<>'deleted' AND created>=? ORDER BY created DESC,id DESC LIMIT 35").bind(await hash(cookie),now()-30*DAY).all();
    return reply({requests:rows.results.map(view)});
  }
  if (!TOKEN.test(cookie)) return reply({error:"Refresh the page to start a guest session."},409);
  const owner = await hash(cookie);
  if (path.startsWith("/api/guest/copilot/")) return copilot(request,env.GUEST_DB,owner,path);
  if (request.method === "POST" && path === "/api/guest/requests") {
    if (request.headers.get("Origin") !== url.origin) return reply({error:"Use this site's form to submit a request."},403);
    return submit(env.GUEST_DB,owner,await body(request,6000));
  }
  const id = path.split("/").at(-1);
  if (request.method === "GET" && path === "/api/guest/requests/"+id && ID.test(id)) {
    const row = await env.GUEST_DB.prepare("SELECT "+COLUMNS+" FROM guest_work WHERE id=? AND owner=? AND state<>'deleted' AND created>=?").bind(id,owner,now()-30*DAY).first();
    if (row) return reply(view(row));
  }
  return reply({error:"Request not found."},404);
}
