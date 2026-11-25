import test from "node:test";
import assert from "node:assert/strict";
import { createRouter } from "../app/entry-router/worker.mjs";

const origin = "https://share.example";
const model = "gradient_boosting_wb_wbc";
const sample = "GSM2417396";
const entry = { model, sample_id: sample, estimate_gy: 0.3718854551243722 };
function site() {
  const route = createRouter();
  const requests = [];
  const ASSETS = { async fetch(request) {
    const path = new URL(request.url).pathname;
    requests.push({ path, cookie: request.headers.get("Cookie"), authorization: request.headers.get("Authorization") });
    if (path === "/published/estimates.json") {
      return Response.json({ catalog: { methods: [model], examples: [{ id: sample }] }, profiles: { [sample]: { [model]: entry } } });
    }
    return new Response(path === "/" ? "Research dashboard" : "asset");
  }};
  return { requests, open: (path = "/", options) => route(new Request(origin + path, options), { ASSETS }) };
}
const prediction = (x) => ({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(x)});

test("anonymous root serves the dashboard and does not send caller credentials to assets", async () => {
  const {open,requests} = site();
  const res=await open("/",{headers:{Cookie:"example-cookie",Authorization:"example-token"}});
  assert.equal(res.status,200);
  assert.equal(res.headers.get("location"),null);
  assert.equal(res.headers.get("set-cookie"),null);
  assert.match(res.headers.get("content-security-policy"), /frame-ancestors 'none'/);
  assert.equal(await res.text(),"Research dashboard");
  assert.deepEqual(requests,[{path:"/",cookie:null,authorization:null}]);
});
test("all six section paths map to a public section, with optional trailing slash",async()=>{
  for(const section of ["overview","datasets","analysis","experiments","predictions","assistant"]) {
    for(const suffix of ["","/"]) assert.equal((await site().open("/"+section+suffix)).headers.get("location"),origin+"/#"+section);
  }
});
test("only recognized public query options are carried through",async()=>{
  const res=await site().open("/experiments?workspace=training&chat=private&model="+model);
  assert.equal(res.headers.get("location"),origin+"/?model="+model+"&workspace=training#experiments");
  assert.equal((await site().open("/?chat=private")).headers.get("location"),origin+"/");
  assert.equal((await site().open("/?model="+model)).status,200);
});
test("optional team entry and access information are distinct",async()=>{
  const {open}=site();
  assert.equal((await open("/team")).headers.get("location"),"https://defender-research-460004627518.us-west1.run.app/");
  const res=await open("/links");assert.match(await res.text(),/Anyone with this link/);
});
test("only an explicit Predict request returns a known profile/model estimate",async()=>{
  const {open,requests}=site();
  const c=await open("/api/prediction-catalog");const text=await c.text();
  assert.doesNotMatch(text,/estimate_gy/);assert.match(text,/GSM2417396/);
  const res=await open("/api/predict",prediction({sample_id:sample,model}));
  assert.equal(res.status,200);assert.deepEqual(await res.json(),entry);
  assert.equal(requests.length,1);
});
test("unlisted inputs, models, arbitrary upload data and oversized bodies are declined",async()=>{
  const {open}=site();
  for(const input of [{sample_id:"other",model},{sample_id:sample,model:"other"},{upload_id:"upload",model},{sample_id:sample,model,extra:1},{sample_id:"x".repeat(1400),model}]) {
    assert.equal((await open("/api/predict",prediction(input))).status,400);
  }
});
test("team APIs and backend-only records are absent; unexpected writes are declined",async()=>{
  const {open,requests}=site();
  for(const path of ["/api/copilot/chats","/api/jobs","/api/uploads","/published/estimates.json","/runtime/frozen/models.json"]) assert.equal((await open(path)).status,404);
  assert.equal((await open("/team",{method:"POST"})).status,405);
  assert.deepEqual(requests,[]);
});
test("HEAD preserves navigation and asset headers without a body",async()=>{
  for(const [path,status] of [["/",200],["/analysis",302],["/links",200],["/robots.txt",200],["/api/jobs",404]]) {
    const res=await site().open(path,{method:"HEAD"});assert.equal(res.status,status);assert.equal(await res.text(),"");
  }
});

test("a video seek reaches the packaged asset and preserves the partial response", async () => {
  let received;
  const ASSETS = {fetch(request) {
    received = request.headers;
    return new Response("clip", {status: 206, headers: {"Content-Range":"bytes 4-7/12", "Content-Type":"video/mp4"}});
  }};
  const res = await createRouter()(new Request(origin + "/static/assets/overview-dna-wide.mp4", {
    headers: {Range:"bytes=4-7", "If-Range":'"film"', Cookie:"example", Authorization:"example"}}), {ASSETS});
  assert.equal(received.get("Range"), "bytes=4-7");
  assert.equal(received.get("If-Range"), '"film"');
  assert.equal(received.get("Cookie"), null);
  assert.equal(received.get("Authorization"), null);
  assert.equal(res.status, 206);
  assert.equal(res.headers.get("Content-Range"), "bytes 4-7/12");
  assert.equal(await res.text(), "clip");
});

test("the homepage six-study link keeps the all-study selection through the edge", async () => {
  const page = await site().open("/?analysis=clustering&pool=all");
  assert.equal(page.status, 200);
  const link = await site().open("/analysis?analysis=clustering&pool=all");
  assert.equal(link.headers.get("location"), origin + "/?analysis=clustering&pool=all#analysis");
});

test("both Overview movies can seek when static storage returns the complete representation", async () => {
  const route = createRouter();
  const ASSETS = {fetch: () => new Response("abcdefghijkl", {headers: {"Content-Type":"video/mp4", ETag:'"film"'}})};
  for (const name of ["overview-dna", "overview-dna-wide"]) {
    const url = origin + "/static/assets/" + name + ".mp4";
    for (const [range, expected, body] of [
      ["bytes=3-6", "bytes 3-6/12", "defg"], ["bytes=-2", "bytes 10-11/12", "kl"],
      ["bytes=9-", "bytes 9-11/12", "jkl"], ["bytes=8-55", "bytes 8-11/12", "ijkl"],
    ]) {
      const res = await route(new Request(url, {headers: {Range:range,"If-Range":'"film"'}}), {ASSETS});
      assert.equal(res.status, 206);
      assert.equal(res.headers.get("Content-Range"), expected);
      assert.equal(res.headers.get("Content-Length"), String(body.length));
      assert.equal(await res.text(), body);
    }
    const out = await route(new Request(url, {headers: {Range:"bytes=12-"}}), {ASSETS});
    assert.equal(out.status, 416);
    assert.equal(out.headers.get("Content-Range"), "bytes */12");
    const changed = await route(new Request(url, {headers:{Range:"bytes=4-6","If-Range":'"older"'}}), {ASSETS});
    assert.equal(changed.status, 200);
    assert.equal(await changed.text(), "abcdefghijkl");
    const head = await route(new Request(url, {method:"HEAD"}), {ASSETS});
    assert.equal(head.headers.get("Accept-Ranges"), "bytes");
    assert.equal(await head.text(), "");
  }
});

test("published study cards and experiment detail links retain their selection after reload", async () => {
  for (const study of ["GSE85570", "GSE97000", "GSE20173"]) {
    const query = "?analysis=clustering&pool=all&study=" + study;
    assert.equal((await site().open("/" + query)).status, 200);
    assert.equal((await site().open("/analysis" + query)).headers.get("location"), origin + "/" + query + "#analysis");
  }
  const query = "?model=A&model-view=details&comparison=final";
  assert.equal((await site().open("/" + query)).status, 200);
  assert.equal((await site().open("/?study=unpublished")).headers.get("location"), origin + "/");
});
