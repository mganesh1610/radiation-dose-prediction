import test from "node:test";
import assert from "node:assert/strict";
import { DatabaseSync } from "node:sqlite";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { guestRoute } from "../app/entry-router/guest-work.mjs";

function setup() {
  const sqlite = new DatabaseSync(":memory:");
  sqlite.exec(readFileSync(new URL("../app/entry-router/guest-schema.sql",import.meta.url),"utf8"));
  class Statement {
    constructor(sql) { this.s = sqlite.prepare(sql); this.args=[]; }
    bind(...args) { this.args=args;return this; }
    first() { return this.s.get(...this.args); }
    all() { return {results:this.s.all(...this.args)}; }
    run() { return {meta:{changes:Number(this.s.run(...this.args).changes)}}; }
  }
  const env={GUEST_DB:{ prepare:s=>new Statement(s),batch:xs=>xs.map(x=>x.all()) },GUEST_KEY_SHA256:createHash("sha256").update("unit-fixture").digest("hex")};
  const origin="https://test.example";
  const call=(path,{data,cookie,auth,site=origin,method=data?"POST":"GET"}={})=>guestRoute(new Request(origin+path,{
    method, headers:{"Content-Type":"application/json",Origin:site,...cookie&&{Cookie:cookie},...auth&&{Authorization:"Bearer "+auth}},
    body:data&&JSON.stringify(data),
  }),env);
  const session=async()=>{const resp=await call("/api/guest/session");assert.match(resp.headers.get("set-cookie"),/HttpOnly; SameSite=Strict/);
    return resp.headers.get("set-cookie").split(";")[0];};
  const claim=async kind=>(await (await call("/internal/guest-work/claim",{data:{kind},auth:"unit-fixture"})).json()).work;
  const finish=(work,state,result)=>call("/internal/guest-work/finish",{auth:"unit-fixture",data:{id:work.id,fence:work.fence,state,result}});
  return {call,session,claim,finish,sqlite};
}
test("Phoenix demo reserves 50 daily places, 20 sends per browser, then resets and expires",async t=>{
  const {call,session,sqlite}=setup();
  let clock=Date.parse("2026-11-24T06:59:59Z");
  t.mock.method(Date,"now",()=>clock);
  const visitors=await Promise.all(Array.from({length:51},session));
  const send=(cookie,key=crypto.randomUUID())=>call("/api/guest/requests",{cookie,
    data:{key,kind:"ai",input:{question:"Demo question"}}});
  const quota=async cookie=>(await (await call("/api/guest/copilot/quota",{cookie})).json()).quota;
  assert.equal((await quota(visitors[0])).limit,10);
  clock+=1000;
  const finishBatch=()=>sqlite.exec("UPDATE guest_work SET state='completed' WHERE state='queued'");
  const firstKey=crypto.randomUUID();
  assert.equal((await send(visitors[0],firstKey)).status,202);
  assert.equal((await send(visitors[0],firstKey)).status,202);
  const first=await Promise.all(visitors.slice(1,50).map(c=>send(c)));
  assert(first.every(r=>r.status===202));
  assert.equal((await (await send(visitors[50])).json()).quota.reason,"visitor_capacity");
  finishBatch();
  for(let turn=2;turn<=20;turn++){
    const responses=await Promise.all(visitors.slice(0,50).map(c=>send(c)));
    assert(responses.every(r=>r.status===202));
    if(turn===20) assert.equal((await responses[0].json()).quota.reason,"visitor_quota");
    finishBatch();
  }
  assert.equal(sqlite.prepare("SELECT COUNT(*) AS n FROM guest_work").get().n,1000);
  const blocked=await send(visitors[0]);assert.equal(blocked.status,429);
  assert.match((await blocked.json()).error,/20 Gemini messages.*midnight Phoenix/);
  assert.equal((await send(visitors[0],firstKey)).status,202);
  sqlite.exec("UPDATE guest_work SET state='deleted'");
  assert.equal((await quota(visitors[0])).remaining,0);
  clock=Date.parse("2026-11-25T07:00:00Z");
  assert.equal((await quota(visitors[0])).remaining,20);
  assert.equal((await send(visitors[50])).status,202);
  clock=Date.parse("2026-12-14T06:59:59Z");
  assert.equal((await quota(visitors[0])).demo,true);
  clock+=1000;assert.equal((await quota(visitors[0])).limit,10);
});
test("guest tasks are idempotent, scoped to the browser and charged only through valid forms",async()=>{
  const {call,session,claim,finish}=setup(), a=await session(), b=await session();
  const data={key:crypto.randomUUID(),kind:"training",input:{method:"gradient_boosting",holdout:"GSE8917"}};
  const post=(cookie,d,site)=>call("/api/guest/requests",{data:d,cookie,site});
  assert.equal((await post(a,data,"https://example.org")).status,403);
  assert.equal((await post(a,{...data,input:{method:"unknown",holdout:"GSE8917"}})).status,400);
  const saved=await (await post(a,data)).json();
  assert.equal(saved.id,(await (await post(a,data)).json()).id);
  assert.equal((await post(a,{...data,input:{method:"pls",holdout:"GSE8917"}})).status,409);
  assert.equal((await post(a,{...data,key:crypto.randomUUID()})).status,429);
  assert.equal((await call("/api/guest/requests/"+saved.id,{cookie:b})).status,404);
  assert.deepEqual((await (await call("/api/guest/session",{cookie:b})).json()).requests,[]);
  assert.equal((await call("/internal/guest-work/claim",{data:{kind:"training"}})).status,404);
  const work=await claim("training"); assert.equal(work.id,saved.id);
  assert.equal(await claim("training"),null);
  assert.equal((await (await finish({...work,fence:crypto.randomUUID()},"completed",{})).json()).updated,false);
  assert.equal((await (await finish(work,"completed",{refit_samples:567})).json()).updated,true);
  assert.equal((await (await call("/api/guest/requests/"+saved.id,{cookie:a})).json()).result.refit_samples,567);
});
test("paid turns get only this visitor's prior completed conversations",async()=>{
  const {call,session,claim,finish}=setup(), a=await session(), b=await session();
  const ask=cookie=>call("/api/guest/requests",{cookie,data:{key:crypto.randomUUID(),kind:"ai",input:{question:"What was evaluated?"}}});
  await ask(a);let work=await claim("ai"); assert.deepEqual(work.history,[]);
  assert.equal((await (await finish(work,"queued",{})).json()).updated,false);
  await finish(work,"completed",{answer:"A public example."});
  await ask(b);work=await claim("ai");assert.deepEqual(work.history,[]);await finish(work,"completed",{answer:"Another browser."});
  await ask(a);work=await claim("ai");assert.deepEqual(work.history,[{question:"What was evaluated?",answer:"A public example."}]);
});

test("new chats, previous guest messages and sources remain scoped to their browser and thread",async()=>{
  const {call,session,claim,finish}=setup(), a=await session(), b=await session();
  const root="/api/guest/copilot/", first=crypto.randomUUID(), second=crypto.randomUUID();
  const send=(thread,text,key=crypto.randomUUID())=>call(root+"chats/"+thread+"/messages",{
    cookie:a,data:{text,request_id:key}});
  const read=async(path,cookie=a)=>(await call(root+path,{cookie})).json();
  await call("/api/guest/requests",{cookie:a,data:{key:crypto.randomUUID(),kind:"ai",input:{question:"Old form"}}});
  let work=await claim("ai");await finish(work,"completed",{answer:"An earlier answer"});
  const key=crypto.randomUUID(), sent=await (await send(first,"First question",key)).json();
  assert.equal((await (await send(first,"First question",key)).json()).run_id,sent.run_id);
  work=await claim("ai");assert.deepEqual(work.history,[]);
  await finish(work,"completed",{answer:"First answer",model:"Gemini",source_version:"test-release"});
  await send(second,"Other thread");work=await claim("ai");assert.deepEqual(work.history,[]);
  await finish(work,"completed",{answer:"Other answer"});
  await send(first,"Follow up");work=await claim("ai");
  assert.deepEqual(work.history,[{question:"First question",answer:"First answer"}]);
  await finish(work,"completed",{answer:"Follow up answer"});
  assert.deepEqual(new Set((await read("chats")).chats.map(c=>c.chat_id)),new Set(["previous",first,second]));
  assert.deepEqual((await read("chats",b)).chats,[]);
  assert.deepEqual((await read("chats/"+first+"/messages")).messages.map(m=>m.text),["First question","First answer","Follow up","Follow up answer"]);
  assert.equal((await read("chats/previous/messages")).messages[0].text,"Old form");
  assert.equal((await read("runs/"+sent.run_id+"/receipt")).source_version,"test-release");
  for(const path of ["chats/"+first+"/messages","runs/"+sent.run_id+"/receipt"])
    assert.equal((await call(root+path,{cookie:b})).status,404);
  assert.equal((await call(root+"chats/"+first,{cookie:a,method:"DELETE",site:"https://elsewhere.example"})).status,403);
  await call(root+"chats/"+first,{cookie:b,method:"DELETE"});
  assert.equal((await read("chats/"+first+"/messages")).messages.length,4);
  await call(root+"chats/"+first,{cookie:a,method:"DELETE"});
  assert.equal((await call(root+"chats/"+first+"/messages",{cookie:a})).status,404);
  assert.equal((await call(root+"runs/"+sent.run_id+"/receipt",{cookie:a})).status,404);
  assert.equal((await read("chats")).chats.length,2);
  assert.equal((await call("/api/guest/requests/"+sent.run_id.slice(4),{cookie:a})).status,404);
  assert.equal((await send(first,"Stale tab")).status,429);
});

test("stopping a guest answer fences late delivery and leaves the training queue alone",async()=>{
  const {call,session,claim,finish}=setup(), owner=await session(), other=await session();
  const root="/api/guest/copilot/", chat=crypto.randomUUID();
  const response=await call(root+"chats/"+chat+"/messages",{cookie:owner,data:{text:"Question",request_id:crypto.randomUUID()}});
  const {run_id}=await response.json(), work=await claim("ai");
  const cancel=cookie=>call(root+"runs/"+run_id+"/cancel",{cookie,data:{}});
  assert.equal((await cancel(other)).status,404);
  assert.equal((await (await cancel(owner)).json()).status,"cancelled");
  assert.equal((await (await finish(work,"completed",{answer:"Late answer"})).json()).updated,false);
  const history=await (await call(root+"chats/"+chat+"/messages",{cookie:owner})).json();
  assert.equal(history.chat.active_run,null);assert.match(history.messages.at(-1).text,/stopped/);
  assert.equal(await claim("ai"),null);assert.equal(await claim("training"),null);
});
