"""Private canonical chat routes. No unsigned identity headers or public CORS."""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from ..contracts.chat import CreateChat,SendMessage,EmptyBody
from ..persistence.control_store import Conflict,QuotaExceeded
from ..persistence.preferences import PreferenceChange, PreferenceDelete
from .startup_timing import phase


def create_app(service, authenticator, preferences=None, protocol_enabled=False, feature_users=None, *, artifacts=None,prediction_enabled=False,training_enabled=False):
    @asynccontextmanager
    async def lifespan(app):
        try:
            started=time.perf_counter()
            prepare=getattr(service.sessions,'prepare',None)
            if prepare is not None:await prepare()
            phase('controls_ready' if hasattr(service.executor,'readiness') else 'sessions_ready',started)
            yield
        finally:
            close=getattr(service.executor,'aclose',None)
            if close is not None:await close()
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)

    @app.middleware("http")
    async def secured(request, call_next):
        started=time.perf_counter()
        try:
            # Cloud Run IAM protects this service. The website's fixed startup
            # probe uses only transport readiness; normal status still verifies
            # the user's signed IAP assertion and current workspace grant.
            if request.method=='GET' and request.url.path in {"/healthz","/api/copilot/runtime-health"}:
                readiness=getattr(service.executor,'readiness',None)
                return JSONResponse({"status":"ok",**(readiness() if readiness is not None else {})})
            request.state.principal=await asyncio.to_thread(authenticator.authenticate,request.headers)
            authenticated=time.perf_counter()
            if request.method not in {"GET","POST","PATCH","DELETE"}:return JSONResponse({"error":"Method not allowed"},status_code=405)
            if request.method in {"POST","PATCH","DELETE"}:
                if (request.headers.get("x-defender-request")!="1" or request.headers.get("origin") not in (None,service.config.origin)
                        or request.headers.get("content-type","").split(";")[0]!="application/json"):
                    raise PermissionError("Same-origin JSON request required")
                chunks=[];size=0
                async for chunk in request.stream():
                    size+=len(chunk)
                    if size>10000:return JSONResponse({"error":"Request too large"},status_code=413)
                    chunks.append(chunk)
                request._body=b''.join(chunks)
            response=await call_next(request)
        except PermissionError:return JSONResponse({"error":"Google sign-in and workspace access required"},status_code=403)
        response.headers["Cache-Control"]="no-store"
        response.headers["X-Content-Type-Options"]="nosniff"
        if request.url.path=="/api/copilot/status":
            finished=time.perf_counter()
            response.headers['Server-Timing']=(f'cp_auth;dur={(authenticated-started)*1000:.3f}, '
                f'cp_handler;dur={(finished-authenticated)*1000:.3f}, cp_total;dur={(finished-started)*1000:.3f}')
        return response

    for error,code,message in [(PermissionError,403,"Access denied"),(FileNotFoundError,404,"Record not found"),
            (Conflict,409,"State changed; retry the same request"),(QuotaExceeded,429,"Pilot allowance or concurrency limit reached"),
            (ValidationError,400,"Unsupported request"),(ValueError,400,"Invalid request"),(RuntimeError,503,"Service unavailable; retry the same request")]:
        async def handler(request, exc, status=code, text=message):return JSONResponse({"error":text},status_code=status)
        app.add_exception_handler(error,handler)

    @app.get("/api/copilot/status")
    async def status(request:Request):
        await asyncio.to_thread(service.authorize,request.state.principal)
        readiness=getattr(service.executor,'readiness',None)
        ready=readiness() if readiness is not None else {'execution_ready':True,'startup_state':'ready'}
        return {"enabled":True,"mode":"adk_a2a",**ready,"release_id":service.config.release_id,"session_ttl_days":30,
                "analysis_enabled":artifacts is not None,
                "training_enabled":training_enabled,"prediction_enabled":prediction_enabled,
                "memory_enabled":preferences is not None and preferences.enabled_for(request.state.principal),
                "protocol_rag_enabled":protocol_enabled and (feature_users is None or request.state.principal.user_id in feature_users)}

    async def memory_service(principal):
        await asyncio.to_thread(service.authorize,principal)
        if preferences is None: raise RuntimeError('Preference memory is not enabled')
        return preferences

    @app.get('/api/copilot/memories')
    async def memories(request:Request):
        return await (await memory_service(request.state.principal)).list(request.state.principal)

    @app.patch('/api/copilot/memories/{key}')
    async def save_memory(key:str, request:Request):
        body=PreferenceChange.model_validate(await request.json())
        return await (await memory_service(request.state.principal)).save(request.state.principal,key,body)

    @app.delete('/api/copilot/memories/{key}')
    async def delete_memory(key:str, request:Request):
        body=PreferenceDelete.model_validate(await request.json())
        return await (await memory_service(request.state.principal)).delete(request.state.principal,key,body.expected_revision)

    @app.post("/api/copilot/chats")
    async def create(request:Request):
        body=CreateChat.model_validate(await request.json())
        return await service.create(request.state.principal,body.request_id)

    @app.get("/api/copilot/chats")
    async def chats(request:Request):return {"chats":await service.list(request.state.principal)}

    @app.get("/api/copilot/chats/{chat_id}/messages")
    async def messages(chat_id:str,request:Request,offset:int=0):
        if not 0<=offset<=1000:raise ValueError("Invalid page")
        return await service.messages(request.state.principal,chat_id,offset)

    @app.post("/api/copilot/chats/{chat_id}/messages")
    async def send(chat_id:str,request:Request):
        body=SendMessage.model_validate(await request.json())
        return await service.send(request.state.principal,chat_id,body.request_id,body.text)

    @app.delete("/api/copilot/chats/{chat_id}")
    async def delete(chat_id:str,request:Request):
        EmptyBody.model_validate(await request.json())
        return await service.delete(request.state.principal,chat_id)

    @app.get("/api/copilot/runs/{run_id}")
    async def run(run_id:str,request:Request):return service.public_run(await service.get_run(request.state.principal,run_id))

    @app.get("/api/copilot/runs/{run_id}/events")
    async def events(run_id:str,request:Request,after:int=0):
        if not 0<=after<=32:raise ValueError("Invalid cursor")
        r=await service.get_run(request.state.principal,run_id)
        return {"run_id":run_id,"status":r["status"],"events":[e for e in r["events"] if e["seq"]>after]}

    @app.get("/api/copilot/runs/{run_id}/receipt")
    async def receipt(run_id:str,request:Request):
        r=await service.get_run(request.state.principal,run_id)
        return {k:r[k] for k in ("run_id","status","model_calls","tool_calls","input_bytes","prompt_tokens","output_tokens","reserved_model_usd","receipts")}

    @app.post("/api/copilot/runs/{run_id}/cancel")
    async def cancel(run_id:str,request:Request):
        EmptyBody.model_validate(await request.json())
        return await service.cancel(request.state.principal,run_id)
    @app.get('/api/copilot/runs/{run_id}/artifacts/{artifact_id}')
    async def artifact(run_id:str,artifact_id:str,request:Request):
        if artifacts is None:raise FileNotFoundError()
        return await artifacts.get(service,request.state.principal,run_id,artifact_id)
    return app
