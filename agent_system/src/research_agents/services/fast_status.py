"""Live authenticated status without importing FastAPI, ADK or the GCS SDK.

Every status request verifies the original signed IAP claims, reads the current
immutable identity binding once, and re-reads the protected grant policy. No
identity/grant/status cache is used. Unknown bindings and all other routes use
the original application, including its binding/CAS and execution safeguards.
"""
import asyncio
import json
import os
import re
import threading
import time
from urllib.parse import quote
from .startup_settings import read_settings
from .startup_timing import phase
from ..policies.access import IapAuthenticator, AccessPolicy, Grant


class LiveBindings:
    def __init__(self,bucket,workspace,credentials):
        self.bucket,self.workspace,self.credentials=bucket,workspace,credentials
        self.local=threading.local();self.sessions=[];self.lock=threading.Lock()

    def read(self,user):
        if not re.fullmatch(r'usr_[a-f0-9]{64}',user):raise PermissionError()
        if not hasattr(self.local,'session'):
            from google.auth.transport.requests import AuthorizedSession
            session=AuthorizedSession(self.credentials,max_refresh_attempts=0)
            self.local.session=session
            with self.lock:self.sessions.append(session)
        key='copilot/v1/'+self.workspace+'/identities/'+user+'.json'
        url='https://storage.googleapis.com/storage/v1/b/'+quote(self.bucket,safe='')+'/o/'+quote(key,safe='')+'?alt=media'
        with self.local.session.get(url,headers={'Accept-Encoding':'identity','Cache-Control':'no-cache'},stream=True,timeout=(2,5),allow_redirects=False) as response:
            if response.status_code==404:return None
            if response.status_code!=200:raise RuntimeError('Identity store unavailable')
            raw=response.raw.read(1_000_001,decode_content=True)
            if len(raw)>1_000_000:raise RuntimeError('Identity record too large')
            value=json.loads(raw)
            if not isinstance(value,dict):raise RuntimeError('Invalid identity record')
            return value

    def close(self):
        for session in self.sessions:session.close()


class FastStatusApp:
    def __init__(self,settings,config,policy,users,bindings,verifier,loader):
        self.settings,self.config,self.policy,self.users=settings,config,policy,frozenset(users or [])
        self.bindings,self.verifier,self.loader=bindings,verifier,loader
        self.full=None;self.full_task=None;self.lifespan=None;self.failed=False;self.closing=False

    def authorize(self,headers):
        verified={}
        def verify(token,audience):
            claims=self.verifier(token,audience);verified['email']=claims.get('email','').lower();return claims
        principal=IapAuthenticator(self.settings['iap_audience'],self.policy.emails,verifier=verify,
            assertion_header='X-Research-IAP-Assertion').authenticate(headers)
        binding=self.bindings.read(principal.user_id)
        if binding is None:return None  # Original binder handles the new identity.
        expected={'user_id':principal.user_id,'email':verified['email']}
        if {k:v for k,v in binding.items() if k!='_control_ops'}!=expected:raise PermissionError('Identity mismatch')
        def current_grant(user,workspace):
            p=self.policy.read()
            if p['workspace']!=workspace or binding['email'] not in [m['email'] for m in p['members'] if m['enabled']]:return None
            return Grant(user,workspace,frozenset(p['releases']),p['revision'])
        AccessPolicy(current_grant).authorize(principal,self.config.workspace,self.config.release_id)
        return principal

    def readiness(self):
        if self.full is not None:return self.full.state.execution_readiness()
        return {'execution_ready':False,'startup_state':'failed' if self.failed else 'warming'}

    def status(self,principal):
        p=self.settings
        return {'enabled':True,'mode':'adk_a2a',**self.readiness(),'release_id':self.config.release_id,'session_ttl_days':30,
            'analysis_enabled':p.get('analysis_enabled',False),'training_enabled':p.get('training_enabled',False),
            'prediction_enabled':p.get('prediction_enabled',False),'memory_enabled':p.get('memory_enabled',False) and principal.user_id in self.users,
            'protocol_rag_enabled':p.get('protocol_rag_enabled',False) and principal.user_id in self.users}

    async def _load(self):
        try:
            app=await asyncio.to_thread(self.loader)
            if app.state.protected_chat_config!=self.config:raise ValueError('Protected configuration changed')
            if self.closing:raise RuntimeError('Research services are closing')
            lifespan=app.router.lifespan_context(app);await lifespan.__aenter__();self.lifespan=lifespan
            self.full=app;return app
        except Exception:
            self.failed=True;raise RuntimeError('Research services unavailable') from None

    async def application(self):
        if self.closing:raise RuntimeError('Research services are closing')
        if self.full_task is None:self.full_task=asyncio.create_task(self._load())
        return await asyncio.wait_for(asyncio.shield(self.full_task),90)

    async def response(self,send,code,payload,started=None):
        data=json.dumps(payload).encode();headers=[(b'content-type',b'application/json'),(b'content-length',str(len(data)).encode()),
            (b'cache-control',b'no-store'),(b'x-content-type-options',b'nosniff')]
        if started is not None:
            ms=(time.perf_counter()-started)*1000
            headers.append((b'server-timing',f'cp_auth;dur={ms:.3f}, cp_handler;dur=0.000, cp_total;dur={ms:.3f}'.encode()))
        await send({'type':'http.response.start','status':code,'headers':headers});await send({'type':'http.response.body','body':data})

    async def __call__(self,scope,receive,send):
        if scope['type']=='lifespan':
            while True:
                message=await receive()
                if message['type']=='lifespan.startup':
                    phase('thin_controls_ready',time.perf_counter());await send({'type':'lifespan.startup.complete'})
                elif message['type']=='lifespan.shutdown':
                    self.closing=True
                    if self.full_task is not None:
                        try:await asyncio.wait_for(asyncio.shield(self.full_task),8)
                        except Exception:
                            self.full_task.cancel();await asyncio.gather(self.full_task,return_exceptions=True)
                    if self.lifespan is not None:await self.lifespan.__aexit__(None,None,None)
                    close=getattr(self.bindings,'close',None)
                    if close:close()
                    await send({'type':'lifespan.shutdown.complete'});return
        if scope['type']=='http' and scope['method']=='GET':
            path=scope['path']
            if path in {'/healthz','/api/copilot/runtime-health'}:
                return await self.response(send,200,{'status':'ok',**self.readiness()})
            if path=='/api/copilot/status':
                started=time.perf_counter();headers={}
                for key,value in scope.get('headers',[]):headers.setdefault(key.decode('latin1').lower(),value.decode('latin1'))
                try:
                    principal=await asyncio.to_thread(self.authorize,{'X-Research-IAP-Assertion':headers.get('x-research-iap-assertion','')})
                    if principal is not None:return await self.response(send,200,self.status(principal),started)
                except PermissionError:return await self.response(send,403,{'error':'Access denied'},started)
                except Exception:return await self.response(send,503,{'error':'Service unavailable; retry the same request'},started)
        try:app=await self.application()
        except Exception:
            if scope['type']=='http':return await self.response(send,503,{'error':'Service unavailable; retry the same request'})
            raise
        await app(scope,receive,send)


def coordinator_app():
    started=time.perf_counter();path=os.environ['COPILOT_CONFIG'];p,config,policy,users=read_settings(path)
    from ..persistence.credentials import cloud_credentials
    credentials=cloud_credentials()
    bindings=LiveBindings(p['control_bucket'],config.workspace,credentials)
    verifier=IapAuthenticator(p['iap_audience'],policy.emails,assertion_header='X-Research-IAP-Assertion').verifier
    def load():
        from .control_bootstrap import coordinator_app as full_app
        return full_app()
    app=FastStatusApp(p,config,policy,users,bindings,verifier,load)
    phase('thin_controls_factory',started);return app
