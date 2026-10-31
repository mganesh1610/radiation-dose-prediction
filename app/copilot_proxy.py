"""Same-origin private chat bridge. Backend re-verifies the original IAP JWT."""
import json
import os
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request
from urllib.error import HTTPError
from flask import request,Response,jsonify,g,has_request_context
import urllib3


def _transport_time(name, operation, *args, **kwargs):
    started=time.perf_counter()
    try:return operation(*args,**kwargs)
    finally:
        if has_request_context() and hasattr(g,'timing'):
            g.timing[name]=g.timing.get(name,0)+time.perf_counter()-started


class TimedHTTPSConnection(urllib3.connection.HTTPSConnection):
    """Numeric phase measurements only; retain urllib3's TLS/auth behavior."""
    def _new_conn(self):return _transport_time('web_tcp',super()._new_conn)
    def connect(self):return _transport_time('web_connect',super().connect)
    def request(self,*args,**kwargs):return _transport_time('web_send',super().request,*args,**kwargs)
    def getresponse(self):return _transport_time('web_wait',super().getresponse)


class TimedHTTPSPool(urllib3.connectionpool.HTTPSConnectionPool):
    ConnectionCls=TimedHTTPSConnection
    def _get_conn(self,*args,**kwargs):
        return _transport_time('web_pool',super()._get_conn,*args,**kwargs)


class PooledResponse:
    """Return consumed connections; never retain authentication on the pool."""
    def __init__(self, response):
        self.response=response
        self.status=response.status
        self.headers=response.headers
    def __enter__(self):return self
    def read(self, size):return self.response.read(size)
    def __exit__(self, *args):
        # Incomplete/oversized/failed bodies must not be reused.
        if not self.response.isclosed():self.response.close()
        self.response.release_conn()


class CopilotTransport:
    def __init__(self):
        # Same bound as website HTTP concurrency. Pool exhaustion is bounded;
        # there are no automatic retries of state-changing chat requests.
        self.pool=urllib3.PoolManager(num_pools=1,maxsize=8,block=True,
                                     cert_reqs='CERT_REQUIRED')
        # Instance-local mapping: do not affect other cloud clients/pools.
        self.pool.pool_classes_by_scheme={**urllib3.poolmanager.pool_classes_by_scheme,
                                          'https':TimedHTTPSPool}
    def open(self, request, timeout):
        response=self.pool.request(request.get_method(),request.full_url,
            body=request.data,headers=dict(request.header_items()),
            timeout=urllib3.Timeout(connect=10,read=timeout),pool_timeout=2,
            retries=False,redirect=False,preload_content=False)
        return PooledResponse(response)


def register_copilot(app, *, token_provider=None, opener=None):
    transport=CopilotTransport() if opener is None else None
    def proxy(path):
        base=os.environ.get('COPILOT_BACKEND_URL','').rstrip('/')
        if not base:
            if path=='status':return jsonify(enabled=False,reason='Managed chat is not enabled on this deployment')
            return jsonify(error='Managed chat is not enabled on this deployment'),503
        parsed=urlsplit(base)
        if parsed.scheme!='https' or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise RuntimeError('Invalid protected backend configuration')
        if not re.fullmatch(r'(status|memories(?:/(response_style|display_format))?|chats|chats/chat-[a-f0-9]{32}(?:/messages)?|runs/run-[a-f0-9]{32}(?:/(events|receipt|cancel|artifacts/plot-[a-f0-9]{32}))?)',path):
            return jsonify(error='Route not found'),404
        g.cloud_stage = 'copilot_service_token'
        token_started=time.perf_counter()
        if token_provider:token=token_provider(base)
        else:
            from google.auth.transport.requests import Request as AuthRequest
            from google.oauth2.id_token import fetch_id_token
            token=fetch_id_token(AuthRequest(),base)
        if hasattr(g,'timing'):g.timing['web_token']=time.perf_counter()-token_started
        url=base+'/api/copilot/'+path
        if request.query_string:
            if len(request.query_string)>100:raise ValueError('Invalid query')
            url+='?'+request.query_string.decode('ascii')
        # Cloud Run removes reserved IAP headers on the next service hop. Carry
        # the original signed assertion in an application header; verify its
        # Google signature and original IAP audience again at the backend.
        headers={'Authorization':'Bearer '+token,'X-Research-IAP-Assertion':request.headers.get('X-Goog-IAP-JWT-Assertion',''),
            'X-Defender-Request':'1','Content-Type':'application/json'}
        if request.headers.get('Origin'):headers['Origin']=request.headers['Origin']
        outgoing=Request(url,data=request.get_data() if request.method in {'POST','PATCH','DELETE'} else None,
            method=request.method,headers=headers)
        g.cloud_stage = 'copilot_transport'
        upstream_started=time.perf_counter()
        try:response=(opener or transport.open)(outgoing,timeout=190)
        except HTTPError as exc:response=exc
        except Exception:raise RuntimeError('Chat backend unavailable') from None
        headers_received=time.perf_counter()
        with response:
            raw=response.read(512001)
            if len(raw)>512000:raise RuntimeError('Chat response too large')
            try:payload=json.loads(raw)
            except ValueError:raise RuntimeError('Invalid chat response') from None
            if hasattr(g,'timing'):
                g.timing['web_headers']=headers_received-upstream_started
                g.timing['web_body']=time.perf_counter()-headers_received
                g.timing['web_upstream']=time.perf_counter()-upstream_started
                # Only fixed numeric metrics from our authenticated service.
                timing=getattr(response,'headers',{}).get('Server-Timing','')
                for name,value in re.findall(r'(cp_auth|cp_handler|cp_total);dur=([0-9]+(?:\.[0-9]+)?)',timing):
                    if 0<=float(value)<=190000:g.timing[name]=float(value)/1000
            return Response(json.dumps(payload),status=response.status,content_type='application/json')
    app.add_url_rule('/api/copilot/<path:path>','copilot_proxy',proxy,methods=['GET','POST','PATCH','DELETE'])
