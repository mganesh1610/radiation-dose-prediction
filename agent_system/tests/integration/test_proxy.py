import io
import json
from flask import Flask
from app.copilot_proxy import register_copilot
from app.copilot_proxy import CopilotTransport


def test_disabled_proxy_keeps_fallback(monkeypatch):
    monkeypatch.delenv('COPILOT_BACKEND_URL',raising=False)
    app=Flask(__name__);register_copilot(app)
    assert app.test_client().get('/api/copilot/status').json['enabled'] is False


def test_private_runtime_health_is_not_a_public_proxy_route(monkeypatch):
    monkeypatch.setenv('COPILOT_BACKEND_URL','https://private-backend.run.app')
    app=Flask(__name__)
    def unexpected(*a,**k):raise AssertionError('Private runtime health must not be proxied')
    register_copilot(app,token_provider=unexpected,opener=unexpected)
    assert app.test_client().get('/api/copilot/runtime-health').status_code==404


def test_proxy_forwards_signed_identity_but_not_cookies(monkeypatch):
    monkeypatch.setenv('COPILOT_BACKEND_URL','https://private-backend.run.app')
    captured=[]
    class Reply(io.BytesIO):status=200
    def open(request,timeout):captured.append(request);return Reply(b'{"enabled":true}')
    app=Flask(__name__);register_copilot(app,token_provider=lambda url:'fixture-service-token',opener=open)
    client=app.test_client();client.set_cookie('session','never-forward')
    r=client.get('/api/copilot/status',headers={'X-Goog-IAP-JWT-Assertion':'fixture-signed-token'})
    assert r.status_code==200
    sent={k.lower():v for k,v in captured[0].header_items()}
    assert sent['x-research-iap-assertion']=='fixture-signed-token'
    assert sent['authorization']=='Bearer fixture-service-token'
    assert 'cookie' not in sent


def test_artifact_proxy_allows_only_typed_same_backend_route(monkeypatch):
    monkeypatch.setenv('COPILOT_BACKEND_URL','https://private-backend.run.app')
    captured=[]
    class Reply(io.BytesIO):status=200
    def open(request,timeout):captured.append(request);return Reply(b'{"schema_version":1}')
    app=Flask(__name__);register_copilot(app,token_provider=lambda url:'fixture-service-token',opener=open)
    client=app.test_client();base='/api/copilot/runs/run-'+'a'*32+'/artifacts/'
    assert client.get(base+'plot-'+'b'*32).status_code==200
    assert captured[0].full_url=='https://private-backend.run.app'+base+'plot-'+'b'*32
    for suffix in ['plot-invalid','..%2Fledger.json','https://other.test','plot-'+'b'*32+'.json']:
        assert client.get(base+suffix).status_code==404
    assert len(captured)==1


def test_pooled_transport_keeps_each_identity_and_disables_redirect_retry(monkeypatch):
    from urllib.request import Request
    calls=[]
    class Reply:
        status=200
        headers={}
        def __init__(self):self.closed=False;self.released=False
        def read(self,size):self.closed=True;return b'{}'
        def isclosed(self):return self.closed
        def close(self):self.closed=True
        def release_conn(self):self.released=True
    replies=[]
    class Pool:
        def __init__(self,**kwargs):assert kwargs==dict(num_pools=1,maxsize=8,block=True,cert_reqs='CERT_REQUIRED')
        def request(self,method,url,**kwargs):
            calls.append((method,url,kwargs));r=Reply();replies.append(r);return r
    monkeypatch.setattr('app.copilot_proxy.urllib3.PoolManager',Pool)
    transport=CopilotTransport()
    for token in ('owner-fixture','reviewer-fixture'):
        request=Request('https://private-backend.run.app/api/copilot/status',headers={'Authorization':token})
        with transport.open(request,190) as response:assert response.read(512001)==b'{}'
    assert [c[2]['headers']['Authorization'] for c in calls]==['owner-fixture','reviewer-fixture']
    assert all(c[2]['retries'] is False and c[2]['redirect'] is False and c[2]['pool_timeout']==2 for c in calls)
    assert all(r.released for r in replies)
    # Broken/oversized response is discarded before releasing its slot.
    with transport.open(Request('https://private-backend.run.app/api/copilot/status'),190):pass
    assert replies[-1].closed and replies[-1].released


def test_real_loopback_transport_reuses_connection_without_identity_or_cookie_leak():
    import threading
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from urllib.request import Request
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def do_GET(self):
            seen.append((self.client_address,self.headers.get('Authorization'),self.headers.get('Cookie')))
            self.send_response(302 if self.path=='/redirect' else 200)
            self.send_header('Content-Length','2')
            self.send_header('Set-Cookie','private=never-store')
            if self.path=='/redirect':self.send_header('Location','/should-not-follow')
            self.end_headers();self.wfile.write(b'{}')
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    transport=CopilotTransport()
    try:
        for identity,path in [('first-fixture','/'),('second-fixture','/'),('third-fixture','/redirect')]:
            request=Request('http://127.0.0.1:'+str(server.server_port)+path,headers={'Authorization':identity})
            with transport.open(request,2) as response:
                assert response.status==(302 if path=='/redirect' else 200)
                assert response.read(512001)==b'{}'
        assert len(seen)==3
        assert seen[0][0]==seen[1][0]==seen[2][0]
        assert [r[1] for r in seen]==['first-fixture','second-fixture','third-fixture']
        assert all(r[2] is None for r in seen)
    finally:
        transport.pool.clear();server.shutdown();server.server_close();thread.join(2)
