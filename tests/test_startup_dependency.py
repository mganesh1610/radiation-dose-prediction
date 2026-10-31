"""Offline checks for bounded startup preparation; no provider or model calls."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
from urllib.parse import parse_qs, urlsplit
import pytest
from app.startup_dependency import StartupDependency, protected_base, start_dependency

BASE='https://research-copilot-460004627518.us-west1.run.app'

class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body);self.status=status;self.headers=headers or {}

def test_parallel_start_calls_probe_only_once_without_user_or_model_requests():
    gate=threading.Event();entered=threading.Event();calls=[];records=[]
    def open(request,timeout):
        calls.append((request,timeout))
        if len(calls)==1:
            entered.set();assert gate.wait(2)
            return Response(b'first.second.third',headers={'Metadata-Flavor':'Google'})
        return Response(b'{"status":"ok","execution_ready":false}')
    startup=StartupDependency(BASE,opener=open,reporter=records.append)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda _:startup.start(),range(8)))
    assert entered.wait(1) and len(calls)==1 and not startup.done.is_set()
    gate.set();assert startup.done.wait(2);startup.thread.join(1);startup.start()
    assert len(calls)==2
    metadata,health=calls
    assert parse_qs(urlsplit(metadata[0].full_url).query)['audience']==[BASE]
    assert metadata[0].headers=={'Metadata-flavor':'Google'} and metadata[1]==2
    assert health[0].full_url==BASE+'/api/copilot/runtime-health' and health[0].get_method()=='GET'
    assert health[0].headers=={'Authorization':'Bearer first.second.third'} and 0<health[1]<=8
    assert records[0]=={'event':'web_startup_dependency','phase':'begin'}
    assert records[-1]['outcome']=='healthy' and 'first.second.third' not in json.dumps(records)
    assert all('/chats' not in r.full_url and '/status' not in r.full_url for r,_ in calls)

@pytest.mark.parametrize('bad',['http://example.run.app','https://example.com','https://u:p@example.run.app',
    'https://example.run.app/path','https://example.run.app?x=1','https://example.run.app#x',
    'https://example.run.app:444','https://example..run.app',''])
def test_invalid_protected_target_rejected_before_identity_request(bad):
    with pytest.raises(ValueError):protected_base(bad)

@pytest.mark.parametrize('kind',['network','metadata_header','metadata_size','token','health_size','health_json'])
def test_failure_is_bounded_sanitized_and_not_retried(kind):
    calls=[];records=[]
    def open(request,timeout):
        calls.append(request.full_url)
        if kind=='network':raise RuntimeError('sensitive diagnostic first.second.third')
        if len(calls)==1:
            if kind=='metadata_header':return Response(b'a.b.c')
            if kind=='metadata_size':return Response(b'a'*16385,headers={'Metadata-Flavor':'Google'})
            if kind=='token':return Response(b'unsafe\n',headers={'Metadata-Flavor':'Google'})
            return Response(b'first.second.third',headers={'Metadata-Flavor':'Google'})
        return Response(b'a'*2049 if kind=='health_size' else b'not-json')
    s=StartupDependency(BASE,opener=open,reporter=records.append).start();assert s.done.wait(2);s.thread.join(1);s.start()
    assert len(calls)==(2 if kind.startswith('health') else 1)
    assert records[-1]['outcome']=='unavailable' and 'sensitive' not in json.dumps(records)

def test_import_and_default_configuration_never_start_network():
    assert start_dependency({}) is None
    assert start_dependency({'K_SERVICE':'web','COPILOT_BACKEND_URL':BASE}) is None
    assert start_dependency({'COPILOT_STARTUP_PREPARE':'1','COPILOT_BACKEND_URL':BASE}) is None
    assert start_dependency({'K_SERVICE':'web','COPILOT_STARTUP_PREPARE':'1','COPILOT_BACKEND_URL':'https://bad.invalid'}) is None
