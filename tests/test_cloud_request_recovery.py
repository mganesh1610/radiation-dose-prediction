"""Fault-injected frontend cloud admission; no network, tokens or training."""
import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1])]
from cloud_io import Cloud,CloudUnavailable,Conflict,PREFIX
from dispatch_control import DispatchControl

@pytest.fixture(autouse=True)
def configured_cloud():
    with patch('cloud_io.PROJECT','example-project'), patch('cloud_io.PROJECT_NUMBER','123'), patch('cloud_io.BUCKET','example'):
        yield


URL='https://storage.googleapis.com/upload/storage/v1/b/example/o?ifGenerationMatch=7'


def client():
    import time
    c=Cloud();c._token='test-only';c._token_time=time.monotonic();return c


def failure(status):
    return urllib.error.HTTPError(URL,status,'provider private message',{},io.BytesIO(b'secret body'))


def test_rate_limited_write_retries_same_generation_and_bytes(caplog):
    c=client();requests=[]
    def send(req,timeout):
        requests.append(req)
        if len(requests)<3:raise failure(429)
        return io.BytesIO(b'ok')
    with patch('urllib.request.urlopen',send),patch('cloud_io.time.sleep'):
        assert c.request(URL,'POST',b'payload')==b'ok'
    assert len(requests)==3 and len({r.full_url for r in requests})==1
    assert all(r.data==b'payload' for r in requests)
    assert c.requests==3 and c.upload_bytes==21
    assert 'status=429' in caplog.text and 'secret' not in caplog.text and 'test-only' not in caplog.text


@pytest.mark.parametrize('error',[failure(503),urllib.error.URLError('private host'),TimeoutError('private body')])
def test_ambiguous_write_is_never_blindly_replayed(error):
    c=client()
    with patch('urllib.request.urlopen',side_effect=error) as send:
        with pytest.raises(CloudUnavailable) as exc:c.request(URL,'POST',b'value')
    assert send.call_count==1 and str(exc.value)=='Cloud operation unavailable'


def test_transient_read_retries_but_per_invocation_guard_is_preserved():
    c=client();c.max_requests=2
    with patch('urllib.request.urlopen',side_effect=failure(503)) as send,patch('cloud_io.time.sleep'):
        with pytest.raises(CloudUnavailable):c.request(URL)
    assert send.call_count==2 and c.requests==2


def test_replaced_generation_is_conflict_never_missing_record():
    c=client()
    with patch.object(c,'metadata',return_value={'generation':'7','size':'2'}),patch.object(c,'request',side_effect=FileNotFoundError):
        with pytest.raises(Conflict):c.read_json(PREFIX+'control/test.json')


def test_dispatch_rebases_after_read_conflict_without_reset():
    class Fake:
        def __init__(self):self.reads=0;self.saved=None
        def read_json(self,name):
            self.reads+=1
            if self.reads==1:raise Conflict('replaced generation')
            return {'requests':18,'jobs':{},'requests_by_owner':{'alice':7}},'10'
        def state(self,name,state,generation):self.saved=state;assert generation=='10'
    c=Fake()
    with patch('dispatch_control.time.sleep'):
        result=DispatchControl(c).reserve_request('alice')
    assert result['requests']==19 and result['requests_by_owner']['alice']==8


def test_frontend_failure_logs_fixed_stage_and_status_only(caplog):
    from app.server import create_app
    class Fake:
        def read_json(self,name):raise CloudUnavailable('storage_get',503)
    app=create_app(lambda **kwargs:Fake(),lambda headers:'test@example.test')
    response=app.test_client().get('/api/copilot/status')
    assert response.status_code==503
    assert 'stage=request_admission' in caplog.text and 'status=503' in caplog.text
    assert 'test@example.test' not in caplog.text


def test_parallel_admission_preserves_both_reservations():
    import copy
    import threading
    from concurrent.futures import ThreadPoolExecutor
    class Fake:
        def __init__(self):
            self.value={'requests':20,'jobs':{},'requests_by_owner':{'alice':20}}
            self.generation=1;self.lock=threading.Lock();self.barrier=threading.Barrier(2);self.reads=0
        def read_json(self,name):
            with self.lock:
                value,generation=copy.deepcopy(self.value),str(self.generation)
                self.reads+=1;first=self.reads<=2
            if first:self.barrier.wait(timeout=3)
            return value,generation
        def state(self,name,state,generation):
            with self.lock:
                if generation!=str(self.generation):raise Conflict('another writer committed')
                self.value=copy.deepcopy(state);self.generation+=1
    cloud=Fake()
    with patch('dispatch_control.time.sleep'),ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:DispatchControl(cloud).reserve_request('alice'),range(2)))
    assert len(results)==2 and cloud.value['requests']==22
    assert cloud.value['requests_by_owner']['alice']==22


def test_uncertain_committed_admission_stops_without_second_increment():
    class Fake:
        def __init__(self):self.value={'requests':4,'jobs':{}};self.writes=0
        def read_json(self,name):return dict(self.value),'1'
        def state(self,name,state,generation):
            self.value=state;self.writes+=1
            raise CloudUnavailable('storage_post',503)
    cloud=Fake()
    with pytest.raises(CloudUnavailable):DispatchControl(cloud).reserve_request()
    assert cloud.writes==1 and cloud.value['requests']==5
