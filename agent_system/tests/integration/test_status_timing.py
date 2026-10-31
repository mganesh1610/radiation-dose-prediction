"""Numeric latency diagnostics must not bypass identity or disclose contents."""
import io
import pytest
from flask import Flask, g
from app.copilot_proxy import register_copilot
from test_chat import setup, headers


def test_backend_timing_preserves_auth_and_revocation():
    client, _, _, enabled = setup()
    assert client.get('/api/copilot/status').status_code == 403
    response = client.get('/api/copilot/status', headers=headers())
    assert response.status_code == 200
    timing = dict(item.strip().split(';dur=') for item in response.headers['Server-Timing'].split(','))
    assert set(timing) == {'cp_auth', 'cp_handler', 'cp_total'}
    assert all(float(value) >= 0 for value in timing.values())
    assert 'alice' not in response.headers['Server-Timing']
    enabled.remove('alice')
    assert client.get('/api/copilot/status', headers=headers()).status_code == 403


def test_proxy_timing_accepts_only_bounded_numeric_metrics(monkeypatch):
    monkeypatch.setenv('COPILOT_BACKEND_URL', 'https://private-backend.run.app')
    class Reply(io.BytesIO):
        status = 200
        headers = {'Server-Timing': 'cp_auth;dur=1.2, cp_handler;dur=2.1, cp_total;dur=999999, secret;desc="private"'}
    app = Flask(__name__)
    @app.before_request
    def start():
        g.timing = {}
    seen = []
    @app.after_request
    def end(response):
        seen.append(g.timing)
        return response
    register_copilot(app, token_provider=lambda url: 'fixture',
                     opener=lambda *args, **kwargs: Reply(b'{"enabled":true}'))
    assert app.test_client().get('/api/copilot/status').status_code == 200
    assert seen[0]['cp_auth'] == pytest.approx(.0012) and seen[0]['cp_handler'] == pytest.approx(.0021)
    assert set(seen[0]) == {'cp_auth', 'cp_handler', 'web_token', 'web_upstream', 'web_headers', 'web_body'}
    assert seen[0]['web_upstream'] >= seen[0]['web_headers'] + seen[0]['web_body']


def test_transport_phase_metrics_are_request_scoped_and_exceptions_propagate(monkeypatch):
    from app.copilot_proxy import _transport_time,CopilotTransport,TimedHTTPSPool
    import urllib3
    original=urllib3.poolmanager.pool_classes_by_scheme.copy()
    transport=CopilotTransport()
    assert transport.pool.pool_classes_by_scheme['https'] is TimedHTTPSPool
    assert urllib3.poolmanager.pool_classes_by_scheme==original
    ticks=iter([1,1.25,2,2.1,3,3.2])
    monkeypatch.setattr('app.copilot_proxy.time.perf_counter',lambda:next(ticks))
    app=Flask(__name__)
    with app.test_request_context():
        g.timing={}
        assert _transport_time('web_wait',lambda:'result')=='result'
        assert g.timing=={'web_wait':.25}
        def failure():raise OSError('fixture')
        with pytest.raises(OSError):_transport_time('web_connect',failure)
        assert g.timing['web_connect']==pytest.approx(.1)
    with app.test_request_context():
        g.timing={}
        _transport_time('web_wait',lambda:None)
        assert g.timing=={'web_wait':pytest.approx(.2)}
    transport.pool.clear()
