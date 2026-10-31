"""One bounded, authenticated dependency probe per Cloud Run worker startup.

This overlaps Coordinator startup with website imports. It has no user identity,
scientific query, model call, retry or periodic keep-alive. Status responses still
come from the normal authenticated Coordinator route, including fresh grants.
"""
import json
import os
import re
import threading
import time
from urllib.parse import urlsplit, urlencode


def protected_base(value):
    base = value.rstrip('/')
    p = urlsplit(base)
    if (p.scheme != 'https' or p.username or p.password or p.path or p.query or p.fragment
            or p.port not in (None, 443) or not p.hostname
            or not re.fullmatch(r'[a-z0-9]+(?:[.-][a-z0-9]+)*\.run\.app', p.hostname)):
        raise ValueError('Invalid protected dependency URL')
    return base


class StartupDependency:
    def __init__(self, base, *, opener=None, reporter=None):
        self.base = protected_base(base)
        self.opener = opener
        self.reporter = reporter or (lambda record: print(json.dumps(record), flush=True))
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = None

    def start(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='copilot-startup', daemon=True)
                self.thread.start()
        return self

    def _run(self):
        started = time.monotonic()
        outcome = 'unavailable'
        stage = 'transport_setup'
        failure = None
        self.reporter({'event': 'web_startup_dependency', 'phase': 'begin'})
        try:
            from urllib.request import Request, build_opener, HTTPRedirectHandler
            class NoRedirect(HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):return None
            opener = self.opener or build_opener(NoRedirect()).open
            stage = 'metadata_identity'
            url = ('http://metadata.google.internal/computeMetadata/v1/instance/'
                   'service-accounts/default/identity?' + urlencode({'audience': self.base, 'format': 'full'}))
            with opener(Request(url, headers={'Metadata-Flavor': 'Google'}), timeout=2) as response:
                stage = 'metadata_validation'
                token = response.read(16385)
                if response.status != 200 or response.headers.get('Metadata-Flavor') != 'Google':
                    raise ValueError('Metadata unavailable')
                if not 1 <= len(token) <= 16384:raise ValueError('Invalid identity response')
                token = token.decode('ascii').strip()
                if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', token):
                    raise ValueError('Invalid identity response')
            remaining = 10 - (time.monotonic() - started)
            if remaining <= 0:raise TimeoutError()
            request = Request(self.base + '/api/copilot/runtime-health', headers={'Authorization': 'Bearer ' + token})
            stage = 'private_health'
            with opener(request, timeout=min(8, remaining)) as response:
                stage = 'health_validation'
                raw = response.read(2049)
                if response.status == 200 and len(raw) <= 2048 and json.loads(raw).get('status') == 'ok':
                    outcome = 'healthy'
        except Exception as error:
            # Numeric/type diagnostics only: no tokens, response bodies or URLs.
            failure = {'type': type(error).__name__}
            if type(getattr(error, 'code', None)) is int:failure['http_status'] = error.code
            close = getattr(error, 'close', None)
            if close is not None:close()
        finally:
            self.done.set()
            self.reporter({'event': 'web_startup_dependency', 'phase': 'end', 'outcome': outcome,
                           'stage': stage, 'failure': failure,
                           'duration_ms': round((time.monotonic() - started) * 1000, 3)})


def start_dependency(environ=None):
    env = os.environ if environ is None else environ
    if not env.get('K_SERVICE') or env.get('COPILOT_STARTUP_PREPARE') != '1':return None
    try:return StartupDependency(env.get('COPILOT_BACKEND_URL', '')).start()
    except ValueError:return None
