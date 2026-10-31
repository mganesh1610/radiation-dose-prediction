"""Private Cloud Run web adapter. Auth fails closed; numerical training is absent."""
import json
import os
import re
import sys
import time
from pathlib import Path

# Start the protected dependency while the website's own imports are loading.
# Gunicorn imports this module in its worker; do not use --preload with threads.
from .startup_dependency import start_dependency
_STARTUP_DEPENDENCY = start_dependency()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from flask import Flask, request, jsonify, g
from cloud_io import Cloud, Conflict, PREFIX
from job_queue import Queue
from application_service import ApplicationService
from training_contract import validate_job, PRESETS, CORE
from dispatch_control import DispatchControl, BudgetExceeded, month_key
from analysis_service import Explorer, validate_options

DATASET = {'bytes': 11489050, 'generation': '1790877713873726',
           'object': 'training/v1/releases/core_fdf0d10c09a29915/core.npz',
           'sha256': 'fdf0d10c09a29915413787383c42a080d85dbbdac2ae4de86b6a8009221ed5e1'}
if os.environ.get('DEFENDER_DATASET_REF'):
    DATASET = json.loads(os.environ['DEFENDER_DATASET_REF'])
    validate_job(dict(schema_version=1, job_id='configuration_check', dataset=DATASET,
                      method='median_null', holdout_study='GSE8917',
                      purpose='engineering_smoke', parameters={}))
AUDIENCE = os.environ.get('DEFENDER_IAP_AUDIENCE', '')


def verify_identity(headers):
    from cachecontrol import CacheControl
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token
    import requests
    # Cache Google's signing keys; never trust the unsigned email header alone.
    if not hasattr(verify_identity, 'transport'):
        verify_identity.transport = Request(session=CacheControl(requests.Session()))
    token = headers.get('X-Goog-IAP-JWT-Assertion', '')
    if not token or not AUDIENCE:
        raise PermissionError('Google sign-in required')
    try:
        claims = id_token.verify_token(token, verify_identity.transport,
            audience=AUDIENCE, certs_url='https://www.gstatic.com/iap/verify/public_key')
    except Exception:
        raise PermissionError('Invalid sign-in assertion') from None
    if claims.get('iss') != 'https://cloud.google.com/iap' or not claims.get('sub'):
        raise PermissionError('Invalid sign-in issuer')
    allowed = set(filter(None, os.environ.get('ALLOWED_EMAILS', '').lower().split(',')))
    email = claims.get('email', '').lower()
    if not email or email not in allowed:
        raise PermissionError('Account is not authorized for this research workspace')
    return email


def create_app(cloud_factory=Cloud, identity=verify_identity, prediction_engine=None):
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    reviewer_limit = os.environ.get('DEFENDER_REVIEWER_MONTHLY_API_LIMIT', '150')
    if not reviewer_limit.isascii() or not reviewer_limit.isdecimal() or not 1 <= int(reviewer_limit) <= 750:
        raise ValueError('Reviewer API allowance must be an integer from 1 to 750')
    reviewer_limit = int(reviewer_limit)
    app.config['MAX_CONTENT_LENGTH'] = 850_000
    explorer = Explorer(DATASET)
    from app.request_admission import RequestAdmission
    admission = RequestAdmission()
    from app.copilot_proxy import register_copilot
    register_copilot(app)

    @app.before_request
    def secure():
        g.request_started = time.perf_counter()
        g.timing = {}
        g.cloud_stage = 'authentication'
        g.owner = identity(request.headers)
        g.timing['web_auth'] = time.perf_counter() - g.request_started
        g.reviewer = g.owner in set(filter(None, os.environ.get('REVIEWER_EMAILS', '').lower().split(',')))
        allowed_delete=request.method=='DELETE' and request.path.startswith(('/api/copilot/chats/','/api/copilot/memories/'))
        allowed_patch=request.method=='PATCH' and request.path.startswith('/api/copilot/memories/')
        if request.method not in ('GET', 'HEAD', 'POST') and not (allowed_delete or allowed_patch):
            return jsonify(error='Method not allowed'), 405
        if request.method in ('POST','PATCH','DELETE'):
            limit = 850_000 if request.path == '/api/uploads' else (10000 if request.path.startswith('/api/copilot/') else 4096)
            request.max_content_length = limit
            if request.content_length and request.content_length > limit:
                return jsonify(error='Request exceeds the route size limit'), 413
            # Custom header + JSON + same-origin browser policy block cross-site form/CSRF requests.
            origin = request.headers.get('Origin')
            expected = 'https://' + request.host
            if (request.headers.get('X-Defender-Request') != '1' or not request.is_json
                    or (origin and origin != expected)):
                return jsonify(error='Same-origin JSON request required'), 403
        if request.path.startswith('/api/'):
            admission_started = time.perf_counter()
            g.cloud_stage = 'request_admission'
            g.cloud = cloud_factory(max_requests=45, max_upload_bytes=1_000_000)
            g.control = DispatchControl(g.cloud)
            g.admission = admission.reserve(g.control, g.owner, reviewer_limit if g.reviewer else None)
            g.timing['web_admission'] = time.perf_counter() - admission_started
            g.cloud_stage = 'application'
            g.service = ApplicationService(g.cloud, Queue(g.cloud), {'dataset': DATASET})

    @app.after_request
    def headers(response):
        if request.path == '/api/copilot/status' and hasattr(g, 'request_started'):
            g.timing['web_total'] = time.perf_counter() - g.request_started
            response.headers['Server-Timing'] = ', '.join(
                f'{name};dur={seconds*1000:.3f}' for name,seconds in g.timing.items())
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    @app.errorhandler(PermissionError)
    def denied(error): return jsonify(error=str(error)), 403
    @app.errorhandler(ValueError)
    def invalid(error): return jsonify(error=str(error)), 400
    @app.errorhandler(FileNotFoundError)
    def missing(error): return jsonify(error='Record not found'), 404
    @app.errorhandler(Conflict)
    def conflict(error): return jsonify(error='State changed; retry the same request'), 409
    @app.errorhandler(BudgetExceeded)
    def budget(error): return jsonify(error=str(error)), 429
    @app.errorhandler(RuntimeError)
    def unavailable(error):
        # Fixed stage/type/status only. No exception messages, prompts, IDs,
        # provider response bodies, request URLs, assertions or tokens.
        app.logger.warning('Cloud operation failed: stage=%s type=%s operation=%s status=%s',
            getattr(g,'cloud_stage','unknown'), type(error).__name__,
            getattr(error,'operation','unclassified'), getattr(error,'status',0))
        return jsonify(error='Cloud operation unavailable; retry the same request with its original ID'), 503

    def owned(job_id):
        match = re.fullmatch(r'web_(\d{4})(\d{2})_[a-f0-9]{32}', job_id)
        if not match: raise ValueError('Invalid application job ID')
        control = DispatchControl(g.cloud, match[1] + '-' + match[2])
        state = g.admission if control.month == g.control.month else control.read()[0]
        entry = state['jobs'].get(job_id)
        if not entry or entry['owner'] != g.owner: raise FileNotFoundError()
        return entry

    def training_bridge():
        if os.environ.get('TRAINING_BRIDGE_ENABLED','false').lower()!='true':
            raise PermissionError('Chat training bridge is not enabled')
        from training_bridge import TrainingBridge
        policy=json.loads(Path(os.environ['TRAINING_BRIDGE_POLICY']).read_text())
        return TrainingBridge(g.cloud,g.owner,DATASET,policy,reviewer=g.reviewer)

    @app.get('/api/training/catalog')
    def training_catalog():
        training_bridge()
        return jsonify(methods=PRESETS,studies=CORE,month=g.control.month,
                       jobs=[k for k,v in g.admission['jobs'].items() if v['owner']==g.owner],
                       training_enabled=os.environ.get('TRAINING_ENABLED','true').lower()=='true',
                       role='reviewer' if g.reviewer else 'researcher')

    @app.post('/api/training/prepare')
    def training_prepare():return jsonify(training_bridge().prepare(request.get_json()))

    @app.post('/api/training/approve')
    def training_approve():
        if os.environ.get('TRAINING_ENABLED','true').lower()!='true':raise PermissionError('New training is paused')
        return jsonify(training_bridge().approve(request.get_json())),202

    @app.get('/api/training/jobs/<job_id>')
    def training_status(job_id):return jsonify(training_bridge().status(job_id))

    @app.post('/api/training/jobs/<job_id>/cancel')
    def training_cancel(job_id):
        if request.get_json()!={}:raise ValueError('Cancellation requires an empty object')
        return jsonify(training_bridge().cancel(job_id))

    @app.get('/')
    def home(): return app.send_static_file('index.html')

    @app.get('/api/catalog')
    def catalog(): return jsonify(explorer.catalog(g.cloud))

    @app.post('/api/analysis')
    def analysis():
        options = validate_options(request.get_json())
        return jsonify(explorer.analyze(g.cloud, options))

    @app.get('/api/bootstrap')
    def bootstrap():
        try:
            worker, _ = g.cloud.read_json(PREFIX + 'control/worker.json')
        except FileNotFoundError:
            worker = {'status': 'not_started', 'checked_at': None}
        # A timestamp is evidence of a past check-in, never a guarantee of current availability.
        return jsonify(**g.service.methods(), studies=CORE, release_id='core_'+DATASET['sha256'][:16],
                       worker=worker, month=g.control.month,
                       jobs=[k for k,v in g.admission['jobs'].items() if v['owner']==g.owner],
                       limits={'api_requests':g.control.http_limit, 'jobs':12}, requests_used=g.admission['requests'],
                       jobs_used=len(g.admission['jobs']),
                       access={'role':'reviewer' if g.reviewer else 'researcher',
                               'job_limit':2 if g.reviewer else 12,
                               'requests_used':g.admission.get('requests_by_owner', {}).get(g.owner,0),
                               'request_limit':reviewer_limit if g.reviewer else g.control.http_limit},
                       training_enabled=os.environ.get('TRAINING_ENABLED','true').lower() == 'true')

    @app.post('/api/jobs')
    def submit():
        if os.environ.get('TRAINING_ENABLED','true').lower() != 'true':
            raise PermissionError('New training is paused. Existing results remain available.')
        body = request.get_json()
        if not isinstance(body,dict) or set(body) != {'job_id','method','holdout_study'}:
            raise ValueError('Expected job_id, method and holdout_study')
        prefix = 'web_' + g.control.month.replace('-', '') + '_'
        if not re.fullmatch(re.escape(prefix) + '[a-f0-9]{32}', str(body['job_id'])):
            raise ValueError('Job ID must use the current admission month')
        if not isinstance(body['method'],str) or body['method'] not in PRESETS: raise ValueError('Unsupported method')
        if not isinstance(body['holdout_study'],str): raise ValueError('Invalid held-out study')
        job = dict(body, schema_version=1, dataset=DATASET, purpose='engineering_smoke',
                   parameters=PRESETS[body['method']])
        validate_job(job)
        # Durable admission precedes job creation. Worker can repair a crash between these writes.
        g.control.admit(job, g.owner, 2 if g.reviewer else None)
        return jsonify(g.service.create_job(body)), 202

    @app.get('/api/jobs/<job_id>')
    def status(job_id):
        entry = owned(job_id)
        try: return jsonify(g.service.get_job(job_id))
        except FileNotFoundError:
            return jsonify(job_id=job_id, status='QUEUED', stage='admitted', method=entry['request']['method'])

    @app.get('/api/jobs/<job_id>/results')
    def results(job_id):
        owned(job_id)
        return jsonify(g.service.results(job_id))

    @app.post('/api/jobs/<job_id>/cancel')
    def cancel(job_id):
        entry = owned(job_id)
        Queue(g.cloud).submit(entry['request'])
        Queue(g.cloud).cancel(job_id)
        return jsonify(g.service.get_job(job_id))

    def predictions():
        nonlocal prediction_engine
        # Lazy imports keep static/authentication requests free of numerical runtime work.
        from frozen_inference import FrozenInference
        from prediction_service import PredictionService
        if prediction_engine is None:
            folder = Path(os.environ.get('FROZEN_MODEL_DIR', str(Path(__file__).resolve().parents[1] / 'runtime/frozen')))
            if not (folder / 'manifest.json').exists():
                raise RuntimeError('Frozen prediction release is unavailable')
            prediction_engine = FrozenInference(folder)
        return PredictionService(g.cloud, prediction_engine, g.owner, g.reviewer)

    @app.get('/api/prediction-catalog')
    def prediction_catalog():
        service = predictions()
        return jsonify(**service.engine.catalog(), role='reviewer' if g.reviewer else 'researcher', uploads=service.list())

    @app.get('/api/prediction-chat-catalog')
    def prediction_chat_catalog():
        service=predictions()
        return jsonify(**service.engine.catalog(),uploads=[{'upload_id':r['upload_id']} for r in service.list(owned_only=True)])

    @app.get('/api/prediction-examples/<sample>')
    def prediction_example(sample):
        engine = predictions().engine
        row = next((r for r in engine.manifest['examples'] if r['id'] == sample), None)
        if not row: raise FileNotFoundError()
        return app.response_class((engine.folder / row['file']).read_bytes(), mimetype='application/json',
                                  headers={'Content-Disposition': 'attachment; filename="' + row['file'] + '"'})

    @app.post('/api/uploads')
    def upload(): return jsonify(predictions().upload(request.get_json())), 201

    @app.get('/api/uploads/<uid>')
    def upload_record(uid): return jsonify(predictions().get(uid, review=not g.reviewer))

    @app.post('/api/uploads/<uid>/review')
    def review_upload(uid): return jsonify(predictions().review(uid, request.get_json()))

    @app.post('/api/predict')
    def predict():
        body = request.get_json()
        if not isinstance(body, dict) or set(body) != {'upload_id', 'model'}:
            raise ValueError('Expected upload_id and model')
        return jsonify(predictions().predict(body['upload_id'], body['model']))

    @app.get('/api/prediction-runtime-check')
    def runtime_check():
        if g.reviewer: raise PermissionError('Runtime checks are researcher-only')
        engine = predictions().engine
        replay = json.loads((engine.folder / 'replay.json').read_text())
        errors = {}
        for sample, expected in replay['expected'].items():
            body = json.loads((engine.folder / (sample + '.json')).read_text())
            x, qc = engine.validate(body)
            if qc['blockers']: raise RuntimeError('Curated replay incompatible')
            panels, _ = engine.panels(x)
            for model, value in expected.items():
                errors[sample + '/' + model] = abs(float(engine.predict_values(model, panels)[0]) - value)
        passed = max(errors.values()) <= replay['tolerance_gy']
        return jsonify(passed=passed, errors_gy=errors, tolerance_gy=replay['tolerance_gy'],
                       release_sha256=engine.release_sha, test_data_used=False), 200 if passed else 503

    return app


app = create_app()
