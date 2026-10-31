"""Small, bounded Google Cloud REST adapter. Tokens are never persisted.

Uses the existing gcloud login; does not create keys, enable APIs or change IAM.
"""
import base64
import hashlib
import json
import os
import logging
import random
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

PROJECT = os.environ.get('DEFENDER_GCP_PROJECT', '')
PROJECT_NUMBER = os.environ.get('DEFENDER_GCP_PROJECT_NUMBER', '')
BUCKET = os.environ.get('DEFENDER_GCS_BUCKET', '')
REGION = 'us-west1'
PREFIX = 'training/v1/'
DATASET = 'radiation_research_v1'


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Conflict(RuntimeError):
    pass


class CloudUnavailable(RuntimeError):
    """Sanitized diagnostics: never retain provider bodies, URLs or credentials."""
    def __init__(self, operation, status=0):
        self.operation, self.status = operation, status
        super().__init__('Cloud operation unavailable')


class Cloud:
    def __init__(self, max_requests=250, max_upload_bytes=100_000_000):
        self.requests = 0
        self.upload_bytes = 0
        self.max_requests = max_requests
        self.max_upload_bytes = max_upload_bytes
        self._token = None
        self._token_time = 0

    def request(self, url, method='GET', data=None, content_type='application/json'):
        if not PROJECT or not PROJECT_NUMBER.isdigit() or not BUCKET:
            raise RuntimeError('Set DEFENDER_GCP_PROJECT, DEFENDER_GCP_PROJECT_NUMBER and DEFENDER_GCS_BUCKET before cloud operations')
        if urllib.parse.urlparse(url).hostname not in ('storage.googleapis.com', 'bigquery.googleapis.com'):
            raise ValueError('Unexpected API host')
        if self.requests >= self.max_requests:
            raise RuntimeError('Per-invocation request guard reached')
        self.requests += 1
        if data is not None:
            self.upload_bytes += len(data)
            if self.upload_bytes > self.max_upload_bytes:
                raise RuntimeError('Per-invocation upload guard reached')
        if self._token is None or time.monotonic() - self._token_time > 2400:
            if os.environ.get('K_SERVICE'):
                request = urllib.request.Request(
                    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                    headers={'Metadata-Flavor': 'Google'})
                with urllib.request.urlopen(request, timeout=5) as response:
                    self._token = json.load(response)['access_token']
            else:
                self._token = subprocess.run(['gcloud', 'auth', 'print-access-token'],
                    capture_output=True, text=True, check=True, timeout=30).stdout.strip()
            self._token_time = time.monotonic()
        req = urllib.request.Request(url, data=data, method=method, headers={
            'Authorization': 'Bearer '+self._token, 'Content-Type': content_type})
        operation = ('storage_' if urllib.parse.urlparse(url).hostname == 'storage.googleapis.com' else 'query_') + method.lower()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if status in (409, 412):
                    raise Conflict('Object generation or job conflict') from None
                if status == 404:
                    raise FileNotFoundError('Cloud object not found') from None
                # Only explicit rate-limit rejection is retried for writes.
                # 5xx/transport failures may have committed; never rebase those
                # mutations as if they had failed. Reads are safe to retry.
                retry = status == 429 or (method == 'GET' and status in (408, 500, 502, 503, 504))
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                status, retry = 0, method == 'GET'
            if not retry or attempt == 3 or self.requests >= self.max_requests:
                raise CloudUnavailable(operation, status) from None
            logging.getLogger(__name__).warning('cloud_retry operation=%s status=%d attempt=%d', operation, status, attempt+1)
            time.sleep(1.1 * 2**attempt + random.uniform(0, .25))
            self.requests += 1
            if data is not None:
                self.upload_bytes += len(data)
                if self.upload_bytes > self.max_upload_bytes:
                    raise CloudUnavailable('upload_guard')

    def api(self, path, method='GET', body=None, service='storage'):
        base = ('https://storage.googleapis.com/storage/v1/' if service == 'storage'
                else 'https://bigquery.googleapis.com/bigquery/v2/')
        return json.loads(self.request(base+path, method, None if body is None else encode(body)))

    def name(self, name):
        if not name.startswith(PREFIX) or '..' in name.split('/') or '\\' in name:
            raise ValueError('Object outside permitted training prefix')
        return urllib.parse.quote(name, safe='')

    def metadata(self, name):
        return self.api(f'b/{BUCKET}/o/{self.name(name)}')

    def read(self, name, generation, expected_sha=None, limit=80_000_000):
        meta = self.api(f'b/{BUCKET}/o/{self.name(name)}?generation={int(generation)}')
        if int(meta['size']) > limit:
            raise ValueError('Object exceeds input limit')
        data = self.request(f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/'
                            f'{self.name(name)}?alt=media&generation={int(generation)}')
        if len(data) != int(meta['size']) or (expected_sha and digest(data) != expected_sha):
            raise ValueError('Cloud payload checksum/size mismatch')
        return data

    def read_json(self, name):
        meta = self.metadata(name)
        if int(meta['size']) > 1_000_000:
            raise ValueError('Object exceeds input limit')
        try:
            data = self.request(f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/'
                                f'{self.name(name)}?alt=media&generation={int(meta["generation"])}')
        except FileNotFoundError:
            # Old generations can disappear when a competing writer replaces
            # the record. This is not evidence of a missing/empty ledger.
            raise Conflict('Object changed during read') from None
        if len(data) != int(meta['size']):
            raise ValueError('Cloud payload size mismatch')
        return json.loads(data), meta['generation']

    def put(self, name, data, generation=0, content_type='application/octet-stream'):
        self.name(name)
        params = urllib.parse.urlencode(dict(uploadType='media', name=name, ifGenerationMatch=str(generation)))
        result = json.loads(self.request(f'https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o?{params}',
                                        'POST', data, content_type))
        md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
        if int(result['size']) != len(data) or result.get('md5Hash') != md5:
            raise ValueError('Upload integrity mismatch')
        return {'object':name, 'generation':result['generation'], 'sha256':digest(data), 'bytes':len(data)}

    def immutable(self, name, data):
        try:
            return self.put(name, data)
        except Conflict:
            meta = self.metadata(name)
            actual = self.read(name, meta['generation'], digest(data))
            return {'object':name, 'generation':meta['generation'], 'sha256':digest(actual), 'bytes':len(actual)}

    def state(self, name, body, generation):
        return self.put(name, encode(body), generation, 'application/json')['generation']

    def preflight(self, extra_bytes):
        bucket = self.api(f'b/{BUCKET}')
        if (bucket['location'] != 'US-WEST1' or bucket.get('storageClass') != 'STANDARD'
                or bucket['projectNumber'] != PROJECT_NUMBER
                or bucket['iamConfiguration'].get('publicAccessPrevention') != 'enforced'
                or not bucket['iamConfiguration']['uniformBucketLevelAccess']['enabled']
                or bucket.get('versioning', {}).get('enabled') or bucket.get('autoclass', {}).get('enabled')):
            raise ValueError('Bucket policy/location changed')
        total = 0
        for mode in ('versions=true', 'softDeleted=true'):
            page = ''
            while True:
                result = self.api(f'b/{BUCKET}/o?{mode}&maxResults=1000'+page)
                total += sum(int(obj['size']) for obj in result.get('items', []))
                if not result.get('nextPageToken'): break
                page = '&pageToken='+urllib.parse.quote(result['nextPageToken'], safe='')
        if total + extra_bytes > 4_000_000_000:
            raise ValueError('4 GB storage guard exceeded, including retained objects')
        return {'stored_bytes_including_retained':total, 'planned_additional_bytes':extra_bytes,
                'projected_bytes':total+extra_bytes, 'billing_cap':False}

    def ensure_dataset(self):
        path = f'projects/{PROJECT}/datasets/{DATASET}'
        try:
            value = self.api(path, service='bq')
        except FileNotFoundError:
            value = self.api(f'projects/{PROJECT}/datasets', 'POST', {
                'datasetReference':{'projectId':PROJECT, 'datasetId':DATASET}, 'location':REGION,
                'description':'Versioned radiation research development metadata; not clinical predictions.',
                'labels':{'purpose':'research', 'schema':'v1'}}, 'bq')
        if value['location'].lower() != REGION:
            raise ValueError('BigQuery location mismatch')
        return value

    def load_table(self, table, fields, source, job_key, expected_rows):
        # GCS source is immutable; its generation and SHA are checked before use.
        self.read(source['object'], source['generation'], source['sha256'], limit=10_000_000)
        job_id = 'load_'+hashlib.sha256(job_key.encode()).hexdigest()[:32]
        ledger_name=PREFIX+'warehouse_loads/'+job_id+'.json'
        try:
            saved,_=self.read_json(ledger_name)
        except FileNotFoundError:
            saved=None
        if saved is not None:
            if saved['source']!=source or saved['table']!=table or saved['expected_rows']!=expected_rows:
                raise ValueError('Load ledger conflicts with request')
            return saved['result']
        path = f'projects/{PROJECT}/jobs'
        body = {'jobReference':{'projectId':PROJECT, 'jobId':job_id, 'location':REGION}, 'configuration':{
            'load':{'destinationTable':{'projectId':PROJECT,'datasetId':DATASET,'tableId':table},
                'sourceUris':[f'gs://{BUCKET}/'+source['object']], 'sourceFormat':'NEWLINE_DELIMITED_JSON',
                'schema':{'fields':fields}, 'writeDisposition':'WRITE_APPEND',
                'createDisposition':'CREATE_IF_NEEDED', 'ignoreUnknownValues':False, 'maxBadRecords':0}}}
        try:
            result = self.api(path, 'POST', body, 'bq')
        except Conflict:
            result = self.api(path+f'/{job_id}?location={REGION}', service='bq')
        # An existing deterministic ID must belong to the same exact request.
        prior = result.get('configuration', {}).get('load', {})
        if prior.get('sourceUris') != body['configuration']['load']['sourceUris'] or prior.get('destinationTable') != body['configuration']['load']['destinationTable']:
            raise ValueError('Existing load job has different inputs')
        for _ in range(30):
            if result['status']['state'] == 'DONE': break
            time.sleep(1)
            result = self.api(path+f'/{job_id}?location={REGION}', service='bq')
        if result['status']['state'] != 'DONE':
            raise RuntimeError('Load still pending; retry the same deterministic job')
        if result['status'].get('errorResult'):
            raise RuntimeError('BigQuery load failed: '+json.dumps(result['status']['errorResult']))
        if int(result['statistics']['load']['outputRows']) != expected_rows:
            raise ValueError('BigQuery loaded row count differs')
        self.immutable(ledger_name,encode({'source':source,'table':table,'expected_rows':expected_rows,'result':result}))
        return result
