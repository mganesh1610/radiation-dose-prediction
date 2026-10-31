"""Upload only checksummed source datasets; retain provenance locally.

Default: preflight only. Use --execute for the authorized upload.
This size guard is not a billing cap; account-wide Free Tier use must be checked.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/public_radiation'
LOG = ROOT / 'context/cloud'
from cloud_io import PROJECT, PROJECT_NUMBER, BUCKET
PREFIX = 'radiation-dose/'


def crc32c(path):
    """Castagnoli CRC used by GCS composite objects (not zlib's CRC-32)."""
    table = []
    for i in range(256):
        c = i
        for _ in range(8): c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
        table.append(c)
    def update(c, data):
        for b in data: c = table[(c ^ b) & 0xff] ^ (c >> 8)
        return c
    assert update(0xffffffff, b'123456789') ^ 0xffffffff == 0xe3069283
    c = 0xffffffff
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024): c = update(c, block)
    return base64.b64encode((c ^ 0xffffffff).to_bytes(4, 'big')).decode()


def main():
    if not PROJECT or not PROJECT_NUMBER.isdigit() or not BUCKET:
        raise ValueError('Configure the cloud project, project number and bucket first')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    records = json.loads((ROOT/'context/dataset-acquisition/downloads.json').read_text())
    planned = []
    for record in records:
        path = (ROOT/record['path']).resolve(strict=True)
        relative = path.relative_to(DATA.resolve())  # Reject anything outside public data.
        if len(relative.parts) != 3 or path.name == 'AGENTS.md':
            raise ValueError(f'Unexpected source layout: {relative}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'Source integrity check failed: {path}')
        with path.open('rb') as stream:
            md5 = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
        planned.append({'local_path':str(path),'object':PREFIX+relative.as_posix(),
                        'bytes':record['bytes'],'sha256':actual,'md5_base64':md5})
    if len({r['object'] for r in planned}) != len(planned):
        raise ValueError('Duplicate destination names')
    token = subprocess.check_output(['gcloud','auth','print-access-token'],text=True).strip()

    def get(endpoint, params=None):
        url = 'https://storage.googleapis.com/storage/v1/'+endpoint
        if params: url += '?'+urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={'Authorization':'Bearer '+token})
        with urllib.request.urlopen(req,timeout=60) as response: return json.load(response)

    def listing():
        items=[];params={'versions':'true'}
        while True:
            result=get('b/'+BUCKET+'/o',params);items.extend(result.get('items',[]))
            if not result.get('nextPageToken'):return items
            params['pageToken']=result['nextPageToken']

    bucket=get('b/'+BUCKET)
    assert bucket['projectNumber']==PROJECT_NUMBER, 'Unexpected owning project'
    assert bucket['location']=='US-WEST1' and bucket['storageClass']=='STANDARD'
    assert bucket['iamConfiguration']['publicAccessPrevention']=='enforced'
    assert bucket['iamConfiguration']['uniformBucketLevelAccess']['enabled']
    assert not bucket.get('versioning',{}).get('enabled',False)
    assert not bucket.get('autoclass',{}).get('enabled',False)
    existing=listing()
    by_name={r['name']:r for r in existing}
    extra_bytes=sum(r['bytes'] for r in planned if r['object'] not in by_name)
    if sum(int(r['size']) for r in existing)+extra_bytes > 4_000_000_000:
        raise ValueError('Destination would exceed the 4-GB working guard')
    for r in planned:
        old=by_name.get(r['object'])
        if old:
            if old.get('md5Hash'):
                matches = old['md5Hash'] == r['md5_base64']
            else:
                r['crc32c_base64'] = crc32c(r['local_path'])
                matches = old.get('crc32c') == r['crc32c_base64']
            if int(old['size'])!=r['bytes'] or not matches:
                raise ValueError('Existing object differs; refusing overwrite: '+r['object'])
    LOG.mkdir(parents=True,exist_ok=True)
    (LOG/'upload-plan.json').write_text(json.dumps({'bucket':BUCKET,'objects':planned},indent=2))
    print(f'Preflight passed: {len(planned)} files, {sum(r["bytes"] for r in planned):,} bytes',flush=True)
    if not args.execute:return
    for number,r in enumerate(planned,1):
        if r['object'] in by_name:
            print(f'{number}/{len(planned)} already verified: {r["object"]}',flush=True)
            continue
        cmd=['gcloud','storage','cp',r['local_path'],'gs://'+BUCKET+'/'+r['object'],
             '--project='+PROJECT,'--if-generation-match=0',
             '--content-md5='+r['md5_base64'],'--storage-class=STANDARD',
             '--content-type=application/octet-stream','--quiet']
        env=dict(os.environ, CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED='False')
        result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        print(f'{number}/{len(planned)} uploaded: {r["object"]}',flush=True)
    remote=listing(); index={r['name']:r for r in remote}
    verified=[]
    for r in planned:
        obj=index[r['object']]
        assert int(obj['size'])==r['bytes'], r['object']
        if obj.get('md5Hash'):
            assert obj['md5Hash']==r['md5_base64'], r['object']
            method='MD5'
        else:
            if 'crc32c_base64' not in r:r['crc32c_base64']=crc32c(r['local_path'])
            assert obj.get('crc32c')==r['crc32c_base64'], r['object']
            method='CRC32C'
        assert obj['storageClass']=='STANDARD', r['object']
        verified.append(dict(r,uri='gs://'+BUCKET+'/'+r['object'],generation=obj['generation'],
                             cloud_md5=obj.get('md5Hash'),cloud_crc32c=obj['crc32c'],verified_checksum=method))
    report={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'bucket':BUCKET,'region':'us-west1','storage_class':'STANDARD',
            'public_access_prevention':'enforced','verified_objects':verified,
            'verified_count':len(verified),'total_bytes':sum(r['bytes'] for r in verified),
            'remote_object_versions':len(remote),'uploads_overwrite_existing':False}
    (LOG/'upload-verification.json').write_text(json.dumps(report,indent=2))
    print(f'VERIFIED {len(verified)} remote objects by byte size and MD5 or CRC32C; local SHA-256 checked.',flush=True)


if __name__=='__main__':main()
