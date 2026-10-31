"""Download original GEO processed matrices and SOFT metadata, never cloud-upload.

Reuses only previously checksummed, gzip-validated files; records provenance locally.
"""
import gzip, hashlib, json, re, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/public_radiation'
LOG = ROOT / 'context/dataset-acquisition'
STUDIES = {'core': ['GSE102971','GSE8917','GSE65292'],
           'auxiliary_candidate': ['GSE85570','GSE97000','GSE20173'],
           'locked_test': ['GSE90909'],
           'conditional_test': ['GSE15341','GSE274788','GSE40640']}

def fetch(url):
    return urllib.request.urlopen(url, timeout=120)

def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            tmp = dest.with_suffix(dest.suffix+'.part')
            with fetch(url) as response, tmp.open('wb') as out:
                while block := response.read(1024*1024): out.write(block)
            if dest.name.endswith('.gz'):
                with gzip.open(tmp,'rb') as stream:
                    while stream.read(1024*1024): pass
            tmp.replace(dest)
            return {'url':url,'path':str(dest.relative_to(ROOT)), 'bytes':dest.stat().st_size,
                    'sha256':hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()}
        except Exception:
            if attempt == 2: raise
            time.sleep(2**attempt)

def main():
    LOG.mkdir(parents=True,exist_ok=True)
    previous = json.loads((LOG/'downloads.json').read_text()) if (LOG/'downloads.json').exists() else []
    records = []
    for role, accessions in STUDIES.items():
        for acc in accessions:
            base = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/'
            folder = OUT/role/acc
            try:
                listing = fetch(base+'matrix/').read().decode()
                names = re.findall(r'href="([^"/]+_series_matrix[^"/]*\.txt\.gz)"',listing)
                urls = [base+'matrix/'+name for name in names]
                urls.append(base+f'soft/{acc}_family.soft.gz')
                if acc == 'GSE274788':
                    urls.append(base+'suppl/GSE274788_gene_count_matrix.txt.gz')
                if acc == 'GSE40640':
                    urls.append(base+'suppl/GSE40640_non-normalized.txt.gz')
                try:
                    supp = fetch(base+'suppl/').read().decode()
                    (LOG/f'{acc}-supplementary.html').write_text(supp)
                except Exception as e: print(acc,'supplementary inventory:',e,flush=True)
                for url in urls:
                    dest = folder/url.rsplit('/',1)[1]
                    old = next((r for r in previous if r.get('url')==url and 'sha256' in r),None)
                    if old and dest.exists() and hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==old['sha256']:
                        records.append(old); print('REUSED',acc,dest.name,flush=True)
                    else:
                        record = download(url,dest); record.update(accession=acc,role=role)
                        records.append(record); print('VERIFIED',acc,dest.name,record['bytes'],flush=True)
                    (LOG/'downloads.json').write_text(json.dumps(records,indent=2))
                if not names: print('NO MATRIX',acc,'inspect supplementary inventory',flush=True)
            except Exception as e:
                records.append({'accession':acc,'role':role,'error':str(e)})
                print('FAILED',acc,str(e),flush=True)
            (LOG/'downloads.json').write_text(json.dumps(records,indent=2))
    try:
        metadata=json.loads(fetch('https://zenodo.org/api/records/22013472').read())
        (LOG/'zenodo-22013472.json').write_text(json.dumps(metadata,indent=2))
        for item in metadata['files']:
            url=item['links']['self'];dest=OUT/'conditional_test/Zenodo22013472'/item['key']
            old=next((r for r in previous if r.get('url')==url and 'sha256' in r),None)
            if old and dest.exists() and hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==old['sha256']:
                record=old
            else:
                record=download(url,dest);record.update(accession='Zenodo22013472',role='conditional_test')
            assert dest.stat().st_size==item['size']
            assert 'md5:'+hashlib.md5(dest.read_bytes()).hexdigest()==item['checksum']
            records.append(record)
    except Exception as e:
        records.append({'accession':'Zenodo22013472','error':str(e)})
    (LOG/'downloads.json').write_text(json.dumps(records,indent=2))
    print('TOTAL_BYTES',sum(r.get('bytes',0) for r in records),flush=True)

if __name__ == '__main__': main()
