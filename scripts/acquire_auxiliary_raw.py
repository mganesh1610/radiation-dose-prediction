"""Acquire the audited auxiliary allowlist, preserving source bytes and provenance."""
import csv
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from acquire_gse102971_raw import fetch, sha

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'context/auxiliary-audit/v1'
LOG = ROOT / 'context/auxiliary-processing/v1'


def selection():
    verification = json.loads((AUDIT/'verification.json').read_text())
    for name in ['raw-selected-files.tsv', 'array-manifest.tsv']:
        path = AUDIT/name
        if sha(path) != verification['artifact_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('Audit input hash changed')
    metadata = {r['gsm']: r for r in csv.DictReader((AUDIT/'array-manifest.tsv').open(), delimiter='\t')}
    rows = []
    for r in csv.DictReader((AUDIT/'raw-selected-files.tsv').open(), delimiter='\t'):
        m = metadata[r['gsm']]
        if m['metadata_eligible_d'] != 'True' or m['study'] != r['study']:
            raise ValueError('Unapproved raw member')
        url = urlparse(r['url'])
        name = Path(url.path).name
        if (url.scheme != 'https' or url.netloc != 'ftp.ncbi.nlm.nih.gov'
                or not url.path.startswith('/geo/samples/') or url.query or url.fragment
                or not name.startswith(r['gsm']) or not name.endswith('.gz')):
            raise ValueError('Unexpected raw source URL')
        path = ROOT/'data/raw_radiation'/r['study']/name
        rows.append(dict(study=r['study'], gsm=r['gsm'], url=r['url'],
                         expected_bytes=int(r['compressed_bytes']), path=str(path.relative_to(ROOT))))
    if Counter(r['study'] for r in rows) != {'GSE85570':412,'GSE97000':12,'GSE20173':10}:
        raise ValueError('Incorrect cohort inventory')
    if len({r['gsm'] for r in rows}) != 434:
        raise ValueError('Duplicate accession')
    return rows


def main():
    LOG.mkdir(parents=True,exist_ok=True)
    rows = selection()
    if shutil.disk_usage(ROOT).free < 8_000_000_000:
        raise RuntimeError('Insufficient temporary disk space')
    manifest=LOG/'downloads.json'
    previous={r['gsm']:r for r in json.loads(manifest.read_text())['files']} if manifest.exists() else {}
    prior=json.loads((AUDIT/'sources/raw-format-downloads.json').read_text())
    examples=prior['files'] if isinstance(prior,dict) else prior
    for r in examples:
        previous.setdefault(r['gsm'],r)
    for r in rows:
        p=ROOT/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
        example=ROOT/'data/auxiliary_raw_format/v1'/p.name
        if not p.exists() and example.exists():
            if r['gsm'] not in previous or sha(example)!=previous[r['gsm']]['sha256']:
                raise ValueError('Raw format example hash mismatch')
            shutil.copyfile(example,p)
    done,errors=[],[]
    print(f'Acquiring {len(rows)} audited files; {sum(r["expected_bytes"] for r in rows):,} compressed bytes',flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,r,previous):r for r in rows}
        for future in as_completed(futures):
            try: done.append(future.result())
            except Exception as exc: errors.append(dict(gsm=futures[future]['gsm'],error=str(exc)))
            result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),complete=len(done)==434 and not errors,
                        scope='audited prospective/paired auxiliary raw only',files=sorted(done,key=lambda x:x['gsm']),errors=errors,
                        source_checksum_note='Publisher-listed byte count and gzip CRC; SHA-256 measured locally, not publisher SHA-256.',
                        selection_sha256=sha(AUDIT/'raw-selected-files.tsv'),code_sha256=sha(Path(__file__)))
            temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(manifest)
            if (len(done)+len(errors))%20==0 or errors: print(f'Verified {len(done)}/434; errors {len(errors)}',flush=True)
    if errors: raise RuntimeError('Acquisition incomplete; see downloads.json')
    print(f'COMPLETE: {sum(r["bytes"] for r in done):,} compressed; {sum(r["expanded_bytes"] for r in done):,} expanded bytes',flush=True)


if __name__=='__main__':main()
