"""Acquire the 100 human GSE102971 per-array source files; no cloud operations.

Source metadata, file inventories and resulting provenance remain local in context/.
"""
import csv
import gzip
import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/raw-pilot/GSE102971'
OUT = ROOT / 'data/raw_radiation/GSE102971'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def samples():
    with (ROOT / 'context/dataset-acquisition/sample-metadata.tsv').open() as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t')
                if r['study'] == 'GSE102971' and r['organism_ch1'] == 'Homo sapiens']
    listing = ROOT / 'context/dataset-acquisition/preprocessing-audit/GSE102971-raw-filelist.txt'
    with listing.open() as f:
        sizes = {r['Name']: int(r['Size']) for r in csv.DictReader(f, delimiter='\t')
                 if r['#Archive/File'] == 'File'}
    output = []
    for r in rows:
        match = re.fullmatch(r'ExVivo_Blood_(\d+)Gy_([A-Z])', r['title'])
        if not match:
            raise ValueError(f'Unexpected title: {r["title"]}')
        dose, group = match.groups()
        traits = dict(x.split(': ', 1) for x in r['characteristics_ch1'].split(' | '))
        assert float(traits['dose'].removesuffix('Gy')) == float(dose)
        url = r['supplementary_file'].replace('ftp://', 'https://')
        assert url.startswith('https://ftp.ncbi.nlm.nih.gov/geo/samples/')
        name = url.rsplit('/', 1)[1]
        assert name.startswith(r['gsm'] + '_') and name in sizes
        assert re.search(rf'_{group}_{dose}Gy\.txt\.gz$', name), name
        output.append(dict(gsm=r['gsm'], title=r['title'], organism='Homo sapiens',
                           dose_gy=float(dose), source_donor_group=group,
                           donor_evidence='consistent GEO title and source filename; no explicit donor field',
                           gender=traits['gender'], time_after_exposure_hours=24,
                           time_evidence='GEO treatment protocol', url=url,
                           expected_bytes=sizes[name], path=str((OUT / name).relative_to(ROOT))))
    assert len(output) == 100 and len({r['gsm'] for r in output}) == 100
    groups = {r['source_donor_group'] for r in output}
    assert len(groups) == 20
    for group in groups:
        group_rows = [r for r in output if r['source_donor_group'] == group]
        assert len(group_rows) == 5
        assert {r['dose_gy'] for r in group_rows} == {0., 2., 5., 6., 7.}
        assert len({r['gender'] for r in group_rows}) == 1
    return sorted(output, key=lambda r: r['gsm'])


def validate(path, expected_bytes):
    if path.stat().st_size != expected_bytes:
        raise ValueError(f'Size mismatch: {path.name}')
    expanded = 0
    with gzip.open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            expanded += len(chunk)
            if expanded > 150_000_000:
                raise ValueError('Unexpected expanded size')
    return dict(bytes=path.stat().st_size, expanded_bytes=expanded,
                sha256=sha(path), gzip_crc_verified=True)


def fetch(row, previous):
    path = ROOT / row['path']
    if path.exists():
        result = validate(path, row['expected_bytes'])
        old = previous.get(row['gsm'])
        if old and result['sha256'] != old['sha256']:
            raise ValueError(f'Previously recorded checksum mismatch: {path.name}')
        return row | result | {'status': 'verified_existing'}
    partial = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(row['url'], timeout=60) as source, partial.open('wb') as dest:
                written = 0
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > row['expected_bytes']:
                        raise ValueError('Response exceeds source-listed size')
                    dest.write(chunk)
            result = validate(partial, row['expected_bytes'])
            partial.replace(path)
            return row | result | {'status': 'downloaded_verified'}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = samples()
    with (LOG / 'samples.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    manifest = LOG / 'downloads.json'
    previous = {r['gsm']: r for r in json.loads(manifest.read_text()).get('files', [])} if manifest.exists() else {}
    completed, errors = [], []
    print(f'Selected {len(rows)} human files; {sum(r["expected_bytes"] for r in rows):,} compressed bytes', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, row, previous): row for row in rows}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                errors.append({'gsm': futures[future]['gsm'], 'error': str(exc)})
            result = dict(study='GSE102971', scope='human per-array sources only',
                          updated_utc=datetime.now(timezone.utc).isoformat(),
                          source_checksum_note='SHA-256 recorded locally; no publisher SHA-256 available. Listed size and gzip CRC checked.',
                          files=sorted(completed, key=lambda r: r['gsm']), errors=errors,
                          complete=len(completed) == 100 and not errors)
            temporary = manifest.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(manifest)
            if (len(completed) + len(errors)) % 10 == 0 or errors:
                print(f'Verified {len(completed)}/100; errors {len(errors)}', flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} downloads failed; consult local manifest')
    print(f'COMPLETE: compressed={sum(r["bytes"] for r in completed):,}; expanded={sum(r["expanded_bytes"] for r in completed):,}', flush=True)


if __name__ == '__main__':
    main()
