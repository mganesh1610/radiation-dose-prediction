"""Download only the previously selected GSE8917/GSE65292 core raw profiles."""
import csv
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from acquire_gse102971_raw import fetch, sha

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/core-reprocessing/v1'


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    labels_path = ROOT / 'context/normalization-validation/v2/core-sample-labels-v2.tsv'
    metadata_path = ROOT / 'context/dataset-acquisition/sample-metadata.tsv'
    labels = list(csv.DictReader(labels_path.open(), delimiter='\t'))
    metadata = {r['gsm']: r for r in csv.DictReader(metadata_path.open(), delimiter='\t')}
    rows = []
    for study in ['GSE8917', 'GSE65292']:
        listing = ROOT / f'context/dataset-acquisition/preprocessing-audit/{study}-raw-filelist.txt'
        sizes = {r['Name']: int(r['Size']) for r in csv.DictReader(listing.open(), delimiter='\t')
                 if r['#Archive/File'] == 'File'}
        out = ROOT / 'data/raw_radiation' / study
        out.mkdir(parents=True, exist_ok=True)
        for label in labels:
            if label['study'] != study:
                continue
            m = metadata[label['gsm']]
            assert m['organism_ch1'] == 'Homo sapiens' and m['channel_count'] == '1'
            url = m['supplementary_file'].replace('ftp://', 'https://')
            assert url.startswith('https://ftp.ncbi.nlm.nih.gov/geo/samples/')
            name = url.rsplit('/', 1)[1]
            assert name.startswith(label['gsm']) and name in sizes and name.endswith('.txt.gz')
            rows.append(label | dict(url=url, expected_bytes=sizes[name], path=str((out/name).relative_to(ROOT))))
    assert Counter(r['study'] for r in rows) == {'GSE8917': 25, 'GSE65292': 20}
    assert len({r['gsm'] for r in rows}) == 45
    assert shutil.disk_usage(ROOT).free > 3_000_000_000
    manifest = LOG/'downloads.json'
    previous = {r['gsm']: r for r in json.loads(manifest.read_text())['files']} if manifest.exists() else {}
    for r in json.loads((ROOT/'context/normalization-validation/v2/sources/raw-format-probes.json').read_text())['files']:
        previous.setdefault(r['gsm'], r)
    done, errors = [], []
    print(f'Selected 45 files: {sum(r["expected_bytes"] for r in rows):,} compressed bytes', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, r, previous): r for r in rows}
        for future in as_completed(futures):
            try:
                done.append(future.result())
            except Exception as exc:
                errors.append(dict(gsm=futures[future]['gsm'], error=str(exc)))
            result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), scope='45 selected core profiles only',
                          complete=len(done)==45 and not errors, files=sorted(done, key=lambda r:r['gsm']), errors=errors,
                          source_checksum_note='Local SHA-256 plus publisher-listed bytes and gzip CRC; not publisher SHA-256.',
                          inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [labels_path,metadata_path]},
                          code_sha256=sha(Path(__file__)))
            tmp = manifest.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(manifest)
            print(f'Verified {len(done)}/45; failures {len(errors)}', flush=True)
    if errors:raise RuntimeError('Acquisition incomplete; see manifest')
    print(f'COMPLETE: compressed {sum(r["bytes"] for r in done):,}; expanded {sum(r["expanded_bytes"] for r in done):,}', flush=True)


if __name__ == '__main__':
    main()
