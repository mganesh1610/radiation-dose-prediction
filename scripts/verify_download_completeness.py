"""Recount the final proposal inventory directly from original files.

No expression-based sample selection, preprocessing, or model fitting.
"""
import csv
import gzip
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/dataset-acquisition'
EXPECTED = {'GSE102971': 100, 'GSE8917': 25, 'GSE65292': 20,
            'GSE85570': 440, 'GSE97000': 12, 'GSE20173': 10,
            'GSE90909': 52, 'GSE15341': 24, 'GSE274788': 12, 'GSE40640': 48}


def eligible(study, title, organism):
    if study == 'GSE102971': return organism == 'Homo sapiens'
    if study == 'GSE8917': return '_24hr_' in title
    if study == 'GSE65292': return title.startswith(('0Gy_', 'Acute_'))
    if study == 'GSE20173': return bool(re.search(r'(?:-|_)1g(?:-|_)', title))
    if study == 'GSE90909': return 'X-ray' in title or 'Control' in title
    return True


def matrix_count(path):
    metadata = {}; all_rows = 0; all_missing = Counter(); selected_missing = Counter()
    selected_numeric = 0; full_numeric = 0; probes = set(); inside = False
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = next(csv.reader([line], delimiter='\t'))
            if line.startswith('!Sample_'):
                metadata.setdefault(row[0], row[1:])
            elif line.startswith('!series_matrix_table_begin'):
                inside = True
                header = next(csv.reader([next(stream)], delimiter='\t'))
                ids = header[1:]
                assert ids == metadata['!Sample_geo_accession'], path
                assert len(ids) == len(set(ids)), path
                indexes = [i for i, title in enumerate(metadata['!Sample_title'])
                           if eligible(path.parent.name, title, metadata['!Sample_organism_ch1'][i])]
                assert len(indexes) == EXPECTED[path.parent.name], (path, len(indexes))
            elif line.startswith('!series_matrix_table_end'):
                inside = False
            elif inside:
                assert len(row) == len(ids) + 1, path
                assert row[0] not in probes, (path, row[0])
                probes.add(row[0]); all_rows += 1
                values = row[1:]
                for val in values:
                    try: finite = math.isfinite(float(val))
                    except ValueError: finite = False
                    if finite: full_numeric += 1
                    else: all_missing[val] += 1
                for i in indexes:
                    val = values[i]
                    try: finite = math.isfinite(float(val))
                    except ValueError: finite = False
                    if finite: selected_numeric += 1
                    else: selected_missing[val] += 1
    return dict(study=path.parent.name, downloaded_profiles=len(ids), selected_profiles=len(indexes),
                selected_gsms=[ids[i] for i in indexes], features=all_rows,
                full_numeric_values=full_numeric, selected_matrix_cells=len(indexes)*all_rows,
                selected_numeric_values=selected_numeric, selected_missing=dict(selected_missing),
                full_missing=dict(all_missing), selection_status='proposal inventory, not final eligibility')


def main():
    manifest = json.loads((LOG/'downloads.json').read_text())
    checks = []; counts = []; listings = []
    for rec in manifest:
        p = ROOT/rec['path']
        with p.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == rec['sha256'] and p.stat().st_size == rec['bytes'], p
        if p.suffix == '.gz':
            with gzip.open(p, 'rb') as stream:
                while stream.read(1024*1024): pass
        checks.append({'path':rec['path'], 'size_and_sha256_ok':True,'gzip_crc_ok':True if p.suffix=='.gz' else None})
        if '_series_matrix' in p.name:
            counts.append(matrix_count(p))
    assert len(checks) == 26
    assert {r['study'] for r in counts} == set(EXPECTED)
    for study in EXPECTED:
        base=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{study[:-3]}nnn/{study}/'
        for sub in ['matrix','suppl']:
            url=base+sub+'/'
            with urllib.request.urlopen(url,timeout=60) as response: html=response.read().decode()
            (LOG/f'{study}-{sub}-recheck.html').write_text(html)
            names=re.findall(r'href="([^"/]+)"',html)
            for name in names:
                if name.startswith(study):
                    listings.append({'study':study,'directory':sub,'file':name,
                        'downloaded':any(Path(r['path']).name==name for r in manifest), 'url':url+name})
    z=json.loads((LOG/'zenodo-22013472.json').read_text())
    for rec in z['files']:
        p=ROOT/'data/public_radiation/conditional_test/Zenodo22013472'/rec['key']
        assert p.stat().st_size == rec['size']
        assert 'md5:'+hashlib.md5(p.read_bytes()).hexdigest() == rec['checksum']
    train=[r for r in counts if r['study'] in list(EXPECTED)[:6]]
    result={'checks':checks,'matrix_counts':counts,'source_listing':listings,
            'six_study_selected_profiles':sum(r['selected_profiles'] for r in train),
            'six_study_numeric_values':sum(r['selected_numeric_values'] for r in train),
            'six_study_matrix_cells':sum(r['selected_matrix_cells'] for r in train),
            'downloaded_bytes':sum(r['bytes'] for r in manifest),
            'zenodo_source_md5_verified':True}
    (LOG/'completeness-check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','matrix_counts','source_listing']},indent=2))
    for r in counts: print(r['study'], r['selected_profiles'], r['features'],r['selected_numeric_values'],r['selected_missing'])
    print('Not downloaded:', [x['file'] for x in listings if not x['downloaded']])


if __name__ == '__main__': main()
