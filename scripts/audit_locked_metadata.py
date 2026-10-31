"""Freeze GSE90909 membership from metadata only; never read expression tables."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'context/locked-eligibility/v1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, obj):
    (LOG / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def parse_metadata(text):
    if '!sample_table_begin' in text.lower() or '!series_matrix_table_begin' in text.lower():
        raise ValueError('Expression tables are not permitted in this metadata input')
    records, current = [], None
    for line in text.splitlines():
        if line.startswith('^SAMPLE = '):
            current = {'gsm': line.split(' = ', 1)[1]}; records.append(current)
        elif line.startswith('!Sample_') and current is not None:
            key, value = line[len('!Sample_'):].split(' = ', 1)
            current.setdefault(key, []).append(value)
    if not records or len({r['gsm'] for r in records}) != len(records):
        raise ValueError('Empty or duplicate metadata')
    return records


def single(row, key):
    values = row[key]
    if len(values) != 1:
        raise ValueError('Ambiguous source field: ' + key)
    return values[0]


def exposure(text):
    if re.fullmatch(r'blood Control(?: rep\d+)?', text):
        return 'control', 0.0
    m = re.fullmatch(r'blood (\d+(?:\.\d+)?)\s*Gy (X-ray|Neutron)(?: rep\d+)?', text)
    if not m:
        raise ValueError('Unknown exposure/dose label: ' + text)
    return ('photon' if m[2] == 'X-ray' else 'neutron'), float(m[1])


def classify(row, listing):
    traits = dict(t.split(': ', 1) for t in row['characteristics_ch1'])
    donor = re.fullmatch(r'Donor (\d+)', traits['donor'])
    if not donor or not 1 <= int(donor[1]) <= 12:
        raise ValueError('Missing or unexpected explicit donor identity')
    number = int(donor[1])
    mode, dose = exposure(single(row, 'title'))
    if (mode, dose) != exposure(single(row, 'source_name_ch1')):
        raise ValueError('Title/source exposure conflict')
    if single(row, 'organism_ch1') != 'Homo sapiens' or traits['tissue'] != 'peripheral blood':
        raise ValueError('Unexpected organism or tissue')
    if single(row, 'platform_id') != 'GPL13497' or single(row, 'channel_count') != '1' or single(row, 'label_ch1') != 'Cy3':
        raise ValueError('Unexpected assay')
    if '24 h' not in single(row, 'growth_protocol_ch1') or 'healthy volunteers' not in single(row, 'treatment_protocol_ch1'):
        raise ValueError('Timing/source protocol requires review')
    url = single(row, 'supplementary_file').replace('ftp://', 'https://')
    filename = url.rsplit('/', 1)[1]
    raw = re.fullmatch(r'(GSM\d+)_CB_HuNEU_DON([1-6A-F])_(.+)\.txt\.gz', filename)
    if not raw or raw[1] != row['gsm'] or not url.startswith('https://ftp.ncbi.nlm.nih.gov/geo/samples/'):
        raise ValueError('Unexpected source filename/host')
    file_donor = int(raw[2]) if raw[2].isdigit() else ord(raw[2]) - ord('A') + 7
    if number != file_donor:
        raise ValueError('Metadata/raw-filename donor disagreement')
    if mode == 'control':
        if raw[3] != 'CTL': raise ValueError('Control filename disagreement')
    else:
        suffix = '_Gy_X' if mode == 'photon' else '_Gy_NEU'
        if not raw[3].endswith(suffix): raise ValueError('Exposure filename disagreement')
        token = raw[3][:-len(suffix)]
        # Only excluded neutron GSM2417361 has the documented filename token 05 for 0.5.
        file_dose = .5 if token == '05' and row['gsm'] == 'GSM2417361' else float(token)
        if file_dose != dose: raise ValueError('Dose filename disagreement')
    if filename not in listing: raise ValueError('Source missing from publisher file list')
    title_rep = int(re.search(r'rep(\d+)$', single(row, 'title'))[1])
    eligible = mode in ('photon', 'control')
    return dict(gsm=row['gsm'], study='GSE90909', title=single(row, 'title'),
                donor_group=f'GSE90909:Donor {number}', donor_number=number,
                donor_basis='explicit_GEO_characteristics_concordant_with_raw_filename',
                title_rep=title_rep, title_rep_differs_from_donor=title_rep != number,
                exposure=mode, dose_gy=dose, time_hours=24, tissue='peripheral_whole_blood',
                platform='GPL13497', eligible=eligible,
                reason='photon_or_matched_control_in_scope' if eligible else 'neutron_outside_primary_photon_scope',
                source_url=url, raw_filename=filename, publisher_bytes=listing[filename],
                raw_local_path='data/raw_radiation/locked/GSE90909/' + filename,
                eligibility_status='metadata_eligible_raw_technical_checks_pending' if eligible else 'excluded_by_exposure')


def main():
    if (LOG / 'eligibility-freeze.json').exists():
        raise ValueError('Eligibility already frozen; do not overwrite')
    metadata_path = LOG / 'sources/GSE90909-metadata.txt'
    listing_path = LOG / 'sources/GSE90909-filelist.txt'
    listing = {}
    for line in listing_path.read_text().splitlines():
        cells = line.split('\t')
        if cells[0] == 'File': listing[cells[1]] = int(cells[3])
    records = parse_metadata(metadata_path.read_text())
    rows = [classify(r, listing) for r in records]
    if len(rows) != 92: raise ValueError('Study metadata count changed')
    selected = [r for r in rows if r['eligible']]
    if len(selected) != 52 or len({r['donor_group'] for r in selected}) != 12:
        raise ValueError('Unexpected primary sample/donor count')
    if len({(r['donor_group'], r['dose_gy']) for r in selected}) != 52:
        raise ValueError('Duplicate donor-dose profile requires adjudication')
    cached = {r['gsm']: r for r in csv.DictReader((ROOT / 'context/dataset-acquisition/sample-metadata.tsv').open(), delimiter='\t') if r['study'] == 'GSE90909'}
    for r in records:
        for key in ['title', 'source_name_ch1', 'characteristics_ch1', 'growth_protocol_ch1', 'platform_id', 'data_processing']:
            if ' | '.join(r[key]) != cached[r['gsm']][key]:
                raise ValueError('Fresh/cached metadata disagreement: ' + key)
    development = json.loads((ROOT / 'data/model_ready/development_raw_f6022cfaaca6d1d9/metadata.json').read_text())
    dev_ids = {r['profile_id'] for r in development['profiles']}
    overlap = {r['gsm'] for r in selected} & dev_ids
    if overlap: raise ValueError('Development accession overlap')
    donor_rows = []
    for n in range(1, 13):
        rr = [r for r in selected if r['donor_number'] == n]
        donor_rows.append(dict(donor_number=n, donor_group=f'GSE90909:Donor {n}',
                          profiles=len(rr), doses_gy=sorted(r['dose_gy'] for r in rr),
                          has_control=any(r['dose_gy'] == 0 for r in rr),
                          file_donor_token=str(n) if n <= 6 else chr(ord('A') + n - 7)))
    write('all-sample-decisions.json', rows)
    write('eligible-samples.json', selected)
    write('donor-manifest.json', donor_rows)
    write('metadata-summary.json', dict(total_arrays=92, selected_arrays=52, excluded_neutron_arrays=40,
          study_local_donors=12, dose_counts=dict(sorted(Counter(r['dose_gy'] for r in selected).items())),
          controls=11, donors_without_control=[r['donor_number'] for r in donor_rows if not r['has_control']],
          title_rep_mismatches=sum(r['title_rep_differs_from_donor'] for r in selected),
          exact_development_accession_overlap=sorted(overlap), cross_study_person_independence='not_provable_from_public_local_donor_ids',
          publisher_raw_bytes=sum(r['publisher_bytes'] for r in selected),
          expression_parsed=False, prediction_run=False, cached_metadata_concordant=True))
    names=['all-sample-decisions.json','eligible-samples.json','donor-manifest.json','metadata-summary.json',
           'sources/GSE90909-metadata.txt','sources/GSE90909-filelist.txt']
    write('eligibility-freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),
          status='METADATA_MEMBERSHIP_FROZEN_BEFORE_RAW_ACQUISITION',sha256={n:sha(LOG/n) for n in names},
          source_sha256=sha(Path(__file__)), excluded_by_expression_or_error=False,
          test_normalization_executed=False, inference_executed=False))
    print('Frozen: 52 photon/control arrays, 12 donors, 40 neutron exclusions; no expression parsed.')


if __name__ == '__main__': main()
