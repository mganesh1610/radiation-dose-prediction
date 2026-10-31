"""Deterministic extraction of public GEO methods; no local context ingestion."""
import gzip
import hashlib
import json
import re
from pathlib import Path

PIPELINE = 'geo-methods-v1'
MODEL = 'gemini-embedding-001'
DIMENSIONS = 768
FIELDS = {'Series_title', 'Series_summary', 'Series_overall_design', 'Sample_treatment_protocol_ch1',
          'Sample_growth_protocol_ch1', 'Sample_extract_protocol_ch1', 'Sample_label_protocol_ch1',
          'Sample_hyb_protocol', 'Sample_scan_protocol', 'Sample_data_processing'}


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def extract_geo(path, study):
    path = Path(path).resolve()
    if 'context' in path.parts or not re.fullmatch(r'GSE\d+', study) or path.name != study + '_family.soft.gz':
        raise ValueError('Only explicit public GEO family archives are accepted')
    raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    sections, sample, title, seen_study = {}, None, '', False
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('^SERIES = '):
                if line.strip().split(' = ')[1] != study: raise ValueError('Study accession mismatch')
                seen_study = True
            if line.startswith('^SAMPLE = '): sample = line.strip().split(' = ')[1]
            if not line.startswith('!') or ' = ' not in line: continue
            field, text = line.rstrip('\r\n')[1:].split(' = ', 1)
            if field not in FIELDS or not text.strip(): continue
            if len(text) > 6000: raise ValueError('Oversized section needs reviewed chunking')
            if field == 'Series_title': title = text
            sections.setdefault((field, text), set())
            if field.startswith('Sample_'):
                if not sample: raise ValueError('Sample method missing accession')
                sections[(field, text)].add(sample)
    if not seen_study or not title: raise ValueError('Missing series metadata')
    document_id = study + '-' + raw_hash[:16]
    base = {'document_id': document_id, 'document_version': raw_hash, 'study_id': study,
            'title': title, 'source_kind': 'public_geo_metadata', 'access_class': 'public',
            'source_url': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + study,
            'source_archive_url': f'https://ftp.ncbi.nlm.nih.gov/geo/series/{study[:-3]}nnn/{study}/soft/{study}_family.soft.gz',
            'pipeline_version': PIPELINE, 'embedding_model': MODEL, 'embedding_dimensions': DIMENSIONS}
    passages = []
    for (section, text), samples in sorted(sections.items()):
        if section == 'Series_title': continue
        chunk_id = 'pc-' + sha(document_id + '\0' + section + '\0' + text)[:24]
        passages.append({**base, 'chunk_id': chunk_id, 'section': section, 'text': text,
                         'text_sha256': sha(text), 'sample_ids': sorted(samples)})
    return base, passages


def build_corpus(paths):
    documents, passages = [], []
    for study, path in sorted(paths.items()):
        document, chunks = extract_geo(path, study)
        documents.append(document); passages.extend(chunks)
    version = 'geo-core-' + sha(json.dumps(passages, sort_keys=True))[:16]
    for p in passages: p.update(corpus_version=version, approved=True)
    return {'version': version, 'pipeline_version': PIPELINE, 'embedding_model': MODEL,
            'embedding_dimensions': DIMENSIONS, 'documents': documents,
            'passages': {p['chunk_id']: {k: v for k, v in p.items() if k != 'text'}
                         for p in passages}}, passages
