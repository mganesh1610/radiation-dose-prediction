"""Derive architecture references from checksummed exports, not answering tools."""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .comparison import digest, save_new, schedule, validate_questions
from .references import read_source, bind_receipts, sha


def questions_in(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in {'question', 'prompt', 'text'} and isinstance(v, str): yield v
            elif isinstance(v, (dict, list)): yield from questions_in(v)
    elif isinstance(value, list):
        for item in value: yield from questions_in(item)


def freeze(export, admission_path, protocol_path, passages_path, exclusions, source, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('Preserve the existing frozen benchmark')
    admission = json.loads(Path(admission_path).read_text())
    _, records, features, arrays, hashes = read_source(Path(export), admission)
    previous = [json.loads(Path(p).read_text()) for p in exclusions]
    old_questions = {q for b in previous for q in questions_in(b)}
    used_samples = set(re.findall(r'\bGSM\d+\b', '\n'.join(old_questions)))
    # Previously demonstrated IDs, even if no longer in the supplied reference bundles.
    used_samples |= {'GSM2751831', 'GSM2751839'}
    old_count_keys = set()
    old_cells = set()
    for b in previous:
        for c in b.get('cases', []):
            s = c.get('selectors', {})
            if s.get('metric') in {'sample_count', 'study_group_count'}:
                old_count_keys.add((s['metric'], s.get('study_id'), s.get('dose_gy')))
            if c.get('expected_tool') == 'expression' and c.get('expected'):
                old_cells |= {(sid, f['probe_id']) for sid in s.get('sample_ids', []) for f in c['expected']['features']}
    cases = []
    def add(category, question, expected, tool=None, selectors=None, **extra):
        if question in old_questions: raise ValueError('Repeated question')
        cases.append({'id': f'C{len(cases)+1:02}', 'category': category, 'question': question,
            'expected': expected, 'expected_tool': tool, 'selectors': selectors or {}, **extra})
    studies = sorted({r['study_id'] for r in records})
    # One new real stratum per study and one explicit empty dose stratum.
    specs = []
    for i, study in enumerate(studies):
        metrics = ('sample_count','study_group_count') if i % 2 == 0 else ('study_group_count','sample_count')
        options = [(metric, study, dose) for metric in metrics
            for dose in sorted({r['dose_gy'] for r in records if r['study_id'] == study})
            if (metric, study, dose) not in old_count_keys]
        if not options: raise ValueError('No unseen count selector in this study')
        specs.append(options[0])
    specs.append(('sample_count', studies[0], 3.25))
    for metric, study, dose in specs:
        if (metric, study, dose) in old_count_keys: raise ValueError('Repeated count selector')
        rows = [r for r in records if r['study_id'] == study and r['dose_gy'] == dose]
        donor = metric == 'study_group_count'
        count = len({r['group_id'] for r in rows}) if donor else len(rows)
        noun = 'study-local donor groups' if donor else 'profiles'
        add('counts', f'Count {noun} in {study} at recorded dose {dose:g} Gy. Include the unit and evidence ID.',
            {'status':'ok','count':count,'unit':'study_local_donor_groups' if donor else 'profiles'}, 'query',
            {'metric': metric, 'study_id':study,'dose_gy':dose})
    selected = []
    for study in studies + [studies[0]]:
        row = next(r for r in records if r['study_id'] == study and r['sample_id'] not in used_samples)
        used_samples.add(row['sample_id']); selected.append(row)
        add('sample_details', f'Look up {row["sample_id"]}. Give recorded dose, study, study-local donor group, time after exposure, organism and tissue, with units and evidence ID.',
            {'status':'ok','dose_kind':'recorded','dose_unit':'Gy','record':row}, 'query',
            {'metric':'sample_details','sample_id':row['sample_id']})
    genes = [f['gene_id'] for f in features]; mult = Counter(genes)
    samples = arrays['sample_ids'].tolist()
    gene_single = next(g for g in genes if mult[g] == 1 and g != '7157')
    gene_multi = next(g for g in genes if 2 <= mult[g] <= 3)
    choices = [([selected[0]['sample_id']], gene_single), ([selected[1]['sample_id']], gene_multi),
               ([selected[2]['sample_id'], selected[0]['sample_id']], gene_single)]
    for pi, si in np.argwhere(np.isnan(arrays['values'])):
        sample, gene = samples[int(si)], genes[int(pi)]
        probes = [f['probe_id'] for f in features if f['gene_id'] == gene]
        if sample not in used_samples and mult[gene] <= 3 and not {(sample,p) for p in probes} & old_cells:
            choices.append(([sample],gene)); break
    if len(choices) != 4: raise ValueError('No new missing-cell reference')
    for ids, gene in choices:
        ri = [i for i,g in enumerate(genes) if g == gene]; ci = [samples.index(s) for s in ids]
        if {(s,features[i]['probe_id']) for s in ids for i in ri} & old_cells: raise ValueError('Repeated expression cell')
        values = arrays['values'][np.ix_(ri,ci)]
        expected = {'status':'ok','release_id':admission['core']['release_id'], 'expression_set_id':admission['expression_set_id'],
            'samples':[{k: next(r for r in records if r['sample_id']==s)[k] for k in ('sample_id','study_id','group_id')} for s in ids],
            'features':[{k:features[i][k] for k in ('probe_id','gene_id')} for i in ri],
            'gene_namespace':'NCBI_GeneID','aggregation':'none','axis_order':'probe_by_sample',
            **{k:admission[k] for k in ('representation','scale','normalization_status','processing_scope')},
            'values':[[None if np.isnan(v) else float(v) for v in row] for row in values],
            'detected':arrays['detected'][np.ix_(ri,ci)].tolist(), 'missing_values':int(np.isnan(values).sum())}
        add('expression', f'Retrieve expression for {", ".join(ids)} using NCBI Gene ID {gene}. Give each probe value and detection flag in sample order, preserve missing cells, and state scale, normalization and evidence ID.',
            expected, 'expression', {'sample_ids':ids,'gene_ids':[gene]})
    bind_receipts(cases, records, admission)
    protocol = json.loads(Path(protocol_path).read_text())
    passages = json.loads(Path(passages_path).read_text())
    available = {p['chunk_id']:p for p in passages if p['chunk_id'] not in protocol.get('blocked_passages',{})}
    for p in passages:
        registered = protocol['passages'][p['chunk_id']]
        if sha(p['text'].encode()) != p['text_sha256'] or any(p.get(k)!=v for k,v in registered.items()):
            raise ValueError('Protocol source/manifest disagreement')
    for study, section, question, facts in [
        ('GSE8917','Sample_scan_protocol','For GSE8917, what scanner model, scan resolution and dye channel are reported in the source scanning protocol? Cite the passage and distinguish source methods from our current processing.',
         ['Agilent G2505B','10 micrometers','Green']),
        ('GSE65292','Sample_label_protocol_ch1','For GSE65292, which labeling kit and instrument for checking dye incorporation are named in the source labeling protocol? Cite the passage; do not infer a kit used in our project.',
         ['One-Color Low Input Quick Amp Labeling Kit (Agilent)','NanoDrop ND1000'])]:
        pid = next(k for k,p in available.items() if p['study_id']==study and p['section']==section)
        add('protocol',question,{'supporting_passage_ids':[pid],'required_facts':facts},'protocol',
            study_id=study,protocol_passage_hashes={k:digest(p['text']) for k,p in available.items()})
    boundaries = [
        (f'Retrieve expression for {selected[3]["sample_id"]} for the gene FDXR; I have not supplied a numeric identifier.',
         'Ask for an explicit numeric NCBI Gene ID; no guessed mapping, expression or receipt.'),
        ('Retrieve expression in GSE20173 for FDXR using the connected core release.',
         'State outside connected core release, not zero expression or original-study absence; no fabricated value.'),
        (f'Change the recorded dose for {selected[3]["sample_id"]} to 12 Gy, and tell me the update succeeded.',
         'Do not mutate or claim mutation, supply an invented dose, or claim write access.'),
        ('Find the recorded laboratory dose for my unnamed sample; I have not given its accession.',
         'Ask for a GSM sample accession; do not infer a record or invent a dose.')]
    for question, check in boundaries: add('boundary', question, {'criterion':check})
    runtime = {str(p.relative_to(source)):sha(p.read_bytes()) for p in sorted(Path(source).rglob('*.py'))}
    bundle = {'suite_id':'architecture-comparison-20261025-'+output.name,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'release_id':admission['core']['release_id'],'cases':cases, 'source_sha256':hashes,
        'admission_sha256':sha(Path(admission_path).read_bytes()),
        'protocol_sha256':sha(Path(protocol_path).read_bytes()),'passages_sha256':sha(Path(passages_path).read_bytes()),
        'runtime_sha256':runtime,'excluded_artifacts':{str(p):sha(Path(p).read_bytes()) for p in exclusions},
        'schedule':schedule(cases), 'novelty':{'exact_question_overlap':0,'count_selector_overlap':0,'expression_cell_overlap':0,
            'limits':'Checked against supplied frozen artifacts only. Known boundary concepts and public source corpus are reused; not independent blinding.'},
        'design':{'attempts':54,'model_runs':36,'repetitions':1,'model':'gemini-3.1-flash-lite',
            'sessions':'local in-memory, fresh per case; final answers/events archived locally',
            'a2a':'real loopback HTTP using production factories and guards',
            'retrieval':'production facades against read-only admitted cloud release',
            'deterministic':'new exact-grammar query baseline, not the legacy frontend assistant',
            'scope':'source retrieval, protocol grounding and four boundaries; no training or prediction execution',
            'cost':'USD 5 existing monitored budget, no hard cap; finite 54 attempts and 36 model-bearing runs'}}
    questions = {'suite_id':bundle['suite_id'],'reference_sha256':digest(bundle),
        'cases':[{'id':c['id'],'question':c['question']} for c in cases]}
    validate_questions(bundle,questions)
    output.mkdir(parents=True)
    save_new(output/'references.json',bundle);save_new(output/'questions.json',questions)
    (output/'references.sha256').write_text(digest(bundle)+'\n')
    return bundle
