"""Annotation/scale compatibility audit. No pooled fit or locked-test reading."""
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from assay_qc import aggregate_median, resolve_gene, platform_probe_id, technical_valid

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/normalization-validation/v2'
OUT=ROOT/'data/processed_pilot/core-compatibility-v1'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tsv(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)


def read_platform(study):
    p=ROOT/f'data/public_radiation/core/{study}/{study}_family.soft.gz'
    metadata={};lines=[]
    with gzip.open(p,'rt') as f:
        for l in f:
            if l.startswith(('!Platform_title','!Platform_last_update_date','!Series_overall_design')):
                k,v=l.strip().split(' = ',1);metadata[k]=v
            if l.startswith('!platform_table_begin'):break
        for l in f:
            if l.startswith('!platform_table_end'):break
            lines.append(l)
    rows=list(csv.DictReader(lines,delimiter='\t'));byid={r['ID']:r for r in rows}
    if len(byid)!=len(rows):raise ValueError('Duplicated platform IDs')
    return rows,byid,metadata,p


def read_matrix(study,selected):
    p=ROOT/f'data/public_radiation/core/{study}/{study}_series_matrix.txt.gz'
    with gzip.open(p,'rt') as f:
        for l in f:
            if l.startswith('!series_matrix_table_begin'):break
        reader=csv.reader(f,delimiter='\t');header=next(reader);ids=header[1:]
        if len(set(ids))!=len(ids) or not set(selected).issubset(ids):raise ValueError('Invalid matrix sample IDs')
        take=[ids.index(s)+1 for s in selected];features=[];data=[]
        for row in reader:
            if row[0].startswith('!series_matrix_table_end'):break
            if len(row)!=len(header):raise ValueError('Ragged matrix row')
            features.append(row[0]);data.append([float(row[i]) if row[i] not in ('','NA','null','NaN') else np.nan for i in take])
    if len(features)!=len(set(features)):raise ValueError('Duplicated matrix feature IDs')
    x=np.asarray(data)
    if np.isinf(x).any():raise ValueError('Infinite processed value')
    return features,x,p


def raw_format(r,platform,feature_key,platform_id):
    p=ROOT/r['path']
    if sha(p)!=r['sha256']:raise ValueError('Raw example checksum mismatch')
    features=[];signals=[];flags=[];sequence_conflicts=0;identity_conflicts=0
    technical_columns=['gIsSaturated','IsManualFlag','gIsFeatNonUnifOL','gIsFeatPopnOL']
    with gzip.open(p,'rt') as f:
        for l in f:
            if l.startswith('FEATURES\t'):header=l.strip().split('\t');break
        for row in csv.reader(f,delimiter='\t'):
            if not row or row[0]!='DATA':continue
            if len(row)!=len(header):raise ValueError('Malformed raw example')
            a=dict(zip(header,row))
            if a['ControlType']!='0':continue
            features.append(a['ProbeName']);signals.append(float(a['gProcessedSignal']))
            flags.append([int(a[k]) for k in technical_columns])
            entry=platform.get(a[feature_key])
            if not entry or platform_probe_id(platform_id,entry)!=a['ProbeName']:
                identity_conflicts+=1
            if entry and 'Sequence' in a and (entry['SEQUENCE'] or '').upper()!=a['Sequence'].upper():
                sequence_conflicts+=1
    x=np.array(signals)
    usable=technical_valid(np.asarray(flags)[:,None,:],technical_columns)[:,0]
    required=['gMedianSignal','gBGMedianSignal','gProcessedSignal','gIsSaturated','IsManualFlag','gIsFeatNonUnifOL','gIsFeatPopnOL','gIsWellAboveBG']
    return dict(gsm=r['gsm'],raw_noncontrol_features=len(x),nonpositive_processed_signals=int((x<=0).sum()),
        technically_usable_features=int(usable.sum()),nonpositive_technically_usable_signals=int(((x<=0)&usable).sum()),
        processed_signal_min=float(x.min()),processed_signal_max=float(x.max()),raw_platform_identity_conflicts=identity_conflicts,
        raw_sequence_column_available='Sequence' in header,
        raw_platform_sequence_conflicts=sequence_conflicts if 'Sequence' in header else None,
        missing_required_adapter_fields=[k for k in required if k not in header],
        scope='one raw example; not a complete cohort audit',path=r['path'],sha256=r['sha256'])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metadata=list(csv.DictReader((ROOT/'context/dataset-acquisition/sample-metadata.tsv').open(),delimiter='\t'))
    gene_path=ROOT/'context/raw-pilot/GSE102971/audit-v1/sources/Homo_sapiens.gene_info.gz'
    with gzip.open(gene_path,'rt') as f:current={r['GeneID']:r for r in csv.DictReader(f,delimiter='\t') if r['#tax_id']=='9606'}
    raw_meta=json.loads((LOG/'sources/raw-format-probes.json').read_text())
    raw_by_study={r['study']:r for r in raw_meta['files']}
    summaries=[];gene_sets={};sequence_maps={};matrix_gene_sets={};all_samples=[];inputs={}
    for study in ['GSE102971','GSE8917','GSE65292']:
        rows,platform,platform_meta,soft=read_platform(study);inputs[str(soft.relative_to(ROOT))]=sha(soft)
        selected=[r for r in metadata if r['study']==study and
                  ((study=='GSE102971' and r['organism_ch1']=='Homo sapiens') or
                   (study=='GSE8917' and '_24hr_' in r['title']) or
                   (study=='GSE65292' and (r['title'].startswith('0Gy_') or r['title'].startswith('Acute_'))))]
        expected={'GSE102971':100,'GSE8917':25,'GSE65292':20}[study]
        if len(selected)!=expected or any(r['organism_ch1']!='Homo sapiens' or r['channel_count']!='1' for r in selected):
            raise ValueError('Incorrect subset or assay type')
        by_probe=defaultdict(list)
        platform_id=selected[0]['platform_id']
        for row in rows:
            if row['CONTROL_TYPE'] in ('FALSE','0'):by_probe[platform_probe_id(platform_id,row)].append(row)
        mappings={p:resolve_gene([r['GENE'] or '' for r in rr],current) for p,rr in by_probe.items()}
        gene_sets[study]={g for g,status in mappings.values() if g}
        sequence_maps[study]={p:{(r['SEQUENCE'] or '').upper() for r in rr} for p,rr in by_probe.items()}
        groups=defaultdict(list)
        for r in selected:
            traits=dict(x.split(': ',1) for x in r['characteristics_ch1'].split(' | ') if ': ' in x)
            group=r['title'].split('_')[-1]
            source_dose=traits.get('dose') or re.search(r'([\d.]+)Gy',r['title'])[1]+' Gy'
            numeric=float(re.search(r'[\d.]+',source_dose)[0])
            canonical=2.23 if study=='GSE65292' and numeric==2.2 else numeric
            label_status='publication_methods_override_preserving_GEO_conflict' if canonical!=numeric else 'source_label'
            all_samples.append(dict(study=study,gsm=r['gsm'],title=r['title'],split_group_id=study+':'+group,
                group_evidence='source title; study design supports donor repeats',time_hours=24,
                source_dose_text=source_dose,source_dose_gy=numeric,candidate_dose_gy=canonical,dose_status=label_status,
                dose_evidence='https://link.springer.com/article/10.1186/s12920-015-0097-x' if canonical!=numeric else 'GEO sample title/characteristics'))
            groups[group].append(r['gsm'])
        if len(groups)!={'GSE102971':20,'GSE8917':5,'GSE65292':5}[study]:raise ValueError('Unexpected groups')
        summary=dict(study=study,samples=expected,study_local_groups=len(groups),platform=selected[0]['platform_id'],
            platform_metadata=platform_meta,noncontrol_platform_probes=len(by_probe),current_annotation_genes=len(gene_sets[study]),
            probe_mapping_statuses=dict(Counter(status for gid,status in mappings.values())))
        if study=='GSE102971':
            p=ROOT/'data/processed_pilot/GSE102971/qc-normalization-v2/gene_candidate.npz'
            z=np.load(p,allow_pickle=False);matrix_gene_sets[study]=set(z['gene_ids'])
            summary.update(representation='QC-before-normalization frozen training-reference quantile; log2; engineering split',
                gene_rows=len(z['gene_ids']),gene_missing_fraction=float(np.isnan(z['values']).mean()),
                readiness='technical implementation checked; biological cross-study evaluation pending')
            inputs[str(p.relative_to(ROOT))]=sha(p)
        else:
            ids,x,matrix_path=read_matrix(study,[r['gsm'] for r in selected]);inputs[str(matrix_path.relative_to(ROOT))]=sha(matrix_path)
            unknown=[i for i in ids if i not in platform]
            if unknown:raise ValueError('Matrix features absent from platform')
            keep=np.array([platform[f]['CONTROL_TYPE'] in ('FALSE','0') for f in ids])
            probe_keys=[platform_probe_id(platform_id,platform[f]) for f,k in zip(ids,keep) if k]
            probe_ids,values,_=aggregate_median(x[keep],probe_keys)
            mapped=np.array([bool(mappings[p][0]) for p in probe_ids])
            genes,values,counts=aggregate_median(values[mapped],[mappings[p][0] for p,k in zip(probe_ids,mapped) if k])
            has_any=np.isfinite(values).any(axis=1);genes,values,counts=genes[has_any],values[has_any],counts[has_any]
            matrix_gene_sets[study]=set(genes)
            np.savez_compressed(OUT/f'{study}_published_gene_view.npz',values=values,gene_ids=genes,sample_ids=np.array([r['gsm'] for r in selected]),
                contributing_probe_counts=counts,representation=np.array('published log2 values; source cohort processing; audit-only view'))
            summary.update(representation='published log2, median-normalized; preserve as received, no second log/normalization',
                matrix_rows=len(ids),matrix_noncontrol_rows=int(keep.sum()),source_missing_fraction=float(np.isnan(x[keep]).mean()),
                gene_rows=len(genes),gene_missing_fraction=float(np.isnan(values).mean()),
                finite_source_quantiles=np.nanquantile(x[keep],[0,.25,.5,.75,1]).tolist(),
                raw_example=raw_format(raw_by_study[study],platform,'FeatureNum' if study=='GSE8917' else 'ProbeName',platform_id))
            summary['readiness']='blocked for shared raw pipeline: nonpositive gProcessedSignal requires study-specific background treatment' if study=='GSE8917' else 'compatible raw columns and array design in one example; full cohort raw validation pending'
        summaries.append(summary)
    platform_intersection=set.intersection(*gene_sets.values());view_intersection=set.intersection(*matrix_gene_sets.values())
    pairs=[]
    studies=list(sequence_maps)
    for i,a in enumerate(studies):
        for b in studies[i+1:]:
            common=set(sequence_maps[a])&set(sequence_maps[b])
            same={p for p in common if sequence_maps[a][p]==sequence_maps[b][p] and sequence_maps[a][p]!={''}}
            pairs.append(dict(a=a,b=b,common_probe_names=len(common),identical_sequence_sets=len(same),sequence_disagreement=len(common-same),
                common_current_annotation_genes=len(gene_sets[a]&gene_sets[b])))
    tsv(LOG/'core-sample-labels-v2.tsv',all_samples)
    tsv(LOG/'common-current-gene-availability.tsv',[dict(gene_id=g,current_symbol=current[g]['Symbol'],
        in_GSE102971_candidate=g in matrix_gene_sets['GSE102971'],in_GSE8917_published=g in matrix_gene_sets['GSE8917'],
        in_GSE65292_published=g in matrix_gene_sets['GSE65292']) for g in sorted(platform_intersection)])
    summary=dict(completed_utc=datetime.now(timezone.utc).isoformat(),scope='three core studies only; annotation/source-scale and two raw-example checks',
        studies=summaries,pairwise_platform_checks=pairs,common_current_annotation_genes=len(platform_intersection),
        common_genes_in_current_candidate_and_published_views=len(view_intersection),
        common_gene_list_role='descriptive audit only; not a selected modeling panel; recompute using training-only availability in future folds',
        dose_override=dict(study='GSE65292',samples=sum(r['dose_status']!='source_label' for r in all_samples),GEO_characteristics=2.2,GEO_summary=2.25,
            candidate_Gy=2.23,evidence='https://link.springer.com/article/10.1186/s12920-015-0097-x',status='publication-supported canonical candidate; original records unchanged'),
        pooling_decision='not interchangeable; do not concatenate for a claim of harmonized training data',
        inputs_sha256=inputs|{str(p.relative_to(ROOT)):sha(p) for p in [gene_path,ROOT/'context/dataset-acquisition/sample-metadata.tsv',LOG/'sources/raw-format-probes.json']},code_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'scripts/assay_qc.py']},
        artifact_files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.glob('*.npz'))])
    (LOG/'compatibility-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['common_current_annotation_genes','common_genes_in_current_candidate_and_published_views','pairwise_platform_checks','dose_override']},indent=2))


if __name__=='__main__':main()
