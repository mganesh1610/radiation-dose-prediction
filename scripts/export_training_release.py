"""Build a curated core release; publish only explicitly generated data artifacts."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from cloud_io import Cloud, PREFIX, digest, encode
from training_contract import SCHEMAS, CORE, validate_data, validate_rows
from core_baseline import inner_splits
from run_core_baselines import load_data, LABELS

ROOT = Path(__file__).resolve().parents[1]


def build():
    data=load_data(); validate_data(data)
    buffer=io.BytesIO(); np.savez_compressed(buffer, **data); payload=buffer.getvalue()
    release='core_'+digest(payload)[:16]
    destination=ROOT/'data/cloud_exports'/release
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'core.npz').write_bytes(payload)
    annotations={r['gsm']:r for r in csv.DictReader(LABELS.open(), delimiter='\t')}
    tables={name:[] for name in ('studies','sample_groups','samples','sample_annotations','features','split_members')}
    for study in sorted(CORE):
        tables['studies'].append(dict(release_id=release,study_id=study,
            source_url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+study,role='core_development'))
    for group in sorted(set(data['split_group_ids'])):
        tables['sample_groups'].append(dict(release_id=release,group_id=str(group),study_id=group.split(':')[0],
            identity_status='inferred_study_local_group_not_independently_verified_global_donor'))
    for sample,study,group,y in zip(data['sample_ids'],data['study_ids'],data['split_group_ids'],data['y']):
        tables['samples'].append(dict(release_id=release,sample_id=str(sample),study_id=str(study),group_id=str(group)))
        tables['sample_annotations'].append(dict(release_id=release,sample_id=str(sample),dose_gy=float(y),
            dose_status=annotations[sample]['dose_status'],time_after_exposure_hours=24.,organism='Homo sapiens',tissue='whole_blood'))
    for probe,gene in zip(data['probe_ids'],data['gene_keys']):
        tables['features'].append(dict(release_id=release,probe_id=str(probe),gene_id=str(gene),gene_namespace='NCBI_GeneID'))
    for held in sorted(CORE):
        outer=np.flatnonzero(data['study_ids'] != held)
        for i,s in enumerate(data['sample_ids']):
            tables['split_members'].append(dict(release_id=release,split_id='outer_'+held,sample_id=str(s),
                partition='validation' if data['study_ids'][i]==held else 'train'))
        for fold,(train,val) in enumerate(inner_splits(data['study_ids'][outer],data['split_group_ids'][outer])):
            for partition,indices in [('train',outer[train]),('validation',outer[val])]:
                for idx in indices:
                    tables['split_members'].append(dict(release_id=release,split_id=f'outer_{held}_inner_{fold}',
                                                       sample_id=str(data['sample_ids'][idx]),partition=partition))
    # Referential and split checks are independent of BigQuery's unenforced keys.
    samples={r['sample_id'] for r in tables['samples']}; groups={r['group_id'] for r in tables['sample_groups']}
    assert len(samples)==145 and len(groups)==30
    assert all(r['group_id'] in groups for r in tables['samples'])
    assert {r['sample_id'] for r in tables['sample_annotations']}==samples
    for split in {r['split_id'] for r in tables['split_members']}:
        members=[r for r in tables['split_members'] if r['split_id']==split]
        lookup={r['sample_id']:r['group_id'] for r in tables['samples']}
        a={lookup[r['sample_id']] for r in members if r['partition']=='train'}
        b={lookup[r['sample_id']] for r in members if r['partition']=='validation'}
        assert not a&b
    records=[]
    for name,rows in tables.items():
        validate_rows(name,rows)
        raw=b''.join(encode(r)+b'\n' for r in rows)
        (destination/(name+'.jsonl')).write_bytes(raw)
        records.append(dict(table=name,rows=len(rows),bytes=len(raw),sha256=digest(raw)))
    manifest=dict(schema_version=1,release_id=release,purpose='core_development',
        samples=145,study_local_groups=30,probes=len(data['probe_ids']),
        core_bytes=len(payload),core_sha256=digest(payload),tables=records,
        processing='per_array_normexp_saddle_offset0; technical_mask_before_fit; no_between_array_normalization',
        locked_test_included=False)
    (destination/'manifest.json').write_bytes(encode(manifest))
    return destination,manifest


def publish(destination,manifest,cloud):
    base=PREFIX+'releases/'+manifest['release_id']+'/'
    receipt_path=ROOT/'context/cloud/integration-v1'/('release-'+manifest['release_id']+'.json')
    guard=cloud.preflight(sum(p.stat().st_size for p in destination.iterdir())+100_000)
    dataset_ref=cloud.immutable(base+'core.npz',(destination/'core.npz').read_bytes())
    rows=[dict(release_id=manifest['release_id'],expression_set_id=manifest['release_id']+'_linear',
        representation='core_common_probes_background_corrected',scale='linear_intensity',processing_scope='per_sample',
        normalization_status='between_array_normalization_deferred_to_training_fold',samples=145,probes=manifest['probes'],
        object=dataset_ref['object'],generation=dataset_ref['generation'],sha256=dataset_ref['sha256'])]
    validate_rows('expression_sets',rows)
    (destination/'expression_sets.jsonl').write_bytes(b''.join(encode(r)+b'\n' for r in rows))
    cloud.ensure_dataset()
    loaded={}
    for record in manifest['tables']+[dict(table='expression_sets',rows=1)]:
        name=record['table'];raw=(destination/(name+'.jsonl')).read_bytes()
        ref=cloud.immutable(base+name+'.jsonl',raw)
        result=cloud.load_table(name,SCHEMAS[name],ref,manifest['release_id']+'_'+name,record['rows'])
        loaded[name]={'source':ref,'rows':record['rows'],'job':result['jobReference'],
                      'load_statistics':result['statistics']['load']}
    public_manifest={**manifest,'dataset':dataset_ref,'warehouse_tables':{k:v['rows'] for k,v in loaded.items()}}
    release_ref=cloud.immutable(base+'release.json',encode(public_manifest))
    receipt=dict(release=release_ref,dataset=dataset_ref,tables=loaded,guard=guard,requests=cloud.requests)
    receipt_path.parent.mkdir(parents=True,exist_ok=True);receipt_path.write_bytes(encode(receipt))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
    directory,manifest=build()
    print(json.dumps(manifest,indent=2),flush=True)
    if args.publish:
        receipt=publish(directory,manifest,Cloud())
        print('Published and row-count verified',receipt['release']['object'],flush=True)
