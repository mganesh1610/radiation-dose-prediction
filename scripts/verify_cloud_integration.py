"""Read-only verification of warehouse keys and cloud model artifact replay."""
import json
from pathlib import Path
import urllib.parse
import numpy as np
from cloud_io import Cloud, PROJECT, DATASET, encode, digest
from training_contract import SCHEMAS, validate_rows, METHODS
from training_worker import job_name
from portable_model import PortablePredictor

ROOT=Path(__file__).resolve().parents[1]


def main():
    cloud=Cloud(max_requests=250,max_upload_bytes=0)
    rows={};meta={}
    for table in SCHEMAS:
        base=f'projects/{PROJECT}/datasets/{DATASET}/tables/{table}'
        meta[table]=cloud.api(base,service='bq');records=[];page=''
        while True:
            response=cloud.api(base+'/data?maxResults=10000'+page,service='bq')
            for r in response.get('rows',[]):
                row={}
                for field,value in zip(meta[table]['schema']['fields'],r['f']):
                    v=value['v'];kind=field['type']
                    if v is not None:
                        if kind in ('INTEGER','INT64'):v=int(v)
                        elif kind in ('FLOAT','FLOAT64'):v=float(v)
                        elif kind in ('BOOLEAN','BOOL'):v=v=='true'
                    row[field['name']]=v
                records.append(row)
            if not response.get('pageToken'):break
            page='&pageToken='+urllib.parse.quote(response['pageToken'],safe='')
        validate_rows(table,records)
        if len(records)!=int(meta[table]['numRows']):raise ValueError('Warehouse count mismatch')
        rows[table]=records
    samples={r['sample_id']:r for r in rows['samples']};groups={r['group_id'] for r in rows['sample_groups']}
    if len(samples)!=145 or len(groups)!=30:raise ValueError('Unexpected core membership')
    assert all(r['group_id'] in groups for r in samples.values())
    assert {r['sample_id'] for r in rows['sample_annotations']}==set(samples)
    for split in {r['split_id'] for r in rows['split_members']}:
        members=[r for r in rows['split_members'] if r['split_id']==split]
        assert {samples[r['sample_id']]['group_id'] for r in members if r['partition']=='train'}.isdisjoint(
            {samples[r['sample_id']]['group_id'] for r in members if r['partition']=='validation'})
    jobs={r['job_id'] for r in rows['model_runs']}
    assert all(r['job_id'] in jobs and r['sample_id'] in samples for r in rows['predictions'])
    assert all(r['job_id'] in jobs for r in rows['model_metrics'])
    checks={}
    for method in METHODS:
        job_id='integration_v1_'+method
        state,_=cloud.read_json(job_name(job_id));assert state['status']=='SUCCEEDED'
        ref=state['result'];committed=json.loads(cloud.read(ref['object'],ref['generation'],ref['sha256']))
        folder=ROOT/'data/cloud_replay'/job_id;folder.mkdir(parents=True,exist_ok=True)
        for name,ref in committed['artifacts'].items():
            if name not in ('model.json','reference.npz','preparation.npz','replay.npz'):raise ValueError('Unexpected model member')
            (folder/name).write_bytes(cloud.read(ref['object'],ref['generation'],ref['sha256']))
        p=PortablePredictor(folder)
        with np.load(folder/'replay.npz',allow_pickle=False) as z:
            predicted=p.predict(z['values'],z['probe_ids']);difference=float(np.max(abs(predicted-z['expected_gy'])))
            np.testing.assert_allclose(predicted,z['expected_gy'],atol=1e-8,rtol=1e-8)
        checks[method]={'reloaded_from_cloud':True,'max_abs_difference_gy':difference,
                        'scientific_prediction_publication_allowed':False}
    report={'table_counts':{k:len(v) for k,v in rows.items()},'table_logical_bytes':{k:int(v['numBytes']) for k,v in meta.items()},
            'cloud_artifact_replay':checks,'keys_and_references_verified':True,'requests':cloud.requests,
            'serving_environment':'Mac CPU, not deployed Linux Cloud Run','locked_test_accessed':False}
    (ROOT/'context/cloud/integration-v1/verification.json').write_bytes(encode(report))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
