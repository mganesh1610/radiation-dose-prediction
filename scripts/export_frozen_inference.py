"""Export trusted frozen artifacts and prove numerical parity without refitting."""
import json,sys,shutil
from pathlib import Path
import numpy as np
import joblib
from frozen_inference import FrozenInference,encode,digest,ADAPTER
from portable_model import estimator_spec
from abcd_models import load_model
from final_model_bundle import predict_panel,raw_panel
from development_preprocessing import load_release
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/cloud_prediction/frozen-v1'
EVIDENCE=ROOT/'context/cloud/prediction-workflow-v1'
MODEL=ROOT/'data/models/final-selection-v2'
FREEZE=ROOT/'context/final-training/v1/model-bundle-freeze.json'
FREEZE_SHA='3b051050b682601006fc965d01ca871aa175799863a48ee73fee314ddd66e725'

def main():
    if digest(FREEZE.read_bytes())!=FREEZE_SHA:raise ValueError('Frozen model ledger changed')
    freeze=json.loads(FREEZE.read_text())
    for name,sha in freeze['files'].items():
        if digest((MODEL/name).read_bytes())!=sha:raise ValueError('Frozen source changed: '+name)
    OUT.mkdir(parents=True,exist_ok=True);EVIDENCE.mkdir(parents=True,exist_ok=True)
    prep=MODEL/'folds/full-development'
    shutil.copyfile(prep/'preprocessing/core.npz',OUT/'core.npz')
    for rep in ('log2','rank'):shutil.copyfile(prep/f'{rep}-features.npz',OUT/f'{rep}-features.npz')
    models={}
    for method in ('A','B','C','D','elastic_net','pls','gradient_boosting'):
        directory=MODEL/f'final/{method}/20261002';r=json.loads((directory/'result.json').read_text())
        if method in ('A','B','C','D'):
            model,saved=load_model(directory/'model.pt');weights=model.state_dict();layers=[]
            for layer in ('encoder.0','encoder.2','dose'):
                layers.append(dict(weight=weights[layer+'.weight'].numpy().tolist(),bias=weights[layer+'.bias'].numpy().tolist(),relu=layer!='dose'))
            spec=dict(kind='neural',layers=layers);artifact=directory/'model.pt'
        else:
            artifact=directory/'model.joblib';spec=estimator_spec(method,joblib.load(artifact),None)
        models[method]=dict(estimator=spec,representation=r['params']['representation'],seed=20261002,
                            target_mean=r['target_mean'],target_scale=r['target_scale'],source_sha256=digest(artifact.read_bytes()))
    median=MODEL/'final/median_null/model.json'
    models['median_null']=dict(value=json.loads(median.read_text())['prediction_gy'],source_sha256=digest(median.read_bytes()),seed=None)
    (OUT/'models.json').write_bytes(encode(models))
    config=json.loads((ROOT/'context/final-selection/v2/config.json').read_text())
    data,metadata=load_release(ROOT/config['release']);core=data['core']
    # One genuine control from each original study, selected for a functionality demo, not an accuracy claim.
    examples=[]
    for study in ('GSE102971','GSE8917','GSE65292'):
        dose={r['profile_id']:r['dose_gy'] for r in metadata['profiles']}
        index=int(np.flatnonzero((core['study_ids']==study)&np.array([dose[str(s)]==0 for s in core['sample_ids']]))[0]);sample=str(core['sample_ids'][index])
        body=dict(schema_version=1,adapter=ADAPTER,sample_id=sample,
                  metadata=dict(study=study,organism='human',tissue='whole_blood',exposure='matched_control',time_hours=24,
                    units='positive_linear_intensity',normalization='background_corrected_not_between_array_normalized',
                    technical_flags='masked_before_normalization',source_url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+sample,
                    provenance='Audited raw Agilent pipeline; sample-independent background correction and technical masking; common annotated probe universe.'),
                  probe_ids=core['probe_ids'].tolist(),values=[float(v) if np.isfinite(v) else None for v in core['values'][:,index]])
        name=sample+'.json';payload=encode(body);(OUT/name).write_bytes(payload)
        examples.append(dict(id=sample,file=name,study=study,sha256=digest(payload),bytes=len(payload),training_overlap=True,
                             observed_dose_gy=0,purpose='Functionality demonstration; not independent validation'))
    names=['core.npz','log2-features.npz','rank-features.npz','models.json']+[r['file'] for r in examples]
    manifest=dict(schema_version=1,release='frozen-research-v1',source_freeze_sha256=FREEZE_SHA,
                  files={n:digest((OUT/n).read_bytes()) for n in names},examples=examples)
    (OUT/'manifest.json').write_bytes(encode(manifest));engine=FrozenInference(OUT)
    ids=core['sample_ids'].tolist();panels,_=engine.panels(core['values']);checks={}
    # Fixed engineering tolerance established before comparison; all 145 development inputs, no test data.
    tolerance=2e-5
    for method in models:
        if method=='median_null':ref=np.full(len(ids),4.45)
        else:
            directory=MODEL/f'final/{method}/20261002';original=raw_panel(MODEL,directory,data,ids,freeze['identity'])
            if not np.allclose(panels[models[method]['representation']],original,atol=1e-12,rtol=0):raise ValueError('Preprocessing mismatch')
            ref=predict_panel(directory,original,freeze['identity'])
        pred=engine.predict_values(method,panels);err=float(np.max(np.abs(pred-ref)))
        if err>tolerance:raise ValueError('Runtime parity failed: '+method+' '+str(err))
        checks[method]=dict(max_absolute_error_gy=err,profiles=len(ids),passed=True)
    # Cloud replay vectors use real curated files; expected values originate from the Mac models.
    replay={}
    for row in examples:
        j=ids.index(row['id']);replay[row['id']]={}
        for method in models:
            if method=='median_null':value=4.45
            else:
                directory=MODEL/f'final/{method}/20261002'
                value=float(predict_panel(directory,raw_panel(MODEL,directory,data,[row['id']],freeze['identity']),freeze['identity'])[0])
            replay[row['id']][method]=value
    (OUT/'replay.json').write_bytes(encode(dict(tolerance_gy=tolerance,expected=replay)))
    manifest['files']['replay.json']=digest((OUT/'replay.json').read_bytes());(OUT/'manifest.json').write_bytes(encode(manifest))
    (EVIDENCE/'mac-parity.json').write_bytes(encode(dict(tolerance_gy=tolerance,checks=checks,examples=examples,release_sha256=digest(encode(manifest)),test_data_used=False,refitting=False)))
    print(json.dumps(dict(bytes=sum(p.stat().st_size for p in OUT.iterdir()),checks=checks)))
if __name__=='__main__':main()
