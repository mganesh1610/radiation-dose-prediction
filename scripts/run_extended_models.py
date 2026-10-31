"""Fixed additional regression/classification comparison on original fold caches."""
import json,sys,time,resource,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import joblib,sklearn,xgboost
from threadpoolctl import threadpool_limits
from abcd_models import balanced_weights
from core_baseline import donor_metrics
from extended_models import REGRESSION,CLASSIFICATION,STOCHASTIC,fit_estimator,binary_metrics
from run_matched_classical import load_fold,FOLDS
from run_abcd_experiments import sha,save_json,write_tsv,verify_files,finish,completed
ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'context/extended-models/v1';OUT=ROOT/'data/models/extended-models-v1'

def score(ids,pred,labels,model):
    y=[labels[s]['dose_gy'] for s in ids];donors=[labels[s]['donor_group'] for s in ids]
    if model in CLASSIFICATION:
        metric=binary_metrics(np.array(y)>0,pred,donors);return 1-metric['balanced_accuracy'],metric
    metric,_=donor_metrics(y,pred,donors);return metric['mae'],metric

def one(fold,p,seed,labels,path,identity,selected=False):
    path.mkdir(parents=True,exist_ok=True)
    if not completed(path,identity):
        x,v,f=load_fold(fold,p['representation']);tr=[labels[s] for s in f['training']];binary=p['model'] in CLASSIFICATION
        actual=np.array([r['dose_gy'] for r in tr]);y=(actual>0).astype(int) if binary else (actual-f['target_mean'])/f['target_scale']
        w=balanced_weights([r['study'] for r in tr],[r['donor_group'] for r in tr],y if binary else None)
        start=time.perf_counter();est=fit_estimator(p,x,y,w,seed);elapsed=time.perf_counter()-start
        def infer(m,z):return m.predict_proba(z)[:,1] if binary else m.predict(z)*f['target_scale']+f['target_mean']
        pred=np.asarray(infer(est,v));ranking,metric=score(f['validation'],pred,labels,p['model']);reload_diff=None;single_diff=None
        if selected:
            joblib.dump(est,path/'model.joblib',compress=3);m=joblib.load(path/'model.joblib');reload_diff=float(np.max(abs(infer(m,v)-pred)))
            single_diff=float(np.max(abs(np.array([infer(m,row[None])[0] for row in v])-pred)))
            if reload_diff>1e-8 or single_diff>1e-5:raise ValueError('Replay mismatch')
        save_json(path/'result.json',dict(params=p,seed=seed,fit_seconds=elapsed,fold=fold['name'],training_ids=f['training'],validation_ids=f['validation'],outputs=pred.tolist(),ranking_loss=ranking,metrics=metric,weights=w.tolist(),target_mean=f['target_mean'],target_scale=f['target_scale'],reload_difference=reload_diff,single_difference=single_diff))
        finish(path,identity)
    return json.loads((path/'result.json').read_text())

def main():
    start=time.perf_counter();cfg=json.loads((LOG/'config.json').read_text());freeze=json.loads((LOG/'protocol-freeze.json').read_text());splits=json.loads((LOG/'splits.json').read_text())
    for n,h in freeze['sha256'].items():assert sha(LOG/n)==h
    verify_files(json.loads((LOG/'protected-before.json').read_text()))
    for c in FOLDS.rglob('complete.json'):
        for n,h in json.loads(c.read_text())['files'].items():assert sha(c.parent/n)==h
    source={n:sha(ROOT/n) for n in ['scripts/extended_models.py','scripts/run_extended_models.py','scripts/run_matched_classical.py','scripts/abcd_models.py','scripts/core_baseline.py','scripts/run_abcd_experiments.py']}
    identity=__import__('hashlib').sha256(json.dumps(dict(protocol=freeze['sha256'],source=source,xgboost=xgboost.__version__),sort_keys=True).encode()).hexdigest()
    if (LOG/'run-start.json').exists():assert json.loads((LOG/'run-start.json').read_text())['identity']==identity
    else:save_json(LOG/'run-start.json',dict(utc=datetime.now(timezone.utc).isoformat(),identity=identity,code_sha256=source,python=sys.version,sklearn=sklearn.__version__,xgboost=xgboost.__version__,numpy=np.__version__,platform=platform.platform(),threads=4))
    meta=json.loads((ROOT/cfg['release']/'metadata.json').read_text());labels={r['profile_id']:r for r in meta['profiles'] if r['role']=='core_development'}
    assert len(labels)==145 and set(labels)==set(splits['core_profiles'])
    selection=[];rows=[];count=0
    for o in splits['outer']:
        study=o['study']
        for model in REGRESSION+CLASSIFICATION:
            candidates=[]
            for p in cfg['candidates'][model]:
                ids=[];pred=[]
                for fold in o['inner']:
                    r=one(fold,p,fold['seed'],labels,OUT/'trials'/fold['name']/model/str(p['id']),identity);count+=1
                    ids+=r['validation_ids'];pred+=r['outputs']
                assert sorted(ids)==sorted(o['training']);loss,metric=score(ids,pred,labels,model)
                candidates.append(dict(params=p,ranking_loss=loss,metrics=metric))
                save_json(LOG/'progress.json',dict(completed_fits=count,study=study,model=model,candidate=p['id'],status='running'))
            best=min(candidates,key=lambda a:(a['ranking_loss'],a['params']['id']));selection.append(dict(study=study,model=model,candidates=candidates,chosen=best));save_json(LOG/'selections.json',selection)
            fold=dict(name=study+'/outer',training=o['training'],validation=o['validation']);seeds=o['seeds'] if model in STOCHASTIC else o['seeds'][:1]
            for si,seed in enumerate(seeds):
                r=one(fold,best['params'],seed,labels,OUT/'selected'/study/model/str(seed),identity,True);count+=1
                for s,v in zip(r['validation_ids'],r['outputs']):
                    a=labels[s];rows.append(dict(model=model,task='classification' if model in CLASSIFICATION else 'regression',seed_index=si,seed=seed,profile_id=s,study=study,donor_group=a['donor_group'],actual_gy=a['dose_gy'],output=v,candidate_id=best['params']['id'],representation=best['params']['representation']))
            write_tsv(LOG/'predictions.tsv',rows);print(study,model,'selected',best['params'],'inner_loss',round(best['ranking_loss'],5),'fits',count,flush=True)
    assert count==1383 and len(rows)==3045
    verify_files(json.loads((LOG/'protected-before.json').read_text()))
    save_json(LOG/'run-complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),identity=identity,inner_fits=1320,outer_fits=63,total_fits=count,prediction_rows=len(rows),wall_seconds=time.perf_counter()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,locked_test_opened=False,cloud_changed=False))
    save_json(LOG/'progress.json',dict(status='complete',completed_fits=count));print('COMPLETE',count,flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=4):main()
    except Exception as e:
        save_json(LOG/'failure.json',dict(utc=datetime.now(timezone.utc).isoformat(),type=type(e).__name__,error=str(e)));raise
