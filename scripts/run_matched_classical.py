"""Matched eight-candidate classical benchmarks on verified A–D fold caches."""
import json,sys,time,platform,resource
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from abcd_models import balanced_weights
from core_baseline import donor_metrics
from matched_classical import fit_classical
from run_abcd_experiments import sha,save_json,write_tsv,verify_files,finish,completed

ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'context/classical-comparison/v1';OUT=ROOT/'data/models/classical-matched-v1'
OLD=ROOT/'context/model-abcd/v1';FOLDS=ROOT/'data/models/abcd-v1/folds'
MODELS=('elastic_net','pls','gradient_boosting')

def load_fold(fold,representation):
    d=FOLDS/fold['name'];f=json.loads((d/'fit.json').read_text())
    if f['training']!=fold['training'] or f['validation']!=fold['validation'] or not f['nontraining_perturbation_invariant']:
        raise ValueError('Fold cache membership mismatch')
    with np.load(d/f'{representation}-panel.npz',allow_pickle=False) as z:
        if list(z['training_ids'])!=fold['training'] or list(z['validation_ids'])!=fold['validation']:raise ValueError('Panel order mismatch')
        x=z['train'];v=z['validation']
    if x.shape!=(len(f['training']),500) or v.shape!=(len(f['validation']),500):raise ValueError('Invalid panel dimensions')
    return x,v,f

def one_fit(fold,params,seed,labels,path,identity,selected=False):
    path.mkdir(parents=True,exist_ok=True)
    if not completed(path,identity):
        x,v,f=load_fold(fold,params['representation']);tr=[labels[s] for s in f['training']]
        weights=balanced_weights([r['study'] for r in tr],[r['donor_group'] for r in tr])
        y=(np.array([r['dose_gy'] for r in tr])-f['target_mean'])/f['target_scale'];started=time.perf_counter()
        model,detail=fit_classical(params,x,y,weights,seed);elapsed=time.perf_counter()-started
        pred=np.asarray(model.predict(v)).reshape(-1)*f['target_scale']+f['target_mean']
        score,_=donor_metrics([labels[s]['dose_gy'] for s in f['validation']],pred,[labels[s]['donor_group'] for s in f['validation']])
        replay=None;single=None
        if selected:
            temp=path/'model.partial';joblib.dump(model,temp,compress=3);temp.replace(path/'model.joblib')
            loaded=joblib.load(path/'model.joblib');replayed=np.asarray(loaded.predict(v)).reshape(-1)*f['target_scale']+f['target_mean']
            replay=float(np.max(abs(pred-replayed)))
            individual=np.array([np.asarray(loaded.predict(row[None])).ravel()[0] for row in v])*f['target_scale']+f['target_mean']
            single=float(np.max(abs(pred-individual)))
            if replay>1e-9 or single>1e-7:raise ValueError('Saved replay mismatch')
        save_json(path/'result.json',dict(params=params,seed=seed,fit_seconds=elapsed,estimator=detail,fold=fold['name'],training_ids=f['training'],
             validation_ids=f['validation'],predicted_gy=pred.tolist(),metrics=score,target_mean=f['target_mean'],target_scale=f['target_scale'],
             sample_weights=weights.tolist(),reload_difference_gy=replay,individual_difference_gy=single))
        finish(path,identity)
    return json.loads((path/'result.json').read_text())

def main():
    start=time.perf_counter();cfg=json.loads((LOG/'config.json').read_text());freeze=json.loads((LOG/'protocol-freeze.json').read_text())
    for name,h in freeze['sha256'].items():
        if sha(LOG/name)!=h:raise ValueError('Protocol changed')
    verify_files(json.loads((LOG/'protected-before.json').read_text()))
    sources=['scripts/run_matched_classical.py','scripts/matched_classical.py','scripts/abcd_models.py','scripts/core_baseline.py','scripts/run_abcd_experiments.py']
    source={s:sha(ROOT/s) for s in sources}
    import hashlib
    identity=hashlib.sha256(json.dumps(dict(protocol=freeze['sha256'],source=source),sort_keys=True).encode()).hexdigest()
    if (LOG/'run-start.json').exists():
        if json.loads((LOG/'run-start.json').read_text())['identity']!=identity:raise ValueError('Run identity changed')
    else:save_json(LOG/'run-start.json',dict(utc=datetime.now(timezone.utc).isoformat(),identity=identity,code_sha256=source,python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,platform=platform.platform(),threads=4))
    splits=json.loads((OLD/'splits.json').read_text());meta=json.loads((ROOT/cfg['release']/'metadata.json').read_text());labels={r['profile_id']:r for r in meta['profiles'] if r['role']=='core_development'}
    if set(labels)!=set(splits['core_profiles']) or len(labels)!=145:raise ValueError('Wrong core membership')
    rows=[];selection=[];count=0
    for outer in splits['outer']:
        study=outer['study']
        for model in MODELS:
            candidates=[]
            for params in cfg['candidates'][model]:
                ids=[];pred=[]
                for fold in outer['inner']:
                    result=one_fit(fold,params,fold['seed'],labels,OUT/'trials'/fold['name']/model/str(params['id']),identity)
                    ids+=result['validation_ids'];pred+=result['predicted_gy'];count+=1
                score,_=donor_metrics([labels[s]['dose_gy'] for s in ids],pred,[labels[s]['donor_group'] for s in ids])
                candidates.append(dict(params=params,metrics=score));print(study,model,params['id'],'inner MAE',round(score['mae'],5),flush=True)
                save_json(LOG/'progress.json',dict(status='running',completed_fits=count,study=study,model=model,candidate=params['id']))
            best=min(candidates,key=lambda r:(r['metrics']['mae'],r['params']['id']))
            selection.append(dict(study=study,model=model,chosen=best,candidates=candidates));save_json(LOG/'selections.json',selection)
            fold=dict(name=f'{study}/outer',training=outer['training'],validation=outer['validation'])
            seeds=outer['seeds'] if model=='gradient_boosting' else outer['seeds'][:1]
            for si,seed in enumerate(seeds):
                result=one_fit(fold,best['params'],seed,labels,OUT/'selected'/study/model/str(seed),identity,True);count+=1
                for s,pred in zip(result['validation_ids'],result['predicted_gy']):
                    label=labels[s];rows.append(dict(model=model,seed_index=si,seed=seed,profile_id=s,study=study,donor_group=label['donor_group'],actual_gy=label['dose_gy'],predicted_gy=pred,candidate_id=best['params']['id'],representation=best['params']['representation']))
            print('Selected',study,model,best['params'],flush=True)
        write_tsv(LOG/'predictions.tsv',rows)
    if count!=375 or len(rows)!=725:raise ValueError('Incomplete training budget')
    verify_files(json.loads((LOG/'protected-before.json').read_text()))
    save_json(LOG/'run-complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),identity=identity,inner_fits=360,outer_fits=15,completed_fits=count,
        prediction_rows=len(rows),wall_seconds=time.perf_counter()-start,process_maxrss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        maxrss_units='bytes on this macOS host',locked_test_opened=False,cloud_changed=False))
    save_json(LOG/'progress.json',dict(status='complete',completed_fits=count));print('COMPLETE',count,'fits',flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=4):main()
    except Exception as exc:
        save_json(LOG/'failure.json',dict(error=str(exc),type=type(exc).__name__,utc=datetime.now(timezone.utc).isoformat()));raise
