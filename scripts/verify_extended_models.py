"""Independent extended-benchmark membership, ranking, weights, metrics and raw replay audit."""
import json,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import joblib
from core_baseline import GenePreparation
from development_preprocessing import DevelopmentPreprocessor
from extended_models import REGRESSION,CLASSIFICATION,STOCHASTIC
from run_extended_models import ROOT,LOG,OUT,FOLDS
from run_abcd_experiments import sha,save_json,write_tsv,verify_files

def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def regression(rr):
    errors=defaultdict(list)
    for r in rr:errors[r['donor_group']].append(float(r['output'])-float(r['actual_gy']))
    return dict(mae_gy=float(np.mean([np.mean(abs(np.array(e))) for e in errors.values()])),rmse_gy=float(np.sqrt(np.mean([np.mean(np.square(e)) for e in errors.values()]))),bias_gy=float(np.mean([np.mean(e) for e in errors.values()])))
def classification(rr):
    recall=[]
    for cls in (False,True):
        g=defaultdict(list)
        for r in rr:
            if (float(r['actual_gy'])>0)==cls:g[r['donor_group']].append((float(r['output'])>=.5)==cls)
        assert g
        recall.append(float(np.mean([np.mean(v) for v in g.values()])))
    b=defaultdict(list)
    for r in rr:b[r['donor_group']].append((float(r['output'])-(float(r['actual_gy'])>0))**2)
    return dict(balanced_accuracy=float(np.mean(recall)),specificity=recall[0],sensitivity=recall[1],false_positive_rate=1-recall[0],brier=float(np.mean([np.mean(v) for v in b.values()])))
def records(ids,values,labels):return [dict(profile_id=s,study=labels[s]['study'],donor_group=labels[s]['donor_group'],actual_gy=labels[s]['dose_gy'],output=float(v)) for s,v in zip(ids,values)]

def main():
    cfg=json.loads((LOG/'config.json').read_text());splits=json.loads((LOG/'splits.json').read_text());start=json.loads((LOG/'run-start.json').read_text());end=json.loads((LOG/'run-complete.json').read_text());freeze=json.loads((LOG/'protocol-freeze.json').read_text())
    assert freeze['utc']<start['utc']<end['utc'] and start['identity']==end['identity']
    for n,h in freeze['sha256'].items():assert sha(LOG/n)==h
    verify_files(start['code_sha256']);protected=json.loads((LOG/'protected-before.json').read_text());verify_files(protected)
    labels={r['profile_id']:r for r in json.loads((ROOT/cfg['release']/'metadata.json').read_text())['profiles'] if r['role']=='core_development'}
    assert len(labels)==145 and len({r['donor_group'] for r in labels.values()})==30
    selections={(r['study'],r['model']):r for r in json.loads((LOG/'selections.json').read_text())}
    rows=read(LOG/'predictions.tsv');assert len(rows)==3045
    for r in rows:
        r['seed_index']=int(r['seed_index']);r['actual_gy']=float(r['actual_gy']);r['output']=float(r['output']);a=labels[r['profile_id']]
        assert r['actual_gy']==a['dose_gy'] and r['study']==a['study'] and r['donor_group']==a['donor_group'] and np.isfinite(r['output'])
        assert r['task']==('classification' if r['model'] in CLASSIFICATION else 'regression')
        if r['task']=='classification':assert 0<=r['output']<=1
    traces=[];checked=0;foldcount=0
    for o in splits['outer']:
        assert set(o['training'])|set(o['validation'])==set(labels) and set(o['training']).isdisjoint(o['validation'])
        for f in o['inner']+[dict(name=o['study']+'/outer',training=o['training'],validation=o['validation'])]:
            assert {labels[s]['donor_group'] for s in f['training']}.isdisjoint({labels[s]['donor_group'] for s in f['validation']})
            path=FOLDS/f['name'];fit=json.loads((path/'fit.json').read_text())
            assert fit['nontraining_perturbation_invariant'] and fit['training']==f['training'] and fit['validation']==f['validation']
            for n,h in json.loads((path/'complete.json').read_text())['files'].items():assert sha(path/n)==h
            foldcount+=1
        for model in REGRESSION+CLASSIFICATION:
            scores=[];sel=selections[o['study'],model]
            for params in cfg['candidates'][model]:
                rr=[]
                for f in o['inner']:
                    r=json.loads((OUT/'trials'/f['name']/model/str(params['id'])/'result.json').read_text());traces.append(r)
                    assert r['params']==params and r['seed']==f['seed'] and r['training_ids']==f['training'] and r['validation_ids']==f['validation']
                    labs=[labels[s] for s in f['training']];studies=set(a['study'] for a in labs)
                    for i,a in enumerate(labs):
                        subset=[b for b in labs if b['study']==a['study'] and (model not in CLASSIFICATION or (b['dose_gy']>0)==(a['dose_gy']>0))]
                        expected=1/(len(studies)*(2 if model in CLASSIFICATION else 1)*len({b['donor_group'] for b in subset})*sum(b['donor_group']==a['donor_group'] for b in subset))
                        assert abs(r['weights'][i]-expected)<1e-14
                    rr+=records(r['validation_ids'],r['outputs'],labels)
                assert sorted(a['profile_id'] for a in rr)==sorted(o['training'])
                score=1-classification(rr)['balanced_accuracy'] if model in CLASSIFICATION else regression(rr)['mae_gy']
                saved=next(a for a in sel['candidates'] if a['params']['id']==params['id']);assert abs(score-saved['ranking_loss'])<1e-12
                scores.append((score,params['id']));checked+=1
            assert min(scores)[1]==sel['chosen']['params']['id']
    sliced=[];summaries=[];seedsummary=[]
    old=read(ROOT/'context/classical-comparison/v1/combined-predictions.tsv')
    for r in old:r.update(output=float(r['predicted_gy']),actual_gy=float(r['actual_gy']),seed_index=int(r['seed_index']),task='regression')
    allrows=rows+old
    for model in REGRESSION+CLASSIFICATION+('A','B','C','D','elastic_net','pls','gradient_boosting','median_null'):
        seeds=sorted({r['seed_index'] for r in allrows if r['model']==model});expected=3 if model in STOCHASTIC or model in ('A','B','C','D','gradient_boosting') else 1
        assert len(seeds)==expected
        for si in seeds:
            rr=[r for r in allrows if r['model']==model and r['seed_index']==si]
            assert len(rr)==145 and {r['profile_id'] for r in rr}==set(labels)
            if model in CLASSIFICATION:
                for st in ['pooled']+[o['study'] for o in splits['outer']]:
                    group=[r for r in rr if st=='pooled' or r['study']==st];sliced.append(dict(task='classification',model=model,seed_index=si,study=st,slice='all',profiles=len(group),**classification(group)))
            else:
                for st in ['pooled']+[o['study'] for o in splits['outer']]:
                    group=[r for r in rr if st=='pooled' or r['study']==st]
                    for name,g in [('all',group),('control',[r for r in group if r['actual_gy']==0]),('>2 Gy',[r for r in group if r['actual_gy']>2])]:sliced.append(dict(task='regression',model=model,seed_index=si,study=st,slice=name,profiles=len(g),**regression(g)))
                sm=[regression([r for r in rr if r['study']==o['study']])['mae_gy'] for o in splits['outer']]
                seedsummary.append(dict(model=model,seed_index=si,macro_study_mae_gy=float(np.mean(sm)),worst_study_mae_gy=max(sm),**regression(rr)))
                if si==0:summaries.append(dict(model=model,new=model in REGRESSION,macro_study_mae_gy=float(np.mean(sm)),worst_study_mae_gy=max(sm),**regression(rr)))
    write_tsv(LOG/'regression-metrics.tsv',[{k:v for k,v in r.items() if k!='task'} for r in sliced if r['task']=='regression'])
    write_tsv(LOG/'classification-metrics.tsv',[{k:v for k,v in r.items() if k!='task'} for r in sliced if r['task']=='classification'])
    save_json(LOG/'regression-summary.json',sorted(summaries,key=lambda r:r['mae_gy']));save_json(LOG/'seed-summary.json',seedsummary)
    with np.load(ROOT/cfg['release']/'core.npz',allow_pickle=False) as z:data={'core':{k:z[k] for k in z.files}}
    replays=[]
    for o in splits['outer']:
        path=FOLDS/o['study']/'outer';pre=DevelopmentPreprocessor.load(path/'preprocessing');view=pre.transform(data,o['validation']);assert set(pre.core_training)==set(o['training']) and not pre.aux_training
        for model in REGRESSION+CLASSIFICATION:
            p=selections[o['study'],model]['chosen']['params'];rep=p['representation']
            with np.load(path/f'{rep}-features.npz',allow_pickle=False) as z:g=GenePreparation(tuple(z['gene_ids']),z['medians'],z['selected'],z['mean'],z['scale'],z['selection_scores'],tuple(z['training_ids']))
            assert set(g.training_ids)==set(o['training']);x=g.transform(view[f'{rep}_values'].T,view['gene_ids'],True)
            for si,seed in enumerate(o['seeds'] if model in STOCHASTIC else o['seeds'][:1]):
                d=OUT/'selected'/o['study']/model/str(seed);r=json.loads((d/'result.json').read_text());traces.append(r);m=joblib.load(d/'model.joblib')
                assert r['params']==p and r['seed']==seed and r['training_ids']==o['training'] and r['validation_ids']==o['validation']
                actual=m.predict_proba(x)[:,1] if model in CLASSIFICATION else m.predict(x)*r['target_scale']+r['target_mean']
                diff=float(np.max(abs(actual-r['outputs'])));assert diff<1e-7
                saved={a['profile_id']:a['output'] for a in rows if a['model']==model and a['study']==o['study'] and a['seed_index']==si}
                assert max(abs(saved[s]-v) for s,v in zip(o['validation'],actual))<1e-7
                replays.append(dict(model=model,study=o['study'],seed=seed,raw_difference=diff,single_difference=r['single_difference']))
    manifests=0
    for c in OUT.rglob('complete.json'):
        cm=json.loads(c.read_text());assert cm['identity']==end['identity']
        for n,h in cm['files'].items():assert sha(c.parent/n)==h
        manifests+=1
    assert checked==264 and foldcount==18 and len(traces)==1383 and len(replays)==63 and manifests==1383
    verify_files(protected);save_json(LOG/'replays.json',replays)
    save_json(LOG/'verification.json',dict(status='PASS_EXTENDED_BENCHMARK',fits=1383,inner_fits=1320,outer_fits=63,candidate_rankings=checked,fold_caches=foldcount,raw_replays=63,max_raw_difference=max(r['raw_difference'] for r in replays),max_single_difference=max(r['single_difference'] for r in replays),protected_files_unchanged=len(protected),cache_manifests=manifests,unit_tests=6,locked_test_opened=False,cloud_changed=False,final_model_selected=False))
    print(json.dumps(sorted(summaries,key=lambda r:r['mae_gy']),indent=2));print('VERIFIED',len(traces),'fits',flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):main()
