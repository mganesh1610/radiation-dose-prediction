"""Independent ranking, membership, raw replay and descriptive clustering audit."""
import csv,json,itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import silhouette_score,adjusted_rand_score
from core_baseline import GenePreparation
from development_preprocessing import DevelopmentPreprocessor
from run_clustering import ROOT,LOG,OUT,MODELS,FOLDS
from run_abcd_experiments import sha,save_json,write_tsv,verify_files

def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def metrics(rows):
    g=defaultdict(list)
    for r in rows:g[r['donor_group']].append(float(r['predicted_gy'])-float(r['actual_gy']))
    if not g:raise ValueError('Empty metric slice')
    return dict(mae_gy=float(np.mean([np.mean(abs(np.array(v))) for v in g.values()])),rmse_gy=float(np.sqrt(np.mean([np.mean(np.square(v)) for v in g.values()]))),bias_gy=float(np.mean([np.mean(v) for v in g.values()])))
def records(ids,pred,labels):return [dict(profile_id=s,study=labels[s]['study'],donor_group=labels[s]['donor_group'],actual_gy=labels[s]['dose_gy'],predicted_gy=float(v)) for s,v in zip(ids,pred)]

def main():
    cfg=json.loads((LOG/'config.json').read_text());splits=json.loads((LOG/'splits.json').read_text());freeze=json.loads((LOG/'protocol-freeze.json').read_text());start=json.loads((LOG/'run-start.json').read_text());end=json.loads((LOG/'run-complete.json').read_text())
    assert freeze['utc']<start['utc']<end['utc'] and start['identity']==end['identity']
    for n,h in freeze['sha256'].items():assert sha(LOG/n)==h
    verify_files(start['code_sha256']);protected=json.loads((LOG/'protected-before.json').read_text());verify_files(protected)
    release=ROOT/cfg['release'];meta=json.loads((release/'metadata.json').read_text());labels={r['profile_id']:r for r in meta['profiles']}
    for r in json.loads((release/'release.json').read_text())['files']:
        assert (release/r['name']).stat().st_size==r['bytes'] and sha(release/r['name'])==r['sha256']
    selections={(r['study'],r['model']):r for r in json.loads((LOG/'selections.json').read_text())};rows=read(LOG/'predictions.tsv');ab=read(LOG/'ablations.tsv')
    for r in rows+ab:
        r['seed_index']=int(r['seed_index']);r['actual_gy']=float(r['actual_gy']);r['predicted_gy']=float(r['predicted_gy']);a=labels[r['profile_id']]
        assert a['role']=='core_development' and a['dose_gy']==r['actual_gy'] and a['study']==r['study'] and a['donor_group']==r['donor_group']
    rankings=0;folds=0;traces=[]
    for o in splits['outer']:
        assert set(o['training']).isdisjoint(o['validation']) and set(o['training'])|set(o['validation'])==set(splits['core_profiles'])
        for f in o['inner']+[dict(name=o['study']+'/outer',training=o['training'],validation=o['validation'])]:
            assert {labels[s]['donor_group'] for s in f['training']}.isdisjoint({labels[s]['donor_group'] for s in f['validation']})
            fit=json.loads((FOLDS/f['name']/'fit.json').read_text());assert fit['training']==f['training'] and fit['validation']==f['validation'] and fit['nontraining_perturbation_invariant'];folds+=1
            for n,h in json.loads((FOLDS/f['name']/'complete.json').read_text())['files'].items():assert sha(FOLDS/f['name']/n)==h
        for model in MODELS:
            choices=[];s=selections[o['study'],model]
            for p in cfg['candidates'][model]:
                rr=[]
                for f in o['inner']:
                    r=json.loads((OUT/'trials'/f['name']/model/str(p['id'])/'result.json').read_text());traces.append(r)
                    assert r['params']==p and r['seed']==f['seed'] and r['training_ids']==f['training'] and r['validation_ids']==f['validation']
                    tr=[labels[i] for i in f['training']]
                    for i,a in enumerate(tr):
                        st=[b for b in tr if b['study']==a['study']]
                        expected=1/(len({b['study'] for b in tr})*len({b['donor_group'] for b in st})*sum(b['donor_group']==a['donor_group'] for b in st))
                        assert abs(r['weights'][i]-expected)<1e-14
                    rr+=records(r['validation_ids'],r['predicted_gy'],labels)
                assert sorted(r['profile_id'] for r in rr)==sorted(o['training'])
                score=metrics(rr)['mae_gy'];saved=next(a for a in s['candidates'] if a['params']['id']==p['id']);assert abs(score-saved['metrics']['mae'])<1e-12
                choices.append((score,p['id']));rankings+=1
            assert min(choices)[1]==s['chosen']['params']['id']
    summary=[];slices=[];seeds=[];deltas=[]
    old=read(ROOT/'context/classical-comparison/v1/combined-predictions.tsv')
    old=[r for r in old if r['model'] in ('elastic_net','A','median_null')]
    ext=read(ROOT/'context/extended-models/v1/predictions.tsv')
    old += [dict(r,predicted_gy=r['output']) for r in ext if r['model']=='ridge']
    for r in old:r['seed_index']=int(r['seed_index'])
    for model in MODELS+('elastic_net','A','ridge','median_null'):
        available=sorted({r['seed_index'] for r in rows+old if r['model']==model})
        for si in available:
            rr=[r for r in rows+old if r['model']==model and r['seed_index']==si];assert len(rr)==145 and {r['profile_id'] for r in rr}==set(splits['core_profiles'])
            studymae=[metrics([r for r in rr if r['study']==o['study']])['mae_gy'] for o in splits['outer']]
            result=dict(model=model,seed_index=si,**metrics(rr),equal_study_mae_gy=float(np.mean(studymae)),worst_study_mae_gy=max(studymae));seeds.append(result)
            if si==0:summary.append(result)
            for study in ['pooled']+[o['study'] for o in splits['outer']]:
                group=[r for r in rr if study=='pooled' or r['study']==study]
                for name,part in [('all',group),('control',[r for r in group if float(r['actual_gy'])==0]),('>2 Gy',[r for r in group if float(r['actual_gy'])>2])]:
                    slices.append(dict(model=model,seed_index=si,study=study,slice=name,profiles=len(part),**metrics(part)))
    for model in ('kmeans_ridge','gmm_ridge'):
        for si in range(3):
            rr=[r for r in rows if r['model']==model and r['seed_index']==si];aa=[r for r in ab if r['parent_model']==model and r['seed_index']==si]
            assert len(aa)==145 and {r['profile_id'] for r in aa}==set(splits['core_profiles'])
            for study in ['pooled']+[o['study'] for o in splits['outer']]:
                r=[a for a in rr if study=='pooled' or a['study']==study];b=[a for a in aa if study=='pooled' or a['study']==study]
                for name,x,y in [('all',r,b),('control',[a for a in r if a['actual_gy']==0],[a for a in b if a['actual_gy']==0]),('>2 Gy',[a for a in r if a['actual_gy']>2],[a for a in b if a['actual_gy']>2])]:
                    m,n=metrics(x),metrics(y);deltas.append(dict(model=model,seed_index=si,study=study,slice=name,cluster_mae_gy=m['mae_gy'],paired_pca_mae_gy=n['mae_gy'],delta_mae_gy=m['mae_gy']-n['mae_gy']))
    write_tsv(LOG/'regression-metrics.tsv',slices);write_tsv(LOG/'paired-ablation.tsv',deltas);save_json(LOG/'regression-summary.json',sorted(summary,key=lambda r:r['mae_gy']));save_json(LOG/'seed-summary.json',seeds)
    with np.load(release/'core.npz',allow_pickle=False) as z:data={'core':{k:z[k] for k in z.files}}
    replays=[]
    for o in splits['outer']:
        d=FOLDS/o['study']/'outer';prep=DevelopmentPreprocessor.load(d/'preprocessing');assert set(prep.core_training)==set(o['training']) and not prep.aux_training
        view=prep.transform(data,o['validation'])
        for model in MODELS:
            p=selections[o['study'],model]['chosen']['params'];rep=p['representation']
            with np.load(d/f'{rep}-features.npz',allow_pickle=False) as z:g=GenePreparation(tuple(z['gene_ids']),z['medians'],z['selected'],z['mean'],z['scale'],z['selection_scores'],tuple(z['training_ids']))
            assert set(g.training_ids)==set(o['training']);x=g.transform(view[f'{rep}_values'].T,view['gene_ids'],True)
            for si,seed in enumerate(o['seeds'][:1] if model=='pca_ridge' else o['seeds']):
                for arm in (['selected','ablations'] if model in ('kmeans_ridge','gmm_ridge') else ['selected']):
                    path=OUT/arm/o['study']/model/str(seed);r=json.loads((path/'result.json').read_text());traces.append(r);m=joblib.load(path/'model.joblib')
                    expected=p if arm=='selected' else dict(id=p['id'],model='pca_ridge',representation=rep,components=10,alpha=p['alpha'])
                    assert r['params']==expected and r['training_ids']==o['training'] and r['validation_ids']==o['validation'] and r['seed']==seed
                    actual=m.predict(x)*r['target_scale']+r['target_mean'];diff=float(np.max(abs(actual-r['predicted_gy'])));assert diff<1e-7
                    saved={a['profile_id']:a['predicted_gy'] for a in (rows if arm=='selected' else ab) if a.get('model',a.get('parent_model'))==model and a['study']==o['study'] and a['seed_index']==si}
                    assert max(abs(saved[s]-v) for s,v in zip(o['validation'],actual))<1e-7
                    replays.append(dict(arm=arm,study=o['study'],model=model,seed=seed,raw_difference_gy=diff,single_difference_gy=r['single_difference']))
    diagnostic=[];stability=[];primary={};assignments=read(LOG/'cluster-assignments.tsv');nassign=0
    diagnostic_root=OUT/'diagnostics'
    with np.load(diagnostic_root/'matrix/views.npz',allow_pickle=False) as a:raw={k:a[k] for k in a.files}
    assert len(raw['profile_ids'])==567 and set(raw['profile_ids'])==set(labels)
    for scope,rep in itertools.product(['core','all'],['log2','rank']):
        viewdir=diagnostic_root/f'{scope}-{rep}';m=joblib.load(viewdir/'view/model.joblib')
        with np.load(viewdir/'view/panel.npz',allow_pickle=False) as a:panel={k:a[k] for k in a.files}
        ids=list(panel['ids']);positions=[list(raw['profile_ids']).index(s) for s in ids]
        assert set(ids)==({s for s in labels if labels[s]['role']=='core_development'} if scope=='core' else set(labels))
        np.testing.assert_allclose(m.transform(raw[f'{rep}_values'][:,positions].T,raw['gene_ids']),panel['z'],atol=1e-10)
        for p in sorted(viewdir.glob('*/result.json')):
            r=json.loads(p.read_text());ix=[ids.index(s) for s in r['ids']];z=panel['z'][ix];labs=np.array(r['labels']);st=[labels[s]['study'] for s in r['ids']];dose=np.array([labels[s]['dose_gy'] for s in r['ids']])
            assert len(set(labs))==r['k'] and sum(r['cluster_sizes'])==len(ix)
            assert abs(silhouette_score(z,labs)-r['silhouette'])<1e-12 and abs(adjusted_rand_score(st,labs)-r['study_ari'])<1e-12
            if r['controls_only']:assert np.all(dose==0)
            else:
                assert abs(adjusted_rand_score(dose>0,labs)-r['exposure_ari'])<1e-12
                assert abs(adjusted_rand_score(np.where(dose==0,0,np.where(dose<=2,1,2)),labs)-r['dose_group_ari'])<1e-12
            model=joblib.load(p.parent/'cluster.joblib');np.testing.assert_array_equal(model.labels_ if r['method']=='ward' else model.predict(z),labs)
            saved=[a for a in assignments if a['scope']==scope and a['representation']==rep and (a['controls_only']=='True')==r['controls_only'] and a['method']==r['method'] and int(a['k'])==r['k'] and int(a['seed_index'])==r['seed_index']]
            assert {a['profile_id']:int(a['cluster']) for a in saved}==dict(zip(r['ids'],r['labels']));nassign+=len(saved)
            diagnostic.append(r)
        for method in ('kmeans','ward','gmm'):
            candidates=[r for r in diagnostic if r['scope']==scope and r['representation']==rep and r['method']==method and not r['controls_only'] and r['seed_index']==0]
            best=min(candidates,key=lambda r:(-r['silhouette'],r['k']));primary[scope+'-'+rep+'-'+method]=best
            for k in cfg['diagnostic_k']:
                rr=sorted([r for r in diagnostic if r['scope']==scope and r['representation']==rep and r['method']==method and not r['controls_only'] and r['k']==k],key=lambda r:r['seed_index'])
                for a,b in itertools.combinations(rr,2):stability.append(dict(scope=scope,representation=rep,method=method,k=k,seed_index_a=a['seed_index'],seed_index_b=b['seed_index'],assignment_ari=float(adjusted_rand_score(a['labels'],b['labels']))))
    save_json(LOG/'descriptive-selected.json',primary);write_tsv(LOG/'seed-stability.tsv',stability)
    counts=0
    for c in OUT.rglob('complete.json'):
        cm=json.loads(c.read_text());assert cm['identity']==end['identity']
        for n,h in cm['files'].items():assert sha(c.parent/n)==h
        counts+=1
    assert rankings==96 and folds==18 and len(traces)==528 and len(replays)==48 and len(diagnostic)==152 and nassign==len(assignments) and counts==685
    verify_files(protected);save_json(LOG/'replays.json',replays)
    result=dict(status='PASS_CLUSTERING_AUDIT',predictive_fits=528,diagnostic_cluster_fits=152,candidate_rankings=rankings,fold_caches=folds,raw_replays=len(replays),max_raw_difference_gy=max(r['raw_difference_gy'] for r in replays),max_single_difference_gy=max(r['single_difference_gy'] for r in replays),diagnostic_assignments_checked=nassign,protected_files_unchanged=len(protected),cache_manifests=counts,unit_tests=5,locked_test_opened=False,cloud_changed=False,final_model_selected=False)
    save_json(LOG/'verification.json',result);print(json.dumps(result,indent=2));print(json.dumps(sorted(summary,key=lambda r:r['mae_gy']),indent=2))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):main()
