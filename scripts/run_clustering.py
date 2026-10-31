"""Frozen local clustering diagnostics and nested prediction comparison."""
import json,sys,time,resource,platform,warnings,itertools
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import joblib,sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score,adjusted_rand_score
from threadpoolctl import threadpool_limits
from abcd_models import balanced_weights
from core_baseline import donor_metrics
from development_preprocessing import DevelopmentPreprocessor,sample_ranks
from clustering_models import ClusterRegressor,VariableGeneView,make_cluster
from run_matched_classical import load_fold,FOLDS
from run_abcd_experiments import sha,save_json,write_tsv,verify_files,finish,completed

ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'context/clustering/v1';OUT=ROOT/'data/models/clustering-v1'
MODELS=('pca_ridge','kmeans_ridge','gmm_ridge','cluster_median')
def utc():return datetime.now(timezone.utc).isoformat()
def metric(ids,pred,labels):return donor_metrics([labels[s]['dose_gy'] for s in ids],pred,[labels[s]['donor_group'] for s in ids])[0]

def predictive_fit(fold,p,seed,labels,path,identity,selected=False):
    path.mkdir(parents=True,exist_ok=True)
    if not completed(path,identity):
        x,v,f=load_fold(fold,p['representation']);tr=[labels[s] for s in f['training']]
        w=balanced_weights([r['study'] for r in tr],[r['donor_group'] for r in tr])
        y=(np.array([r['dose_gy'] for r in tr])-f['target_mean'])/f['target_scale']
        start=time.perf_counter();m=ClusterRegressor(**{k:v for k,v in p.items() if k not in ('id','representation')},seed=seed).fit(x,y,w)
        elapsed=time.perf_counter()-start;pred=m.predict(v)*f['target_scale']+f['target_mean']
        if not np.isfinite(pred).all():raise ValueError('Nonfinite output')
        reload_diff=single_diff=None
        if selected:
            joblib.dump(m,path/'model.joblib',compress=3);loaded=joblib.load(path/'model.joblib')
            reload_diff=float(np.max(abs(loaded.predict(v)*f['target_scale']+f['target_mean']-pred)))
            single_diff=float(np.max(abs(np.array([loaded.predict(a[None])[0] for a in v])*f['target_scale']+f['target_mean']-pred)))
            if reload_diff>1e-8 or single_diff>1e-7:raise ValueError('Model reload mismatch')
        save_json(path/'result.json',dict(params=p,seed=seed,fit_seconds=elapsed,fold=fold['name'],training_ids=f['training'],validation_ids=f['validation'],weights=w.tolist(),target_mean=f['target_mean'],target_scale=f['target_scale'],predicted_gy=pred.tolist(),metrics=metric(f['validation'],pred,labels),reload_difference=reload_diff,single_difference=single_diff))
        finish(path,identity)
    return json.loads((path/'result.json').read_text())

def predictive(cfg,splits,metadata,identity):
    labels={r['profile_id']:r for r in metadata['profiles'] if r['role']=='core_development'}
    assert set(labels)==set(splits['core_profiles']) and len(labels)==145
    selections=[];rows=[];ablations=[];count=0
    for o in splits['outer']:
        for model in MODELS:
            candidates=[]
            for p in cfg['candidates'][model]:
                ids=[];pred=[]
                for f in o['inner']:
                    r=predictive_fit(f,p,f['seed'],labels,OUT/'trials'/f['name']/model/str(p['id']),identity)
                    count+=1;ids+=r['validation_ids'];pred+=r['predicted_gy']
                assert sorted(ids)==sorted(o['training'])
                candidates.append(dict(params=p,metrics=metric(ids,pred,labels)))
            best=min(candidates,key=lambda a:(a['metrics']['mae'],a['params']['id']))
            selections.append(dict(study=o['study'],model=model,candidates=candidates,chosen=best));save_json(LOG/'selections.json',selections)
            f=dict(name=o['study']+'/outer',training=o['training'],validation=o['validation'])
            for si,seed in enumerate(o['seeds'][:1] if model=='pca_ridge' else o['seeds']):
                r=predictive_fit(f,best['params'],seed,labels,OUT/'selected'/o['study']/model/str(seed),identity,True);count+=1
                for s,v in zip(r['validation_ids'],r['predicted_gy']):
                    a=labels[s];rows.append(dict(model=model,seed_index=si,seed=seed,profile_id=s,study=a['study'],donor_group=a['donor_group'],actual_gy=a['dose_gy'],predicted_gy=v))
                if model in ('kmeans_ridge','gmm_ridge'):
                    p=dict(id=best['params']['id'],model='pca_ridge',representation=best['params']['representation'],components=10,alpha=best['params']['alpha'])
                    r=predictive_fit(f,p,seed,labels,OUT/'ablations'/o['study']/model/str(seed),identity,True);count+=1
                    for s,v in zip(r['validation_ids'],r['predicted_gy']):
                        a=labels[s];ablations.append(dict(parent_model=model,seed_index=si,seed=seed,profile_id=s,study=a['study'],donor_group=a['donor_group'],actual_gy=a['dose_gy'],predicted_gy=v))
            write_tsv(LOG/'predictions.tsv',rows);write_tsv(LOG/'ablations.tsv',ablations) if ablations else None
            save_json(LOG/'progress.json',dict(stage='prediction',fits=count,study=o['study'],model=model));print(o['study'],model,'innerMAE',round(best['metrics']['mae'],4),'fits',count,flush=True)
    assert count==528 and len(rows)==1450 and len(ablations)==870
    return count

def diagnostics(cfg,metadata,identity):
    dest=OUT/'diagnostics';dest.mkdir(exist_ok=True)
    core_ids=sorted(r['profile_id'] for r in metadata['profiles'] if r['role']=='core_development')
    aux_ids=sorted(r['profile_id'] for r in metadata['profiles'] if r['role']!='core_development')
    labels={r['profile_id']:r for r in metadata['profiles']}
    matrix_dir=dest/'matrix';matrix_dir.mkdir(exist_ok=True)
    if not completed(matrix_dir,identity):
        with np.load(ROOT/cfg['release']/'core.npz',allow_pickle=False) as z:data={'core':{k:z[k] for k in z.files}}
        for study in ('GSE20173','GSE85570','GSE97000'):
            data[study]={'profile_ids':np.array([],dtype=str)}
        pre=DevelopmentPreprocessor.fit(data,metadata,core_ids);v=pre.transform(data,core_ids);pre.save(matrix_dir/'core-reference')
        old=ROOT/'data/models/abcd-v1/auxiliary'
        for n,h in json.loads((old/'complete.json').read_text())['files'].items():assert sha(old/n)==h
        with np.load(old/'gene-views.npz',allow_pickle=False) as a:
            assert list(a['profile_ids'])==aux_ids
            lookup={str(g):i for i,g in enumerate(a['gene_ids'])}
            ax=a['log2_values'][[lookup[str(g)] for g in v['gene_ids']]]
        x=np.column_stack([v['log2_values'],ax]);ids=core_ids+aux_ids
        assert x.shape[1]==567 and set(ids)==set(labels)
        np.savez_compressed(matrix_dir/'views.npz',gene_ids=v['gene_ids'],profile_ids=ids,log2_values=x,rank_values=sample_ranks(x))
        save_json(matrix_dir/'scope.json',dict(diagnostic_only=True,dose_labels_used_in_fitting=False,core_ids=core_ids,auxiliary_ids=aux_ids,raw_release_core_sha256=sha(ROOT/cfg['release']/'core.npz'),auxiliary_cache_sha256=sha(old/'gene-views.npz')))
        finish(matrix_dir,identity)
    with np.load(matrix_dir/'views.npz',allow_pickle=False) as a:raw={k:a[k] for k in a.files}
    rows=[];assignments=[];count=0;seeds=cfg['diagnostic_seeds']
    for scope in ('core','all'):
        ids=core_ids if scope=='core' else list(raw['profile_ids']);pos=[list(raw['profile_ids']).index(s) for s in ids]
        records=[labels[s] for s in ids];studies=np.array([a['study'] for a in records]);donors=np.array([a['donor_group'] for a in records]);doses=np.array([a['dose_gy'] for a in records]);w=balanced_weights(studies,donors)
        for rep in ('log2','rank'):
            viewdir=dest/f'{scope}-{rep}';viewdir.mkdir(exist_ok=True)
            if not completed(viewdir/'view',identity):
                (viewdir/'view').mkdir(exist_ok=True)
                model=VariableGeneView().fit(raw[f'{rep}_values'][:,pos].T,raw['gene_ids'],w)
                z=model.transform(raw[f'{rep}_values'][:,pos].T,raw['gene_ids'])
                joblib.dump(model,viewdir/'view/model.joblib',compress=3)
                np.savez_compressed(viewdir/'view/panel.npz',ids=ids,z=z,panel=model.panel(raw[f'{rep}_values'][:,pos].T,raw['gene_ids']),selected_genes=np.asarray(raw['gene_ids'])[model.selected_],weights=w)
                finish(viewdir/'view',identity)
            with np.load(viewdir/'view/panel.npz',allow_pickle=False) as a:z=a['z']
            for controls in (False,True):
                ix=np.flatnonzero(doses==0) if controls else np.arange(len(ids));zz=z[ix];ww=balanced_weights(studies[ix],donors[ix])
                for method in ('kmeans','ward','gmm'):
                    for k in ([len(set(studies))] if controls else cfg['diagnostic_k']):
                        for si,seed in enumerate(seeds[:1] if controls or method=='ward' else seeds):
                            path=viewdir/f'{"controls" if controls else "all"}-{method}-k{k}-s{si}';path.mkdir(exist_ok=True)
                            if not completed(path,identity):
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter('always')
                                    m=AgglomerativeClustering(n_clusters=k,linkage='ward') if method=='ward' else make_cluster(method,k,seed)
                                    if method=='kmeans':m.fit(zz,sample_weight=ww*len(ww));lab=m.predict(zz)
                                    elif method=='gmm':m.fit(zz);assert m.converged_;lab=m.predict(zz)
                                    else:lab=m.fit_predict(zz)
                                if caught:raise ValueError('Diagnostic fit warning: '+'; '.join(str(a.message) for a in caught))
                                assert len(set(lab))>=2
                                joblib.dump(m,path/'cluster.joblib',compress=3)
                                save_json(path/'result.json',dict(scope=scope,representation=rep,controls_only=controls,method=method,k=k,seed_index=si,seed=seed,ids=[ids[i] for i in ix],labels=lab.tolist(),silhouette=float(silhouette_score(zz,lab)),study_ari=float(adjusted_rand_score(studies[ix],lab)),exposure_ari=None if controls else float(adjusted_rand_score(doses[ix]>0,lab)),dose_group_ari=None if controls else float(adjusted_rand_score(np.where(doses[ix]==0,0,np.where(doses[ix]<=2,1,2)),lab)),donor_ari=float(adjusted_rand_score(donors[ix],lab)),cluster_sizes=np.bincount(lab,minlength=k).tolist()))
                                finish(path,identity)
                            r=json.loads((path/'result.json').read_text());count+=1
                            rows.append({k:v for k,v in r.items() if k not in ('ids','labels','cluster_sizes')}|{'cluster_sizes':json.dumps(r['cluster_sizes'])})
                            for s,c in zip(r['ids'],r['labels']):assignments.append(dict(scope=scope,representation=rep,controls_only=controls,method=method,k=k,seed_index=si,profile_id=s,cluster=c,study=labels[s]['study'],donor_group=labels[s]['donor_group'],dose_gy=labels[s]['dose_gy']))
            print('Diagnostic',scope,rep,'fits',count,flush=True)
    assert count==152
    write_tsv(LOG/'diagnostic-metrics.tsv',rows);write_tsv(LOG/'cluster-assignments.tsv',assignments)
    return count

def main():
    start=time.perf_counter();cfg=json.loads((LOG/'config.json').read_text());freeze=json.loads((LOG/'protocol-freeze.json').read_text());splits=json.loads((LOG/'splits.json').read_text())
    for n,h in freeze['sha256'].items():assert sha(LOG/n)==h
    protected=json.loads((LOG/'protected-before.json').read_text());verify_files(protected)
    for c in FOLDS.rglob('complete.json'):
        for n,h in json.loads(c.read_text())['files'].items():assert sha(c.parent/n)==h
    source={n:sha(ROOT/n) for n in ['scripts/clustering_models.py','scripts/run_clustering.py','scripts/run_matched_classical.py','scripts/development_preprocessing.py','scripts/core_baseline.py','scripts/abcd_models.py','scripts/run_abcd_experiments.py']}
    identity=__import__('hashlib').sha256(json.dumps(dict(protocol=freeze['sha256'],source=source),sort_keys=True).encode()).hexdigest()
    if (LOG/'run-start.json').exists():assert json.loads((LOG/'run-start.json').read_text())['identity']==identity
    else:save_json(LOG/'run-start.json',dict(utc=utc(),identity=identity,code_sha256=source,python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,platform=platform.platform(),threads=4))
    metadata=json.loads((ROOT/cfg['release']/'metadata.json').read_text())
    df=diagnostics(cfg,metadata,identity);pf=predictive(cfg,splits,metadata,identity)
    verify_files(protected)
    save_json(LOG/'run-complete.json',dict(utc=utc(),identity=identity,predictive_fits=pf,diagnostic_cluster_fits=df,wall_seconds=time.perf_counter()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,locked_test_opened=False,cloud_changed=False))
    save_json(LOG/'progress.json',dict(status='complete',predictive_fits=pf,diagnostic_fits=df));print('COMPLETE',pf,df,flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(limits=4):main()
    except Exception as e:
        save_json(LOG/'failure.json',dict(utc=utc(),type=type(e).__name__,error=str(e)));raise
