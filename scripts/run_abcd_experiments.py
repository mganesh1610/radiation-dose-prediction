"""Run frozen nested A–D development experiments locally, with auditable fold caches."""
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from assay_qc import aggregate_median
from core_baseline import GenePreparation, donor_metrics
from development_preprocessing import (AUX, CORE, DevelopmentPreprocessor, check_training_ids,
                                       load_release, sample_ranks)
from frozen_affymetrix import FrozenAffymetrix
from masked_quantile import MaskedFrozenQuantile
from abcd_models import balanced_weights, tensor, fit_network, predict, save_model, load_model

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/model-abcd/v1';OUT=ROOT/'data/models/abcd-v1'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def write_tsv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)

def utc():return datetime.now(timezone.utc).isoformat()

def verify_files(records):
    for name,expected in records.items():
        if sha(ROOT/name)!=expected:raise ValueError(f'Protected hash changed: {name}')

def manifest(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='complete.json' and not p.name.endswith('.partial')}

def completed(directory,identity):
    p=directory/'complete.json'
    if not p.exists():return False
    m=json.loads(p.read_text())
    if m['identity']!=identity:raise ValueError('Cache identity changed')
    for name,h in m['files'].items():
        if sha(directory/name)!=h:raise ValueError('Cache content changed')
    return True

def finish(directory,identity):save_json(directory/'complete.json',dict(identity=identity,files=manifest(directory)))

def auxiliary_cache(data,metadata,ids,identity):
    dest=OUT/'auxiliary';dest.mkdir(exist_ok=True)
    if not completed(dest,identity):
        check_training_ids(metadata['profiles'],ids,AUX)
        genes=np.array(metadata['annotation_gene_ids']);matrix=np.full((len(genes),len(ids)),np.nan)
        target={s:i for i,s in enumerate(ids)};records={}
        for study in sorted(AUX):
            print('Fitting auxiliary training reference',study,flush=True);start=time.perf_counter()
            a=data[study];training=sorted(set(a['profile_ids'])&set(ids))
            if study=='GSE85570':
                model=FrozenAffymetrix.fit(a['values'],a['probesets'],a['physical_indices'],a['sample_ids'],a['profile_ids'],training)
                y=model.transform(a['values'],a['probesets'],a['physical_indices']);features=model.probeset_ids
            else:
                model=MaskedFrozenQuantile.fit(a['values'],np.isfinite(a['values']),a['probe_ids'],a['sample_ids'],list(a['sample_ids']))
                y=np.log2(model.transform(a['values'],np.isfinite(a['values']),a['probe_ids']));features=a['probe_ids']
            model.save(dest/f'{study}.npz')
            keys=np.array([metadata['mapping'][study].get(str(p),'') for p in features]);keep=keys!=''
            mapped,y,_=aggregate_median(y[keep],keys[keep]);names,y,_=aggregate_median(y.T,a['profile_ids']);y=y.T
            rows={str(g):i for i,g in enumerate(mapped)};ix=np.array([rows[str(g)] for g in genes]);cols=[target[str(s)] for s in names]
            matrix[:,cols]=y[ix]
            records[study]=dict(training_profiles=training,arrays=len(a['sample_ids']),seconds=time.perf_counter()-start)
        if np.isinf(matrix).any() or not np.isfinite(matrix).any(axis=0).all():raise ValueError('Invalid auxiliary views')
        np.savez_compressed(dest/'gene-views.npz',gene_ids=genes,profile_ids=np.array(ids),log2_values=matrix)
        save_json(dest/'fit-membership.json',records);finish(dest,identity)
    with np.load(dest/'gene-views.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}

def fit_core(data,metadata,train,labels):
    # Only training columns reach the fitted core adapter and feature selector.
    a=data['core'];take=np.isin(a['sample_ids'],train)
    sliced=dict(a)
    for key in ['values','detected']:sliced[key]=a[key][:,take]
    for key in ['sample_ids','study_ids','split_group_ids']:sliced[key]=a[key][take]
    fit_data=dict(data);fit_data['core']=sliced
    prep=DevelopmentPreprocessor.fit(fit_data,metadata,train)
    view=prep.transform({'core':sliced},train)
    y=np.array([labels[p]['dose_gy'] for p in train]);studies=np.array([labels[p]['study'] for p in train])
    gene_preps={rep:GenePreparation.fit(view[f'{rep}_values'].T,view['gene_ids'],y,studies,train,500) for rep in ['log2','rank']}
    return prep,gene_preps,float(y.mean()),float(y.std())

def signature(prep,genes,mean,scale):
    return dict(reference=hashlib.sha256(prep.core.reference.target.tobytes()).hexdigest(),
                keep=hashlib.sha256(prep.core.keep.tobytes()).hexdigest(),genes=list(prep.gene_ids),
                preparation={r:p.fingerprint() for r,p in genes.items()},mean=mean,scale=scale)

def prepare_fold(data,metadata,labels,aux,fold,identity):
    dest=OUT/'folds'/fold['name'];dest.mkdir(parents=True,exist_ok=True)
    if not completed(dest,identity):
        start=time.perf_counter();train=fold['training'];val=fold['validation']
        if set(labels[s]['donor_group'] for s in train)&set(labels[s]['donor_group'] for s in val):raise ValueError('Donor leakage')
        p,g,mean,scale=fit_core(data,metadata,train,labels)
        if scale<=0:raise ValueError('Constant training dose')
        # Independent perturbation of ALL nontraining core values, flags and labels.
        changed=dict(data);a=dict(data['core']);nontrain=~np.isin(a['sample_ids'],train)
        a['values']=a['values'].copy();a['values'][:,nontrain]=987654.
        a['detected']=a['detected'].copy();a['detected'][:,nontrain]=False;changed['core']=a
        changed_labels={k:(dict(v,dose_gy=8765.) if k not in train else v) for k,v in labels.items()}
        p2,g2,m2,s2=fit_core(changed,metadata,train,changed_labels)
        if signature(p,g,mean,scale)!=signature(p2,g2,m2,s2):raise ValueError('Nontraining contamination')
        del changed,a,p2,g2
        p.save(dest/'preprocessing')
        view=p.transform(data,train+val)
        aux_rows={str(g):i for i,g in enumerate(aux['gene_ids'])}
        aux_log=aux['log2_values'][[aux_rows[x] for x in p.gene_ids]]
        aux_views={'log2':aux_log,'rank':sample_ranks(aux_log)}
        checks={}
        for rep in ['log2','rank']:
            gp=g[rep];gp.save(dest/f'{rep}-features.npz')
            x=gp.transform(view[f'{rep}_values'].T,view['gene_ids'],True)
            ax=gp.transform(aux_views[rep].T,view['gene_ids'],True)
            if x.shape[1]!=500 or not np.isfinite(x).all() or not np.isfinite(ax).all():raise ValueError('Invalid selected panel')
            np.savez_compressed(dest/f'{rep}-panel.npz',train=x[:len(train)],validation=x[len(train):],aux=ax,
                                training_ids=train,validation_ids=val,auxiliary_ids=aux['profile_ids'])
            checks[rep]=dict(preparation_fingerprint=gp.fingerprint(),genes=[str(gp.gene_ids[i]) for i in gp.selected],
                             missing_core_cells=int(np.isnan(view[f'{rep}_values']).sum()),missing_auxiliary_cells=int(np.isnan(aux_views[rep]).sum()))
        save_json(dest/'fit.json',dict(training=train,validation=val,target_mean=mean,target_scale=scale,
                 raw_gene_count=len(p.gene_ids),selected_genes=500,nontraining_perturbation_invariant=True,
                 representations=checks,seconds=time.perf_counter()-start))
        finish(dest,identity)
        print('Prepared',fold['name'],len(train),len(val),'genes',len(p.gene_ids),flush=True)
    fit=json.loads((dest/'fit.json').read_text());return dest,fit

def make_batch(panel,fit,labels,strategy):
    tr=[labels[s] for s in fit['training']]
    core_y=np.array([r['dose_gy'] for r in tr]);e=np.array([r['exposure'] for r in tr]);s=[r['study'] for r in tr];d=[r['donor_group'] for r in tr]
    batch=dict(core_x=panel['train'],core_y=(core_y-fit['target_mean'])/fit['target_scale'],core_exposure=e,
               core_weights=balanced_weights(s,d),core_binary_weights=balanced_weights(s,d,e))
    if strategy!='A':
        ar=[labels[str(s)] for s in panel['auxiliary_ids']];ae=np.array([r['exposure'] for r in ar]);ss=[r['study'] for r in ar];dd=[r['donor_group'] for r in ar]
        batch.update(aux_x=panel['aux'],aux_exposure=ae,aux_binary_weights=balanced_weights(ss,dd,ae))
        if strategy=='D':
            ay=np.array([r['dose_gy'] for r in ar])
            if not set(ay).issubset({0.,2.}) or any(r['dose_status'] in ('unknown','missing') for r in ar):raise ValueError('Unverified auxiliary doses')
            batch.update(aux_y=(ay-fit['target_mean'])/fit['target_scale'],aux_weights=balanced_weights(ss,dd))
    return {k:tensor(v) for k,v in batch.items()}

def one_fit(fold_dir,fit,labels,params,seed,path,identity,save_weights=False):
    path.mkdir(parents=True,exist_ok=True)
    if not completed(path,identity):
        with np.load(fold_dir/f"{params['representation']}-panel.npz",allow_pickle=False) as z:panel={k:z[k] for k in z.files}
        batch=make_batch(panel,fit,labels,params['strategy'])
        model,trace=fit_network(batch,params,seed)
        values=predict(model,panel['validation'],fit['target_mean'],fit['target_scale'])
        actual=np.array([labels[s]['dose_gy'] for s in fit['validation']]);groups=np.array([labels[s]['donor_group'] for s in fit['validation']])
        metric,_=donor_metrics(actual,values,groups)
        replay=None;single=None
        if save_weights:
            save_model(path/'model.pt',model,params,seed,fit['target_mean'],fit['target_scale'])
            loaded,saved=load_model(path/'model.pt')
            replay=float(np.max(abs(values-predict(loaded,panel['validation'],saved['target_mean'],saved['target_scale']))))
            single=float(np.max(abs(values-np.array([predict(loaded,row[None],saved['target_mean'],saved['target_scale'])[0] for row in panel['validation']]))))
            if replay>1e-6 or single>1e-4:raise ValueError('Saved-model replay failed')
        save_json(path/'result.json',dict(params=params,seed=seed,fold=str(fold_dir.relative_to(OUT)),fit_seconds=trace['seconds'],trace=trace,
                   validation_ids=fit['validation'],prediction_gy=values.tolist(),metrics=metric,
                   reload_max_difference_gy=replay,individual_max_difference_gy=single))
        finish(path,identity)
    return json.loads((path/'result.json').read_text())

def main():
    started=time.perf_counter();config=json.loads((LOG/'config.json').read_text());splits=json.loads((LOG/'splits.json').read_text())
    freeze=json.loads((LOG/'protocol-freeze.json').read_text())
    for name,h in freeze['sha256'].items():
        if sha(LOG/name)!=h:raise ValueError('Frozen protocol changed')
    sources=['scripts/abcd_models.py','scripts/run_abcd_experiments.py','scripts/development_preprocessing.py',
             'scripts/frozen_affymetrix.py','scripts/core_transfer.py','scripts/masked_quantile.py','scripts/assay_qc.py','scripts/core_baseline.py']
    code={s:sha(ROOT/s) for s in sources};identity=hashlib.sha256(json.dumps(dict(protocol=freeze['sha256'],code=code),sort_keys=True).encode()).hexdigest()
    if (LOG/'run-start.json').exists():
        if json.loads((LOG/'run-start.json').read_text())['identity']!=identity:raise ValueError('Run implementation changed; create a separate version')
    else:
        save_json(LOG/'run-start.json',dict(utc=utc(),identity=identity,source_sha256=code,python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),device='cpu',threads=4,protocol_sha256=freeze['sha256']))
    verify_files(json.loads((LOG/'protected-before.json').read_text()))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    release=ROOT/config['release']
    if sha(release/'release.json')!=config['release_sha256']:raise ValueError('Release changed')
    data,metadata=load_release(release);labels={r['profile_id']:r for r in metadata['profiles']}
    if list(data['core']['sample_ids'])!=splits['core_profiles']:raise ValueError('Core identity order mismatch')
    aux=auxiliary_cache(data,metadata,splits['auxiliary_training'],identity)
    rows=[];selections=[];count=0
    for outer in splits['outer']:
        st=outer['study'];print('OUTER',st,flush=True)
        prepared={}
        for fold in outer['inner']:prepared[fold['name']]=prepare_fold(data,metadata,labels,aux,fold,identity)
        for strategy in 'ABCD':
            scores=[]
            for params in config['candidates'][strategy]:
                pred=[];ids=[]
                for fold in outer['inner']:
                    d,f=prepared[fold['name']]
                    r=one_fit(d,f,labels,params,fold['seed'],OUT/'trials'/fold['name']/strategy/str(params['id']),identity)
                    pred.extend(r['prediction_gy']);ids.extend(r['validation_ids']);count+=1
                score,_=donor_metrics([labels[s]['dose_gy'] for s in ids],pred,[labels[s]['donor_group'] for s in ids])
                scores.append(dict(params=params,inner_metrics=score));print(st,strategy,params['id'],'inner MAE',round(score['mae'],4),flush=True)
                save_json(LOG/'progress.json',dict(utc=utc(),status='running',completed_strategy_fits=count,outer=st,strategy=strategy,candidate=params['id']))
            best=min(scores,key=lambda r:(r['inner_metrics']['mae'],r['params']['id']))
            selection=dict(outer=st,strategy=strategy,chosen=best,candidates=scores);selections.append(selection)
            save_json(LOG/'selections.json',selections)
            fold=dict(name=f'{st}/outer',training=outer['training'],validation=outer['validation'])
            d,f=prepare_fold(data,metadata,labels,aux,fold,identity)
            for si,seed in enumerate(outer['seeds']):
                r=one_fit(d,f,labels,best['params'],seed,OUT/'selected'/st/strategy/str(seed),identity,True);count+=1
                for name,pred in zip(r['validation_ids'],r['prediction_gy']):
                    label=labels[name];rows.append(dict(strategy=strategy,seed_index=si,seed=seed,profile_id=name,study=st,donor_group=label['donor_group'],actual_gy=label['dose_gy'],predicted_gy=pred,candidate_id=best['params']['id'],representation=best['params']['representation']))
            print('Selected',st,strategy,best['params'],flush=True)
        null=float(np.median([labels[s]['dose_gy'] for s in outer['training']]))
        for name in outer['validation']:
            r=labels[name];rows.append(dict(strategy='median_null',seed_index=0,seed=0,profile_id=name,study=st,donor_group=r['donor_group'],actual_gy=r['dose_gy'],predicted_gy=null,candidate_id=-1,representation='none'))
        write_tsv(LOG/'predictions.tsv',rows)
    if count!=516:raise ValueError(f'Incomplete budget {count}')
    verify_files(json.loads((LOG/'protected-before.json').read_text()))
    save_json(LOG/'run-complete.json',dict(utc=utc(),identity=identity,completed_strategy_fits=count,inner_fits=480,outer_fits=36,
               wall_seconds_this_invocation=time.perf_counter()-started,prediction_rows=len(rows),protected_files_unchanged=len(json.loads((LOG/'protected-before.json').read_text())),locked_test_opened=False,cloud_changed=False))
    save_json(LOG/'progress.json',dict(utc=utc(),status='complete',completed_strategy_fits=count))
    print('COMPLETE',count,'fits;',len(rows),'prediction rows',flush=True)

if __name__=='__main__':main()
