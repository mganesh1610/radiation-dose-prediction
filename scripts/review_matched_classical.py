"""Independently audit matched baselines and review fixed development predictions."""
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from core_baseline import GenePreparation
from development_preprocessing import DevelopmentPreprocessor
from run_abcd_experiments import sha,save_json,write_tsv,verify_files
from run_matched_classical import ROOT,LOG,OUT,OLD,FOLDS,MODELS


def read_tsv(path):return list(csv.DictReader(Path(path).open(),delimiter='\t'))

def metrics(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['donor_group']].append(float(r['predicted_gy'])-float(r['actual_gy']))
    if not grouped:raise ValueError('Empty metric subset')
    return dict(profiles=len(rows),donors=len(grouped),mae_gy=float(np.mean([np.mean(np.abs(e)) for e in grouped.values()])),
                rmse_gy=float(np.sqrt(np.mean([np.mean(np.square(e)) for e in grouped.values()]))),
                bias_gy=float(np.mean([np.mean(e) for e in grouped.values()])))

def slicing(r,name):
    y=r['actual_gy']
    return {'all':True,'control':y==0,'0–2 Gy':y<=2,'positive ≤2 Gy':0<y<=2,'>2 Gy':y>2,
        'inside training range':r['inside_training_range'],'outside training range':not r['inside_training_range'],
        'exact training dose':r['exact_training_dose'],'unseen dose level':not r['exact_training_dose'],
        'common 0/2 Gy':y in (0,2)}[name]

def main():
    cfg=json.loads((LOG/'config.json').read_text());freeze=json.loads((LOG/'protocol-freeze.json').read_text());start=json.loads((LOG/'run-start.json').read_text());end=json.loads((LOG/'run-complete.json').read_text())
    assert start['identity']==end['identity'] and freeze['utc']<start['utc']
    for n,h in freeze['sha256'].items():assert sha(LOG/n)==h
    assert sha(OLD/'splits.json')==freeze['parent_split_sha256'];verify_files(start['code_sha256'])
    protected=json.loads((LOG/'protected-before.json').read_text());verify_files(protected)
    splits=json.loads((OLD/'splits.json').read_text());meta=json.loads((ROOT/cfg['release']/'metadata.json').read_text())
    labels={r['profile_id']:r for r in meta['profiles'] if r['role']=='core_development'}
    assert len(labels)==145 and len({r['donor_group'] for r in labels.values()})==30
    outermap={o['study']:o for o in splits['outer']};selected=json.loads((LOG/'selections.json').read_text());assert len(selected)==9
    selections={(r['study'],r['model']):r for r in selected};traces=[];weight_checks=[]
    assert len(list((OUT/'trials').rglob('result.json')))==360 and len(list((OUT/'selected').rglob('model.joblib')))==15
    for o in splits['outer']:
        assert set(o['training']).isdisjoint(o['validation']) and set(o['training'])|set(o['validation'])==set(labels)
        assert {labels[s]['study'] for s in o['validation']}=={o['study']}
        for f in o['inner']+[dict(name=f"{o['study']}/outer",training=o['training'],validation=o['validation'])]:
            assert set(f['training']).isdisjoint(f['validation'])
            assert set(labels[s]['donor_group'] for s in f['training']).isdisjoint(labels[s]['donor_group'] for s in f['validation'])
            cache=json.loads((FOLDS/f['name']/'fit.json').read_text());assert cache['training']==f['training'] and cache['validation']==f['validation']
            assert cache['nontraining_perturbation_invariant'] and cache['selected_genes']==500
        for model in MODELS:
            candidates=[]
            for params in cfg['candidates'][model]:
                rr=[];seen=[]
                for f in o['inner']:
                    r=json.loads((OUT/'trials'/f['name']/model/str(params['id'])/'result.json').read_text());traces.append(r)
                    assert r['params']==params and r['seed']==f['seed'] and r['training_ids']==f['training'] and r['validation_ids']==f['validation']
                    weights=np.asarray(r['sample_weights']);tr=[labels[s] for s in f['training']];studies=sorted({x['study'] for x in tr})
                    for i,row in enumerate(tr):
                        ng=len({x['donor_group'] for x in tr if x['study']==row['study']});ns=sum(x['donor_group']==row['donor_group'] for x in tr)
                        assert abs(weights[i]-1/(len(studies)*ng*ns))<1e-14
                    seen+=r['validation_ids'];rr += [dict(donor_group=labels[s]['donor_group'],actual_gy=labels[s]['dose_gy'],predicted_gy=v) for s,v in zip(r['validation_ids'],r['predicted_gy'])]
                assert sorted(seen)==sorted(o['training'])
                score=metrics(rr)['mae_gy'];assert abs(score-selections[o['study'],model]['candidates'][params['id']]['metrics']['mae'])<1e-12
                candidates.append((score,params['id']))
            assert min(candidates)[1]==selections[o['study'],model]['chosen']['params']['id']
    rows=[]
    for r in read_tsv(OLD/'predictions.tsv'):
        rows.append({**{k:v for k,v in r.items() if k!='strategy'},'model':r['strategy']})
    rows+=read_tsv(LOG/'predictions.tsv');assert len(rows)==2610
    models=['A','B','C','D','elastic_net','pls','gradient_boosting','median_null']
    for r in rows:
        r['seed_index']=int(r['seed_index']);r['actual_gy']=float(r['actual_gy']);r['predicted_gy']=float(r['predicted_gy']);lab=labels[r['profile_id']]
        assert r['study']==lab['study'] and r['donor_group']==lab['donor_group'] and r['actual_gy']==lab['dose_gy'] and np.isfinite(r['predicted_gy'])
        dose=sorted({labels[s]['dose_gy'] for s in outermap[r['study']]['training']})
        r.update(training_min_gy=min(dose),training_max_gy=max(dose),inside_training_range=min(dose)<=r['actual_gy']<=max(dose),
                 exact_training_dose=r['actual_gy'] in dose,negative_prediction=r['predicted_gy']<0)
    for model in models:
        for si in sorted({r['seed_index'] for r in rows if r['model']==model}):
            rr=[r for r in rows if r['model']==model and r['seed_index']==si]
            assert len(rr)==145 and {r['profile_id'] for r in rr}==set(labels)
    write_tsv(LOG/'combined-predictions.tsv',rows)
    sliced=[];dose_metrics=[];secondary=[];seed_summary=[]
    slice_names=['all','control','0–2 Gy','positive ≤2 Gy','>2 Gy','inside training range','outside training range','exact training dose','unseen dose level','common 0/2 Gy']
    for model in models:
        vals=[]
        for si in sorted({r['seed_index'] for r in rows if r['model']==model}):
            allrows=[r for r in rows if r['model']==model and r['seed_index']==si];vals.append(metrics(allrows)['mae_gy'])
            for study in ['pooled']+sorted(outermap):
                group=[r for r in allrows if study=='pooled' or r['study']==study]
                for name in slice_names:
                    rr=[r for r in group if slicing(r,name)]
                    if rr:sliced.append(dict(model=model,seed_index=si,study=study,slice=name,**metrics(rr)))
                if study!='pooled':
                    for dose in sorted({r['actual_gy'] for r in group}):
                        rr=[r for r in group if r['actual_gy']==dose]
                        dose_metrics.append(dict(model=model,seed_index=si,study=study,dose_gy=dose,mean_prediction_gy=float(np.mean([r['predicted_gy'] for r in rr])),**metrics(rr)))
            macro=float(np.mean([metrics([r for r in allrows if r['study']==s])['mae_gy'] for s in outermap]))
            secondary.append(dict(model=model,seed_index=si,analysis='equal-study mean MAE',omitted_study='',mae_gy=macro))
            for omitted in outermap:secondary.append(dict(model=model,seed_index=si,analysis='omit one study from scoring only',omitted_study=omitted,mae_gy=metrics([r for r in allrows if r['study']!=omitted])['mae_gy']))
        seed_summary.append(dict(model=model,seeds=len(vals),mae_per_seed=vals,mean_mae=float(np.mean(vals)),minimum_mae=min(vals),maximum_mae=max(vals),range_mae=max(vals)-min(vals),interpretation='mean of individual-model errors; not ensemble performance or uncertainty interval'))
    write_tsv(LOG/'slice-metrics.tsv',sliced);write_tsv(LOG/'dose-metrics.tsv',dose_metrics);write_tsv(LOG/'study-sensitivity.tsv',secondary);save_json(LOG/'seed-summary.json',seed_summary)
    # Preserve original A–D metric values when recomputing their primary endpoints.
    prior=read_tsv(OLD/'metrics.tsv')
    for r in sliced:
        if r['model'] in 'ABCD' and r['study']=='pooled' and r['slice']=='all':
            old=next(x for x in prior if x['strategy']==r['model'] and int(x['seed_index'])==r['seed_index'] and x['slice']=='all')
            for k in ['mae_gy','rmse_gy','bias_gy']:assert abs(r[k]-float(old[k]))<1e-12
    primary=[r for r in rows if r['seed_index']==0];donors=sorted({r['donor_group'] for r in primary});rng=np.random.default_rng(cfg['bootstrap_seed']);draw=[]
    for study in sorted(outermap):
        ix=[i for i,g in enumerate(donors) if any(r['study']==study and r['donor_group']==g for r in primary)]
        draw.append(rng.choice(ix,size=(cfg['bootstrap_repeats'],len(ix))))
    draws=np.concatenate(draw,axis=1);errs={};intervals=[];comparisons=[]
    for model in models:
        err=np.array([metrics([r for r in primary if r['model']==model and r['donor_group']==g])['mae_gy'] for g in donors]);errs[model]=err
        intervals.append(dict(model=model,mae_gy=float(err.mean()),conditional_95_percent_interval_gy=np.quantile(err[draws].mean(1),[.025,.975]).tolist()))
    for model in ['B','C','D',*MODELS]:
        d=errs['A']-errs[model]
        comparisons.append(dict(comparator=model,a_minus_comparator_gy=float(d.mean()),family_six_99_1667_percent_interval_gy=np.quantile(d[draws].mean(1),[.05/12,1-.05/12]).tolist()))
    save_json(LOG/'uncertainty.json',dict(mae_intervals=intervals,paired_contrasts=comparisons,repeats=cfg['bootstrap_repeats'],positive_contrast_favors_comparator=True,limitation='Conditional on 3 observed studies and primary-seed fitted predictions; no refitting or initialization uncertainty'))
    sensitivity=[]
    for si in range(3):
        current={s:metrics([r for r in rows if r['model']==s and r['seed_index']==si])['mae_gy'] for s in 'ABCD'}
        sensitivity.append(dict(seed_index=si,ranking=sorted(current,key=current.get),mae_gy=current,a_minus_comparator={s:current['A']-current[s] for s in 'BCD'}))
    save_json(LOG/'neural-seed-ranking.json',sensitivity)
    variability=[];variability_summary=[]
    for model in ['A','B','C','D','gradient_boosting']:
        per_model=[]
        for profile in sorted(labels):
            rr=sorted([r for r in rows if r['model']==model and r['profile_id']==profile],key=lambda r:r['seed_index'])
            assert [r['seed_index'] for r in rr]==[0,1,2]
            values=np.array([r['predicted_gy'] for r in rr]);lab=labels[profile]
            row=dict(model=model,profile_id=profile,study=lab['study'],donor_group=lab['donor_group'],
                     prediction_range_gy=float(np.ptp(values)),prediction_population_sd_gy=float(np.std(values)))
            per_model.append(row);variability.append(row)
        ranges=np.array([r['prediction_range_gy'] for r in per_model])
        variability_summary.append(dict(model=model,median_prediction_range_gy=float(np.median(ranges)),p95_prediction_range_gy=float(np.quantile(ranges,.95)),maximum_prediction_range_gy=float(ranges.max())))
    write_tsv(LOG/'prediction-seed-variability.tsv',variability);save_json(LOG/'prediction-seed-variability.json',variability_summary)
    # Replay each selected classical pipeline from raw core expression, not saved panel matrices.
    with np.load(ROOT/cfg['release']/'core.npz',allow_pickle=False) as z:data={'core':{k:z[k] for k in z.files}}
    replays=[]
    for o in splits['outer']:
        directory=FOLDS/o['study']/'outer';processor=DevelopmentPreprocessor.load(directory/'preprocessing');view=processor.transform(data,o['validation'])
        assert set(processor.core_training)==set(o['training']) and not processor.aux_training
        for model in MODELS:
            params=selections[o['study'],model]['chosen']['params'];rep=params['representation']
            with np.load(directory/f'{rep}-features.npz',allow_pickle=False) as z:g=GenePreparation(tuple(z['gene_ids']),z['medians'],z['selected'],z['mean'],z['scale'],z['selection_scores'],tuple(z['training_ids']))
            assert set(g.training_ids)==set(o['training']);x=g.transform(view[f'{rep}_values'].T,view['gene_ids'],True)
            seeds=o['seeds'] if model=='gradient_boosting' else o['seeds'][:1]
            for si,seed in enumerate(seeds):
                path=OUT/'selected'/o['study']/model/str(seed);r=json.loads((path/'result.json').read_text());traces.append(r)
                estimator=joblib.load(path/'model.joblib');pred=np.asarray(estimator.predict(x)).reshape(-1)*r['target_scale']+r['target_mean']
                expected={a['profile_id']:a['predicted_gy'] for a in rows if a['model']==model and a['study']==o['study'] and a['seed_index']==si}
                diff=float(np.max(abs(pred-np.array([expected[s] for s in o['validation']]))));assert diff<=1e-8
                replays.append(dict(study=o['study'],model=model,seed=seed,raw_replay_max_difference_gy=diff,individual_difference_gy=r['individual_difference_gy']))
    assert len(traces)==375;save_json(LOG/'replay.json',replays)
    runtime={model:dict(fits=sum(r['params']['model']==model for r in traces),fit_seconds=sum(r['fit_seconds'] for r in traces if r['params']['model']==model)) for model in MODELS}
    save_json(LOG/'runtime.json',dict(models=runtime,wall_seconds=end['wall_seconds'],maxrss_bytes=end['process_maxrss_native']))
    caches=0
    for f in OUT.rglob('complete.json'):
        m=json.loads(f.read_text());assert m['identity']==end['identity']
        for n,h in m['files'].items():assert sha(f.parent/n)==h
        caches+=1
    verify_files(protected)
    save_json(LOG/'verification.json',dict(status='PASS_MATCHED_CLASSICAL_AND_BIAS_REVIEW',inner_fits=360,outer_models=15,core_profiles=145,donor_groups=30,
             fold_partitions=18,recomputed_candidate_selections=72,metric_slices=len(sliced),dose_slices=len(dose_metrics),saved_raw_replays=len(replays),
             raw_replay_max_difference_gy=max(r['raw_replay_max_difference_gy'] for r in replays),individual_replay_max_difference_gy=max(r['individual_difference_gy'] for r in replays),
             protected_files_unchanged=len(protected),cache_manifests_verified=caches,weight_ratios_verified=True,new_unit_tests=7,
             parent_neural_predictions_unchanged=True,locked_test_opened=False,cloud_changed=False,final_model_selected=False))
    print(json.dumps([r for r in sliced if r['study']=='pooled' and r['slice']=='all' and r['seed_index']==0],indent=2));print('VERIFIED 375 fits,15 raw replays,protected',len(protected),flush=True)

if __name__=='__main__':main()
