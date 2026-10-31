"""Independent membership, selection, replay and metrics audit for A–D v1."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from abcd_models import load_model,predict
from core_baseline import GenePreparation
from development_preprocessing import DevelopmentPreprocessor,load_release
from run_abcd_experiments import ROOT,LOG,OUT,sha,save_json,write_tsv,verify_files


def metrics(rows):
    grouped={}
    for r in rows:grouped.setdefault(r['donor_group'],[]).append(float(r['predicted_gy'])-float(r['actual_gy']))
    return dict(mae_gy=float(np.mean([np.mean(np.abs(e)) for e in grouped.values()])),
                rmse_gy=float(np.sqrt(np.mean([np.mean(np.square(e)) for e in grouped.values()]))),
                bias_gy=float(np.mean([np.mean(e) for e in grouped.values()])),donors=len(grouped),profiles=len(rows))

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    end=json.loads((LOG/'run-complete.json').read_text());start=json.loads((LOG/'run-start.json').read_text());freeze=json.loads((LOG/'protocol-freeze.json').read_text())
    assert end['identity']==start['identity'];assert freeze['frozen_utc']<start['utc'];verify_files(start['source_sha256'])
    for name,h in freeze['sha256'].items():assert sha(LOG/name)==h
    config=json.loads((LOG/'config.json').read_text());splits=json.loads((LOG/'splits.json').read_text());selections=json.loads((LOG/'selections.json').read_text())
    rows=list(csv.DictReader((LOG/'predictions.tsv').open(),delimiter='\t'));m=json.loads((ROOT/config['release']/'metadata.json').read_text());labels={r['profile_id']:r for r in m['profiles']}
    assert len(rows)==1885 and len(selections)==12
    assert len(list((OUT/'trials').rglob('result.json')))==480
    assert len(list((OUT/'selected').rglob('model.pt')))==36
    selected={(r['outer'],r['strategy']):r for r in selections};traces=[]
    for r in rows:
        assert r['profile_id'] in splits['core_profiles'];lab=labels[r['profile_id']]
        assert r['study']==lab['study'] and r['donor_group']==lab['donor_group'] and float(r['actual_gy'])==lab['dose_gy']
        assert np.isfinite(float(r['predicted_gy']))
    for st in 'ABCD':
        for si in range(3):
            rr=[r for r in rows if r['strategy']==st and int(r['seed_index'])==si]
            assert sorted(r['profile_id'] for r in rr)==sorted(splits['core_profiles'])
    for outer in splits['outer']:
        train=set(outer['training']);val=set(outer['validation'])
        assert train.isdisjoint(val) and train|val==set(splits['core_profiles'])
        assert {labels[x]['study'] for x in val}=={outer['study']}
        inn=[]
        for f in outer['inner']:
            assert set(f['training']).isdisjoint(f['validation'])
            assert set(f['training'])|set(f['validation'])==train
            assert set(labels[x]['donor_group'] for x in f['training']).isdisjoint(labels[x]['donor_group'] for x in f['validation'])
            inn+=f['validation']
        assert sorted(inn)==sorted(train)
        for strategy in 'ABCD':
            results=[]
            for params in config['candidates'][strategy]:
                inner_rows=[]
                for f in outer['inner']:
                    result=json.loads((OUT/'trials'/f['name']/strategy/str(params['id'])/'result.json').read_text())
                    assert result['params']==params and result['seed']==f['seed'] and result['validation_ids']==f['validation']
                    traces.append(result)
                    inner_rows.extend(dict(donor_group=labels[s]['donor_group'],actual_gy=labels[s]['dose_gy'],predicted_gy=p) for s,p in zip(result['validation_ids'],result['prediction_gy']))
                mae=metrics(inner_rows)['mae_gy'];record=selected[outer['study'],strategy]['candidates'][params['id']]
                assert abs(mae-record['inner_metrics']['mae'])<1e-12
                results.append((mae,params['id']))
            assert min(results)[1]==selected[outer['study'],strategy]['chosen']['params']['id']
    summary=[]
    for strategy in ['A','B','C','D','median_null']:
        for si in ([0] if strategy=='median_null' else range(3)):
            group=[r for r in rows if r['strategy']==strategy and int(r['seed_index'])==si]
            slices=[('all',group),('0–2 Gy',[r for r in group if float(r['actual_gy'])<=2]),('>2 Gy',[r for r in group if float(r['actual_gy'])>2])]
            slices +=[(st,[r for r in group if r['study']==st]) for st in sorted({r['study'] for r in group})]
            slices +=[(f'dose={dose:g} Gy',[r for r in group if float(r['actual_gy'])==dose]) for dose in sorted({float(r['actual_gy']) for r in group})]
            for name,part in slices:summary.append(dict(strategy=strategy,seed_index=si,slice=name,**metrics(part)))
    write_tsv(LOG/'metrics.tsv',summary)
    # Paired error contrasts: same donors across A–D, preserving study sizes.
    primary=[r for r in rows if int(r['seed_index'])==0];donors=sorted({r['donor_group'] for r in primary});errors={}
    for strategy in 'ABCD':
        errors[strategy]=np.array([metrics([r for r in primary if r['strategy']==strategy and r['donor_group']==g])['mae_gy'] for g in donors])
    rng=np.random.default_rng(config['bootstrap_seed']);indices=[]
    for study in sorted({r['study'] for r in primary}):
        ix=np.array([i for i,g in enumerate(donors) if any(r['study']==study and r['donor_group']==g for r in primary)])
        indices.append(rng.choice(ix,size=(config['bootstrap_repeats'],len(ix)),replace=True))
    draws=np.concatenate(indices,axis=1);comparisons=[]
    for strategy in 'BCD':
        delta=errors['A']-errors[strategy];dist=delta[draws].mean(axis=1)
        comparisons.append(dict(comparison=f'A minus {strategy}',mean_mae_difference_gy=float(delta.mean()),
                      bonferroni_98_333_percent_interval_gy=np.quantile(dist,[.05/6,1-.05/6]).tolist(),
                      direction='positive favors auxiliary strategy',bootstrap_repeats=config['bootstrap_repeats'],
                      limitation='conditional on 3 observed studies and fitted models; no refitting'))
    save_json(LOG/'paired-comparisons.json',comparisons)
    # Full raw -> saved preprocessing -> saved feature selector -> saved network replay.
    data,metadata=load_release(ROOT/config['release']);replays=[];fold_records=[]
    for outer in splits['outer']:
        directory=OUT/'folds'/outer['study']/'outer';prep=DevelopmentPreprocessor.load(directory/'preprocessing')
        fit=json.loads((directory/'fit.json').read_text());view=prep.transform(data,outer['validation'])
        for strategy in 'ABCD':
            par=selected[outer['study'],strategy]['chosen']['params'];rep=par['representation']
            with np.load(directory/f'{rep}-features.npz',allow_pickle=False) as z:
                gp=GenePreparation(tuple(z['gene_ids']),z['medians'],z['selected'],z['mean'],z['scale'],z['selection_scores'],tuple(z['training_ids']))
            assert set(gp.training_ids)==set(outer['training']) and set(prep.core_training)==set(outer['training'])
            x=gp.transform(view[f'{rep}_values'].T,view['gene_ids'],True)
            for si,seed in enumerate(outer['seeds']):
                path=OUT/'selected'/outer['study']/strategy/str(seed)
                net,saved=load_model(path/'model.pt');v=predict(net,x,saved['target_mean'],saved['target_scale'])
                result=json.loads((path/'result.json').read_text());traces.append(result)
                difference=float(np.max(abs(v-np.asarray(result['prediction_gy']))));assert difference<=1e-6
                expected={r['profile_id']:float(r['predicted_gy']) for r in rows if r['strategy']==strategy and int(r['seed_index'])==si and r['study']==outer['study']}
                assert max(abs(v[i]-expected[s]) for i,s in enumerate(outer['validation']))<=1e-6
                replays.append(dict(study=outer['study'],strategy=strategy,seed=seed,raw_replay_max_difference_gy=difference,individual_replay_max_difference_gy=result['individual_max_difference_gy']))
        for f in outer['inner']+[dict(name=f"{outer['study']}/outer")]:
            fit_record=json.loads((OUT/'folds'/f['name']/'fit.json').read_text());assert fit_record['nontraining_perturbation_invariant']
            fold_records.append(dict(fold=f['name'],genes=fit_record['raw_gene_count'],training=len(fit_record['training']),validation=len(fit_record['validation']),seconds=fit_record['seconds']))
    assert len(traces)==516;assert len({r['trace']['parameters'] for r in traces})==1
    runtime={s:dict(fits=sum(r['params']['strategy']==s for r in traces),fit_seconds=sum(r['fit_seconds'] for r in traces if r['params']['strategy']==s),optimizer_steps=sum(p['steps'] for r in traces if r['params']['strategy']==s for p in r['trace']['phases'])) for s in 'ABCD'}
    save_json(LOG/'runtime.json',dict(strategies=runtime,parameters_per_network=traces[0]['trace']['parameters'],folds=fold_records,wall_seconds=end['wall_seconds_this_invocation']))
    save_json(LOG/'replay.json',replays)
    protected=json.loads((LOG/'protected-before.json').read_text());verify_files(protected)
    # Validate all completion-manifest payload hashes (candidate/selected/preprocessing caches).
    n=0
    for p in OUT.rglob('complete.json'):
        rec=json.loads(p.read_text());assert rec['identity']==end['identity']
        for name,h in rec['files'].items():assert sha(p.parent/name)==h
        n+=1
    save_json(LOG/'verification.json',dict(status='PASS_INITIAL_NESTED_DEVELOPMENT',strategy_fits=516,inner_fits=480,outer_models=36,outer_studies=3,inner_folds_per_outer=5,
              core_profiles=145,core_donor_groups=30,auxiliary_profiles=422,auxiliary_donor_groups=211,fold_preprocessing_checks=len(fold_records),
              checkpoint_raw_replays=len(replays),max_raw_replay_difference_gy=max(r['raw_replay_max_difference_gy'] for r in replays),
              max_individual_replay_difference_gy=max(r['individual_replay_max_difference_gy'] for r in replays),
              protected_files_unchanged=len(protected),completed_cache_manifests=n,protocol_precedes_prediction_fitting=True,
              locked_test_opened=False,cloud_changed=False,production_model_selected=False,
              limitations=['Only three core studies','All auxiliary doses are 0/2 Gy','No independent auxiliary validation','Cross-study donor identity not independently established','Bootstrap conditional on observed studies without model refitting','Final proposal/PPT hypothesis wording unresolved']))
    print(json.dumps([r for r in summary if r['seed_index']==0 and r['slice']=='all'],indent=2))
    print('VERIFIED',len(replays),'raw checkpoint replays,',len(protected),'protected files')

if __name__=='__main__':main()
