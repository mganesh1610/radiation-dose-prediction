"""Independent membership, ranking, replay and comparative-error checks."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch

from abcd_models import load_model, predict
from development_preprocessing import DevelopmentPreprocessor
from transfer_robust import load_features, panel_transform
from run_abcd_experiments import sha, save_json, write_tsv, verify_files
from run_transfer_robust import ROOT, LOG, OUT, MODELS, read_npz


def read(path):
    return list(csv.DictReader(Path(path).open(), delimiter='\t'))


def metrics(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r['donor_group']].append(float(r['predicted_gy'])-float(r['actual_gy']))
    if not grouped:
        raise ValueError('Empty metric subset')
    return dict(profiles=len(records), donors=len(grouped),
        mae_gy=float(np.mean([np.mean(np.abs(v)) for v in grouped.values()])),
        rmse_gy=float(np.sqrt(np.mean([np.mean(np.square(v)) for v in grouped.values()]))),
        bias_gy=float(np.mean([np.mean(v) for v in grouped.values()])))


def score_independently(records):
    assert len(records) == len({r['profile_id'] for r in records})
    losses = []
    for study in sorted({r['study'] for r in records}):
        rr = [r for r in records if r['study']==study]
        losses.append(.5*metrics(rr)['mae_gy'] + .25*metrics([r for r in rr if r['actual_gy']==0])['mae_gy']
                      + .25*metrics([r for r in rr if r['actual_gy']>2])['mae_gy'])
    return float(np.mean(losses))


def summarize_predictions(allrows):
    output = []
    keys = sorted({(r['version'],r['model'],r['arm'],r['seed_index']) for r in allrows})
    for version, model, arm, si in keys:
        rr = [r for r in allrows if (r['version'],r['model'],r['arm'],r['seed_index'])==(version,model,arm,si)]
        assert len(rr)==145 and len({r['profile_id'] for r in rr})==145
        for study in ['pooled','GSE102971','GSE65292','GSE8917']:
            st = rr if study=='pooled' else [r for r in rr if r['study']==study]
            slices = {'all':st,'control':[r for r in st if r['actual_gy']==0],
                      '0–2 Gy':[r for r in st if r['actual_gy']<=2],'>2 Gy':[r for r in st if r['actual_gy']>2]}
            for name, sub in slices.items():
                output.append(dict(version=version, model=model, arm=arm, seed_index=si, study=study, slice=name, **metrics(sub)))
    return output


def main():
    torch.set_num_threads(4)
    cfg = json.loads((LOG/'config.json').read_text()); splits = json.loads((LOG/'splits.json').read_text())
    freeze = json.loads((LOG/'protocol-freeze.json').read_text()); start = json.loads((LOG/'run-start.json').read_text()); end = json.loads((LOG/'run-complete.json').read_text())
    assert freeze['utc'] < start['utc'] < end['utc'] and start['identity']==end['identity']
    verify_files(start['code_sha256'])
    for n,h in freeze['sha256'].items():
        assert sha(LOG/n)==h
    protected = json.loads((LOG/'protected-before.json').read_text()); verify_files(protected)
    meta = json.loads((ROOT/cfg['release']/'metadata.json').read_text())
    labels = {r['profile_id']:r for r in meta['profiles']}; core = {s:r for s,r in labels.items() if r['role']=='core_development'}
    assert len(core)==145 and len({r['donor_group'] for r in core.values()})==30
    selected = {(r['study'],r['model']):r for r in json.loads((LOG/'selections.json').read_text())}
    traces, preparation, score_checks, replays = [], [], [], []
    assert len(selected)==21 and len(list((OUT/'trials').rglob('result.json')))==1176
    assert len(list((OUT/'selected').rglob('result.json')))==102
    data = {'core':read_npz(ROOT/cfg['release']/'core.npz')}
    for outer in splits['outer']:
        study = outer['study']
        assert set(outer['training']).isdisjoint(outer['validation'])
        assert set(outer['training'])|set(outer['validation'])==set(core)
        assert {labels[s]['study'] for s in outer['validation']}=={study}
        for fold in outer['inner']+[outer['outer_fold']]:
            path = OUT/'folds'/fold['name']; f = json.loads((path/'fit.json').read_text())
            assert f['training']==fold['training'] and f['validation']==fold['validation']
            assert set(f['training']).isdisjoint(f['validation'])
            assert {labels[s]['donor_group'] for s in f['training']}.isdisjoint({labels[s]['donor_group'] for s in f['validation']})
            assert f['nontraining_perturbation_invariant'] and f['auxiliary_training_ids']==splits['auxiliary_training']
            if fold['design']=='source':
                assert {labels[s]['study'] for s in f['training']}.isdisjoint({labels[s]['study'] for s in f['validation']})
                assert len({labels[s]['study'] for s in f['training']})==1
            else:
                assert f['parent_original_preparation_exact'] is True
            for rep in ['log2','rank']:
                for selector in ['original','stable']:
                    name=rep+'-'+selector; gp=load_features(path/f'{name}-features.npz')
                    assert set(gp.training_ids)==set(f['training']) and gp.fingerprint()==f['preparations'][name]['fingerprint']
                    assert len(gp.selected)==f['preparations'][name]['selected_genes']
                    panel=read_npz(path/f'{name}-panel.npz')
                    assert panel['train'].shape==(len(f['training']),500) and panel['aux'].shape==(422,500)
                    assert list(panel['training_ids'])==f['training'] and list(panel['validation_ids'])==f['validation']
                    assert list(panel['auxiliary_ids'])==splits['auxiliary_training']
                    for key in ['train','validation','aux']:
                        assert np.isfinite(panel[key]).all()
                        assert np.all(panel[key][:,len(gp.selected):]==0)
                    if selector=='stable':
                        z=read_npz(path/f'{name}-selection-evidence.npz')
                        c=z['correlations']; shift=z['control_shift_sd']
                        expected=np.all(abs(c)>=.1,0)&(np.all(c>0,0)|np.all(c<0,0))&(shift<=1)&np.isfinite(shift)
                        assert np.array_equal(expected,z['eligible'])
                        order=sorted(np.flatnonzero(expected),key=lambda i:(-z['scores'][i],str(gp.gene_ids[i])))[:500]
                        np.testing.assert_array_equal(order,gp.selected)
                        assert (c.shape[0]==1)==f['preparations'][name]['single_source_fallback']
                        if c.shape[0]==1:
                            assert np.all(shift[expected]==0)
                    preparation.append(dict(fold=fold['name'],design=fold['design'],representation=rep,selector=selector,
                        training_profiles=len(f['training']),validation_profiles=len(f['validation']),active_genes=len(gp.selected),
                        raw_genes=f['raw_genes'],single_source=(fold['design']=='source')))
        for model in MODELS:
            sel=selected[study,model]; scores=[]
            for params in cfg['candidates'][model]:
                by_design={'donor':[],'source':[]}
                for fold in outer['inner']:
                    r=json.loads((OUT/'trials'/fold['name']/model/str(params['id'])/'result.json').read_text());traces.append(r)
                    assert r['params']==params and r['seed']==fold['seed'] and r['training_ids']==fold['training'] and r['validation_ids']==fold['validation']
                    for s,v in zip(r['validation_ids'],r['predicted_gy']):
                        by_design[fold['design']].append(dict(profile_id=s,study=labels[s]['study'],donor_group=labels[s]['donor_group'],actual_gy=labels[s]['dose_gy'],predicted_gy=v))
                for rr in by_design.values():
                    assert sorted(r['profile_id'] for r in rr)==sorted(outer['training'])
                score=.5*(score_independently(by_design['donor'])+score_independently(by_design['source']))
                expected=next(c for c in sel['candidates'] if c['params']['id']==params['id'])
                assert abs(score-expected['score'])<1e-12
                scores.append((score,params['id'],params['selector']))
            assert min(scores)[1]==sel['chosen']['params']['id']
            for arm in ['original','stable']:
                assert min(s for s in scores if s[2]==arm)[1]==sel['arms'][arm]['params']['id']
            score_checks.append(dict(study=study,model=model,verified_candidates=len(scores)))
        # Fresh raw-input transforms, then every saved model and seed.
        fd=OUT/'folds'/outer['outer_fold']['name']; p=DevelopmentPreprocessor.load(fd/'preprocessing')
        assert set(p.core_training)==set(outer['training']) and not p.aux_training
        view=p.transform(data,outer['validation'])
        for model in MODELS:
            for arm in ['original','stable']:
                params=selected[study,model]['arms'][arm]['params']; name=params['representation']+'-'+arm
                gp=load_features(fd/f'{name}-features.npz'); x=panel_transform(gp,view[params['representation']+'_values'].T,view['gene_ids'])
                seeds=outer['seeds'] if model in ['A','B','C','D','gradient_boosting'] else outer['seeds'][:1]
                for seed in seeds:
                    path=OUT/'selected'/study/model/arm/str(seed);r=json.loads((path/'result.json').read_text());traces.append(r)
                    assert r['params']==params and r['seed']==seed and r['validation_ids']==outer['validation'] and r['training_ids']==outer['training']
                    if model in ['A','B','C','D']:
                        estimator,ck=load_model(path/'model.pt'); v=predict(estimator,x,ck['target_mean'],ck['target_scale'])
                        assert sum(a.numel() for a in estimator.parameters())==33138
                    else:
                        estimator=joblib.load(path/'model.joblib'); v=np.asarray(estimator.predict(x)).ravel()*r['target_scale']+r['target_mean']
                    error=float(max(abs(v-np.asarray(r['predicted_gy']))));assert error<1e-5
                    replays.append(dict(study=study,model=model,selector=arm,seed=seed,max_raw_replay_difference_gy=error,single_max_difference_gy=r['single_max_difference_gy']))
    assert len(traces)==1278 and len(replays)==102 and len(preparation)==96
    fresh=read(LOG/'predictions.tsv'); prior=read(ROOT/'context/classical-comparison/v1/combined-predictions.tsv'); allrows=[]
    for version,records in [('v1',prior),('v2',fresh)]:
        for r in records:
            if r['model'] not in MODELS:continue
            x=dict(r,version=version,arm='reference' if version=='v1' else r['selector_arm'],seed_index=int(r['seed_index']),actual_gy=float(r['actual_gy']),predicted_gy=float(r['predicted_gy']))
            lab=core[x['profile_id']]
            assert x['actual_gy']==lab['dose_gy'] and x['study']==lab['study'] and x['donor_group']==lab['donor_group'] and np.isfinite(x['predicted_gy'])
            if version=='v2':
                sel=selected[x['study'],x['model']];expected=(x['arm']==sel['chosen']['params']['selector'])
                assert (r['joint_selected']=='True')==expected
                if expected:allrows.append(dict(x,arm='joint'))
            allrows.append(x)
    table=summarize_predictions(allrows);write_tsv(LOG/'metrics.tsv',table)
    compact=[{k:r[k] for k in ['version','model','arm','seed_index','profile_id','study','donor_group','actual_gy','predicted_gy']} for r in allrows]
    write_tsv(LOG/'comparison-predictions.tsv',compact)
    def metric(version,model,seed=0,study='pooled',subset='all',arm=None):
        arm=arm or ('reference' if version=='v1' else 'joint')
        return next(r for r in table if r['version']==version and r['model']==model and r['arm']==arm and r['seed_index']==seed and r['study']==study and r['slice']==subset)
    comparison=[];seedsum=[];studies=sorted({r['study'] for r in core.values()})
    for model in MODELS:
        n=3 if model in ['A','B','C','D','gradient_boosting'] else 1
        macro={v:[float(np.mean([metric(v,model,si,st)['mae_gy'] for st in studies])) for si in range(n)] for v in ['v1','v2']}
        breaches=[]
        for st in studies:
            for subset in ['control','>2 Gy']:
                a,b=metric('v1',model,study=st,subset=subset),metric('v2',model,study=st,subset=subset)
                if b['mae_gy']>a['mae_gy']+1e-9:breaches.append(dict(study=st,slice=subset,v1_mae_gy=a['mae_gy'],v2_mae_gy=b['mae_gy']))
        passes=macro['v2'][0]<macro['v1'][0] and not breaches and np.mean(macro['v2'])<np.mean(macro['v1'])
        comparison.append(dict(model=model,v1_donor_mae_gy=metric('v1',model)['mae_gy'],v2_donor_mae_gy=metric('v2',model)['mae_gy'],
            v1_macro_study_mae_gy=macro['v1'][0],v2_macro_study_mae_gy=macro['v2'][0],
            v1_control_mae_gy=metric('v1',model,subset='control')['mae_gy'],v2_control_mae_gy=metric('v2',model,subset='control')['mae_gy'],
            v1_high_mae_gy=metric('v1',model,subset='>2 Gy')['mae_gy'],v2_high_mae_gy=metric('v2',model,subset='>2 Gy')['mae_gy'],
            strict_guard_pass=bool(passes),guard_breaches=breaches))
        seedsum.append(dict(model=model,macro_study_mae_by_seed=macro,mean_macro_by_version={v:float(np.mean(x)) for v,x in macro.items()},
            interpretation='Mean of errors from separate models, not an ensemble. No best-seed selection.'))
    save_json(LOG/'comparison.json',comparison);save_json(LOG/'seed-summary.json',seedsum)
    # Conditional paired donor bootstrap; report multiplicity rather than select on it.
    donors=sorted({r['donor_group'] for r in core.values()}); rng=np.random.default_rng(cfg['bootstrap_seed']);draws=[]
    for st in studies:
        ix=[i for i,d in enumerate(donors) if any(r['study']==st and r['donor_group']==d for r in core.values())]
        draws.append(rng.choice(ix,size=(cfg['bootstrap_repeats'],len(ix))))
    draw=np.concatenate(draws,axis=1);intervals=[]
    for model in MODELS:
        errors={}
        for v in ['v1','v2']:
            arm='reference' if v=='v1' else 'joint'
            rr=[r for r in allrows if r['version']==v and r['model']==model and r['arm']==arm and r['seed_index']==0]
            errors[v]=np.array([metrics([r for r in rr if r['donor_group']==d])['mae_gy'] for d in donors])
        delta=errors['v2']-errors['v1'];sample=delta[draw].mean(1)
        intervals.append(dict(model=model,v2_minus_v1_mae_gy=float(delta.mean()),conditional_95_interval_gy=np.quantile(sample,[.025,.975]).tolist(),
            family_seven_99_2857_interval_gy=np.quantile(sample,[.05/14,1-.05/14]).tolist()))
    save_json(LOG/'uncertainty.json',dict(comparisons=intervals,repeats=cfg['bootstrap_repeats'],negative_favors_v2=True,
        limitation='Exploratory comparison informed by earlier core outcomes; fixed predictions/three studies, no refit or seed uncertainty. Intervals are not confirmatory evidence.'))
    write_tsv(LOG/'preparation-audit.tsv',preparation);save_json(LOG/'replays.json',replays)
    cache_count=0
    for p in OUT.rglob('complete.json'):
        j=json.loads(p.read_text());assert j['identity']==end['identity']
        for n,h in j['files'].items():assert sha(p.parent/n)==h
        cache_count+=1
    verify_files(protected)
    save_json(LOG/'verification.json',dict(status='PASS_LIMITED_TRANSFER_COMPARISON',new_inner_fits=1176,new_outer_fits=102,
        preprocessing_partitions=24,held_expression_and_label_perturbation_checks=24,parent_reference_and_original_panel_matches=18,
        selector_representation_preparations=96,candidate_rankings_verified=168,saved_raw_pipeline_replays=102,
        max_raw_replay_difference_gy=max(r['max_raw_replay_difference_gy'] for r in replays),
        max_single_replay_difference_gy=max(r['single_max_difference_gy'] for r in replays),
        protected_files_unchanged=len(protected),cache_manifests_verified=cache_count,unit_tests=8,
        locked_test_opened=False,cloud_changed=False,final_model_selected=False))
    print(json.dumps(comparison,indent=2),flush=True)
    print('Verified 1278 fits,102 raw replays,24 isolated preprocessing partitions.',flush=True)


if __name__=='__main__':main()
