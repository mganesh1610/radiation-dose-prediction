"""Verify six-study integration on a donor-separated engineering split; no model fit."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from development_preprocessing import DevelopmentPreprocessor,load_release,check_training_ids,CORE,AUX
from core_transfer import CoreTransfer
from masked_quantile import MaskedFrozenQuantile
from frozen_affymetrix import FrozenAffymetrix
from audit_auxiliary import save_json,tsv,sha

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/auxiliary-processing/v1'
OUT=ROOT/'data/processed_pilot/auxiliary-raw-v1'


def delta(a,b):
    if not np.array_equal(np.isnan(a),np.isnan(b)):raise ValueError('Missingness mismatch')
    return float(np.nanmax(abs(a-b)))


def main():
    start=time.perf_counter()
    engineering=json.loads((LOG/'engineering-validation.json').read_text())
    if not engineering['checks_pass'] or engineering['profiles']!=422:raise ValueError('Auxiliary checks incomplete')
    release=ROOT/json.loads((LOG/'release-location.json').read_text())['path']
    data,config=load_release(release);metadata=config['profiles']
    valdonors=set()
    for study in CORE|AUX:
        valdonors.update(sorted({r['donor_group'] for r in metadata if r['study']==study})[::5])
    training=[r['profile_id'] for r in metadata if r['donor_group'] not in valdonors]
    coretraining=[r['profile_id'] for r in metadata if r['profile_id'] in training and r['study'] in CORE]
    auxtraining=[r['profile_id'] for r in metadata if r['profile_id'] in training and r['study'] in AUX]
    check_training_ids(metadata,coretraining,CORE);check_training_ids(metadata,auxtraining,AUX)
    coreonly=DevelopmentPreprocessor.fit(data,config,coretraining)
    aux={}
    for s in sorted(AUX):
        m=(FrozenAffymetrix if s=='GSE85570' else MaskedFrozenQuantile).load(OUT/f'{s}_engineering_reference.npz')
        if s=='GSE85570':expected=set(m.fit_profile_ids)
        else:
            a=data[s];expected={p for sm,p in zip(a['sample_ids'],a['profile_ids']) if sm in m.fit_sample_ids}
        wanted={r['profile_id'] for r in metadata if r['study']==s and r['profile_id'] in training}
        if expected!=wanted:raise ValueError('Engineering fit donor membership mismatch')
        aux[s]=m
    model=DevelopmentPreprocessor(coreonly.core,aux,coreonly.gene_ids,sorted(coretraining),sorted(auxtraining),config['mapping'])
    profiles=[r['profile_id'] for r in metadata]
    result=model.transform(data,profiles)
    print('Combined',len(profiles),'profiles x',len(result['gene_ids']),'core-training-selected genes',flush=True)
    model.save(OUT/'engineering_shared_transform')
    replay=DevelopmentPreprocessor.load(OUT/'engineering_shared_transform')
    reloaded=replay.transform(data,profiles)
    checks=dict(reload_log2_difference=delta(result['log2_values'],reloaded['log2_values']),
                reload_rank_difference=delta(result['rank_values'],reloaded['rank_values']))
    # Probe both a technical-repeat profile and one validation donor profile from each source.
    selected=[]
    for study in sorted(CORE|AUX):
        selected.append(next(r['profile_id'] for r in metadata if r['study']==study and r['donor_group'] in valdonors))
    selected.append(next(r['profile_id'] for r in metadata if len(r['source_arrays'])>1))
    take=[profiles.index(p) for p in selected]
    batched=model.transform(data,selected[::-1])
    checks['reordered_subset_rank_difference']=delta(batched['rank_values'],result['rank_values'][:,take[::-1]])
    checks['single_profile_rank_difference']=max(delta(model.transform(data,[p])['rank_values'],result['rank_values'][:,profiles.index(p):profiles.index(p)+1]) for p in selected)
    coreids=[r['profile_id'] for r in metadata if r['study'] in CORE]
    aresult=coreonly.transform(data,coreids)
    checks['A_vs_BCD_core_rank_difference']=delta(aresult['rank_values'],result['rank_values'][:,:145])
    # Core reference fit cannot read nontraining expression/detection cells.
    a=data['core'];train=np.isin(a['sample_ids'],coretraining);x=a['values'].copy();x[:,~train]=np.nan;det=a['detected'].copy();det[:,~train]=False
    independent=CoreTransfer.fit(x,np.isfinite(x),det,a['probe_ids'],a['gene_keys'],a['sample_ids'],a['study_ids'],coretraining)
    checks['core_nontraining_reference_influence']=delta(independent.reference.target,model.core.reference.target)
    if not np.array_equal(independent.keep,model.core.keep):raise ValueError('Core selection leakage')
    if any(v!=0 for v in checks.values()):raise ValueError(f'Integrated invariance failed {checks}')
    train_mask=np.array([p in training for p in profiles]);x=result['rank_values'];coremask=np.array([r['study'] in CORE for r in metadata])
    medians=np.nanmedian(x[:,train_mask&coremask],axis=1)
    if not np.isfinite(medians).all():raise ValueError('Training-empty gene')
    finite=np.where(np.isfinite(x),x,medians[:,None])
    if not np.isfinite(finite).all():raise ValueError('Nonfinite downstream training input')
    np.savez_compressed(OUT/'engineering_shared_gene_view.npz',**result,training_mask=train_mask,
                        core_mask=coremask,core_training_imputer=medians,finite_rank_values=finite,
                        representation='donor-split engineering preview; not a globally reusable model training matrix')
    qc=[]
    for j,r in enumerate(metadata):
        vals=result['log2_values'][:,j];ok=np.isfinite(vals)
        qc.append(dict(profile_id=r['profile_id'],study=r['study'],donor_group=r['donor_group'],dose_gy=r['dose_gy'],
                       role='engineering_train' if train_mask[j] else 'engineering_validation',
                       genes=len(vals),missing_genes=int((~ok).sum()),missing_fraction=float((~ok).mean()),
                       median_log2=float(np.median(vals[ok])),iqr_log2=float(np.subtract(*np.quantile(vals[ok],[.75,.25])))))
    # Outcome-independent robust review flags; no deletion or winsorization.
    for study in CORE|AUX:
        rows=[r for r in qc if r['study']==study]
        for metric in ['missing_fraction','median_log2','iqr_log2']:
            v=np.array([r[metric] for r in rows]);center=np.median(v);mad=np.median(abs(v-center));scale=1.4826*mad
            for r,value in zip(rows,v):
                r[metric+'_robust_review']=bool(abs(value-center)>3*scale) if scale>0 else bool(value!=center)
    tsv(LOG/'shared-profile-qc.tsv',qc)
    summary=dict(status='PASS_ENGINEERING_NOT_PREDICTIVE_VALIDATION',profiles=567,core_profiles=145,auxiliary_profiles=422,
                 study_local_donors=241,annotation_genes=len(config['annotation_gene_ids']),engineering_selected_genes=len(result['gene_ids']),
                 training_profiles=int(train_mask.sum()),validation_profiles=int((~train_mask).sum()),checks=checks,
                 rank_missing_cells=int(np.isnan(x).sum()),max_profile_missing_fraction=float(np.isnan(x).mean(axis=0).max()),
                 no_automatic_sample_exclusions=True,imputation_scope='engineering core-training donors only',
                 raw_corrected_release=str(release.relative_to(ROOT)),new_predictors_fitted=False,locked_test_access=False,
                 runtime_seconds=time.perf_counter()-start)
    save_json(LOG/'shared-validation.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
