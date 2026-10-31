"""Donor-separated engineering validation of auxiliary frozen preprocessing."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from assay_qc import aggregate_median
from audit_auxiliary import sha,save_json,tsv
from masked_quantile import MaskedFrozenQuantile
from frozen_affymetrix import FrozenAffymetrix

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/auxiliary-processing/v1'
OUT=ROOT/'data/processed_pilot/auxiliary-raw-v1'


def collapse_profiles(values,profile_ids):
    names,result,counts=aggregate_median(values.T,profile_ids)
    return names,result.T,counts.T


def selected_donors(donors):
    valid=sorted(set(donors))[::5]
    return ~np.isin(donors,valid),valid


def max_delta(a,b):
    if not np.array_equal(np.isnan(a),np.isnan(b)):raise ValueError('Missingness changed')
    return float(np.nanmax(abs(a-b)))


def main():
    start=time.perf_counter();reports=[];all_labels=[];artifacts=[]
    label_rows=list(csv.DictReader((ROOT/'context/auxiliary-audit/v1/array-manifest.tsv').open(),delimiter='\t'))
    label_map={r['profile_id']:r for r in label_rows if r['metadata_eligible_d']=='True'}
    maprows=list(csv.DictReader((LOG/'full-platform-mapping.tsv').open(),delimiter='\t'))
    for study in ['GSE97000','GSE20173','GSE85570']:
        mapping={r['feature_id']:r['gene_id'] for r in maprows if r['study']==study}
        clock=time.perf_counter()
        if study=='GSE85570':
            z=np.load(OUT/f'{study}_axes.npz',allow_pickle=False)
            ps=z['probesets'];phys=z['physical_indices'];samples=z['sample_ids'];profiles=z['profile_ids'];donors=z['donor_ids']
            n,m=len(ps),len(samples)
            x=np.memmap(OUT/'GSE85570_rma_background.bin',dtype='<f8',mode='r',shape=(m,n)).T
            train,valid=selected_donors(donors);training_profiles=sorted(set(profiles[train]))
            print('Fitting frozen Affymetrix training reference:',len(training_profiles),'profiles',flush=True)
            model=FrozenAffymetrix.fit(x,ps,phys,samples,profiles,training_profiles)
            model.save(OUT/f'{study}_engineering_reference.npz')
            print('Affymetrix fit complete; transforming all arrays',flush=True)
            y=model.transform(x,ps,phys);feature_ids=model.probeset_ids
            # Physically omit every held-out column for an independent fit-membership check.
            independent=FrozenAffymetrix.fit(x[:,train],ps,phys,samples[train],profiles[train],training_profiles)
            fitting_delta=max(max_delta(model.target,independent.target),max_delta(model.effects,independent.effects))
            replay=FrozenAffymetrix.load(OUT/f'{study}_engineering_reference.npz')
            # Complete reload application is covered; single/order checks use all validation arrays.
            reload_delta=max_delta(replay.transform(x,ps,phys),y)
            ix=np.flatnonzero(~train)
            order_delta=max_delta(model.transform(x[:,ix[::-1]],ps,phys),y[:,ix[::-1]])
            single_delta=max(max_delta(model.transform(x[:,j:j+1],ps,phys),y[:,j:j+1]) for j in ix)
            extra=dict(max_median_polish_iterations=int(model.iterations.max()),pm_rows=n,
                       summarizer='project training-frozen PM quantile + median-polish effects, not published fRMA vectors',
                       fit_profiles=len(training_profiles),fit_arrays=int(train.sum()))
        else:
            z=np.load(OUT/f'{study}_probe_inputs.npz',allow_pickle=False)
            x=z['values'];feature_ids=z['probe_ids'];samples=z['sample_ids'];profiles=z['profile_ids'];donors=z['donor_ids']
            train,valid=selected_donors(donors);mask=np.isfinite(x)
            # No duplicate profiles in these two studies.
            if len(set(profiles))!=len(profiles):raise ValueError('Unexpected Agilent technical repeat')
            model=MaskedFrozenQuantile.fit(x,mask,feature_ids,samples,list(samples[train]))
            model.save(OUT/f'{study}_engineering_reference.npz')
            y=np.log2(model.transform(x,mask,feature_ids))
            changed=x.copy();changed[:,~train]=np.nan;mm=mask.copy();mm[:,~train]=False
            independent=MaskedFrozenQuantile.fit(changed,mm,feature_ids,samples,list(samples[train]))
            fitting_delta=max_delta(model.target,independent.target)
            reload_delta=max_delta(np.log2(MaskedFrozenQuantile.load(OUT/f'{study}_engineering_reference.npz').transform(x,mask,feature_ids)),y)
            order_delta=max_delta(np.log2(model.transform(x[:,::-1],mask[:,::-1],feature_ids)),y[:,::-1])
            single_delta=max(max_delta(np.log2(model.transform(x[:,j:j+1],mask[:,j:j+1],feature_ids)),y[:,j:j+1]) for j in range(x.shape[1]))
            extra=dict(fit_profiles=int(train.sum()),fit_arrays=int(train.sum()))
        checks=dict(nontraining_fit_influence=fitting_delta,reload_difference=reload_delta,
                    order_difference=order_delta,single_sample_difference=single_delta)
        if any(v!=0 for v in checks.values()):raise ValueError(f'Frozen invariance failed {checks}')
        # Gene mapping is externally pinned; controls/ambiguous/obsolete features excluded.
        keys=np.array([mapping.get(p,'') for p in feature_ids]);mapped=keys!=''
        genes,gvalues,counts=aggregate_median(y[mapped],keys[mapped])
        profile_axis,collapsed,contributing_arrays=collapse_profiles(gvalues,profiles)
        profile_train=np.array([label_map[p]['donor_group'] not in valid for p in profile_axis])
        np.savez_compressed(OUT/f'{study}_engineering_genes.npz',gene_ids=genes,profile_ids=profile_axis,values=collapsed,
                            training_mask=profile_train,contributing_arrays=contributing_arrays,
                            representation='engineering donor-split transform; refit for every different training split')
        for i,p in enumerate(profile_axis):
            r=label_map[p]
            all_labels.append(dict(study=study,profile_id=p,donor_group=r['donor_group'],dose_gy=float(r['canonical_dose_gy']),
                                   engineering_role='train' if profile_train[i] else 'validation',
                                   mapped_genes=len(genes),missing_fraction=float(np.isnan(collapsed[:,i]).mean())))
        reports.append(dict(study=study,arrays=len(samples),profiles=len(profile_axis),donors=len(set(donors)),
                            engineering_validation_donors=valid,mapped_genes=len(genes),checks=checks,**extra,
                            missing_gene_cells=int(np.isnan(collapsed).sum()),runtime_seconds=time.perf_counter()-clock))
        save_json(LOG/'engineering-progress.json',dict(studies=reports))
        print(study,'passed',checks,'profiles',len(profile_axis),flush=True)
    tsv(LOG/'engineering-profile-qc.tsv',all_labels)
    for p in sorted(OUT.glob('*engineering*.npz')):artifacts.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    save_json(LOG/'engineering-validation.json',dict(studies=reports,profiles=len(all_labels),checks_pass=True,
               scope='preprocessing engineering validation only; no predictor fitted',runtime_seconds=time.perf_counter()-start,artifacts=artifacts))


if __name__=='__main__':main()
