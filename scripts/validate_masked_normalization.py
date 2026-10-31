"""Validate pre-normalization QC masking on the GSE102971 engineering split."""
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from assay_qc import aggregate_median, fit_probe_filter
from frozen_quantile import FrozenQuantile
from masked_quantile import MaskedFrozenQuantile

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/normalization-validation/v2'
OUT=ROOT/'data/processed_pilot/GSE102971/qc-normalization-v2'
PILOT=ROOT/'context/raw-pilot/GSE102971'
AUDIT=PILOT/'audit-v1'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def distribution(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),maximum=float(x.max()))


def main():
    start=time.perf_counter();LOG.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    prior=json.loads((PILOT/'summary.json').read_text());audit=json.loads((AUDIT/'summary.json').read_text())
    for a in prior['artifact_files']+audit['artifact_files']:
        if sha(ROOT/a['path'])!=a['sha256']:raise ValueError('Prior artifact changed')
    c=np.load(ROOT/'data/processed_pilot/GSE102971/v1/canonical_source.npz',allow_pickle=False)
    a=np.load(ROOT/'data/processed_pilot/GSE102971/audit-v1/quality_flags.npz',allow_pickle=False)
    old=np.load(ROOT/'data/processed_pilot/GSE102971/v1/engineering_log2_qn.npz',allow_pickle=False)
    for k in ['sample_ids','feature_ids']:
        if not np.array_equal(c[k],old[k]) or not np.array_equal(c[k],a[k]):raise ValueError('Axis mismatch')
    raw=c['values'][:,:,0];valid=a['technical_valid'];features=tuple(c['feature_ids']);samples=tuple(c['sample_ids'])
    train=old['training_mask'];training=tuple(c['sample_ids'][train]);old_logged=old['values']
    q=MaskedFrozenQuantile.fit(raw,valid,features,samples,training)
    linear=q.transform(raw,valid,features);logged=np.log2(linear)
    q.save(OUT/'masked_quantile_reference.npz')
    np.savez_compressed(OUT/'feature_log2_qn.npz',values=logged,feature_ids=c['feature_ids'],sample_ids=c['sample_ids'],training_mask=train)
    probes,probe_values,_=aggregate_median(logged,c['probe_names'])
    detected=a['flags'][:,:,list(a['flag_columns']).index('gIsWellAboveBG')].astype(float)
    _,detection,_=aggregate_median(detected,c['probe_names'],valid)
    keep,vf,df=fit_probe_filter(probe_values,np.isfinite(detection)&(detection>=.5),train)
    mapping=list(csv.DictReader((AUDIT/'probe-map.tsv').open(),delimiter='\t'))
    if [r['probe_id'] for r in mapping]!=probes.tolist():raise ValueError('Probe order mismatch')
    if not np.array_equal(keep,np.array([r['qc_eligible']=='True' for r in mapping])):raise ValueError('Unexpected QC catalog change')
    selected=keep&np.array([bool(r['current_gene_id']) for r in mapping])
    selected_gene_keys=[r['current_gene_id'] for r,k in zip(mapping,selected) if k]
    gene_ids,genes,contributors=aggregate_median(probe_values[selected],selected_gene_keys)
    eligible=np.isfinite(genes[:,train]).mean(axis=1)>=.95
    gene_ids,genes,contributors=gene_ids[eligible],genes[eligible],contributors[eligible]
    gene_before=np.load(ROOT/'data/processed_pilot/GSE102971/audit-v1/gene_candidate.npz',allow_pickle=False)
    before_values=gene_before['values'];gene_symbols=gene_before['gene_symbols']
    if not np.array_equal(gene_ids,gene_before['gene_ids']):raise ValueError('Unexpected gene catalog change')
    np.savez_compressed(OUT/'gene_candidate.npz',values=genes,gene_ids=gene_ids,gene_symbols=gene_symbols,sample_ids=c['sample_ids'],training_mask=train,contributing_probe_counts=contributors)
    # Perturbations verify exclusion before reference fitting AND before ranking.
    changed=raw.copy();changed[~valid]=1e12
    q_changed=MaskedFrozenQuantile.fit(changed,valid,features,samples,training)
    target_mask_delta=float(np.max(abs(q.target-q_changed.target)))
    output_mask_delta=float(np.nanmax(abs(q.transform(changed,valid,features)-linear)))
    # This deliberate flawed control must react; otherwise the test is uninformative.
    legacy=FrozenQuantile.fit(raw,features,samples,training)
    flawed=FrozenQuantile.fit(changed,features,samples,training)
    flawed_delta=float(np.max(abs(flawed.target-legacy.target)))
    changed=raw.copy();changed[:,~train]=np.nan;changed_mask=valid.copy();changed_mask[:,~train]=False
    q_validation=MaskedFrozenQuantile.fit(changed,changed_mask,features,samples,training)
    target_validation_delta=float(np.max(abs(q.target-q_validation.target)))
    single_delta=0.
    for j in np.flatnonzero(~train):
        single=q.transform(raw[:,[j]],valid[:,[j]],features)
        single_delta=max(single_delta,float(np.nanmax(abs(single-linear[:,[j]]))))
    reordered=q.transform(raw[:,::-1],valid[:,::-1],features)
    reload=MaskedFrozenQuantile.load(OUT/'masked_quantile_reference.npz').transform(raw,valid,features)
    checks=dict(masked_value_target_max_change=target_mask_delta,masked_value_usable_output_max_change=output_mask_delta,
        validation_value_and_mask_target_max_change=target_validation_delta,single_vs_batch_max_change=single_delta,
        reversed_batch_max_change=float(np.nanmax(abs(reordered-linear[:,::-1]))),artifact_reload_max_change=float(np.nanmax(abs(reload-linear))),
        missing_cells_equal_mask=bool(np.array_equal(np.isnan(linear),~valid)),
        flawed_unmasked_control_target_change=flawed_delta)
    if any(checks[k]!=0 for k in checks if k.endswith('max_change')) or not checks['missing_cells_equal_mask'] or flawed_delta<=0:
        raise ValueError('Normalization invariance check failed')
    # Five complete donor-group holdouts exercise repeated fitting without modeling.
    group_rows=list(csv.DictReader((AUDIT/'sample-groups.tsv').open(),delimiter='\t'))
    group_by_gsm={r['gsm']:r['split_group_id'] for r in group_rows};groups=sorted(set(group_by_gsm.values()))
    fold_checks=[]
    for fold in range(5):
        held=set(groups[fold::5]);ids=[s for s in samples if group_by_gsm[s] not in held]
        qm=MaskedFrozenQuantile.fit(raw,valid,features,samples,ids)
        xx=raw.copy();mm=valid.copy();val=np.array([group_by_gsm[s] in held for s in samples]);xx[:,val]=np.nan;mm[:,val]=False
        qmm=MaskedFrozenQuantile.fit(xx,mm,features,samples,ids)
        fold_checks.append(dict(fold=fold,training_samples=len(ids),held_out_groups=sorted(held),target_max_change=float(np.max(abs(qm.target-qmm.target)))))
    if any(r['target_max_change']!=0 for r in fold_checks):raise ValueError('Fold leakage check failed')
    # Alternative masks/methods are diagnostics on a fixed gene catalog, not tuned winners.
    strict=valid&~a['background_warning'];strict_q=MaskedFrozenQuantile.fit(raw,strict,features,samples,training)
    strict_log=np.log2(strict_q.transform(raw,strict,features))
    _,strict_probe,_=aggregate_median(strict_log,c['probe_names']);_,strict_genes,strict_counts=aggregate_median(strict_probe[selected],selected_gene_keys)
    strict_genes=strict_genes[eligible];strict_counts=strict_counts[eligible]
    sample_medians=np.nanmedian(np.where(valid,raw,np.nan),axis=0);median_ref=float(np.median(sample_medians[train]))
    median_log=np.log2(np.where(valid,raw*median_ref/sample_medians[None,:],np.nan))
    _,median_probe,_=aggregate_median(median_log,c['probe_names']);_,median_genes,_=aggregate_median(median_probe[selected],selected_gene_keys);median_genes=median_genes[eligible]
    np.savez_compressed(OUT/'median_reference.npz',linear_median_reference=np.array(median_ref),fit_sample_ids=np.array(training),feature_ids=c['feature_ids'])
    # Describe intensity dependence of flags; do not assume missing at random.
    training_values=raw[:,train];training_valid=valid[:,train]
    boundaries=np.quantile(training_values,np.linspace(0,1,11));deciles=[]
    for i in range(10):
        mask=(training_values>=boundaries[i])&(training_values<=boundaries[i+1] if i==9 else training_values<boundaries[i+1])
        deciles.append(dict(decile=i+1,lower=float(boundaries[i]),upper=float(boundaries[i+1]),flagged_fraction=float((~training_valid[mask]).mean())))
    # Compare within-donor 7-vs-0 log2 changes on training groups only; descriptive.
    meta={r['gsm']:r for r in csv.DictReader((PILOT/'samples.tsv').open(),delimiter='\t')}
    paired=[]
    for group in sorted({meta[s]['source_donor_group'] for s in training}):
        lo=next(i for i,s in enumerate(samples) if meta[s]['source_donor_group']==group and float(meta[s]['dose_gy'])==0)
        hi=next(i for i,s in enumerate(samples) if meta[s]['source_donor_group']==group and float(meta[s]['dose_gy'])==7)
        paired.append((lo,hi))
    def paired_change(matrix):return np.nanmedian(np.column_stack([matrix[:,hi]-matrix[:,lo] for lo,hi in paired]),axis=1)
    fc_old=paired_change(before_values);fc_new=paired_change(genes)
    finite=np.isfinite(fc_old)&np.isfinite(fc_new)
    strong=finite&(abs(fc_old)>=.5)
    sensitivity=dict(gene_absolute_log2_change_vs_postmask=distribution(abs(genes-before_values)),
        feature_absolute_log2_change_vs_postmask=distribution(abs(logged-old_logged)),
        gene_absolute_log2_change_background_strict=distribution(abs(strict_genes-genes)),
        background_strict_missing_gene_cells=int(np.isnan(strict_genes).sum()),
        background_strict_cells_changed_more_than_one_log2=int(np.sum(abs(strict_genes-genes)>1)),
        gene_absolute_log2_change_frozen_median=distribution(abs(median_genes-genes)),
        training_paired_7_vs_0_change_correlation=float(np.corrcoef(fc_old[finite],fc_new[finite])[0,1]),
        training_paired_change_sign_flips_of_old_abs_change_at_least_half_log2=int(np.sum(np.sign(fc_old[strong])!=np.sign(fc_new[strong]))),
        training_paired_change_genes_above_half_log2=int(strong.sum()),
        technical_flag_rate_by_training_intensity_decile=deciles)
    strict_delta=np.where(np.isfinite(strict_genes)&np.isfinite(genes),abs(strict_genes-genes),-1.)
    top=np.argsort(strict_delta,axis=None)[-50:][::-1]
    with (LOG/'background-sensitive-cells.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['gene_id','symbol','sample_id','primary_log2','background_strict_log2','abs_change','primary_probe_contributors','strict_probe_contributors'])
        for k in top:
            i,j=np.unravel_index(k,strict_delta.shape)
            w.writerow([gene_ids[i],gene_symbols[i],samples[j],genes[i,j],strict_genes[i,j],strict_delta[i,j],contributors[i,j],strict_counts[i,j]])
    with (LOG/'gene-normalization-sensitivity.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['gene_id','symbol','median_abs_log2_change','train_paired_7_vs_0_old','train_paired_7_vs_0_new'])
        for i,g in enumerate(gene_ids):w.writerow([g,gene_symbols[i],float(np.nanmedian(abs(genes[i]-before_values[i]))),fc_old[i],fc_new[i]])
    summary=dict(completed_utc=datetime.now(timezone.utc).isoformat(),method='masked-endpoint-quantile-v1; linear normalization then log2',
        study='GSE102971',training_samples=int(train.sum()),validation_samples=int((~train).sum()),features=len(features),
        candidate_genes=len(gene_ids),missing_feature_cells=int(np.isnan(logged).sum()),missing_gene_cells=int(np.isnan(genes).sum()),
        checks=checks,donor_fold_checks=fold_checks,sensitivity=sensitivity,
        status='technical exclusion and training independence validated; biological and cross-study performance not established',
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [PILOT/'summary.json',AUDIT/'summary.json',AUDIT/'probe-map.tsv',PILOT/'split.json']},
        code_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'scripts/masked_quantile.py',ROOT/'scripts/assay_qc.py']},
        artifact_files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.glob('*.npz'))],
        runtime_seconds=time.perf_counter()-start)
    (LOG/'normalization-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    axes[0].plot(np.linspace(0,100,len(q.target)),np.log2(legacy.target),label='Prior reference')
    axes[0].plot(np.linspace(0,100,len(q.target)),np.log2(q.target),label='Exclude flags first')
    axes[0].set(xlabel='Quantile (%)',ylabel='log2 reference intensity',title='Training-fitted references');axes[0].legend()
    deltas=np.nanmedian(abs(genes-before_values),axis=0)
    axes[1].bar(np.arange(100),deltas,color=np.where(train,'#21887d','#c38a32'))
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color='#21887d',label='Training'),Patch(color='#c38a32',label='Validation')],fontsize=8)
    axes[1].set(xlabel='Sample index',ylabel='Median absolute gene change (log2)',title='Effect of moving QC before normalization')
    axes[2].bar(range(1,11),[100*r['flagged_fraction'] for r in deciles],color='#21887d')
    axes[2].set(xlabel='Training raw-signal intensity decile',ylabel='Technical flags (%)',title='Check for intensity-dependent missingness')
    fig.suptitle('GSE102971 | normalization validation | no dose-model performance claim',fontsize=15)
    fig.savefig(LOG/'normalization-checks.png',dpi=150);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['candidate_genes','missing_gene_cells','checks','runtime_seconds']},indent=2))


if __name__=='__main__':main()
