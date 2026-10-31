"""Complete core background products and audit preprocessing transfer by study.

No predictive model, test-cohort refit, or cloud operation is performed.
"""
import csv
import gzip
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from assay_qc import aggregate_median, platform_probe_id, resolve_gene
from check_core_compatibility import read_platform, sha, tsv
from core_transfer import CoreTransfer

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'context/core-reprocessing/v1'
OUT=ROOT/'data/processed_pilot/core-reprocessing-v1'
STUDIES=['GSE102971','GSE8917','GSE65292']


def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=int(len(x)),minimum=float(x.min()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max()))


def annotation_maps():
    path=ROOT/'context/raw-pilot/GSE102971/audit-v1/sources/Homo_sapiens.gene_info.gz'
    with gzip.open(path,'rt') as f:current={r['GeneID']:r for r in csv.DictReader(f,delimiter='\t') if r['#tax_id']=='9606'}
    mappings={};sequences={};inputs={str(path.relative_to(ROOT)):sha(path)}
    for study,gpl in zip(STUDIES,['GPL10332','GPL1708','GPL13497']):
        rows,_,_,path=read_platform(study);inputs[str(path.relative_to(ROOT))]=sha(path)
        grouped=defaultdict(list)
        for r in rows:
            if r['CONTROL_TYPE'] in ('FALSE','0'):grouped[platform_probe_id(gpl,r)].append(r)
        mappings[study]={p:resolve_gene([r['GENE'] or '' for r in rr],current) for p,rr in grouped.items()}
        sequences[study]={p:{r['SEQUENCE'].upper() for r in rr} for p,rr in grouped.items()}
    shared=set.intersection(*(set(mappings[s]) for s in STUDIES));rows=[];accepted=[]
    for probe in sorted(shared):
        gene_ids=[mappings[s][probe][0] for s in STUDIES]
        same_sequence=all(sequences[s][probe]==sequences[STUDIES[0]][probe] for s in STUDIES) and '' not in sequences[STUDIES[0]][probe]
        status='accepted' if same_sequence and len(set(gene_ids))==1 and gene_ids[0] else 'inconsistent_or_unresolved_gene_mapping'
        if not same_sequence:status='sequence_conflict_or_missing'
        row=dict(probe_id=probe,status=status,**{s+'_gene_id':g for s,g in zip(STUDIES,gene_ids)},
                 current_symbol=current[gene_ids[0]]['Symbol'] if status=='accepted' else '')
        rows.append(row)
        if status=='accepted':accepted.append(row)
    tsv(LOG/'shared-probe-annotation.tsv',rows)
    return accepted,mappings,current,inputs,len(shared)


def main():
    started=time.perf_counter()
    prepared=json.loads((LOG/'prepared.json').read_text())
    for a in prepared['artifacts']:
        if sha(ROOT/a['path'])!=a['sha256']:raise ValueError('Prepared input artifact changed')
    checks=list(csv.DictReader((LOG/'background-checks.tsv').open(),delimiter='\t'))
    params=list(csv.DictReader((LOG/'background-parameters.tsv').open(),delimiter='\t'))
    if len(params)!=145 or any(r['convergence']!='0' or r['warnings'] for r in params):raise ValueError('Unaccepted background fit')
    for row in checks:
        if any(float(row[k])!=0 for k in ['api_max_difference','reversed_batch_max_difference','single_sample_max_difference','nonpositive_corrected']):
            raise ValueError('Background checks failed')
    accepted,maps,current,inputs,shared_count=annotation_maps()
    common=np.array([r['probe_id'] for r in accepted]);keys=np.array([r['GSE102971_gene_id'] for r in accepted])
    matrices=[];detections=[];all_samples=[];all_studies=[];source_reports=[]
    label_path=ROOT/'context/normalization-validation/v2/core-sample-labels-v2.tsv'
    labels={r['gsm']:r for r in csv.DictReader(label_path.open(),delimiter='\t')}
    for study in STUDIES:
        a=np.load(OUT/f'{study}_axes_flags.npz',allow_pickle=False)
        n,m=len(a['feature_ids']),len(a['sample_ids'])
        y=np.fromfile(OUT/f'{study}_normexp.bin',dtype='<f8').reshape((m,n)).T
        valid=a['technical_valid'];sample_ids=a['sample_ids'];probes=a['probe_names']
        if not np.array_equal(np.isnan(y),~valid) or not np.isfinite(y[valid]).all() or np.any(y[valid]<=0):raise ValueError('Invalid corrected output')
        probe_ids,signal,counts=aggregate_median(y,probes)
        _,detection,_=aggregate_median(a['detected'].astype(float),probes,valid)
        detected=np.isfinite(detection)&(detection>=.5)
        ix={p:i for i,p in enumerate(probe_ids)};take=[ix[p] for p in common]
        matrices.append(signal[take]);detections.append(detected[take])
        all_samples.extend(sample_ids);all_studies.extend([study]*m)
        mapped=np.array([bool(maps[study][p][0]) for p in probe_ids])
        genes,full_gene,gcounts=aggregate_median(np.log2(signal[mapped]),[maps[study][p][0] for p in probe_ids[mapped]])
        np.savez_compressed(OUT/f'{study}_corrected_gene_view.npz',values=full_gene,gene_ids=genes,sample_ids=sample_ids,
                            contributing_probes=gcounts,representation=np.array('per-array limma normexp saddle offset0; log2; median probes; no between-array normalization/filtering'))
        np.savez_compressed(OUT/f'{study}_corrected_probe_view.npz',values=signal,probe_ids=probe_ids,detected=detected,
                            sample_ids=sample_ids,contributing_spots=counts)
        pr=[r for r in params if r['study']==study]
        source_reports.append(dict(study=study,samples=m,physical_features=n,corrected_physical_cells=int(valid.sum()),
            missing_physical_cells=int((~valid).sum()),corrected_signal=stats(y),probes=len(probe_ids),unfiltered_current_genes=len(genes),
            unfiltered_gene_missing_cells=int(np.isnan(full_gene).sum()),
            normexp_sigma=stats([float(r['sigma']) for r in pr]),normexp_mu=stats([float(r['mu']) for r in pr])))
    x=np.column_stack(matrices);det=np.column_stack(detections);usable=np.isfinite(x)
    samples=np.array(all_samples);studies=np.array(all_studies)
    if len(set(samples))!=145 or set(samples)!=set(labels):raise ValueError('Incomplete/duplicate core sample set')
    groups=np.array([labels[s]['split_group_id'] for s in samples]);doses=np.array([float(labels[s]['candidate_dose_gy']) for s in samples])
    np.savez_compressed(OUT/'common_probe_background_corrected.npz',values=x,detected=det,probe_ids=common,gene_keys=keys,
                        sample_ids=samples,study_ids=studies,split_group_ids=groups)
    fold_reports=[];pca_rows=[];sample_rows=[];projection_data=[]
    for held in STUDIES:
        train=studies!=held;test=~train;training=list(samples[train])
        q=CoreTransfer.fit(x,usable,det,common,keys,samples,studies,training)
        genes,y,_=q.transform(x,usable,common)
        refpath=OUT/f'holdout_{held}_reference.npz';q.save(refpath)
        np.savez_compressed(OUT/f'holdout_{held}_gene_matrix.npz',values=y,gene_ids=genes,sample_ids=samples,training_mask=train)
        changed=x.copy();changed[:,test]=np.nan;mm=usable.copy();mm[:,test]=False;dd=det.copy();dd[:,test]=False
        qp=CoreTransfer.fit(changed,mm,dd,common,keys,samples,studies,training)
        target_delta=float(np.max(abs(q.reference.target-qp.reference.target)))
        if not np.array_equal(q.keep,qp.keep):raise ValueError('Held-out selection influence')
        single_delta=0.
        for j in np.flatnonzero(test):
            one=q.transform(x[:,j:j+1],usable[:,j:j+1],common)[1]
            single_delta=max(single_delta,float(np.nanmax(abs(one-y[:,j:j+1]))))
        reversed_delta=float(np.nanmax(abs(q.transform(x[:,::-1],usable[:,::-1],common)[1]-y[:,::-1])))
        reload_delta=float(np.nanmax(abs(CoreTransfer.load(refpath).transform(x,usable,common)[1]-y)))
        if any(v!=0 for v in [target_delta,single_delta,reversed_delta,reload_delta]):raise ValueError('Transfer independence failed')
        inner=[]
        for fold in range(5):
            validation_groups=set()
            for st in sorted(set(studies[train])):
                validation_groups.update(sorted(set(groups[train&(studies==st)]))[fold::5])
            inner_train=train&~np.isin(groups,list(validation_groups));inner_ids=list(samples[inner_train])
            iq=CoreTransfer.fit(x,usable,det,common,keys,samples,studies,inner_ids)
            xx=x.copy();xx[:,~inner_train]=np.nan;vm=usable.copy();vm[:,~inner_train]=False;vd=det.copy();vd[:,~inner_train]=False
            iq2=CoreTransfer.fit(xx,vm,vd,common,keys,samples,studies,inner_ids)
            delta=float(np.max(abs(iq.reference.target-iq2.reference.target)))
            if delta!=0 or not np.array_equal(iq.keep,iq2.keep):raise ValueError('Inner fold influence')
            inner.append(dict(fold=fold,training_samples=int(inner_train.sum()),validation_groups=sorted(validation_groups),
                              eligible_probes=int(iq.keep.sum()),target_max_change=delta,selection_unchanged=True))
        # PCA is a descriptive, training-fitted display, not a dose predictor.
        impute=np.nanmedian(y[:,train],axis=1)
        if not np.isfinite(impute).all():raise ValueError('Training-empty gene')
        filled=np.where(np.isfinite(y),y,impute[:,None])
        variance=filled[:,train].var(axis=1);panel=np.argsort(-variance,kind='stable')[:min(2000,(variance>0).sum())]
        mean=filled[panel][:,train].mean(axis=1);centered=filled[panel].T-mean
        _,singular,vt=np.linalg.svd(centered[train],full_matrices=False)
        coordinates=centered@vt[:2].T;evr=singular[:2]**2/np.sum(singular**2)
        residual=np.sqrt(np.mean((centered-coordinates@vt[:2])**2,axis=1))
        np.savez_compressed(OUT/f'holdout_{held}_pca_display.npz',gene_ids=genes[panel],imputer=impute[panel],mean=mean,
                            components=vt[:2],explained_variance_ratio=evr,coordinates=coordinates,sample_ids=samples)
        # Study-balanced control medians; descriptive, never used to fit/choose methods.
        train_control_median=np.mean(np.column_stack([np.nanmedian(y[:,(studies==st)&(doses==0)],axis=1) for st in sorted(set(studies[train]))]),axis=1)
        held_control_median=np.nanmedian(y[:,test&(doses==0)],axis=1)
        control_shift=abs(held_control_median-train_control_median)
        q50=CoreTransfer.fit(x+50,usable,det,common,keys,samples,studies,training)
        genes50,y50,_=q50.transform(x+50,usable,common)
        if not np.array_equal(genes,genes50) or not np.array_equal(q.keep,q50.keep):raise ValueError('Offset changed catalog')
        sensitivity=stats(abs(y50-y))
        for j,s in enumerate(samples):
            selected=x[q.keep,j];ok=np.isfinite(selected)
            outside=(selected[ok]<q.reference.target[0])|(selected[ok]>q.reference.target[-1])
            sample_rows.append(dict(held_study=held,study=studies[j],gsm=s,role='training' if train[j] else 'held_out',
                missing_gene_fraction=float(np.isnan(y[:,j]).mean()),outside_reference_intensity_range_fraction=float(outside.mean()),
                two_pc_reconstruction_rms=float(residual[j])))
            pca_rows.append(dict(held_study=held,study=studies[j],gsm=s,split_group_id=groups[j],dose_gy=float(doses[j]),
                                 role='training' if train[j] else 'held_out',pc1=float(coordinates[j,0]),pc2=float(coordinates[j,1])))
        fold=dict(held_study=held,training_studies=sorted(set(studies[train])),training_samples=int(train.sum()),held_samples=int(test.sum()),
            eligible_probes=int(q.keep.sum()),genes=len(genes),training_missing_gene_fraction=float(np.isnan(y[:,train]).mean()),
            held_missing_gene_fraction=float(np.isnan(y[:,test]).mean()),held_max_sample_missing_fraction=float(np.isnan(y[:,test]).mean(axis=0).max()),
            checks=dict(heldout_target_max_change=target_delta,selection_unchanged=True,single_batch_max_change=single_delta,
                        reverse_batch_max_change=reversed_delta,reload_max_change=reload_delta),inner_donor_group_checks=inner,
            pca=dict(training_selected_genes=len(panel),explained_variance_ratio=evr.tolist(),
                     training_residual=stats(residual[train]),held_residual=stats(residual[test])),
            absolute_control_median_shift_log2=stats(control_shift),offset50_absolute_change_log2=sensitivity,
            interpretation='computational preprocessing transfer tested; residual study differences and predictive generalization require evaluation')
        fold_reports.append(fold);projection_data.append((held,coordinates,evr,train))
        print(f'Held {held}: {q.keep.sum()} probes, {len(genes)} genes; held missing {fold["held_missing_gene_fraction"]:.4%}; invariants pass',flush=True)
    tsv(LOG/'fold-sample-qc.tsv',sample_rows);tsv(LOG/'pca-projections.tsv',pca_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'GSE102971':'#167c80','GSE8917':'#cd762c','GSE65292':'#7562ac'}
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,(held,xy,evr,train) in zip(axes,projection_data):
        for st in STUDIES:
            take=studies==st
            ax.scatter(xy[take,0],xy[take,1],c=colors[st],marker='^' if st==held else 'o',s=27,label=st+(' held out' if st==held else ' train'),alpha=.8)
        ax.set(title='Held out: '+held,xlabel=f'PC1 ({evr[0]:.1%} training variance)',ylabel=f'PC2 ({evr[1]:.1%})');ax.legend(fontsize=8)
    fig.suptitle('Core study transfer | PCA fitted on training studies only | no prediction accuracy claim',fontsize=15)
    fig.savefig(LOG/'cross-study-projection.png',dpi=150);plt.close(fig)
    for path in [label_path,LOG/'prepared.json',LOG/'background-parameters.tsv',LOG/'background-checks.tsv',LOG/'METHOD.md']:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    summary=dict(completed_utc=datetime.now(timezone.utc).isoformat(),scope='145 core profiles; background processing plus leave-one-study-out preprocessing audit',
        background_method='limma normexp saddle; per-array median foreground-minus-median background; offset0; technical exclusions before fitting',
        background_fits=145,source_studies=source_reports,shared_probe_names=shared_count,consistent_sequence_and_gene_probes=len(common),
        common_annotation_gene_count=len(set(keys)),annotation_only_universe=True,folds=fold_reports,
        status='preprocessing implementation checked; study transfer remains an empirical modeling question',
        inputs_sha256=inputs,code_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'scripts/core_transfer.py',ROOT/'scripts/masked_quantile.py',ROOT/'scripts/assay_qc.py',ROOT/'scripts/background_core.R']},
        artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.glob('*')) if p.is_file()],
        runtime_seconds=time.perf_counter()-started)
    (LOG/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE',summary['runtime_seconds'],'seconds',flush=True)


if __name__=='__main__':main()
