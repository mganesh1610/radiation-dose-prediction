"""Exploratory within-study paired-donor contrasts; never used for model selection."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from frozen_inference import FrozenInference,encode,digest
ROOT=Path(__file__).resolve().parents[1]

def bh(p):
    p=np.asarray(p,dtype=float);out=np.full(p.shape,np.nan);idx=np.flatnonzero(np.isfinite(p))
    order=idx[np.argsort(p[idx],kind='stable')];m=len(order)
    if m:out[order]=np.minimum(1,np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1])
    return out

def paired(differences):
    x=np.asarray(differences,float);n=np.isfinite(x).sum(axis=1);mean=np.full(len(n),np.nan);sd=mean.copy();p=mean.copy()
    for i in range(len(n)):
        v=x[i,np.isfinite(x[i])]
        if len(v):mean[i]=v.mean()
        if len(v)>=3:
            sd[i]=v.std(ddof=1)
            # A zero sample variance does not provide a usable t variance estimate.
            if sd[i]>1e-12:p[i]=2*student_t.sf(abs(mean[i])/(sd[i]/np.sqrt(len(v))),df=len(v)-1)
    return mean,n,p,bh(p)

def clean(a):return [None if not np.isfinite(v) else round(float(v),7) for v in a]

def build():
    source=ROOT/'data/cloud_exports/core_fdf0d10c09a29915/core.npz'
    if digest(source.read_bytes())!='fdf0d10c09a29915413787383c42a080d85dbbdac2ae4de86b6a8009221ed5e1':raise ValueError('Core source changed')
    with np.load(source,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    engine=FrozenInference(ROOT/'data/cloud_prediction/frozen-v1')
    genes,values,_=engine.core.transform(a['values'],np.isfinite(a['values']),a['probe_ids'])
    rows=[]
    for study in sorted(set(a['study_ids'])):
        take=a['study_ids']==study;groups=sorted(set(a['split_group_ids'][take]));doses=sorted(set(a['y'][take]));curves=[];paired_by_dose={}
        for dose in doses:
            per_donor=[];names=[]
            for group in groups:
                cols=take&(a['split_group_ids']==group)&(a['y']==dose)
                if not cols.any():continue
                v=values[:,cols];counts=np.isfinite(v).sum(axis=1)
                avg=np.divide(np.nansum(v,axis=1),counts,out=np.full(v.shape[0],np.nan),where=counts>0)
                per_donor.append(avg);names.append(group)
            x=np.stack(per_donor,axis=1);paired_by_dose[dose]=(names,x);n=np.isfinite(x).sum(axis=1)
            mean=np.divide(np.nansum(x,axis=1),n,out=np.full(len(n),np.nan),where=n>0)
            ss=np.nansum((x-mean[:,None])**2,axis=1)
            sd=np.sqrt(np.divide(ss,n-1,out=np.full(len(n),np.nan),where=n>1))
            curves.append(dict(dose_gy=float(dose),mean_log2=clean(mean),sd_log2=clean(sd),donors=n.tolist()))
        contrasts=[]
        controls,control_values=paired_by_dose[0]
        for dose in doses:
            if dose==0:continue
            names,x=paired_by_dose[dose];common=sorted(set(names)&set(controls))
            difference=np.stack([x[:,names.index(g)]-control_values[:,controls.index(g)] for g in common],axis=1)
            effects,n,p,q=paired(difference);eligible=np.flatnonzero(np.isfinite(p))
            order=sorted(eligible,key=lambda i:(q[i],p[i],str(genes[i])))[:100]
            contrasts.append(dict(dose_gy=float(dose),matched_donors=len(common),testable_genes=len(eligible),
                significant_q05=int(np.sum(q<.05)),reported_rows=len(order),
                rows=[dict(gene_id=str(genes[i]),mean_log2_difference=float(effects[i]),paired_donors=int(n[i]),p=float(p[i]),q=float(q[i])) for i in order]))
        rows.append(dict(study=str(study),profiles=int(take.sum()),donors=len(groups),curves=curves,contrasts=contrasts))
    out=dict(schema_version=1,release='core-donor-statistics-v1',purpose='exploratory_development_only',final_test_included=False,
             source_sha256=digest(source.read_bytes()),preprocessing_release_sha256=engine.release_sha,genes=genes.tolist(),studies=rows,
             methods='Common eligible probes; frozen core-reference quantile normalization, log2 and median probe-to-GeneID aggregation. Repeated profiles at a donor/dose are averaged before summaries. Dose curves show donor means and SD, not confidence or prediction intervals. Contrasts pair each donor’s dose with their own control. Two-sided one-sample t test of within-donor log2 differences; at least three complete donor pairs and nonzero variance required. Benjamini–Hochberg adjustment across every testable gene separately within each study/dose contrast; top 100 rows displayed.',
             limitations=['Exploratory full-development summaries, not independent model validation or training-fold features.',
               'Study-local donor labels are not verified identities across studies; studies are not pooled for inference.',
               'Small donor counts and normality assumptions limit t-test inference. Missingness is handled per gene; no sample is automatically removed.',
               'Multiplicity adjustment is within each contrast, not across all studies, doses or exploratory analyses.',
               'Normalized expression differences are assay-level associations, not proof of causality or clinical effects. Curve arrays are rounded to seven decimals.'],
             method_sources=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html','https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html'])
    payload=encode(out);(ROOT/'app/static/statistical-results.json').write_bytes(payload)
    evidence=ROOT/'context/cloud/prediction-workflow-v1/statistical-export.json'
    evidence.write_bytes(encode(dict(bytes=len(payload),sha256=digest(payload),genes=len(genes),contrasts=sum(len(s['contrasts']) for s in rows),test_data_used=False)))
    print(evidence.read_text())
if __name__=='__main__':build()
