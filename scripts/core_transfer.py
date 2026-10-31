"""Training-only eligibility and equal-study frozen-reference preprocessing.

The fixed input probe universe comes from external platform annotations, not
held-out expression. This module does not fit dose predictors or batch correction.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from assay_qc import aggregate_median, fit_probe_filter
from masked_quantile import MaskedFrozenQuantile


@dataclass(frozen=True)
class CoreTransfer:
    probe_ids: tuple
    gene_keys: tuple
    keep: np.ndarray
    reference: MaskedFrozenQuantile

    @classmethod
    def fit(cls, values, usable, detected, probe_ids, gene_keys, sample_ids, study_ids, training_ids,
            min_valid=.95, min_detected=.20):
        x=np.asarray(values);mask=np.asarray(usable);det=np.asarray(detected)
        if (x.ndim!=2 or mask.shape!=x.shape or det.shape!=x.shape or mask.dtype!=bool or det.dtype!=bool
                or len(probe_ids)!=x.shape[0] or len(gene_keys)!=x.shape[0]
                or len(set(probe_ids))!=len(probe_ids) or any(not str(g) for g in gene_keys)
                or len(sample_ids)!=x.shape[1] or len(set(sample_ids))!=len(sample_ids)
                or len(study_ids)!=x.shape[1] or not training_ids
                or len(set(training_ids))!=len(training_ids) or not set(training_ids).issubset(sample_ids)):
            raise ValueError('Invalid transfer inputs/membership')
        train=np.array([s in training_ids for s in sample_ids]);studies=np.asarray(study_ids)
        train_studies=sorted(set(studies[train]));keep=np.ones(x.shape[0],dtype=bool)
        for study in train_studies:
            take=train&(studies==study)
            kept,_,_=fit_probe_filter(np.where(mask[:,take],x[:,take],np.nan),det[:,take],np.ones(take.sum(),bool),min_valid,min_detected)
            keep &= kept
        if keep.sum()<2:raise ValueError('Fewer than two eligible common probes')
        features=tuple(np.asarray(probe_ids)[keep]);refs=[]
        for study in train_studies:
            ids=[s for s,st in zip(sample_ids,study_ids) if st==study and s in training_ids]
            refs.append(MaskedFrozenQuantile.fit(x[keep],mask[keep],features,sample_ids,ids).target)
        target=np.stack(refs).mean(axis=0);target.setflags(write=False)
        reference=MaskedFrozenQuantile(target,features,tuple(sorted(training_ids)))
        keep.setflags(write=False)
        return cls(tuple(probe_ids),tuple(gene_keys),keep,reference)

    def transform(self, values, usable, probe_ids):
        if tuple(probe_ids)!=self.probe_ids:raise ValueError('Input probe order mismatch')
        x=np.asarray(values);mask=np.asarray(usable)
        if x.ndim!=2 or x.shape!=mask.shape or x.shape[0]!=len(probe_ids):raise ValueError('Invalid transfer dimensions')
        normalized=self.reference.transform(x[self.keep],mask[self.keep],self.reference.feature_ids)
        genes,result,counts=aggregate_median(np.log2(normalized),np.asarray(self.gene_keys)[self.keep])
        return genes,result,counts

    def save(self,path):
        path=Path(path)
        self.reference.save(path)
        with np.load(path,allow_pickle=False) as z:contents={k:z[k] for k in z.files}
        np.savez_compressed(path,**contents,all_probe_ids=np.asarray(self.probe_ids),all_gene_keys=np.asarray(self.gene_keys),
                            selected_mask=self.keep,transfer_method=np.array('equal-study-common-probe-v1'))

    @classmethod
    def load(cls,path):
        q=MaskedFrozenQuantile.load(path)
        with np.load(path,allow_pickle=False) as z:
            if str(z['transfer_method'])!='equal-study-common-probe-v1':raise ValueError('Unknown transfer method')
            ids=tuple(z['all_probe_ids']);keys=tuple(z['all_gene_keys']);keep=z['selected_mask'].copy()
        if (keep.dtype!=bool or keep.shape!=(len(ids),) or len(keys)!=len(ids) or len(set(ids))!=len(ids)
                or tuple(np.asarray(ids)[keep])!=q.feature_ids or any(not str(g) for g in keys)):
            raise ValueError('Invalid transfer artifact')
        return cls(ids,keys,keep,q)
