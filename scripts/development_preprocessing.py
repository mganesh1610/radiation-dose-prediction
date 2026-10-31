"""Fold-aware raw-release preprocessing for the six development studies.

Training IDs are biological profiles; donor-complete membership is enforced.
No labels enter normalization and no fitted model is shipped as a global matrix.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from assay_qc import aggregate_median
from core_transfer import CoreTransfer
from masked_quantile import MaskedFrozenQuantile
from frozen_affymetrix import FrozenAffymetrix

CORE={'GSE102971','GSE8917','GSE65292'}
AUX={'GSE85570','GSE97000','GSE20173'}


def check_training_ids(metadata,training_ids,allowed):
    if not training_ids or len(set(training_ids))!=len(training_ids):raise ValueError('Empty/duplicate training membership')
    lookup={r['profile_id']:r for r in metadata}
    if len(lookup)!=len(metadata) or not set(training_ids).issubset(lookup):raise ValueError('Unknown development profile')
    if any(lookup[s]['study'] not in allowed for s in training_ids):raise ValueError('Forbidden training role')
    donors={lookup[s]['donor_group'] for s in training_ids}
    complete={r['profile_id'] for r in metadata if r['donor_group'] in donors}
    if complete!=set(training_ids):raise ValueError('Donor doses/repeats cannot be split')


def sample_ranks(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or np.isinf(x).any():raise ValueError('Invalid gene matrix')
    result=np.full_like(x,np.nan)
    for j in range(x.shape[1]):
        finite=np.isfinite(x[:,j]);n=int(finite.sum())
        if n<2:raise ValueError('Too few finite genes')
        result[finite,j]=(rankdata(x[finite,j],method='average')-.5)/n
    return result


def load_release(directory,verify=True):
    root=Path(directory);manifest=json.loads((root/'release.json').read_text())
    if manifest['schema_version']!=1 or manifest['locked_test_included']:raise ValueError('Invalid development release')
    if verify:
        for row in manifest['files']:
            p=(root/row['name']).resolve()
            if p.parent!=root.resolve():raise ValueError('Unsafe release path')
            with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
            if p.stat().st_size!=row['bytes'] or h!=row['sha256']:raise ValueError('Release integrity failure')
    config=json.loads((root/'metadata.json').read_text())
    if set(r['study'] for r in config['profiles'])!=CORE|AUX or len(config['profiles'])!=567:
        raise ValueError('Incorrect development membership')
    data={}
    for name in ['core','GSE97000','GSE20173','GSE85570']:
        with np.load(root/f'{name}.npz',allow_pickle=False) as z:data[name]={k:z[k] for k in z.files}
    a=data['GSE85570'];a['values']=np.memmap(root/'GSE85570-background.bin',dtype='<f8',mode='r',
                      shape=(len(a['sample_ids']),len(a['probesets']))).T
    return data,config


class DevelopmentPreprocessor:
    def __init__(self,core,aux,gene_ids,core_training,aux_training,mapping):
        self.core=core;self.aux=aux;self.gene_ids=tuple(gene_ids)
        self.core_training=tuple(core_training);self.aux_training=tuple(aux_training);self.mapping=mapping

    @classmethod
    def fit(cls,data,config,core_training,aux_training=()):
        check_training_ids(config['profiles'],core_training,CORE)
        if aux_training:check_training_ids(config['profiles'],aux_training,AUX)
        a=data['core']
        core=CoreTransfer.fit(a['values'],np.isfinite(a['values']),a['detected'],a['probe_ids'],a['gene_keys'],
                              a['sample_ids'],a['study_ids'],core_training)
        genes=sorted(set(np.asarray(core.gene_keys)[core.keep])&set(config['annotation_gene_ids']))
        if len(genes)<2:raise ValueError('Insufficient common training-selected genes')
        aux={}
        for study in sorted(AUX):
            a=data[study];training=sorted(set(a['profile_ids'])&set(aux_training))
            if not training:continue
            if study=='GSE85570':
                aux[study]=FrozenAffymetrix.fit(a['values'],a['probesets'],a['physical_indices'],a['sample_ids'],a['profile_ids'],training)
            else:
                samples=list(a['sample_ids'][np.isin(a['profile_ids'],training)])
                aux[study]=MaskedFrozenQuantile.fit(a['values'],np.isfinite(a['values']),a['probe_ids'],a['sample_ids'],samples)
        return cls(core,aux,genes,sorted(core_training),sorted(aux_training),config['mapping'])

    def transform(self,data,profile_ids):
        if not profile_ids or len(set(profile_ids))!=len(profile_ids):raise ValueError('Invalid prediction membership')
        output=np.full((len(self.gene_ids),len(profile_ids)),np.nan);target={p:i for i,p in enumerate(profile_ids)};seen=set()
        for study in ['core']+sorted(self.aux):
            a=data[study];profiles=a['sample_ids'] if study=='core' else a['profile_ids']
            take=np.flatnonzero(np.isin(profiles,profile_ids))
            if not len(take):continue
            if study=='core':
                v=a['values'][:,take];genes,y,_=self.core.transform(v,np.isfinite(v),a['probe_ids']);names=a['sample_ids'][take]
            else:
                if study=='GSE85570':
                    y=self.aux[study].transform(a['values'][:,take],a['probesets'],a['physical_indices']);features=self.aux[study].probeset_ids
                else:
                    v=a['values'][:,take];y=np.log2(self.aux[study].transform(v,np.isfinite(v),a['probe_ids']));features=a['probe_ids']
                keys=np.array([self.mapping[study].get(str(p),'') for p in features]);keep=keys!=''
                genes,y,_=aggregate_median(y[keep],keys[keep])
                names,yy,_=aggregate_median(y.T,profiles[take]);y=yy.T
            rows={str(g):i for i,g in enumerate(genes)}
            present=[(k,rows[g]) for k,g in enumerate(self.gene_ids) if g in rows]
            for j,p in enumerate(names):
                if p in seen:raise ValueError('Duplicate result profile')
                seen.add(p)
                for k,source in present:output[k,target[p]]=y[source,j]
        if seen!=set(profile_ids):raise ValueError('Profile missing or source not fitted')
        return dict(gene_ids=np.array(self.gene_ids),profile_ids=np.array(profile_ids),log2_values=output,rank_values=sample_ranks(output))

    def save(self,directory):
        root=Path(directory);root.mkdir(parents=True,exist_ok=True);self.core.save(root/'core.npz')
        for study,model in self.aux.items():model.save(root/f'{study}.npz')
        (root/'processing.json').write_text(json.dumps(dict(method='development-fold-raw-v1',gene_ids=self.gene_ids,
                 core_training=self.core_training,aux_training=self.aux_training,mapping=self.mapping,aux_studies=sorted(self.aux)),indent=2)+'\n')

    @classmethod
    def load(cls,directory):
        root=Path(directory);m=json.loads((root/'processing.json').read_text())
        if m['method']!='development-fold-raw-v1':raise ValueError('Unsupported preprocessing method')
        aux={s:(FrozenAffymetrix if s=='GSE85570' else MaskedFrozenQuantile).load(root/f'{s}.npz') for s in m['aux_studies']}
        return cls(CoreTransfer.load(root/'core.npz'),aux,m['gene_ids'],m['core_training'],m['aux_training'],m['mapping'])
