"""Training-profile frozen PM quantiles and median-polish probe effects.

This is a project implementation, not platform-specific published fRMA vectors.
Raw inputs must already have per-array RMA background correction.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


def fingerprint(ps,physical):
    h=hashlib.sha256()
    for name,index in zip(ps,physical):h.update(f'{name}\t{int(index)}\n'.encode())
    return h.hexdigest()


def quantile_apply(column,target):
    x=np.asarray(column,dtype=float)
    if x.shape!=target.shape or not np.isfinite(x).all() or np.any(x<=0):
        raise ValueError('Invalid corrected PM column')
    order=np.argsort(x,kind='stable');v=x[order]
    starts=np.r_[0,np.flatnonzero(np.diff(v)!=0)+1]
    lengths=np.diff(np.r_[starts,len(v)])
    out=np.empty_like(x)
    out[order]=np.repeat(np.add.reduceat(target,starts)/lengths,lengths)
    return out


def median_polish_batch(values,eps=0.01,maxiter=200):
    """Vectorized stats::medpolish convention: sets x probes x samples."""
    z=np.array(values,dtype=float,copy=True)
    if z.ndim!=3 or not np.isfinite(z).all() or min(z.shape)<1:
        raise ValueError('Invalid median-polish input')
    n,p,m=z.shape;r=np.zeros((n,p));c=np.zeros((n,m));t=np.zeros(n)
    old=np.zeros(n);active=np.ones(n,dtype=bool);iterations=np.zeros(n,dtype=int)
    for it in range(1,maxiter+1):
        ix=np.flatnonzero(active)
        if not len(ix):break
        a=z[ix];rr=r[ix];cc=c[ix];tt=t[ix]
        dr=np.median(a,axis=2);a-=dr[:,:,None];rr+=dr
        delta=np.median(cc,axis=1);cc-=delta[:,None];tt+=delta
        dc=np.median(a,axis=1);a-=dc[:,None,:];cc+=dc
        delta=np.median(rr,axis=1);rr-=delta[:,None];tt+=delta
        new=np.sum(abs(a),axis=(1,2));done=(new==0)|(abs(new-old[ix])<eps*new)
        z[ix]=a;r[ix]=rr;c[ix]=cc;t[ix]=tt;old[ix]=new;iterations[ix]=it;active[ix[done]]=False
    if active.any():raise ValueError(f'Median polish failed to converge for {active.sum()} sets')
    return r,t,c,iterations


@dataclass(frozen=True)
class FrozenAffymetrix:
    target: np.ndarray
    effects: np.ndarray
    probeset_ids: np.ndarray
    lengths: np.ndarray
    map_sha256: str
    fit_profile_ids: tuple
    fit_sample_ids: tuple
    iterations: np.ndarray

    @classmethod
    def fit(cls,values,probesets,physical,sample_ids,profile_ids,training_profiles):
        x=np.asarray(values);ps=np.asarray(probesets);samples=np.asarray(sample_ids);profiles=np.asarray(profile_ids)
        if (x.ndim!=2 or x.shape!=(len(ps),len(samples)) or len(profiles)!=len(samples)
                or len(physical)!=len(ps) or len(set(samples))!=len(samples)
                or not training_profiles or len(set(training_profiles))!=len(training_profiles)
                or not set(training_profiles).issubset(profiles)):
            raise ValueError('Invalid Affymetrix identity/membership')
        starts=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1];lengths=np.diff(np.r_[starts,len(ps)])
        names=ps[starts]
        if len(set(names))!=len(names):raise ValueError('Probe sets must be contiguous')
        groups=[np.array(sorted(np.flatnonzero(profiles==p),key=lambda j:samples[j])) for p in sorted(training_profiles)]
        target=np.zeros(len(ps))
        for g in groups:
            curve=np.zeros(len(ps))
            for j in g:
                v=x[:,j]
                if not np.isfinite(v).all() or np.any(v<=0):raise ValueError('Invalid training PM values')
                curve+=np.sort(v)
            target+=curve/len(g)
        target/=len(groups)
        # Store training-only normalized columns; technical repeats receive one profile vote.
        y=np.empty((len(ps),len(groups)))
        for k,g in enumerate(groups):
            cols=[np.log2(quantile_apply(x[:,j],target)) for j in g]
            y[:,k]=np.median(np.stack(cols),axis=0)
        effects=np.empty(len(ps));iterations=np.empty(len(names),dtype=int)
        for length in sorted(set(lengths)):
            take=np.flatnonzero(lengths==length)
            for off in range(0,len(take),256):
                sets=take[off:off+256];positions=starts[sets,None]+np.arange(length)[None,:]
                r,_,_,it=median_polish_batch(y[positions])
                effects[positions]=r;iterations[sets]=it
        return cls(target,effects,names,lengths,fingerprint(ps,physical),tuple(sorted(training_profiles)),
                   tuple(sorted(samples[np.isin(profiles,training_profiles)])),iterations)

    def transform(self,values,probesets,physical):
        x=np.asarray(values)
        if x.ndim!=2 or x.shape[0]!=len(self.target) or fingerprint(probesets,physical)!=self.map_sha256:
            raise ValueError('Affymetrix feature identity/order mismatch')
        starts=np.r_[0,np.cumsum(self.lengths)[:-1]]
        out=np.empty((len(starts),x.shape[1]))
        for j in range(x.shape[1]):
            residual=np.log2(quantile_apply(x[:,j],self.target))-self.effects
            for length in sorted(set(self.lengths)):
                take=np.flatnonzero(self.lengths==length)
                positions=starts[take,None]+np.arange(length)[None,:]
                out[take,j]=np.median(residual[positions],axis=1)
        return out

    def save(self,path):
        np.savez_compressed(path,target=self.target,effects=self.effects,probeset_ids=self.probeset_ids,
                            lengths=self.lengths,map_sha256=self.map_sha256,fit_profile_ids=self.fit_profile_ids,
                            fit_sample_ids=self.fit_sample_ids,iterations=self.iterations,
                            method='project-frozen-pm-medianpolish-v1')

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['method'])!='project-frozen-pm-medianpolish-v1':raise ValueError('Unsupported method')
            a=cls(z['target'],z['effects'],z['probeset_ids'],z['lengths'],str(z['map_sha256']),
                  tuple(z['fit_profile_ids']),tuple(z['fit_sample_ids']),z['iterations'])
        if (not np.isfinite(a.target).all() or np.any(a.target<=0) or np.any(np.diff(a.target)<0)
                or not np.isfinite(a.effects).all() or len(a.effects)!=len(a.target)
                or a.lengths.sum()!=len(a.target) or np.any(a.lengths<1)
                or len(a.probeset_ids)!=len(a.lengths) or len(set(a.probeset_ids))!=len(a.probeset_ids)
                or not a.fit_profile_ids or not a.fit_sample_ids):
            raise ValueError('Invalid frozen artifact')
        return a
