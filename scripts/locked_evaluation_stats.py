"""Prespecified donor-level locked-evaluation statistics; no fitting or exclusions."""
import numpy as np
from scipy import stats


def donor_errors(actual, predicted, groups):
    y,p,g=np.asarray(actual,float),np.asarray(predicted,float),np.asarray(groups,str)
    if y.ndim!=1 or p.shape!=y.shape or g.shape!=y.shape or not len(y) or not np.isfinite([y,p]).all():
        raise ValueError('Missing/nonfinite evaluation row; no complete-case deletion allowed')
    names=sorted(set(g));residual=p-y;rows=[]
    for name in names:
        e=residual[g==name]
        rows.append(dict(donor_group=name,profiles=len(e),mae=float(abs(e).mean()),
                         mse=float(np.square(e).mean()),bias=float(e.mean())))
    return rows,dict(profiles=len(y),donors=len(names),
        donor_mae=float(np.mean([r['mae'] for r in rows])),
        donor_rmse=float(np.sqrt(np.mean([r['mse'] for r in rows]))),
        donor_bias=float(np.mean([r['bias'] for r in rows])),
        pooled_mae=float(abs(residual).mean()),pooled_rmse=float(np.sqrt(np.square(residual).mean())),
        pooled_bias=float(residual.mean()))


def bootstrap_interval(values,indices):
    a=np.asarray(values,float);ix=np.asarray(indices)
    if a.ndim!=1 or not np.isfinite(a).all() or ix.ndim!=2 or ix.shape[1]!=len(a):
        raise ValueError('Invalid paired donor bootstrap')
    estimates=a[ix].mean(axis=1)
    lo,hi=np.quantile(estimates,[.025,.975],method='linear')
    return dict(low=float(lo),high=float(hi),confidence=.95,degenerate=bool(lo==hi))


def paired_comparison(a,b,indices,independence_supported=True):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.ndim!=1 or a.shape!=b.shape:raise ValueError('Unpaired donor vectors')
    d=a-b;n=len(d);reasons=[]
    if n<2:reasons.append('fewer_than_two_donors')
    if not np.isfinite(d).all():reasons.append('nonfinite_required_error')
    if n>=2 and np.isfinite(d).all() and np.std(d,ddof=1)==0:reasons.append('zero_difference_variance')
    if not independence_supported:reasons.append('unsupported_donor_independence')
    result=dict(donors=n,mean_difference=float(d.mean()) if np.isfinite(d).all() and n else None,
       direction='A_minus_comparator',alternative='less',estimable=not reasons,reasons=reasons,
       raw_p=None,bookkeeping_p=1.,t=None,df=n-1 if n>=2 else None,simultaneous_t_ci=None,
       bootstrap_ci=bootstrap_interval(d,indices) if n and np.isfinite(d).all() else None)
    if not reasons:
        test=stats.ttest_rel(a,b,alternative='less',nan_policy='raise')
        se=float(np.std(d,ddof=1)/np.sqrt(n));critical=float(stats.t.ppf(1-.05/6,n-1))
        result.update(raw_p=float(test.pvalue),bookkeeping_p=float(test.pvalue),t=float(test.statistic),
          standard_error=se,simultaneous_t_ci=dict(low=float(d.mean()-critical*se),
            high=float(d.mean()+critical*se),confidence=1-.05/3))
    return result


def holm_three(comparisons):
    if set(comparisons)!={'B','C','D'}:raise ValueError('The primary family must contain B/C/D')
    ordered=sorted(comparisons,key=lambda k:(comparisons[k]['bookkeeping_p'],k));previous=0.
    for i,name in enumerate(ordered):
        row=comparisons[name];p=row['bookkeeping_p']
        if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid family p value')
        previous=min(1.,max(previous,(3-i)*p))
        row.update(holm_adjusted_p=previous,holm_reject=bool(row['estimable'] and previous<=.05),
                   p_is_observed=row['estimable'])
    return comparisons
