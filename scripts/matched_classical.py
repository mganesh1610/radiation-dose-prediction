"""Source-weighted classical estimators on the frozen A–D fold panels."""
from fractions import Fraction
from functools import reduce
from math import gcd,lcm
import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet


def integer_weight_counts(weights, max_rows=10000):
    """Exact bounded replication for weighted PLS moments, not extra subjects."""
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or len(w)<2 or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('Invalid positive source weights')
    w=w/w.sum();fractions=[Fraction(float(v)).limit_denominator(1000000) for v in w]
    if max(abs(float(f)-v) for f,v in zip(fractions,w))>1e-12:raise ValueError('Weights cannot be represented exactly')
    denominator=lcm(*(f.denominator for f in fractions))
    counts=[f.numerator*(denominator//f.denominator) for f in fractions];divisor=reduce(gcd,counts)
    counts=[n//divisor for n in counts]
    if sum(counts)>max_rows:raise ValueError('Excessive weighted PLS replication')
    counts=np.array(counts,dtype=int)
    if not np.allclose(counts/counts.sum(),w,atol=1e-12,rtol=0):raise ValueError('Weight-ratio mismatch')
    return counts


def fit_classical(params,x,y,weights,seed):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);weights=np.asarray(weights,dtype=float)
    if (x.ndim!=2 or y.shape!=(len(x),) or weights.shape!=y.shape or not np.isfinite(x).all()
        or not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights<=0)):
        raise ValueError('Invalid training data')
    name=params['model'];replicated_rows=None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if name=='elastic_net':
            model=ElasticNet(alpha=params['alpha'],l1_ratio=params['l1_ratio'],fit_intercept=True,
                             tol=1e-6,max_iter=100000,selection='cyclic')
            model.fit(x,y,sample_weight=weights)
        elif name=='pls':
            n=params['n_components']
            if n<1 or n>=len(x) or n>x.shape[1]:raise ValueError('PLS components exceed original dimensions')
            counts=integer_weight_counts(weights);replicated_rows=int(counts.sum())
            model=PLSRegression(n_components=n,scale=False,max_iter=1000,tol=1e-6)
            model.fit(np.repeat(x,counts,axis=0),np.repeat(y,counts))
        elif name=='gradient_boosting':
            model=GradientBoostingRegressor(max_depth=params['max_depth'],n_estimators=params['n_estimators'],
                       learning_rate=.05,min_samples_leaf=3,subsample=1.,random_state=seed,loss='squared_error')
            model.fit(x,y,sample_weight=weights)
        else:raise ValueError('Unknown estimator')
    if caught:raise ValueError('Fit warning: '+'; '.join(str(w.message) for w in caught))
    if not np.isfinite(np.asarray(model.predict(x))).all():raise ValueError('Nonfinite fitted predictions')
    return model,dict(original_training_profiles=len(x),computational_pls_rows=replicated_rows,
                     estimator_class=type(model).__name__,n_iter=np.asarray(getattr(model,'n_iter_',[])).tolist())
