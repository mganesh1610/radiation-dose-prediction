"""Solve weighted Elastic Net and Lasso in their original feature units."""
import numpy as np
import osqp
from scipy import sparse


def fit_regularized(x, y, weights, alpha, l1_ratio):
    """An epigraph formulation of squared error plus the L1/L2 penalties."""
    x, y, w = map(lambda a: np.asarray(a, dtype=float), (x, y, weights))
    w = w / w.sum()
    center = w @ x
    mean = w @ y
    centered = x-center
    scale = np.maximum(np.sqrt(w @ (centered**2)), 1e-12)
    design = centered / scale
    gram = (design.T*w) @ design + np.diag(alpha*(1-l1_ratio)/scale**2)
    width = len(scale)
    quadratic = sparse.block_diag((sparse.triu(gram), sparse.csc_matrix((width,width))), format='csc')
    linear = np.r_[-design.T @ (w*(y-mean)), alpha*l1_ratio/scale]
    eye = sparse.eye(width,format='csc')
    constraints = sparse.bmat([[eye,-eye],[-eye,-eye]],format='csc')
    solver = osqp.OSQP()
    solver.setup(P=quadratic,q=linear,A=constraints,l=np.full(2*width,-np.inf),
                 u=np.zeros(2*width),eps_abs=1e-8,eps_rel=1e-8,polishing=True,
                 max_iter=250000,verbose=False)
    result = solver.solve(raise_error=True)
    if result.info.status != 'solved':
        raise ValueError('Weighted regularized fit did not converge: '+result.info.status)
    coef = result.x[:width]/scale
    return coef, float(mean-center@coef)
