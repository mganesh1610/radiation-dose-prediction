"""Training-only cluster representations for bounded research comparisons."""
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def checked_weights(x, weights):
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    if (x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all()
            or w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w <= 0)):
        raise ValueError('Invalid finite matrix or positive weights')
    return x, w / w.sum()


class WeightedPCA:
    def __init__(self, components=10):
        self.components = components

    def fit(self, x, weights):
        x, w = checked_weights(x, weights)
        if not 1 <= self.components <= min(x.shape):
            raise ValueError('Invalid PCA dimension')
        self.center_ = np.average(x, axis=0, weights=w)
        _, singular, axes = np.linalg.svd((x-self.center_) * np.sqrt(w[:, None]), full_matrices=False)
        self.axes_ = axes[:self.components]
        self.variance_ratio_ = singular[:self.components]**2 / np.sum(singular**2)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.center_) or not np.isfinite(x).all():
            raise ValueError('Invalid PCA prediction input')
        return (x-self.center_) @ self.axes_.T


class VariableGeneView:
    """Unsupervised diagnostic selection. Never receives dose labels."""
    def __init__(self, genes=500, components=10):
        self.genes, self.components = genes, components

    def fit(self, x, gene_ids, weights):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(gene_ids) or np.isinf(x).any():
            raise ValueError('Invalid gene matrix')
        present = np.isfinite(x).any(axis=0)
        self.medians_ = np.zeros(x.shape[1])
        self.medians_[present] = np.nanmedian(x[:, present], axis=0)
        filled = np.where(np.isfinite(x), x, self.medians_)
        _, w = checked_weights(filled, weights)
        center = np.average(filled, axis=0, weights=w)
        variance = np.average((filled-center)**2, axis=0, weights=w)
        eligible = np.flatnonzero(present & (variance > 1e-12))
        self.selected_ = np.array(sorted(eligible, key=lambda j: (-variance[j], str(gene_ids[j])))[:self.genes])
        if len(self.selected_) != self.genes:
            raise ValueError('Insufficient variable genes')
        self.gene_ids_ = tuple(gene_ids)
        self.scaler_ = StandardScaler().fit(filled[:, self.selected_], sample_weight=w)
        panel = self.scaler_.transform(filled[:, self.selected_])
        self.pca_ = WeightedPCA(self.components).fit(panel, w)
        return self

    def panel(self, x, gene_ids):
        if tuple(gene_ids) != self.gene_ids_:
            raise ValueError('Gene identity/order mismatch')
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(gene_ids) or np.isinf(x).any():
            raise ValueError('Invalid gene prediction input')
        a = np.where(np.isfinite(x), x, self.medians_)[:, self.selected_]
        return self.scaler_.transform(a)

    def transform(self, x, gene_ids):
        return self.pca_.transform(self.panel(x, gene_ids))


def weighted_median(y, w):
    order = np.argsort(y, kind='stable')
    return float(np.asarray(y)[order][np.searchsorted(np.cumsum(np.asarray(w)[order]), np.sum(w)/2, side='left')])


def make_cluster(kind, k, seed):
    if kind == 'kmeans':
        return KMeans(n_clusters=k, n_init=20, max_iter=500, tol=1e-6, random_state=seed)
    if kind == 'gmm':
        return GaussianMixture(n_components=k, covariance_type='diag', reg_covar=.001,
                               n_init=5, max_iter=1000, tol=1e-4, random_state=seed)
    raise ValueError('Unknown clustering method')


class ClusterRegressor:
    def __init__(self, model, components=10, k=2, alpha=1., seed=0):
        self.model, self.components, self.k, self.alpha, self.seed = model, components, k, alpha, seed

    def fit(self, x, y, sample_weight):
        x, w = checked_weights(x, sample_weight)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(x),) or not np.isfinite(y).all():
            raise ValueError('Invalid training target')
        if self.model not in ('pca_ridge', 'kmeans_ridge', 'gmm_ridge', 'cluster_median'):
            raise ValueError('Unknown predictor')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.pca_ = WeightedPCA(self.components).fit(x, w)
            z = self.pca_.transform(x)
            if self.model != 'pca_ridge':
                if not 2 <= self.k < len(x):
                    raise ValueError('Invalid cluster count')
                kind = 'gmm' if self.model == 'gmm_ridge' else 'kmeans'
                self.cluster_ = make_cluster(kind, self.k, self.seed)
                if kind == 'kmeans':
                    self.cluster_.fit(z, sample_weight=w*len(w))
                else:
                    # sklearn GMM has no sample_weight; this limitation is explicit.
                    self.cluster_.fit(z)
                    if not self.cluster_.converged_:
                        raise ValueError('Mixture failed to converge')
            if self.model == 'cluster_median':
                labels = self.cluster_.predict(z)
                self.global_median_ = weighted_median(y, w)
                self.medians_ = np.array([weighted_median(y[labels == i], w[labels == i])
                                         if np.any(labels == i) else self.global_median_ for i in range(self.k)])
            else:
                a = self.features(z)
                self.scaler_ = StandardScaler().fit(a, sample_weight=w)
                self.regressor_ = Ridge(alpha=self.alpha, solver='svd').fit(self.scaler_.transform(a), y,
                                                                          sample_weight=w*len(w))
        if caught:
            raise ValueError('Fit warning: '+'; '.join(str(t.message) for t in caught))
        return self

    def features(self, z):
        if self.model == 'pca_ridge':
            return z
        if self.model == 'kmeans_ridge':
            return np.column_stack([z, self.cluster_.transform(z)])
        if self.model == 'gmm_ridge':
            return np.column_stack([z, self.cluster_.predict_proba(z)])
        raise ValueError('Median predictor has no Ridge features')

    def predict(self, x):
        z = self.pca_.transform(x)
        if self.model == 'cluster_median':
            return self.medians_[self.cluster_.predict(z)]
        return self.regressor_.predict(self.scaler_.transform(self.features(z)))
