"""Single-response PLS with positive sample weights and no replicated rows."""
import numpy as np


class WeightedPLS:
    def __init__(self, components):
        self.components = components

    def fit(self, x, y, weights):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(weights, dtype=float)
        if (x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape
                or not 0 < self.components <= min(len(x) - 1, x.shape[1])
                or not all(np.isfinite(a).all() for a in (x, y, w)) or np.any(w <= 0)):
            raise ValueError("Invalid weighted PLS inputs")
        w = w / w.sum()
        center = np.average(x, axis=0, weights=w)
        mean = np.dot(w, y)
        residual_x, residual_y = x - center, y - mean
        directions, loadings, responses = [], [], []
        for _ in range(self.components):
            direction = residual_x.T @ (w * residual_y)
            norm = np.linalg.norm(direction)
            if norm <= np.finfo(float).eps:
                break
            direction /= norm
            scores = residual_x @ direction
            weighted_scores = w * scores
            variance = weighted_scores @ scores
            if variance <= np.finfo(float).eps:
                break
            loading = residual_x.T @ weighted_scores / variance
            response = residual_y @ weighted_scores / variance
            residual_x -= np.outer(scores, loading)
            residual_y -= scores * response
            directions.append(direction)
            loadings.append(loading)
            responses.append(response)
        self.coef_ = np.zeros(x.shape[1])
        if directions:
            axes = np.array(directions).T
            self.coef_ = axes @ np.linalg.pinv(np.array(loadings) @ axes) @ responses
        self.intercept_ = float(mean - center @ self.coef_)
        return self

    def predict(self, x):
        return np.asarray(x) @ self.coef_ + self.intercept_
