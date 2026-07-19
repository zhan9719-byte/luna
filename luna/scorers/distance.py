"""Distance-based OOD scorers."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.covariance import LedoitWolf  # noqa: F401 (kept for reference)


def _shrunk_cov_inv(X, shrink=0.1, eps=1e-6):
    """Inverse of a fixed-shrinkage covariance (fast Ledoit-Wolf approximation).

    Empirical covariance shrunk toward a scaled identity. ~10x faster than
    sklearn LedoitWolf on high-dim embeddings and robust for small classes;
    matches the validated notebook's np.cov + ridge approach.
    """
    S = np.cov(X, rowvar=False)
    d = S.shape[0]
    mu = np.trace(S) / d
    S = (1.0 - shrink) * S + shrink * mu * np.eye(d)
    return np.linalg.inv(S + eps * np.eye(d))
from sklearn.neighbors import NearestNeighbors

from luna.scorers.base import BaseScorer, register_scorer


@register_scorer
class MahalGlobalScorer(BaseScorer):
    """Mahalanobis distance to the global embedding mean."""
    name = "mahal_global"
    family = "distance"
    requires = {"embeddings"}

    def fit(self, train_data):
        emb = train_data["embeddings"]
        self._mean = emb.mean(axis=0)
        self._cov_inv = _shrunk_cov_inv(emb)
        self._fitted = True
        return self

    def score(self, data):
        d = data["embeddings"] - self._mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self._cov_inv, d))


@register_scorer
class MahalClassScorer(BaseScorer):
    """Minimum Mahalanobis distance to any class centroid."""
    name = "mahal_class"
    family = "distance"
    requires = {"embeddings", "labels"}

    def fit(self, train_data):
        emb, lab = train_data["embeddings"], train_data["labels"]
        self._class_means = {}
        self._class_cov_inv = {}
        for c in np.unique(lab):
            mask = lab == c
            if mask.sum() < 5:
                continue
            ec = emb[mask]
            self._class_means[c] = ec.mean(axis=0)
            self._class_cov_inv[c] = _shrunk_cov_inv(ec)
        self._fitted = True
        return self

    def score(self, data):
        emb = data["embeddings"]
        N = emb.shape[0]
        result = np.full(N, np.inf)
        for c, mean in self._class_means.items():
            dc = emb - mean
            mh = np.sqrt(np.einsum("ij,jk,ik->i", dc, self._class_cov_inv[c], dc))
            result = np.minimum(result, mh)
        return result


@register_scorer
class MahalWithinScorer(BaseScorer):
    """Mahalanobis distance to the predicted class centroid."""
    name = "mahal_within"
    family = "distance"
    requires = {"embeddings", "logits", "labels"}

    def fit(self, train_data):
        emb, lab = train_data["embeddings"], train_data["labels"]
        self._class_means = {}
        self._class_cov_inv = {}
        for c in np.unique(lab):
            mask = lab == c
            if mask.sum() < 5:
                continue
            ec = emb[mask]
            self._class_means[c] = ec.mean(axis=0)
            self._class_cov_inv[c] = _shrunk_cov_inv(ec)
        self._fitted = True
        return self

    def score(self, data):
        emb = data["embeddings"]
        preds = softmax(data["logits"], axis=1).argmax(axis=1)
        N = emb.shape[0]
        result = np.zeros(N)
        for c in self._class_means:
            mask = preds == c
            if not mask.any():
                continue
            dc = emb[mask] - self._class_means[c]
            result[mask] = np.sqrt(
                np.einsum("ij,jk,ik->i", dc, self._class_cov_inv[c], dc)
            )
        return result


@register_scorer
class KNNScorer(BaseScorer):
    """Mean k-nearest-neighbor distance in embedding space.

    Args:
        n_neighbors: number of neighbors (default 10).
    """
    name = "knn"
    family = "distance"
    requires = {"embeddings"}

    def __init__(self, n_neighbors: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

    def fit(self, train_data):
        # brute (BLAS) is far faster than ball_tree for high-dim (~192) embeddings
        # and returns identical neighbors; ball_tree degenerates in high dimensions.
        self._knn = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm="brute", n_jobs=-1
        ).fit(train_data["embeddings"])
        self._fitted = True
        return self

    def score(self, data):
        dists, _ = self._knn.kneighbors(data["embeddings"])
        return dists.mean(axis=1)


@register_scorer
class CosineScorer(BaseScorer):
    """1 - max cosine similarity to class projection centroids."""
    name = "cosine"
    family = "distance"
    requires = {"projections", "labels"}

    def fit(self, train_data):
        proj, lab = train_data["projections"], train_data["labels"]
        self._centroids = {}
        for c in np.unique(lab):
            mask = lab == c
            if mask.sum() < 2:
                continue
            mean = proj[mask].mean(axis=0)
            self._centroids[c] = mean / (np.linalg.norm(mean) + 1e-8)
        self._fitted = True
        return self

    def score(self, data):
        proj = data["projections"]
        pn = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)
        cent_matrix = np.stack(list(self._centroids.values()))
        sims = pn @ cent_matrix.T
        return 1.0 - sims.max(axis=1)


@register_scorer
class TypicalityScorer(BaseScorer):
    """Distance to predicted class center, normalized by class radius."""
    name = "typicality"
    family = "distance"
    requires = {"embeddings", "logits", "labels"}

    def fit(self, train_data):
        emb, lab = train_data["embeddings"], train_data["labels"]
        self._class_means = {}
        self._class_radii = {}
        for c in np.unique(lab):
            mask = lab == c
            if mask.sum() < 5:
                continue
            self._class_means[c] = emb[mask].mean(axis=0)
            dists = np.linalg.norm(emb[mask] - self._class_means[c], axis=1)
            self._class_radii[c] = np.percentile(dists, 95)
        self._fitted = True
        return self

    def score(self, data):
        emb = data["embeddings"]
        preds = softmax(data["logits"], axis=1).argmax(axis=1)
        N = emb.shape[0]
        result = np.zeros(N)
        for c in self._class_means:
            mask = preds == c
            if not mask.any():
                continue
            d = np.linalg.norm(emb[mask] - self._class_means[c], axis=1)
            result[mask] = d / max(self._class_radii[c], 1e-8)
        return result
