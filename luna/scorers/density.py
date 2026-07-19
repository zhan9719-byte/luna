"""Density-based OOD scorers (require only embeddings)."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from luna.scorers.base import BaseScorer, register_scorer


@register_scorer
class GMMScorer(BaseScorer):
    """Negative log-likelihood from a Gaussian Mixture Model.

    Args:
        n_components: number of GMM components (default 14).
        pca_dim: PCA dimension for preprocessing (default 32, None to skip).
    """
    name = "gmm"
    family = "density"
    requires = {"embeddings"}

    def __init__(self, n_components: int = 14, pca_dim: int | None = 32, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.pca_dim = pca_dim

    def fit(self, train_data):
        emb = train_data["embeddings"]
        if self.pca_dim is not None and emb.shape[1] > self.pca_dim:
            self._pca = PCA(n_components=self.pca_dim, random_state=42).fit(emb)
            emb = self._pca.transform(emb)
        else:
            self._pca = None
        self._gmm = GaussianMixture(
            n_components=self.n_components, covariance_type="full",
            max_iter=500, random_state=42,
        ).fit(emb)
        self._fitted = True
        return self

    def score(self, data):
        emb = data["embeddings"]
        if self._pca is not None:
            emb = self._pca.transform(emb)
        return -self._gmm.score_samples(emb)


@register_scorer
class IsolationForestScorer(BaseScorer):
    """Isolation Forest anomaly score.

    Args:
        n_estimators: number of trees (default 300).
        pca_dim: PCA dimension for preprocessing (default 32, None to skip).
    """
    name = "iforest"
    family = "density"
    requires = {"embeddings"}

    def __init__(self, n_estimators: int = 300, pca_dim: int | None = 32, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.pca_dim = pca_dim

    def fit(self, train_data):
        emb = train_data["embeddings"]
        if self.pca_dim is not None and emb.shape[1] > self.pca_dim:
            self._pca = PCA(n_components=self.pca_dim, random_state=42).fit(emb)
            emb = self._pca.transform(emb)
        else:
            self._pca = None
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators, contamination=0.05, random_state=42,
        ).fit(emb)
        self._fitted = True
        return self

    def score(self, data):
        emb = data["embeddings"]
        if self._pca is not None:
            emb = self._pca.transform(emb)
        return -self._iforest.score_samples(emb)


@register_scorer
class LOFScorer(BaseScorer):
    """Local Outlier Factor (novelty detection mode).

    Args:
        n_neighbors: number of neighbors (default 20).
        pca_dim: PCA dimension for preprocessing (default 32, None to skip).
    """
    name = "lof"
    family = "density"
    requires = {"embeddings"}

    def __init__(self, n_neighbors: int = 20, pca_dim: int | None = 32, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.pca_dim = pca_dim

    def fit(self, train_data):
        emb = train_data["embeddings"]
        if self.pca_dim is not None and emb.shape[1] > self.pca_dim:
            self._pca = PCA(n_components=self.pca_dim, random_state=42).fit(emb)
            emb = self._pca.transform(emb)
        else:
            self._pca = None
        self._lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, novelty=True, contamination=0.05,
            algorithm="brute", n_jobs=-1,
        ).fit(emb)
        self._fitted = True
        return self

    def score(self, data):
        emb = data["embeddings"]
        if self._pca is not None:
            emb = self._pca.transform(emb)
        return -self._lof.score_samples(emb)


@register_scorer
class PCAReconScorer(BaseScorer):
    """PCA reconstruction error.

    Args:
        n_components: variance to retain (default 0.95) or int for fixed dims.
    """
    name = "pca_recon"
    family = "density"
    requires = {"embeddings"}

    def __init__(self, n_components: float | int = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

    def fit(self, train_data):
        emb = train_data["embeddings"]
        self._pca = PCA(n_components=self.n_components, random_state=42).fit(emb)
        self._fitted = True
        return self

    def score(self, data):
        emb = data["embeddings"]
        recon = self._pca.inverse_transform(self._pca.transform(emb))
        return ((emb - recon) ** 2).sum(axis=1)
