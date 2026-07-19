"""Hybrid OOD scorers combining multiple signal sources."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA

from luna.scorers.base import BaseScorer, register_scorer


@register_scorer
class HECScorer(BaseScorer):
    """Hierarchical Entropy Consistency.

    Combines fine-head entropy, coarse-head entropy, and hierarchy
    inconsistency (fine prediction vs coarse probability).

    Args:
        fine_to_coarse: dict mapping fine class index -> coarse class index.
            Required for this scorer to work.
        weights: (w_fine, w_coarse, w_incons) weights (default (1.0, 0.5, 1.0)).
    """
    name = "hec"
    family = "hybrid"
    requires = {"logits", "coarse_logits"}

    def __init__(
        self,
        fine_to_coarse: dict[int, int] | None = None,
        weights: tuple[float, float, float] = (1.0, 0.5, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fine_to_coarse = fine_to_coarse
        self.weights = weights

    def fit(self, train_data):
        if self.fine_to_coarse is None:
            n_fine = train_data["logits"].shape[1]
            n_coarse = train_data["coarse_logits"].shape[1]
            # Default: identity mapping (fine class i -> coarse class min(i, n_coarse-1))
            self.fine_to_coarse = {i: min(i, n_coarse - 1) for i in range(n_fine)}
        self._fitted = True
        return self

    def score(self, data):
        fine_probs = softmax(data["logits"], axis=1)
        coarse_probs = softmax(data["coarse_logits"], axis=1)
        preds = fine_probs.argmax(axis=1)
        N = len(preds)

        h_fine = -(fine_probs * np.log(fine_probs + 1e-10)).sum(axis=1)
        h_coarse = -(coarse_probs * np.log(coarse_probs + 1e-10)).sum(axis=1)
        parent = np.array([self.fine_to_coarse[p] for p in preds])
        h_incons = 1.0 - coarse_probs[np.arange(N), parent]

        w = self.weights
        return w[0] * h_fine + w[1] * h_coarse + w[2] * h_incons


@register_scorer
class MRSScorer(BaseScorer):
    """Manifold Residual Score — min per-class PCA reconstruction error.

    Args:
        n_components: PCA components per class (default 12).
        min_samples: minimum class size to fit PCA (default 32).
    """
    name = "mrs"
    family = "hybrid"
    requires = {"embeddings", "labels"}

    def __init__(self, n_components: int = 12, min_samples: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.min_samples = min_samples

    def fit(self, train_data):
        emb, lab = train_data["embeddings"], train_data["labels"]
        D = emb.shape[1]
        self._class_pca = {}
        for c in np.unique(lab):
            mask = lab == c
            nc = mask.sum()
            if nc < self.min_samples:
                continue
            k = min(self.n_components, nc - 1, D)
            if k < 1:
                continue
            self._class_pca[c] = PCA(n_components=k, random_state=42).fit(emb[mask])
        self._fitted = True
        return self

    def score(self, data):
        emb = data["embeddings"]
        N = emb.shape[0]
        result = np.full(N, np.inf)
        for c, pca in self._class_pca.items():
            recon = pca.inverse_transform(pca.transform(emb))
            err = np.linalg.norm(emb - recon, axis=1)
            result = np.minimum(result, err)
        return result
