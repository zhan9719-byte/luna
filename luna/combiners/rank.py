"""Mean-rank ensemble combiner (unsupervised)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from luna.combiners.base import BaseCombiner


class MeanRankCombiner(BaseCombiner):
    """Unsupervised mean-rank ensemble.

    Averages rank-normalized scores across all methods.
    No labelled anomaly data needed — the fit() stores the ID score
    distributions for rank normalization at prediction time.
    """

    def __init__(self):
        self._id_sorted: np.ndarray | None = None
        self._n_id: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MeanRankCombiner:
        # Store only the ID (y==0) scores for rank normalization
        id_mask = y == 0
        id_X = X[id_mask]
        self._n_id = id_X.shape[0]
        # Store sorted ID scores per method for fast searchsorted
        self._id_sorted = np.sort(id_X, axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, M = X.shape
        ranks = np.empty_like(X, dtype=float)
        for j in range(M):
            # Rank each test score in the ID distribution
            ranks[:, j] = np.searchsorted(
                self._id_sorted[:, j], X[:, j]
            ) / self._n_id
        return ranks.mean(axis=1)

    def cross_val_auroc(self, X, y, n_splits=5, seed=42):
        # Rank combiner uses only ID distribution, so CV is simpler
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        aurocs = []
        for train_idx, test_idx in cv.split(X, y):
            clone = MeanRankCombiner()
            clone.fit(X[train_idx], y[train_idx])
            preds = clone.predict(X[test_idx])
            aurocs.append(roc_auc_score(y[test_idx], preds))
        return np.array(aurocs)
