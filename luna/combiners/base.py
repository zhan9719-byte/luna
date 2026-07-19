"""Base combiner class for ensembling OOD scores."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class BaseCombiner(ABC):
    """Abstract base for all score combiners.

    A combiner takes a matrix of OOD scores (N, M) and produces a single
    anomaly score per sample in [0, 1].
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseCombiner:
        """Fit on labelled data (0 = in-distribution, 1 = anomaly).

        Args:
            X: score matrix of shape (N, M).
            y: binary labels (0/1).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1] for each sample.

        Args:
            X: score matrix of shape (N, M).
        """
        ...

    def cross_val_auroc(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42,
    ) -> np.ndarray:
        """Run stratified k-fold CV and return per-fold AUROC.

        Args:
            X: score matrix of shape (N, M).
            y: binary labels.
            n_splits: number of CV folds.
            seed: random seed.

        Returns:
            Array of shape (n_splits,) with AUROC per fold.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        aurocs = []
        for train_idx, test_idx in cv.split(X, y):
            clone = self._clone()
            clone.fit(X[train_idx], y[train_idx])
            preds = clone.predict(X[test_idx])
            aurocs.append(roc_auc_score(y[test_idx], preds))
        return np.array(aurocs)

    def _clone(self) -> BaseCombiner:
        """Create a fresh instance with the same hyperparameters."""
        return self.__class__(**self._init_kwargs())

    def _init_kwargs(self) -> dict:
        """Return kwargs to reconstruct this instance. Override in subclasses."""
        return {}
