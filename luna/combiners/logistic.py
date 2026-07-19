"""Logistic Regression combiner for OOD scores."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from luna.combiners.base import BaseCombiner


class LogisticRegressionCombiner(BaseCombiner):
    """Logistic Regression with standard scaling for combining OOD scores.

    Args:
        C: regularization strength (default 1.0).
        max_iter: maximum iterations (default 2000).
        random_state: random seed (default 42).
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        random_state: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._scaler = None
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionCombiner:
        self._scaler = StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X)
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    def _init_kwargs(self) -> dict:
        return dict(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
