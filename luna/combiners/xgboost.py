"""XGBoost-based combiner for OOD scores."""
from __future__ import annotations

import numpy as np

from luna.combiners.base import BaseCombiner


class XGBoostCombiner(BaseCombiner):
    """XGBoost classifier for combining OOD scores.

    Args:
        n_estimators: number of boosting rounds (default 100).
        max_depth: maximum tree depth (default 4).
        learning_rate: boosting learning rate (default 0.05).
        random_state: random seed (default 42).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostCombiner:
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError(
                "XGBoost is required for XGBoostCombiner. "
                "Install it with: pip install luna[xgboost]"
            ) from e
        # Compute scale_pos_weight for class imbalance
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        spw = n_neg / max(n_pos, 1)
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=spw,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def _init_kwargs(self) -> dict:
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
