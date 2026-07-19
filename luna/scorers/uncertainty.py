"""Uncertainty-based OOD scorers (require logits)."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax, logsumexp

from luna.scorers.base import BaseScorer, register_scorer


@register_scorer
class MSPScorer(BaseScorer):
    """Maximum Softmax Probability — negative max softmax prob."""
    name = "msp"
    family = "uncertainty"
    requires = {"logits"}

    def fit(self, train_data):
        self._fitted = True
        return self

    def score(self, data):
        probs = softmax(data["logits"], axis=1)
        return -probs.max(axis=1)


@register_scorer
class EntropyScorer(BaseScorer):
    """Shannon entropy of the softmax distribution."""
    name = "entropy"
    family = "uncertainty"
    requires = {"logits"}

    def fit(self, train_data):
        self._fitted = True
        return self

    def score(self, data):
        probs = softmax(data["logits"], axis=1)
        return -(probs * np.log(probs + 1e-10)).sum(axis=1)


@register_scorer
class EnergyScorer(BaseScorer):
    """Negative log-sum-exp energy score."""
    name = "energy"
    family = "uncertainty"
    requires = {"logits"}

    def fit(self, train_data):
        self._fitted = True
        return self

    def score(self, data):
        return -logsumexp(data["logits"], axis=1)


@register_scorer
class ODINScorer(BaseScorer):
    """ODIN — temperature-scaled negative max softmax probability.

    Args:
        temperature: temperature for scaling (default 1.5).
    """
    name = "odin"
    family = "uncertainty"
    requires = {"logits"}

    def __init__(self, temperature: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def fit(self, train_data):
        self._fitted = True
        return self

    def score(self, data):
        scaled = softmax(data["logits"] / self.temperature, axis=1)
        return -scaled.max(axis=1)
