"""Base scorer class and registry for OOD detection methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# Global registry: name -> scorer class
SCORER_REGISTRY: dict[str, type[BaseScorer]] = {}


def register_scorer(cls: type[BaseScorer]) -> type[BaseScorer]:
    """Decorator that registers a scorer class by its `name` attribute."""
    SCORER_REGISTRY[cls.name] = cls
    return cls


def get_available_scorers(data_keys: set[str]) -> list[str]:
    """Return names of scorers whose requirements are satisfied by `data_keys`."""
    return [
        name for name, cls in SCORER_REGISTRY.items()
        if cls.requires.issubset(data_keys)
    ]


def get_scorer_families() -> dict[str, list[str]]:
    """Return {family_name: [scorer_name, ...]} from the registry."""
    families: dict[str, list[str]] = {}
    for name, cls in SCORER_REGISTRY.items():
        families.setdefault(cls.family, []).append(name)
    return families


class BaseScorer(ABC):
    """Abstract base for all OOD scorers.

    Subclasses must define:
        name:     short identifier (e.g. "msp")
        family:   one of "uncertainty", "distance", "density", "hybrid"
        requires: set of data-dict keys needed (e.g. {"embeddings", "logits"})

    Convention: higher score = more anomalous.
    """
    name: str = ""
    family: str = ""
    requires: set[str] = set()

    def __init__(self, **kwargs: Any):
        self._fitted = False
        self._config = kwargs

    @abstractmethod
    def fit(self, train_data: dict[str, np.ndarray]) -> BaseScorer:
        """Fit scorer statistics on labelled training data.

        Args:
            train_data: dict with keys like "embeddings", "logits", "labels", etc.
        """
        ...

    @abstractmethod
    def score(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Compute anomaly scores for new data.

        Args:
            data: same key structure as train_data (labels not required).

        Returns:
            1-D array of shape (N,) — higher = more anomalous.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, family={self.family!r})"
