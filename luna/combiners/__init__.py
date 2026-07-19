"""Score combiner implementations."""
from luna.combiners.base import BaseCombiner
from luna.combiners.lightgbm import LightGBMCombiner
from luna.combiners.xgboost import XGBoostCombiner
from luna.combiners.logistic import LogisticRegressionCombiner
from luna.combiners.rank import MeanRankCombiner

COMBINER_REGISTRY: dict[str, type[BaseCombiner]] = {
    "lightgbm": LightGBMCombiner,
    "xgboost": XGBoostCombiner,
    "logistic": LogisticRegressionCombiner,
    "rank": MeanRankCombiner,
}


def get_combiner(name: str | BaseCombiner, **kwargs) -> BaseCombiner:
    """Instantiate a combiner by name or return a pre-built instance.

    Args:
        name: combiner name ("lightgbm", "xgboost", "logistic", "rank")
              or a BaseCombiner instance.
        **kwargs: passed to the combiner constructor.
    """
    if isinstance(name, BaseCombiner):
        return name
    if name not in COMBINER_REGISTRY:
        raise ValueError(
            f"Unknown combiner '{name}'. Available: {list(COMBINER_REGISTRY.keys())}. "
            "Or pass a BaseCombiner instance."
        )
    return COMBINER_REGISTRY[name](**kwargs)


__all__ = [
    "BaseCombiner",
    "LightGBMCombiner",
    "XGBoostCombiner",
    "LogisticRegressionCombiner",
    "MeanRankCombiner",
    "COMBINER_REGISTRY",
    "get_combiner",
]
