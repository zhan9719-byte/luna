"""FAR-based threshold calibration for anomaly detection."""
from __future__ import annotations

import numpy as np


def calibrate_threshold(id_scores: np.ndarray, target_far: float = 0.01) -> float:
    """Compute threshold on in-distribution scores at a target false-alarm rate.

    The threshold is set so that approximately `target_far` fraction of the
    in-distribution (validation) scores exceed it.

    Args:
        id_scores: anomaly scores for the in-distribution (validation) set.
        target_far: desired false-alarm rate (default 0.01 = 1%).

    Returns:
        Threshold value.
    """
    sorted_desc = np.sort(id_scores)[::-1]
    k = max(0, int(np.ceil(target_far * len(sorted_desc))) - 1)
    return float(sorted_desc[k])


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Flag samples exceeding the threshold.

    Args:
        scores: anomaly scores.
        threshold: detection threshold.

    Returns:
        Boolean array — True for flagged (anomalous) samples.
    """
    return scores > threshold
