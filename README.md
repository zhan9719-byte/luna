# LUNA — Library for Uncertainty and Novelty Analysis

Classifier-agnostic **out-of-distribution / novelty detection** for any model that
produces embeddings and logits. LUNA computes **16 anomaly scores in four families**
— uncertainty (MSP, entropy, energy, ODIN), distance (Mahalanobis global/class/within,
kNN, cosine, typicality), density (GMM, isolation forest, LOF, PCA-reconstruction),
hybrid (HEC, MRS) — and combines them with a trainable ensemble (LightGBM by default),
thresholded at a user-chosen false-alarm rate.

No assumption is made about the upstream model: anything that yields a
per-sample embedding vector (and optionally logits / projection heads) works —
CNNs, transformers, autoencoders, in any domain.

## Install

```bash
pip install git+https://github.com/asasli/LUNA          # library
pip install "luna-ood[viz] @ git+https://github.com/asasli/LUNA"   # + UMAP/plotly extras
# or for development:
git clone https://github.com/asasli/LUNA && pip install -e ./LUNA
```

## Quick start

```python
import numpy as np
from luna import LUNAPipeline

# Your model's outputs, as plain arrays:
train = {"embeddings": Ztr, "logits": Ltr, "labels": ytr}   # in-distribution train
val   = {"embeddings": Zva, "logits": Lva, "labels": yva}   # in-distribution holdout
query = {"embeddings": Zq,  "logits": Lq}                   # anything you want to score

pipe = LUNAPipeline(combiner="lightgbm", target_far=0.01)
pipe.fit(train)                       # fits the 16 scorers on ID statistics

scores = pipe.score(query)            # dict: {"msp": ..., "mahal_within": ..., ...}

# Optional: supervised combiner if you have a small set of known outliers
pipe.fit_combiner(val, known_outliers)
out = pipe.predict(query)             # combined score + flags at target FAR
```

Optional input keys: `coarse_logits` (hierarchical heads), `projections`
(contrastive projection head) — scorers that need a missing key are skipped
automatically. `family_analysis(id_data, ood_data)` gives per-family AUROC,
detections and flags; `report(...)` produces a ranked anomaly table.

## What's inside

```
luna/
  pipeline.py      LUNAPipeline: fit / score / fit_combiner / predict / family_analysis
  scorers/         16 scorers in 4 families (uncertainty, distance, density, hybrid)
  combiners/       lightgbm (default), logistic, rank-average, xgboost
  thresholding.py  FAR-calibrated thresholds
  viz.py           detection map, complementarity map, ROC/PR, AUROC bars, UMAP
  style.py         publication figure standard (final-size figures, light+dark themes)
  report.py        ranked anomaly reports (CSV / LaTeX)
```

## Evaluation protocol (recommended)

When benchmarking detection on a known set of outliers, avoid leakage: use a
stratified k-fold over the outliers so no object is scored by a combiner that saw
it in training, and calibrate thresholds on a held-out in-distribution split. See
the [ASTRAnet](https://github.com/<user>/ASTRAnet) pipeline for a complete,
worked example of this protocol on astronomical spectra (where LUNA serves as the
anomaly-detection layer of a transient classifier).

## Demo

`notebooks/demo.ipynb` builds a toy 2-class Gaussian problem, trains a tiny model,
and walks through scoring, family analysis and FAR-thresholded detection.
