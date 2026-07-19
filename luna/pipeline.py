"""LUNAPipeline — orchestrates scorers, combiner, thresholding, and reporting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from luna.scorers import SCORER_REGISTRY, get_available_scorers, get_scorer_families
from luna.combiners import get_combiner, BaseCombiner
from luna.thresholding import calibrate_threshold, apply_threshold


class LUNAPipeline:
    """End-to-end anomaly detection pipeline.

    Orchestrates:
      1. Scorer fitting on training data
      2. OOD score computation
      3. Combiner training on ID vs OOD scores
      4. Threshold calibration at a target false-alarm rate
      5. Full prediction (scores + ensemble + flags)
      6. Report + visualization generation

    Example:
        pipe = LUNAPipeline(scorers=["knn", "iforest", "msp"], combiner="lightgbm")
        pipe.fit(train_data)
        pipe.fit_combiner(val_data, rare_data)
        result = pipe.predict(new_data)
    """

    def __init__(
        self,
        scorers: list[str] | None = None,
        combiner: str | BaseCombiner = "lightgbm",
        target_far: float = 0.01,
        fine_to_coarse: dict[int, int] | None = None,
        scorer_kwargs: dict[str, dict] | None = None,
        combiner_kwargs: dict | None = None,
    ):
        """
        Args:
            scorers: list of scorer names to use, or None for auto-detection.
            combiner: combiner name or BaseCombiner instance.
            target_far: false-alarm rate for threshold calibration (default 1%).
            fine_to_coarse: mapping fine class idx -> coarse class idx (for HEC).
            scorer_kwargs: per-scorer kwargs, e.g. {"odin": {"temperature": 2.0}}.
            combiner_kwargs: kwargs passed to the combiner constructor.
        """
        self._requested_scorers = scorers
        self._combiner_spec = combiner
        self.target_far = target_far
        self.fine_to_coarse = fine_to_coarse
        self._scorer_kwargs = scorer_kwargs or {}
        self._combiner_kwargs = combiner_kwargs or {}

        self._scorers: dict[str, object] = {}
        self._combiner: BaseCombiner | None = None
        self._threshold: float | None = None
        self._active_methods: list[str] = []
        self._fitted = False
        self._combiner_fitted = False

    @property
    def active_methods(self) -> list[str]:
        """Names of scorers that are currently active."""
        return list(self._active_methods)

    @property
    def families(self) -> dict[str, list[str]]:
        """Active scorers grouped by family."""
        fam = {}
        for name in self._active_methods:
            family = self._scorers[name].family
            fam.setdefault(family, []).append(name)
        return fam

    def fit(self, train_data: dict[str, np.ndarray]) -> LUNAPipeline:
        """Fit scorer statistics on training data.

        Args:
            train_data: dict with keys like "embeddings", "logits", "labels", etc.
                "embeddings" and "labels" are required.

        Returns:
            self
        """
        data_keys = {k for k, v in train_data.items() if v is not None}

        # Determine which scorers to use
        if self._requested_scorers is not None:
            method_names = self._requested_scorers
        else:
            method_names = get_available_scorers(data_keys)

        # Instantiate and fit scorers
        self._scorers = {}
        self._active_methods = []
        for name in method_names:
            if name not in SCORER_REGISTRY:
                raise ValueError(
                    f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY.keys())}"
                )
            cls = SCORER_REGISTRY[name]
            if not cls.requires.issubset(data_keys):
                missing = cls.requires - data_keys
                print(f"  [LUNA] Skipping '{name}' — missing: {missing}")
                continue

            kwargs = self._scorer_kwargs.get(name, {})
            if name == "hec" and self.fine_to_coarse is not None:
                kwargs.setdefault("fine_to_coarse", self.fine_to_coarse)

            scorer = cls(**kwargs)
            import time as _t
            _t0 = _t.time()
            scorer.fit(train_data)
            print(f"  [LUNA] fit {name} in {_t.time()-_t0:.1f}s", flush=True)
            self._scorers[name] = scorer
            self._active_methods.append(name)

        self._fitted = True
        print(f"  [LUNA] Fitted {len(self._active_methods)} scorers: {self._active_methods}")
        return self

    def score(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute all active OOD scores on new data.

        Args:
            data: dict with the same keys used during fit (labels not required).

        Returns:
            Dict mapping scorer name -> (N,) score array.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .score().")

        # NOTE: no id()-based memoization — Python recycles array memory ids, so a
        # fresh temporary array can collide with a GC'd one and return wrong cached
        # scores. Always recompute (fast with brute-kNN + shrinkage-cov scorers).
        scores = {}
        for name in self._active_methods:
            scores[name] = self._scorers[name].score(data)
        return scores

    def _scores_to_matrix(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Stack score dict into (N, M) matrix in canonical order."""
        return np.column_stack([scores[m] for m in self._active_methods])

    def fit_combiner(
        self,
        id_data: dict[str, np.ndarray],
        ood_data: dict[str, np.ndarray],
        verbose: bool = True,
    ) -> LUNAPipeline:
        """Fit the combiner on ID (validation) vs OOD (rare/anomaly) data.

        Args:
            id_data: in-distribution data dict.
            ood_data: out-of-distribution / anomaly data dict.
            verbose: print AUROC and threshold info.

        Returns:
            self
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .fit_combiner().")

        id_scores = self.score(id_data)
        ood_scores = self.score(ood_data)

        X_id = self._scores_to_matrix(id_scores)
        X_ood = self._scores_to_matrix(ood_scores)

        X = np.vstack([X_id, X_ood])
        y = np.concatenate([
            np.zeros(X_id.shape[0], dtype=int),
            np.ones(X_ood.shape[0], dtype=int),
        ])

        # Instantiate combiner
        self._combiner = get_combiner(self._combiner_spec, **self._combiner_kwargs)

        # Cross-validation
        if verbose:
            cv_aurocs = self._combiner.cross_val_auroc(X, y)
            print(f"  [LUNA] Combiner CV AUROC = {cv_aurocs.mean():.4f} "
                  f"+/- {cv_aurocs.std():.4f}")

        # Full fit
        self._combiner.fit(X, y)
        ens = self._combiner.predict(X)
        ens_id = ens[:X_id.shape[0]]

        if verbose:
            full_auroc = roc_auc_score(y, ens)
            print(f"  [LUNA] Full-fit AUROC = {full_auroc:.4f}")

        # Calibrate threshold
        self._threshold = calibrate_threshold(ens_id, self.target_far)
        if verbose:
            n_fp = (ens_id > self._threshold).sum()
            print(f"  [LUNA] Threshold @ FAR={self.target_far:.1%}: {self._threshold:.4f} "
                  f"(val FP={n_fp}/{len(ens_id)})")

        self._combiner_fitted = True
        return self

    def predict(self, data: dict[str, np.ndarray]) -> dict:
        """Full prediction: scores + ensemble + threshold + flags.

        Args:
            data: input data dict.

        Returns:
            Dict with:
                "scores": dict of per-scorer arrays
                "ensemble_score": (N,) combined anomaly score
                "flagged": (N,) boolean array
                "threshold": float
        """
        if not self._combiner_fitted:
            raise RuntimeError("Call .fit_combiner() before .predict().")

        scores = self.score(data)
        X = self._scores_to_matrix(scores)
        ens = self._combiner.predict(X)
        flagged = apply_threshold(ens, self._threshold)

        return {
            "scores": scores,
            "ensemble_score": ens,
            "flagged": flagged,
            "threshold": self._threshold,
        }

    def family_analysis(
        self,
        id_data: dict[str, np.ndarray],
        ood_data: dict[str, np.ndarray],
    ) -> dict:
        """Per-family AUROC and detection analysis.

        Args:
            id_data: in-distribution data.
            ood_data: out-of-distribution data.

        Returns:
            Dict with per-family AUROC, detection counts, and flag arrays.
        """
        id_scores = self.score(id_data)
        ood_scores = self.score(ood_data)
        n_id = len(next(iter(id_scores.values())))
        n_ood = len(next(iter(ood_scores.values())))
        y = np.concatenate([np.zeros(n_id), np.ones(n_ood)])

        families = self.families
        result = {}

        for fam, methods in families.items():
            # Mean-rank ensemble within family
            rank_id = np.zeros(n_id)
            rank_ood = np.zeros(n_ood)
            for m in methods:
                full = np.concatenate([id_scores[m], ood_scores[m]])
                r = full.argsort().argsort().astype(float) / max(1, len(full) - 1)
                rank_id += r[:n_id]
                rank_ood += r[n_id:]
            rank_id /= len(methods)
            rank_ood /= len(methods)

            full_rank = np.concatenate([rank_id, rank_ood])
            auroc = roc_auc_score(y, full_rank)

            # Threshold at same FAR
            thr = calibrate_threshold(rank_id, self.target_far)
            flags = rank_ood > thr

            result[fam] = {
                "auroc": auroc,
                "detected": int(flags.sum()),
                "total": n_ood,
                "det_rate": flags.mean(),
                "flags": flags,
            }

        return result

    def report(
        self,
        id_data: dict[str, np.ndarray],
        ood_data: dict[str, np.ndarray],
        out_dir: str | Path,
        class_names: list[str] | None = None,
        meta: list[dict] | None = None,
    ) -> None:
        """Generate full report: CSV tables, LaTeX, and figures.

        Args:
            id_data: in-distribution data.
            ood_data: out-of-distribution data.
            out_dir: output directory.
            class_names: list of class name strings.
            meta: per-sample metadata dicts for the OOD data
                  (keys like "obj_id", "class").
        """
        from luna.report import (
            generate_anomaly_report,
            generate_category_ablation,
            write_latex_table,
        )
        from luna.viz import (
            plot_detection_map,
            plot_complementarity_map,
        )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compute scores and predictions
        pred = self.predict(ood_data)
        fam_analysis = self.family_analysis(id_data, ood_data)

        # Per-method flags for detection map
        id_scores = self.score(id_data)
        ood_scores = pred["scores"]
        method_flags = {}
        for m in self._active_methods:
            thr = calibrate_threshold(id_scores[m], self.target_far)
            method_flags[m] = ood_scores[m] > thr

        # Family flags
        fam_flags = {f: info["flags"] for f, info in fam_analysis.items()}

        # True classes from meta
        if meta is not None:
            true_classes = np.array([m.get("class", "?") for m in meta])
        else:
            true_classes = None

        # Anomaly report CSV
        report_df = generate_anomaly_report(
            scores=ood_scores,
            ensemble_score=pred["ensemble_score"],
            flags=pred["flagged"],
            meta=meta,
            class_names=class_names,
            logits=ood_data.get("logits"),
            family_flags=fam_flags,
        )
        report_df.to_csv(out_dir / "anomaly_report.csv", index=False)

        # Category ablation
        ablation_df = generate_category_ablation(fam_analysis)
        ablation_df.to_csv(out_dir / "category_ablation.csv", index=False)

        # LaTeX table
        write_latex_table(report_df, out_dir / "anomaly_report_published.tex")

        # Detection map
        n_ood = len(pred["ensemble_score"])
        plot_detection_map(
            method_flags=method_flags,
            ensemble_flags=pred["flagged"],
            true_classes=true_classes,
            family_flags=fam_flags,
            out_path=out_dir / "detection_map",
            target_far=self.target_far,
        )

        # Complementarity map
        if true_classes is not None:
            plot_complementarity_map(
                family_flags=fam_flags,
                true_classes=true_classes,
                out_path=out_dir / "complementarity_map",
            )

        print(f"  [LUNA] Report written to {out_dir}")
