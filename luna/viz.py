"""Visualization functions for LUNA anomaly detection reports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def plot_detection_map(
    method_flags: dict[str, np.ndarray],
    ensemble_flags: np.ndarray,
    true_classes: np.ndarray | None,
    family_flags: dict[str, np.ndarray],
    out_path: str | Path,
    target_far: float = 0.01,
    cv_auroc: tuple[float, float] | None = None,
) -> None:
    """Per-anomaly x per-method detection heatmap.

    Three panels:
      Top:    per-method detection (sorted by AUROC-proxy = detection rate)
      Middle: per-family mini-ensemble detection
      Bottom: bar showing how many families caught each anomaly

    Args:
        method_flags: {method_name: (N,) bool array}
        ensemble_flags: (N,) bool array from the combiner
        true_classes: (N,) string array of true class labels (or None)
        family_flags: {family_name: (N,) bool array}
        out_path: base path (extensions .pdf and .png are appended)
        target_far: for title annotation
        cv_auroc: (mean, std) of CV AUROC for title (optional)
    """
    out_path = Path(out_path)
    n_rare = len(ensemble_flags)
    method_names = sorted(method_flags.keys(), key=lambda m: -method_flags[m].sum())
    fam_names = list(family_flags.keys())

    # Build matrices
    n_rows = 1 + len(method_names)
    mat = np.zeros((n_rows, n_rare), dtype=int)
    mat[0] = ensemble_flags.astype(int)
    for j, m in enumerate(method_names, start=1):
        mat[j] = method_flags[m].astype(int)

    fam_mat = np.stack([family_flags[f].astype(int) for f in fam_names], axis=0)
    n_fams_per_sample = fam_mat.sum(axis=0)

    # Sort columns by class then by total hits
    if true_classes is not None:
        sort_idx = np.lexsort((-mat.sum(0), true_classes))
        cls_sorted = true_classes[sort_idx]
    else:
        sort_idx = np.argsort(-mat.sum(0))
        cls_sorted = None

    mat_sorted = mat[:, sort_idx]
    fam_mat_sorted = fam_mat[:, sort_idx]
    n_fams_sorted = n_fams_per_sample[sort_idx]

    # Plot (paper standard: final physical size, serif 9 pt, no titles)
    from .style import paper_theme as _pt
    _ctx = _pt(False); _ctx.__enter__()
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(PAPER_WIDE, n_rows * 0.135 + 2.1),
        gridspec_kw={"height_ratios": [n_rows, max(len(fam_names), 1), 1.5]},
        sharex=True,
    )
    cmap = mcolors.ListedColormap(["#f0f0f0", "#1a6b5e"])

    ax_top.imshow(mat_sorted, aspect="auto", cmap=cmap, interpolation="nearest")
    yt = ["Ensemble"] + method_names
    ax_top.set_yticks(range(n_rows))
    ax_top.set_yticklabels(yt, fontsize=6)
    ax_top.axhline(0.5, color="royalblue", lw=1.4, ls="--", alpha=0.7)

    ax_mid.imshow(fam_mat_sorted, aspect="auto", cmap=cmap, interpolation="nearest")
    ax_mid.set_yticks(range(len(fam_names)))
    ax_mid.set_yticklabels([f"{f} ensemble" for f in fam_names], fontsize=6)

    # Bottom bar
    if cls_sorted is not None:
        unique_cls = sorted(set(cls_sorted))
        cm = plt.cm.Set2(np.linspace(0, 1, max(len(unique_cls), 1)))
        cls_color = {c: cm[i] for i, c in enumerate(unique_cls)}
        bar_colors = [cls_color[c] for c in cls_sorted]
        ax_bot.bar(range(n_rare), n_fams_sorted, color=bar_colors, width=1.0,
                   edgecolor="none")
        ax_bot.legend(
            handles=[Patch(facecolor=cls_color[c], label=c) for c in unique_cls],
            loc="upper center", bbox_to_anchor=(0.5, -0.45),
            ncol=min(6, len(unique_cls)), fontsize=6, frameon=False,
            handlelength=1.2, columnspacing=0.9,
        )
    else:
        ax_bot.bar(range(n_rare), n_fams_sorted, color="steelblue", width=1.0,
                   edgecolor="none")

    ax_bot.set_ylabel("# families\ncaught", fontsize=8)
    # no xlabel: the caption describes the axis; keeps the class legend clear

    fig.tight_layout()
    fig.savefig(str(out_path) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _ctx.__exit__(None, None, None)


def plot_complementarity_map(
    family_flags: dict[str, np.ndarray],
    true_classes: np.ndarray,
    out_path: str | Path,
) -> None:
    """Stacked bar: per true class, how many caught by 0/1/2/.../K families.

    Args:
        family_flags: {family_name: (N,) bool array}
        true_classes: (N,) string array of true class labels
        out_path: base path (.pdf and .png appended)
    """
    out_path = Path(out_path)
    fam_names = list(family_flags.keys())
    fam_mat = np.stack([family_flags[f].astype(int) for f in fam_names], axis=0)
    n_fams_per_sample = fam_mat.sum(axis=0)

    cls_levels = sorted(set(true_classes))
    n_fam_levels = len(fam_names) + 1

    fig, ax = plt.subplots(figsize=(PAPER_COL, PAPER_COL * 0.72))
    width = 0.7
    bottoms = np.zeros(len(cls_levels))
    colors_k = plt.cm.viridis(np.linspace(0.15, 0.9, n_fam_levels))

    for k in range(n_fam_levels):
        heights = []
        for c in cls_levels:
            m = true_classes == c
            heights.append(int(((n_fams_per_sample == k) & m).sum()))
        ax.bar(cls_levels, heights, bottom=bottoms, color=colors_k[k],
               label=f"{k} family/ies", width=width, edgecolor="white", lw=0.5)
        bottoms += np.array(heights, dtype=float)

    ax.set_ylabel("# samples")
    ax.set_xlabel("True class")
    ax.legend(title="caught by", loc="upper right", fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(str(out_path) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_diagnostic(
    id_pvals: np.ndarray,
    ood_pvals: np.ndarray,
    out_path: str | Path,
    alpha_grid: np.ndarray | None = None,
) -> None:
    """ECDF of conformal p-values (ID vs OOD) + coverage diagnostic.

    Args:
        id_pvals: (N_id,) marginal or Mondrian p-values for ID data.
        ood_pvals: (N_ood,) p-values for OOD data.
        out_path: output file path.
    """
    out_path = Path(out_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ECDF
    for vals, name, color in [
        (id_pvals, "In-distribution", "#2c7fb8"),
        (ood_pvals, "OOD", "#d7301f"),
    ]:
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        ax1.plot(x, y, lw=2.0, label=name, color=color)
    ax1.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Uniform (ideal ID)")
    ax1.set_xlabel("Conformal p-value")
    ax1.set_ylabel("ECDF")
    ax1.set_title("p-value distribution")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    # Coverage vs nominal
    if alpha_grid is None:
        alpha_grid = np.linspace(0.005, 0.20, 30)
    emp_cov = np.array([(id_pvals > a).mean() for a in alpha_grid])
    nom_cov = 1.0 - alpha_grid
    det_rate = np.array([(ood_pvals < a).mean() for a in alpha_grid])

    ax2.plot(nom_cov, emp_cov, "o-", lw=1.8, color="#2c7fb8", label="Empirical ID coverage")
    ax2.plot([0.7, 1.0], [0.7, 1.0], "k--", lw=1.0, alpha=0.5)
    ax2r = ax2.twinx()
    ax2r.plot(nom_cov, det_rate, "s-", lw=1.5, color="#d7301f", alpha=0.8,
              label="OOD detection rate")
    ax2r.set_ylabel("OOD detection rate", color="#d7301f")
    ax2r.set_ylim(0, 1.02)
    ax2.set_xlabel("Nominal coverage (1 - alpha)")
    ax2.set_ylabel("Empirical ID coverage")
    ax2.set_title("Conformal calibration")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_predset_sizes(
    sizes_id: dict[float, np.ndarray],
    sizes_ood: dict[float, np.ndarray],
    out_path: str | Path,
    n_classes: int | None = None,
) -> None:
    """Prediction set size distributions across alpha values.

    Args:
        sizes_id: {alpha: (N_id,) array of set sizes}
        sizes_ood: {alpha: (N_ood,) array of set sizes}
        out_path: output file path.
        n_classes: total number of classes (for x-axis).
    """
    out_path = Path(out_path)
    alphas = sorted(sizes_id.keys())
    if n_classes is None:
        n_classes = max(
            max(s.max() for s in sizes_id.values()),
            max(s.max() for s in sizes_ood.values()),
        )

    fig, axes = plt.subplots(
        1, len(alphas), figsize=(3.4 * len(alphas), 4), sharey=True,
    )
    if len(alphas) == 1:
        axes = [axes]

    for ax, a in zip(axes, alphas):
        bins = np.arange(0.5, n_classes + 1.5, 1)
        ax.hist(sizes_id[a], bins=bins, alpha=0.65, density=True,
                label="ID", color="#2c7fb8", edgecolor="white")
        ax.hist(sizes_ood[a], bins=bins, alpha=0.65, density=True,
                label="OOD", color="#d7301f", edgecolor="white")
        ax.set_xlabel("Prediction set size")
        ax.set_title(f"alpha = {a:.2f}, target cov. {1-a:.2f}")
        ax.set_xticks(range(1, n_classes + 1))
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("density")
            ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)




# ════════════════════════════════════════════════════════════════════
# Extra plots — watermelon palette, paper fonts, light + dark themes.
# Each function emits both a light (paper: _light.pdf/.png) and dark
# (slides: _dark.png) version via luna.style.save_fig.
# ════════════════════════════════════════════════════════════════════
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)
from luna.style import (themed, theme_colors, save_fig, cmap,
                        ACCENT, ACCENT_DEEP, QUALITATIVE,
                        paper_theme, PAPER_COL, PAPER_WIDE)

THEMES = (False, True)   # light, dark


def plot_roc_pr(id_scores, ood_scores, out_dir, themes=THEMES):
    """ROC and PR curves for the ensemble anomaly score (ID=0, OOD=1)."""
    out_dir = Path(out_dir)
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    s = np.r_[id_scores, ood_scores]
    fpr, tpr, _ = roc_curve(y, s); auroc = roc_auc_score(y, s)
    prec, rec, _ = precision_recall_curve(y, s); ap = average_precision_score(y, s)
    for dark in themes:
        c = theme_colors(dark)
        with themed(dark):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, color=ACCENT, label=f"AUROC = {auroc:.4f}")
            ax.plot([0, 1], [0, 1], "--", color=c["fg_dim"], lw=1.3)
            ax.axvline(0.01, color=ACCENT_DEEP, ls=":", lw=1.6, label="1% FAR")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("ROC — ensemble anomaly score"); ax.legend(loc="lower right")
            save_fig(fig, out_dir / "roc_ensemble", dark)
        with themed(dark):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(rec, prec, color=ACCENT_DEEP, label=f"AP = {ap:.4f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("Precision–Recall — ensemble"); ax.legend(loc="lower left")
            save_fig(fig, out_dir / "pr_ensemble", dark)
    return {"auroc": float(auroc), "ap": float(ap)}


def plot_family_auroc_bar(family_analysis, out_dir, themes=THEMES):
    """Bar chart of per-family AUROC."""
    out_dir = Path(out_dir)
    fams = list(family_analysis); vals = [family_analysis[f]["auroc"] for f in fams]
    order = np.argsort(vals)[::-1]
    fams = [fams[i] for i in order]; vals = [vals[i] for i in order]
    for dark in themes:
        c = theme_colors(dark)
        with themed(dark):
            fig, ax = plt.subplots(figsize=(max(6, len(fams) * 1.3), 5))
            bars = ax.bar(fams, vals, color=ACCENT, edgecolor=c["fg"], lw=0.8)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=12, color=c["fg"])
            ax.set_ylim(min(0.5, min(vals) - 0.05), 1.0)
            ax.set_ylabel("AUROC"); ax.set_title("Per-family anomaly AUROC")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            save_fig(fig, out_dir / "auroc_bar", dark)


def plot_umap(train_emb, ood_emb, train_labels, class_names, out_dir,
              max_train=4000, themes=THEMES):
    """UMAP of train embeddings (by class) + OOD overlay. Plotly HTML + dual PNG."""
    out_dir = Path(out_dir)
    try:
        import umap
    except Exception as e:
        print(f"  [umap] skipped ({e})"); return
    rng = np.random.default_rng(42)
    n = min(max_train, len(train_emb))
    idx = rng.choice(len(train_emb), n, replace=False)
    Z = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(
        np.vstack([train_emb[idx], ood_emb]))
    Ztr, Zood = Z[:n], Z[n:]
    tl = np.asarray(train_labels)[idx]
    for dark in themes:
        c = theme_colors(dark)
        with themed(dark):
            fig, ax = plt.subplots(figsize=(8, 7))
            for ci, name in enumerate(class_names):
                m = tl == ci
                if m.any():
                    ax.scatter(Ztr[m, 0], Ztr[m, 1], s=6,
                               color=QUALITATIVE[ci % len(QUALITATIVE)],
                               label=name, alpha=0.55, edgecolors="none")
            ax.scatter(Zood[:, 0], Zood[:, 1], s=40, color=c["fg"], marker="x",
                       linewidths=1.5, label="rare / OOD")
            ax.legend(markerscale=1.6, loc="best", framealpha=0.85)
            ax.set_title("UMAP — embeddings"); ax.set_xticks([]); ax.set_yticks([])
            save_fig(fig, out_dir / "umap", dark)
    try:
        import plotly.graph_objects as go
        fig2 = go.Figure()
        for ci, name in enumerate(class_names):
            m = tl == ci
            if m.any():
                fig2.add_trace(go.Scattergl(
                    x=Ztr[m, 0], y=Ztr[m, 1], mode="markers", name=name,
                    marker=dict(size=5, opacity=0.6,
                                color=QUALITATIVE[ci % len(QUALITATIVE)])))
        fig2.add_trace(go.Scattergl(x=Zood[:, 0], y=Zood[:, 1], mode="markers",
                       name="rare/OOD", marker=dict(size=10, color="black", symbol="x")))
        fig2.update_layout(title="UMAP — embeddings (interactive)",
                           template="plotly_white", font=dict(size=15))
        fig2.write_html(str(out_dir / "umap.html"))
    except Exception as e:
        print(f"  [umap html] skipped ({e})")


def plot_confusion(cm_counts, class_names, title, out_dir, stem="confusion_test",
                   themes=THEMES):
    """Watermelon row-normalized confusion matrix, light + dark, paper fonts."""
    out_dir = Path(out_dir)
    N = len(class_names)
    cm_counts = np.asarray(cm_counts)
    row = cm_counts.sum(1, keepdims=True).clip(min=1)
    norm = cm_counts / row
    acc = np.trace(cm_counts) / cm_counts.sum() * 100
    macro = np.diag(norm).mean() * 100
    for dark in themes:
        c = theme_colors(dark)
        with paper_theme(dark):
            fig, ax = plt.subplots(figsize=(PAPER_COL, PAPER_COL * 0.92),
                                   facecolor=c["bg"])
            ax.set_facecolor(c["bg"])
            ax.imshow(norm, cmap=cmap(dark), vmin=0, vmax=1)
            for i in range(N):
                for j in range(N):
                    if cm_counts[i, j]:
                        hot = norm[i, j] > (0.4 if dark else 0.55)
                        tc = "white" if hot else (c["fg_dim"] if dark else "black")
                        ax.text(j, i - 0.13, f"{norm[i,j]*100:.0f}%", ha="center",
                                va="center", color=tc, fontsize=7, fontweight="bold")
                        ax.text(j, i + 0.24, f"({int(cm_counts[i,j])})", ha="center",
                                va="center", color=tc, fontsize=5.5)
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                             edgecolor=c["border"], lw=1.2))
            ax.set_xticks(range(N)); ax.set_yticks(range(N))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
            save_fig(fig, out_dir / stem, dark)   # no in-figure title: caption carries it
