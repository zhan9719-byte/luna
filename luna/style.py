"""Publication / presentation plot styling for LUNA figures.

- Watermelon colour scheme (light + dark colormaps, accent + qualitative palette).
- Large fonts tuned for papers.
- A `dark` flag for presentation slides (dark background, transparent dark PNG).

Use `themed(dark)` as a context manager, `theme_colors(dark)` for fg/bg/accent, and
`save_fig(fig, path_stem, dark)` to write the right files. The convention across the
pipeline is to emit BOTH a light version (`_light.pdf` + `_light.png`, for papers)
and a dark version (`_dark.png`, transparent, for slides).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Watermelon palette ────────────────────────────────────────────────
WATERMELON_LIGHT = LinearSegmentedColormap.from_list("watermelon_light", [
    "#fdf2f8", "#fce7f3", "#fbcfe8", "#f472b6", "#e11d6a", "#9f1239"])
WATERMELON_DARK = LinearSegmentedColormap.from_list("watermelon_dark", [
    "#1a0a10", "#4c0f27", "#9f1239", "#e11d6a", "#f472b6", "#fce7f3"])

ACCENT = "#e11d6a"          # primary single-series colour (watermelon pink-red)
ACCENT_DEEP = "#9f1239"     # secondary

# Qualitative palette for per-class scatter/lines (watermelon -> teal spread).
QUALITATIVE = ["#9f1239", "#e11d6a", "#f472b6", "#f59e0b", "#10b981",
               "#0ea5e9", "#6366f1", "#a855f7", "#64748b", "#14b8a6"]

def cmap(dark: bool = False):
    return WATERMELON_DARK if dark else WATERMELON_LIGHT


def theme_colors(dark: bool = False) -> dict:
    if dark:
        return dict(bg="#0d0d0d", fg="white", fg_dim="#aaaaaa",
                    grid="#2a2a2a", border="white", accent=ACCENT)
    return dict(bg="white", fg="black", fg_dim="#555555",
                grid="#dddddd", border="#222222", accent=ACCENT)


# ── Fonts: large, paper-grade ─────────────────────────────────────────
def _font_rc(dark: bool, scale: float) -> dict:
    fg = "white" if dark else "black"
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 16 * scale,
        "axes.titlesize": 19 * scale,
        "axes.labelsize": 17 * scale,
        "xtick.labelsize": 14 * scale,
        "ytick.labelsize": 14 * scale,
        "legend.fontsize": 14 * scale,
        "figure.titlesize": 20 * scale,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.4,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,   # serif fonts lack U+2212; use ASCII hyphen
        "text.color": fg, "axes.labelcolor": fg,
        "xtick.color": fg, "ytick.color": fg, "axes.edgecolor": fg,
    }


@contextmanager
def themed(dark: bool = False, scale: float = 1.0):
    """Apply the LUNA paper/slide theme within a context."""
    base = "dark_background" if dark else "default"
    with plt.style.context(base):
        with mpl.rc_context(_font_rc(dark, scale)):
            yield


def save_fig(fig, path_stem, dark: bool = False):
    """Save a figure with the right suffix/format for its theme.

    light -> <stem>_light.pdf + <stem>_light.png   (papers)
    dark  -> <stem>_dark.png  (transparent, slides)
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    bg = "#0d0d0d" if dark else "white"
    if dark:
        fig.savefig(stem.parent / f"{stem.name}_dark.png", dpi=300,
                    bbox_inches="tight", facecolor=bg)
    else:
        fig.savefig(stem.parent / f"{stem.name}_light.pdf", bbox_inches="tight", facecolor=bg)
        fig.savefig(stem.parent / f"{stem.name}_light.png", dpi=300,
                    bbox_inches="tight", facecolor=bg)
    plt.close(fig)


# ── Paper-standard figures ────────────────────────────────────────────
# Author every figure at its FINAL physical size so in-figure text is the same
# size in print everywhere (aastex631 two-column: body 10 pt):
#   single column -> PAPER_COL wide;  full width (figure*) -> PAPER_WIDE.
# No in-figure titles — captions carry them.
PAPER_COL = 3.5     # inches (aastex \columnwidth)
PAPER_WIDE = 7.1    # inches (aastex \textwidth)


def _paper_rc(dark: bool) -> dict:
    fg = "white" if dark else "black"
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9,      # unused: paper figures carry no titles
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
        "text.color": fg, "axes.labelcolor": fg,
        "xtick.color": fg, "ytick.color": fg, "axes.edgecolor": fg,
    }


@contextmanager
def paper_theme(dark: bool = False):
    """Fixed-final-size paper figures: 9 pt text, no titles (see PAPER_COL/WIDE)."""
    base = "dark_background" if dark else "default"
    with plt.style.context(base):
        with mpl.rc_context(_paper_rc(dark)):
            yield
