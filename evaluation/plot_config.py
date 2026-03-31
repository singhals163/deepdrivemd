"""Shared plotting configuration for all study figures.

Edit this file to tweak fonts, colors, sizes, and output format
for all figures at once. Individual plot scripts import from here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Directories ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Output format ────────────────────────────────────────────────
# Set to [".pdf"] for paper, [".png"] for quick preview, or both
OUTPUT_FORMATS = [".pdf", ".png"]

# ── Color palette ────────────────────────────────────────────────
BLUE = "#2196F3"
BLUE_LIGHT = "#64B5F6"
ORANGE = "#FF9800"
ORANGE_LIGHT = "#FFB74D"
GREEN = "#4CAF50"
GREEN_LIGHT = "#81C784"
GREEN_DARK = "#2E7D32"
RED = "#F44336"
RED_LIGHT = "#FFCDD2"
GRAY = "#BDBDBD"
GRAY_DARK = "#333333"

# ── Matplotlib RC params ─────────────────────────────────────────
RCPARAMS = {
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style():
    """Apply the shared style to matplotlib. Call at top of each plot script."""
    plt.rcParams.update(RCPARAMS)


def savefig(fig, name: str):
    """Save figure in all configured formats, overwriting existing files.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Filename stem without extension (e.g. "fig_study1a_latency").
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in OUTPUT_FORMATS:
        path = FIGURES_DIR / f"{name}{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  -> {path}")
