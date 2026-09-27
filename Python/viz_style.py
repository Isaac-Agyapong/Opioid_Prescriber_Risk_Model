"""Chart style for the early-warning project: warm paper background, charcoal ink, violet for the model,
grey for the baseline rule, amber for risk. (Deliberately different from the other two projects.)"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

VIOLET, VIOLET_LIGHT, AMBER, GREY, INK, INK_2 = "#5B3FD1", "#B9AEF0", "#E08A00", "#A8A29E", "#1C1917", "#57534E"
PAPER, GRID = "#FBF8F3", "#E7E2DA"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "Image"
IMAGE_DIR.mkdir(exist_ok=True)


def apply():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (9, 4.8),
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 26, "legend.frameon": False, "lines.linewidth": 2.4,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=10.5, va="bottom")


def pct(ax, axis="y", decimals=0):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1, decimals=decimals))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / f"{name}.png", bbox_inches="tight")
