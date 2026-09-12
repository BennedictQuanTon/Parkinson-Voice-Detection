"""Figure defaults and a single save helper.

Every figure goes out as both PNG (for the notebook) and PDF (for the paper),
so the write-up never depends on re-running a notebook to get vector output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Okabe-Ito: distinguishable under the common forms of colour vision deficiency.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",
]

#: Fixed across every figure in the project, so colour means the same thing.
GROUP_COLOURS = {
    "PD": "#D55E00",
    "eHC": "#0072B2",
    "yHC": "#009E73",
}
GROUP_MARKERS = {"PD": "o", "eHC": "s", "yHC": "^"}

ARM_COLOURS = {
    "A_recording": "#D55E00",
    "B_folder_path": "#E69F00",
    "C_subject_key": "#0072B2",
}
ARM_LINESTYLES = {
    "A_recording": "-",
    "B_folder_path": "--",
    "C_subject_key": "-.",
}


def use_house_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "pdf.fonttype": 42,  # embed as TrueType so editors can open it
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, name: str, caption: str | None = None) -> Path:
    """Save ``fig`` as ``figures/<name>.png`` and ``.pdf``; return the PNG path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{name}.png"
    fig.savefig(png)
    fig.savefig(FIGURES / f"{name}.pdf")
    if caption:
        (FIGURES / f"{name}.txt").write_text(caption.strip() + "\n")
    print(f"  saved {png.name} (+ .pdf)")
    return png


def raincloud(ax, groups: dict[str, list[float]], ylabel: str) -> None:
    """Strip plot over a box, for sample sizes where a box alone hides too much."""
    import numpy as np

    for i, (name, values) in enumerate(groups.items()):
        values = np.asarray(values, dtype=float)
        colour = GROUP_COLOURS.get(name, OKABE_ITO[i])
        ax.boxplot(
            values,
            positions=[i],
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": colour, "alpha": 0.25, "edgecolor": colour},
            medianprops={"color": colour, "linewidth": 2},
            whiskerprops={"color": colour},
            capprops={"color": colour},
        )
        jitter = np.random.default_rng(0).normal(0, 0.06, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=26,
            color=colour,
            marker=GROUP_MARKERS.get(name, "o"),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=f"{name} (n={len(values)})",
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups.keys())
    ax.set_ylabel(ylabel)
