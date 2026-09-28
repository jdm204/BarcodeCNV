"""Render the frozen 5% depth-mixture evaluation; no inference or truth fitting.

From barcodecnv/: uv run python docs/figures/render_calling_evaluation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(HERE / "calling-evaluation.csv", dtype={"depth": str})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout="constrained")
    for panel, label, color in [
        ("histories", "Five histories", "#26776d"),
        ("focal_bam", "Focal: held-out fitted reference", "#c46421"),
        ("focal_oracle", "Focal: generating reference", "#4c6d9c"),
    ]:
        order = ["500", "5000", "20000" if panel == "histories" else "full"]
        rows = data[data.panel == panel].set_index("depth").loc[order]
        axes[0].plot(
            range(3),
            100 * rows.diploid_false_alteration,
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(
            range(3), 100 * rows.events_detected / rows.events_tested, "o-", color=color
        )
    for ax in axes:
        ax.set_xticks(range(3), ["Low", "Medium", "High"])
        ax.set_xlabel("Molecular depth")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Truth-diploid genes called altered (%)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("True events detected (%)")
    axes[1].set_ylim(0, 105)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=1,
        frameon=False,
    )
    fig.savefig(HERE / "calling-evaluation.png", dpi=320)
    plt.close(fig)
    saved = np.load(HERE / "rbl1-group-calls.npz")
    p = saved["probabilities"]
    chrom = saved["chromosome"]
    colors = np.array([to_rgb(c) for c in ["#44a58f", "#e98116", "#416eb0", "#a54ca3"]])
    confidence = np.clip((p.max(-1) - 0.25) / 0.75, 0, 1)
    rgb = 1 - confidence[:, :, None] * (1 - colors[p.argmax(-1)])
    starts = np.r_[0, np.flatnonzero(chrom[1:] != chrom[:-1]) + 1]
    stops = np.r_[starts[1:], len(chrom)]
    fig, ax = plt.subplots(figsize=(10, 2.6), layout="constrained")
    ax.imshow(rgb, interpolation="nearest", aspect="auto")
    for j in starts[1:]:
        ax.axvline(j - 0.5, color="black", alpha=0.25, lw=0.6)
    ax.axhline(0.5, color="black", lw=0.8)
    ax.set_yticks(
        range(len(p)),
        [f"G{g}: {n} cells" for g, n in zip(saved["groups"], saved["cells"])],
    )
    ax.set_xticks(
        (starts + stops - 1) / 2,
        [c.removeprefix("chr") for c in chrom[starts]],
        fontsize=8,
    )
    ax.set_xlabel("Chromosome (columns are retained genes)")
    ax.set_title("RBL1 group pseudobulks | 5% depth outlier mixture", fontsize=12)
    fig.legend(
        handles=[
            Patch(color=c, label=n)
            for c, n in zip(colors, ["Diploid", "Gain", "Loss", "CN-LOH"])
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.savefig(HERE / "rbl1-group-calls.png", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
