"""Production figures use observed data and fitted summaries, never CN truth."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

COLORS = ("#43a58e", "#ec861b", "#3e70b7", "#a454a0")


def plot_group_calls(bundle, calls, path):
    """Separate descriptive consensus, model probability, and member conflict."""
    if not len(calls.groups):
        fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
        ax.text(
            0.5,
            0.5,
            "No resolved groups; barcode CN calls remain available.",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
    else:
        markers = bundle.gene_markers
        chrom = np.asarray(bundle.grid.chrom)[markers]
        ticks = []
        names = []
        ends = []
        for c in dict.fromkeys(chrom):
            ix = np.flatnonzero(chrom == c)
            ticks.append((ix[0] + ix[-1]) / 2)
            names.append(c.removeprefix("chr"))
            ends.append(ix[-1] + 0.5)
        labels = [
            f"G{g} ({b} barcodes, {n} cells)"
            for g, b, n in zip(calls.groups, calls.barcodes, calls.cells)
        ]
        fig, axes = plt.subplots(
            4, 1, figsize=(16, max(1.8, len(labels) * 0.3) * 4), layout="constrained"
        )
        for ax, p, title in zip(
            axes[:2],
            (calls.consensus[:, markers], calls.pooled[:, markers]),
            (
                "Cell-weighted barcode consensus; saturation = weighted state probability",
                "Group pseudobulk CN/phase model; saturation = conditional state probability",
            ),
        ):
            colors = np.asarray([to_rgb(c) for c in COLORS])[p.argmax(-1)]
            alpha = np.clip((p.max(-1) - 0.25) / 0.75, 0, 1)
            ax.imshow(
                1 - alpha[:, :, None] * (1 - colors),
                aspect="auto",
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=11)
        for ax, values, title in zip(
            axes[2:],
            (calls.uncertain_cell_fraction, calls.pooled_conflict_cell_fraction),
            (
                f"Cells in uncertain barcodes (maximum barcode state probability < {calls.confident_threshold:g})",
                "Cells in confidently conflicting barcodes (different class from pooled group call)",
            ),
        ):
            im = ax.imshow(
                values[:, markers],
                aspect="auto",
                interpolation="nearest",
                cmap="Reds",
                vmin=0,
                vmax=1,
            )
            fig.colorbar(im, ax=ax, label="Cell fraction", fraction=0.02, pad=0.01)
            ax.set_title(title, fontsize=11)
        for ax in axes:
            ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
            ax.set_xticks(ticks, names, fontsize=8)
            ax.set_xlabel("Chromosome")
            for end in ends[:-1]:
                ax.axvline(end, color="black", linewidth=0.3, alpha=0.3)
        fig.legend(
            handles=[
                Patch(facecolor=c, label=n)
                for c, n in zip(COLORS, ("Diploid", "Gain", "Loss", "CN-LOH"))
            ],
            loc="outside lower center",
            ncol=4,
            frameon=False,
        )
        fig.suptitle(
            "Group CN reporting: fixed membership; noise fitted to group pseudobulks; unresolved barcodes excluded",
            fontsize=13,
        )
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_signal(result, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    null = result["null_scores"]
    if len(null):
        ax.hist(
            null[:, 0],
            bins=min(30, max(5, int(np.sqrt(len(null))))),
            color="#8ba8bd",
            label="Shuffled cell–barcode labels",
        )
        ax.axvline(
            result["scores"]["joint"],
            color="#b54236",
            linewidth=2,
            label="Observed labels",
        )
        ax.set_title(
            f"Barcode-associated regional signal: permutation p = {result['pvalue']:.4g}"
        )
        ax.legend(fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "Not assessable: fewer than two lineage barcodes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )
    ax.set(xlabel="Regional count statistic", ylabel="Permutations")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_run(result, path, signal=None):
    """Vertically stacked modalities with identical barcode/genomic order."""
    data = result["bundle"]
    order = result["order"]
    labels = result["groups"][order]
    p = result["fit"].classes[:, data.gene_markers][order]
    classes = p.argmax(axis=-1)
    confidence = p.max(axis=-1)
    alpha = np.clip((confidence - 0.25) / 0.75, 0, 1)
    colors = np.array([to_rgb(c) for c in COLORS])[classes]
    rgb = 1 - alpha[:, :, None] * (1 - colors)
    chrom = np.array([data.grid.chrom[t] for t in data.gene_markers])
    ticks = []
    names = []
    ends = []
    for c in dict.fromkeys(chrom):
        ix = np.flatnonzero(chrom == c)
        ticks.append((ix[0] + ix[-1]) / 2)
        names.append(c.removeprefix("chr"))
        ends.append(ix[-1] + 0.5)
    panels = [
        (
            "Self-relative smoothed expression",
            result["self_expression"][:, order].T,
            "RdBu_r",
            "Relative expression (log2)",
        ),
        (
            "External-reference smoothed expression",
            result["external_expression"][:, order].T,
            "RdBu_r",
            "Relative expression (log2)",
        ),
    ]
    hf = result.get("hf_features")
    if hf is not None:
        lookup = {key: i for i, key in enumerate(hf["regions"])}
        indices = np.array(
            [
                lookup.get((c, int(data.grid.start[t] - 1) // hf["stride_bp"]), -1)
                for c, t in zip(chrom, data.gene_markers)
            ]
        )
        values = hf["x"][np.maximum(indices, 0)][:, order].T.copy()
        coverage = hf["coverage"][np.maximum(indices, 0)][:, order].T
        values[:, indices < 0] = np.nan
        values[coverage == 0] = np.nan
        panels.append(
            (
                f"Smoothed haplotype fraction ({hf['window_bp'] / 1e6:g} Mb windows)",
                values,
                "PuOr_r",
                "2 × HF − 1, centered across barcodes",
            )
        )
    cn_title = (
        "CN state; saturation = probability with CN and phase jointly integrated"
        if result["fit"].posterior.phase.ndim == 2
        else "CN state; saturation = conditional probability"
    )
    panels.append((cn_title, rgb, None, None))
    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(15, max(3.3, len(order) * 0.13) * len(panels)),
        gridspec_kw={"width_ratios": [1, 0.022]},
        layout="constrained",
        squeeze=False,
    )
    stability = np.minimum(result["weighted"]["stability"], result["core"]["stability"])
    if result.get("hf_stability") is not None:
        stability = np.minimum(stability, result["hf_stability"])
    stability = stability[order]
    barcodes = [f"{data.barcodes[b]}  ({result['cells'][b]} cells)" for b in order]
    for (ax, strip), (title, values, cmap, colorlabel) in zip(axes, panels):
        if cmap:
            im = ax.imshow(
                values,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-1,
                vmax=1,
            )
            fig.colorbar(
                im, ax=ax, location="bottom", shrink=0.35, pad=0.02, label=colorlabel
            )
        else:
            ax.imshow(values, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(ticks, names, fontsize=8, rotation=90)
        ax.set_xlabel("Chromosome")
        ax.set_yticks(np.arange(len(order)), barcodes, fontsize=7)
        for boundary in ends[:-1]:
            ax.axvline(boundary, color="black", linewidth=0.3, alpha=0.25)
        strip.imshow(
            stability[:, None],
            aspect="auto",
            cmap="Greys",
            vmin=0.5,
            vmax=1,
            interpolation="nearest",
        )
        strip.set_title("Split\nstability\n0.5–1.0", fontsize=8)
        strip.set_xticks([])
        strip.set_yticks([])
        for target in (ax, strip):
            for boundary in np.flatnonzero(labels[1:] != labels[:-1]) + 0.5:
                target.axhline(boundary, color="black", linewidth=0.8)
        for group in dict.fromkeys(labels):
            ix = np.flatnonzero(labels == group)
            strip.text(
                1.1,
                ix.mean(),
                "Unresolved" if group == 0 else f"G{group}",
                fontsize=8,
                va="center",
                transform=strip.get_yaxis_transform(),
            )
    fig.legend(
        handles=[
            Patch(facecolor=c, label=n)
            for c, n in zip(COLORS, ("Diploid", "Gain", "Loss", "CN-LOH"))
        ],
        loc="outside lower center",
        ncol=4,
        fontsize=10,
        frameon=False,
    )
    title = (
        "BarcodeCNV: expression, haplotype fractions and conditional CN inference"
        if hf is not None
        else "BarcodeCNV: expression and conditional CN inference"
    )
    if signal is not None:
        title += (
            f"\nBarcode signal permutation p = {signal['pvalue']:.4g}"
            if signal["pvalue"] is not None
            else "\nBarcode signal: not assessable with fewer than two lineage barcodes"
        )
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
