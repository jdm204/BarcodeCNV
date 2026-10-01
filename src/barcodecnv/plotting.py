"""Production figures use observed data and fitted summaries, never CN truth."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from .genomic_plot import (
    MISSING_COLOR,
    genome_layout,
    interval_heatmap,
    label_genome_axis,
    marker_heatmap,
)

COLORS = ("#43a58e", "#ec861b", "#3e70b7", "#a454a0")


def _finish(fig, path, **options):
    """Return an open figure for notebooks; close only figures saved by this call."""
    if path is not None:
        try:
            fig.savefig(path, dpi=220, **options)
        finally:
            plt.close(fig)
    return fig


def plot_group_calls(bundle, calls, path=None, *, genome="hg38", chromosome_sizes=None):
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
        markers = np.arange(len(bundle.grid))
        layout = genome_layout(
            bundle.grid, genome=genome, chromosome_sizes=chromosome_sizes
        )
        labels = [
            f"G{g} ({b} barcodes, {n} cells)"
            for g, b, n in zip(calls.groups, calls.barcodes, calls.cells)
        ]
        fig, panels = plt.subplots(
            4,
            2,
            figsize=(16, max(1.8, len(labels) * 0.3) * 4),
            layout="constrained",
            gridspec_kw={"width_ratios": [1, 0.025]},
            squeeze=False,
        )
        axes, colorbars = panels[:, 0], panels[:, 1]
        for ax in colorbars[:2]:
            ax.set_visible(False)
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
            marker_heatmap(
                ax, 1 - alpha[:, :, None] * (1 - colors), bundle.grid, markers, layout
            )
            ax.set_title(title, fontsize=11)
        for ax, cax, values, title in zip(
            axes[2:],
            colorbars[2:],
            (calls.uncertain_cell_fraction, calls.pooled_conflict_cell_fraction),
            (
                f"Cells in uncertain barcodes (maximum barcode state probability < {calls.confident_threshold:g})",
                "Cells in confidently conflicting barcodes (different class from pooled group call)",
            ),
        ):
            im = marker_heatmap(
                ax, values, bundle.grid, markers, layout, cmap="Reds", vmin=0, vmax=1
            )
            fig.colorbar(im, cax=cax, label="Cell fraction")
            ax.set_title(title, fontsize=11)
        for ax in axes:
            ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
            label_genome_axis(ax, layout, coordinate_axis=ax is axes[0])
        fig.legend(
            handles=[
                Patch(facecolor=c, label=n)
                for c, n in zip(COLORS, ("Diploid", "Gain", "Loss", "CN-LOH"))
            ]
            + [Patch(facecolor=MISSING_COLOR, label="No displayed data")],
            loc="outside lower center",
            ncol=5,
            frameon=False,
        )
        fig.suptitle(
            "Group CN reporting: fixed membership; noise fitted to group pseudobulks; unresolved barcodes excluded",
            fontsize=13,
        )
    return _finish(fig, path, bbox_inches="tight", facecolor="white")


def plot_signal(result, path=None):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    null = result.null_scores
    if len(null):
        ax.hist(
            null[:, 0],
            bins=min(30, max(5, int(np.sqrt(len(null))))),
            color="#8ba8bd",
            label="Shuffled cell–barcode labels",
        )
        ax.axvline(
            result.scores.joint,
            color="#b54236",
            linewidth=2,
            label="Observed labels",
        )
        ax.set_title(
            f"Barcode-associated regional signal: permutation p = {result.pvalue:.4g}"
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
    return _finish(fig, path)


def plot_run(result, path=None, signal=None, *, genome="hg38", chromosome_sizes=None):
    """Vertically stacked modalities with identical barcode/genomic order."""
    data = result.bundle
    order = result.order
    labels = result.groups[order]
    p = result.fit.classes[order]
    classes = p.argmax(axis=-1)
    confidence = p.max(axis=-1)
    alpha = np.clip((confidence - 0.25) / 0.75, 0, 1)
    colors = np.array([to_rgb(c) for c in COLORS])[classes]
    rgb = 1 - alpha[:, :, None] * (1 - colors)
    layout = genome_layout(data.grid, genome=genome, chromosome_sizes=chromosome_sizes)
    panels = [
        (
            "Self-relative smoothed expression",
            result.self_expression[:, order].T,
            "RdBu_r",
            "Relative expression (log2)",
            "genes",
        ),
        (
            "External-reference smoothed expression",
            result.external_expression[:, order].T,
            "RdBu_r",
            "Relative expression (log2)",
            "genes",
        ),
    ]
    hf = result.hf_features
    if hf is not None:
        values = hf["x"][:, order].T.copy()
        values[hf["coverage"][:, order].T == 0] = np.nan
        panels.append(
            (
                f"Smoothed haplotype fraction ({hf['window_bp'] / 1e6:g} Mb windows)",
                values,
                "PuOr_r",
                "2 × HF − 1, centered across barcodes",
                "hf",
            )
        )
    cn_title = (
        "CN state; saturation = probability with CN and phase jointly integrated"
        if result.fit.posterior.phase.ndim == 2
        else "CN state; saturation = conditional probability"
    )
    panels.append((cn_title, rgb, None, None, "markers"))
    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(15, max(3.3, len(order) * 0.13) * len(panels)),
        gridspec_kw={"width_ratios": [1, 0.022]},
        layout="constrained",
        squeeze=False,
    )
    cn_support = (
        result.clone_calls is not None and result.clone_calls.method != "expression"
    )
    support_title = (
        "Self/HF\nsupport\n0.5–1.0"
        if cn_support and result.clone_calls.evidence.startswith("self_expression")
        else "CN/HF\nsupport\n0.5–1.0"
        if cn_support
        else "Split\nstability\n0.5–1.0"
    )
    stability = (
        result.clone_calls.stability.copy()
        if cn_support
        else np.minimum(result.weighted["stability"], result.core["stability"])
    )
    if result.hf_stability is not None:
        stability = np.minimum(stability, result.hf_stability)
    stability[result.groups == 0] = np.nan
    stability = stability[order]
    barcodes = [f"{data.barcodes[b]}  ({result.cells[b]} cells)" for b in order]
    for (ax, strip), (title, values, cmap, colorlabel, geometry) in zip(axes, panels):
        options = dict(cmap=cmap, vmin=-1, vmax=1) if cmap else {}
        if geometry == "hf":
            chrom = [c for c, _ in hf["regions"]]
            left = np.array([i for _, i in hf["regions"]]) * hf["stride_bp"]
            im = interval_heatmap(
                ax, values, chrom, left, left + hf["stride_bp"], layout, **options
            )
        else:
            markers = (
                data.gene_markers if geometry == "genes" else np.arange(len(data.grid))
            )
            im = marker_heatmap(ax, values, data.grid, markers, layout, **options)
        if cmap:
            fig.colorbar(
                im, ax=ax, location="bottom", shrink=0.35, pad=0.02, label=colorlabel
            )
        ax.set_title(title, fontsize=12)
        label_genome_axis(ax, layout, coordinate_axis=ax is axes[0, 0])
        ax.set_yticks(np.arange(len(order)), barcodes, fontsize=7)
        strip.imshow(
            stability[:, None],
            aspect="auto",
            cmap=plt.get_cmap("Greys").with_extremes(bad=MISSING_COLOR),
            vmin=0.5,
            vmax=1,
            interpolation="nearest",
        )
        strip.set_title(support_title, fontsize=8)
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
        ]
        + [Patch(facecolor=MISSING_COLOR, label="No displayed data")],
        loc="outside lower center",
        ncol=5,
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
            f"\nBarcode signal permutation p = {signal.pvalue:.4g}"
            if signal.pvalue is not None
            else "\nBarcode signal: not assessable with fewer than two lineage barcodes"
        )
    fig.suptitle(title, fontsize=14)
    return _finish(fig, path, bbox_inches="tight", facecolor="white")
