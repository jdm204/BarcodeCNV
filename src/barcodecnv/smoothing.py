"""InferCNV 1.28 preprocessing through step 14, ported from the checked Julia adapter.

See Broad Institute infercnv/R/inferCNV_ops.R and the checked-in R fixtures.
No denoising, reference-variation filtering, HMM or gene selection occurs here.
"""

import numpy as np
from scipy.ndimage import convolve1d


def infercnv_smoothing(
    counts,
    chromosomes,
    *,
    reference=None,
    window_length=101,
    max_centered_threshold=3.0,
):
    counts = np.asarray(counts, dtype=float)
    chrom = np.asarray(chromosomes, dtype=str)
    if counts.ndim != 2 or min(counts.shape) < 1 or len(chrom) != counts.shape[0]:
        raise ValueError("counts must be gene × sample with one chromosome per gene")
    if (
        not isinstance(window_length, int)
        or window_length < 1
        or window_length % 2 != 1
    ):
        raise ValueError("window length must be positive and odd")
    if not 0 < max_centered_threshold < np.inf:
        raise ValueError("centering threshold must be finite and positive")
    if reference is None:
        data = counts.copy()
        refs = slice(None)
    else:
        ref = np.asarray(reference, dtype=float)
        if ref.ndim == 1:
            ref = ref[:, None]
        if ref.ndim != 2 or ref.shape[0] != len(chrom) or ref.shape[1] < 1:
            raise ValueError("reference must share the gene axis")
        data = np.column_stack((counts, ref))
        refs = slice(counts.shape[1], None)
    if np.any(~np.isfinite(data)) or np.any(data < 0):
        raise ValueError("counts and reference must be finite and nonnegative")
    keep = (data > 0).any(axis=1)
    if not keep.any():
        raise ValueError("no expressed genes")
    data = data[keep]
    selected = chrom[keep]
    libraries = data.sum(axis=0)
    if np.any(libraries <= 0):
        raise ValueError("each sample needs a positive library")
    data = np.log2(1 + data * (np.median(libraries) / libraries))
    data -= data[:, refs].mean(axis=1, keepdims=True)
    data = np.clip(data, -max_centered_threshold, max_centered_threshold)
    radius = window_length // 2
    kernel = radius + 1 - np.abs(np.arange(-radius, radius + 1))
    smoothed = np.empty_like(data)
    for chromosome in dict.fromkeys(selected):
        ix = np.flatnonzero(selected == chromosome)
        if ix[-1] - ix[0] + 1 != len(ix):
            raise ValueError("chromosome blocks must be contiguous")
        mass = convolve1d(
            np.ones(len(ix)), kernel.astype(float), mode="constant", cval=0.0
        )
        smoothed[ix] = (
            convolve1d(
                data[ix], kernel.astype(float), axis=0, mode="constant", cval=0.0
            )
            / mass[:, None]
        )
    smoothed -= np.median(smoothed, axis=0)
    smoothed -= smoothed[:, refs].mean(axis=1, keepdims=True)
    result = np.full(counts.shape, np.nan)
    result[keep] = smoothed[:, : counts.shape[1]]
    return result


def infercnv_features(requests, genes, barcodes, *, work_parent=None):
    """In-process replacement for the former Julia bootstrap file bridge."""
    result = {}
    for key, counts, library, width in requests:
        reference = np.median(counts / library[None, :], axis=1)
        if reference.sum() <= 0:
            raise ValueError("cohort median has no expression support")
        reference *= np.median(counts.sum(axis=0)) / reference.sum()
        result[key] = infercnv_smoothing(
            counts, genes.chromosome, reference=reference, window_length=width
        )
    return result
