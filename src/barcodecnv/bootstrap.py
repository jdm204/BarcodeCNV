"""Whole-cell resampling; no resampling of smoothed gene columns."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from .smoothing import infercnv_features


def aggregate_cells(cells, samples):
    """Sum whole-cell bootstrap multiplicities without densifying the cell matrix."""
    rows = np.concatenate(samples)
    columns = np.repeat(np.arange(len(samples)), [len(ix) for ix in samples])
    weights = csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, columns)),
        shape=(cells.shape[1], len(samples)),
    )
    values = cells @ weights
    return values.toarray() if issparse(values) else np.asarray(values)


def feature_blocks(genes, width=25):
    blocks = []
    for c in pd.unique(genes.chromosome):
        ix = np.flatnonzero(genes.chromosome.to_numpy() == c)
        blocks.extend(ix[i : i + width] for i in range(0, len(ix), width))
    return blocks


def bootstrap_features(
    cells,
    library,
    membership,
    genes,
    barcodes,
    *,
    replicates=64,
    seed=270929,
    work_parent=None,
    feature_function=infercnv_features,
):
    """Recompute the self reference and InferCNV transform for each cell draw.

    Gene rows must already follow genomic order. Membership is zero-based into
    the explicit barcode axis. Library sizes cover the original assay universe.
    """
    library = np.asarray(library, dtype=float)
    membership = np.asarray(membership)
    if replicates < 20:
        raise ValueError("at least 20 whole-cell bootstrap replicates are required")
    if cells.shape != (len(genes), len(library)) or membership.shape != library.shape:
        raise ValueError("cell counts, annotations and library axes disagree")
    if (
        not np.issubdtype(membership.dtype, np.integer)
        or np.any(membership < 0)
        or np.any(membership >= len(barcodes))
    ):
        raise ValueError("membership must index the supplied barcode axis")
    values = cells.data if issparse(cells) else np.asarray(cells)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values != np.floor(values))
    ):
        raise ValueError("expression must contain nonnegative integer counts")
    if (
        not np.isfinite(library).all()
        or np.any(library <= 0)
        or np.any(np.asarray(cells.sum(axis=0)).ravel() > library)
    ):
        raise ValueError("positive whole-library counts must cover selected counts")
    members = [np.flatnonzero(membership == b) for b in range(len(barcodes))]
    if any(len(ix) == 0 for ix in members):
        raise ValueError("each barcode requires at least one observed cell")
    rng = np.random.default_rng(seed)
    blocks = feature_blocks(genes)

    def requests():
        for r in range(-1, replicates):
            samples = (
                members
                if r == -1
                else [rng.choice(ix, len(ix), replace=True) for ix in members]
            )
            yield (
                f"draw_{r}",
                aggregate_cells(cells, samples),
                np.array([library[ix].sum() for ix in samples]),
                101,
            )

    features = feature_function(requests(), genes, barcodes, work_parent=work_parent)

    def reduce(x):
        return np.array([np.nanmean(x[ix], axis=0) * np.sqrt(len(ix)) for ix in blocks])

    x = reduce(features["draw_-1"])
    boot = np.array([reduce(features[f"draw_{r}"]) for r in range(replicates)])
    valid = np.isfinite(x).all(1) & np.isfinite(boot).all(axis=(0, 2))
    if not valid.any():
        raise ValueError("no finite expression features remain after preprocessing")
    return dict(
        x=x[valid],
        boot=boot[:, valid],
        self_expression=features["draw_-1"],
        barcodes=np.asarray(barcodes, dtype=str),
        cells=np.array([len(ix) for ix in members]),
    )
