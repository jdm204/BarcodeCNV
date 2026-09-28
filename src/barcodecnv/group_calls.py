"""Reporting-only group summaries and a homogeneous CN/phase model.

Sum member cells into one expression/allele pseudobulk per reporting group,
fit dispersion at this aggregation level, and infer each group CN/phase path.
Membership is fixed. Cell weights belong only to descriptive summaries.
"""

from dataclasses import dataclass

import numpy as np

from .data import owned
from .depth_controls import DepthOptions
from .fitting import DispersionFit, FitOptions
from .joint import fit_joint_dispersion, infer_joint


@dataclass(frozen=True)
class GroupCalls:
    groups: np.ndarray
    cells: np.ndarray
    barcodes: np.ndarray
    consensus: np.ndarray
    pooled: np.ndarray
    phase: np.ndarray
    uncertain_cell_fraction: np.ndarray
    consensus_conflict_cell_fraction: np.ndarray
    pooled_conflict_cell_fraction: np.ndarray
    confident_threshold: float
    dispersion: DispersionFit | None
    depth_options: DepthOptions = DepthOptions()

    def __post_init__(self):
        for name in ("groups", "cells", "barcodes"):
            object.__setattr__(self, name, owned(getattr(self, name), np.int64))
        for name in (
            "consensus",
            "pooled",
            "phase",
            "uncertain_cell_fraction",
            "consensus_conflict_cell_fraction",
            "pooled_conflict_cell_fraction",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, owned(value, float, ndim=np.ndim(value)))


def summarize_groups(
    bundle,
    model,
    probabilities,
    labels,
    *,
    options=FitOptions(),
    confident_threshold=0.95,
    depth_options=DepthOptions(),
):
    """Summarize final labels; group zero is excluded from count aggregation.

    Example: summarize_groups(bundle, model, fit.classes, final_labels).
    Consensus is a cell-weighted descriptive average of barcode probabilities.
    Pooled calls use summed cell counts and independently fitted group noise.
    """
    labels = owned(labels, np.int64)
    cells = np.bincount(bundle.membership, minlength=len(bundle.barcodes))
    p = np.asarray(probabilities, dtype=float)
    if (
        labels.shape != (len(bundle.barcodes),)
        or cells.shape != labels.shape
        or np.any(labels < 0)
        or np.any(cells <= 0)
    ):
        raise ValueError("labels and positive cell counts must cover every barcode")
    if (
        p.shape != (len(bundle.barcodes), len(bundle.grid), 4)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(-1), 1.0)
        or not 0.5 < confident_threshold <= 1.0
    ):
        raise ValueError("invalid barcode class probabilities or confidence threshold")
    groups = np.unique(labels[labels > 0])
    n, t = len(groups), len(bundle.grid)
    consensus = np.empty((n, t, 4))
    uncertain, conflict, pooled_conflict = (np.empty((n, t)) for _ in range(3))
    sizes, members = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    pooled, phase = np.empty((0, t, 4)), np.empty((0, len(bundle.loci)))
    noise = None
    if n:
        data = bundle.prepared(labels, exclude_unresolved=True)
        noise = fit_joint_dispersion(
            data, model, options=options, depth_options=depth_options
        )
        if not noise.converged:
            raise RuntimeError("group pseudobulk dispersion search did not converge")
        result = infer_joint(data, model, noise.parameters, depth_options=depth_options)
        pooled, phase = result.classes, result.phase
    for i, group in enumerate(groups):
        ix = labels == group
        weights = cells[ix] / cells[ix].sum()
        values = p[ix]
        consensus[i] = np.einsum("b,btc->tc", weights, values)
        confident = values.max(-1) >= confident_threshold
        calls = values.argmax(-1)
        uncertain[i] = weights @ (~confident)
        conflict[i] = weights @ (confident & (calls != consensus[i].argmax(-1)))
        pooled_conflict[i] = weights @ (confident & (calls != pooled[i].argmax(-1)))
        sizes[i], members[i] = cells[ix].sum(), ix.sum()
    return GroupCalls(
        groups,
        sizes,
        members,
        consensus,
        pooled,
        phase,
        uncertain,
        conflict,
        pooled_conflict,
        confident_threshold,
        noise,
        depth_options,
    )
