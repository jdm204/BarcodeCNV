"""Count-weighted haplotype smoothing and local group refinement.

Adapts PBPC2's whole-cell bootstrap and recursive expression splitter to signed
haplotype fractions. Phase is a fixed MAP estimate. Cell resampling and weak
beta draws overlap as uncertainty sources; stability is not a calibrated clone
posterior. No truth, filesystem state or mutable global fit state enters here.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .bootstrap import aggregate_cells
from .grouping import common_refinement, recursive_groups

BASE_BIN_BP = 2_000_000
WINDOW_RADIUS = 2
WINDOW_BP = BASE_BIN_BP * (2 * WINDOW_RADIUS + 1)


def region_counts(bundle, phase, width=10_000_000):
    if not isinstance(width, int) or width < 1:
        raise ValueError("window width must be a positive integer")
    chrom = [bundle.grid.chrom[t] for t in bundle.loci.marker]
    keys = list(zip(chrom, (bundle.loci.position - 1) // width))
    unique = list(dict.fromkeys(keys))
    index = {k: i for i, k in enumerate(unique)}
    rows = np.array([index[k] for k in keys])
    q = np.asarray(phase)
    if (
        q.shape != (len(bundle.loci),)
        or np.any(~np.isfinite(q))
        or np.any((q < 0) | (q > 1))
    ):
        raise ValueError("phase must contain one probability per SNP")
    # Conditional MAP phase: balanced loci can be deeply covered while their
    # orientation is unidentified. Do not equate phase entropy with no coverage.
    # This approximation omits uncertainty/correlation in the fitted phase.
    weight = bundle.het
    direct = csr_matrix(
        (weight * (q >= 0.5), (rows, np.arange(len(q)))), shape=(len(unique), len(q))
    )
    reverse = csr_matrix(
        (weight * (q < 0.5), (rows, np.arange(len(q)))), shape=direct.shape
    )
    return (
        direct @ bundle.h1 + reverse @ bundle.h2,
        direct @ bundle.h2 + reverse @ bundle.h1,
        unique,
    )


def features_from_counts(
    a, b, membership, B, *, replicates=64, seed=42, smoother=None, resampling_seed=None
):
    if replicates < 20:
        raise ValueError("at least 20 whole-cell bootstrap replicates are required")
    membership = np.asarray(membership)
    if (
        a.shape != b.shape
        or membership.shape != (a.shape[1],)
        or not np.issubdtype(membership.dtype, np.integer)
        or np.any((membership < 0) | (membership >= B))
    ):
        raise ValueError("allele counts and barcode membership axes disagree")
    members = [np.flatnonzero(membership == i) for i in range(B)]
    if any(len(ix) == 0 for ix in members):
        raise ValueError("every barcode needs observed cells")
    rng = np.random.default_rng(seed)
    cell_rng = rng
    if resampling_seed is not None:
        # Match expression's cell draws without beta draws advancing that stream.
        cell_rng = np.random.default_rng(resampling_seed)
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    aa = aggregate_cells(a, members)
    bb = aggregate_cells(b, members)
    x = (
        2 * (aa + 0.5) / (aa + bb + 1) - 1
        if smoother is None
        else 2 * (smoother @ (aa + 0.5)) / (smoother @ (aa + bb + 1)) - 1
    )
    boot = []
    for _ in range(replicates):
        samples = [cell_rng.choice(ix, len(ix), replace=True) for ix in members]
        aa = aggregate_cells(a, samples)
        bb = aggregate_cells(b, samples)
        theta = rng.beta(aa + 0.5, bb + 0.5)
        # Smooth the same base-bin draw into all overlapping windows, retaining
        # their covariance. Each base bin has the explicit Beta(1/2,1/2) prior.
        value = (
            theta
            if smoother is None
            else (smoother @ (theta * (aa + bb + 1))) / (smoother @ (aa + bb + 1))
        )
        boot.append(2 * value - 1)
    # Centering is part of the transform and is repeated inside each bootstrap.
    boot = np.array(boot)
    x -= np.median(x, axis=1, keepdims=True)
    boot -= np.median(boot, axis=2, keepdims=True)
    return x, boot


def window_smoother(regions, radius=2):
    lookup = {key: i for i, key in enumerate(regions)}
    rows = []
    cols = []
    for i, (chrom, pos) in enumerate(regions):
        for j in range(pos - radius, pos + radius + 1):
            if (chrom, j) in lookup:
                rows.append(i)
                cols.append(lookup[(chrom, j)])
    return csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(regions), len(regions))
    )


def rolling_features(bundle, phase, *, replicates=64, seed=42, resampling_seed=None):
    """10 Mb boxcar on 2 Mb base bins, never crossing chromosomes.

    Nearby output windows share base-bin beta draws; they are not independent
    measurements. This differs slightly from the 10 Mb bin prior: five covered
    base bins contribute five weak priors rather than a single regional prior.
    """
    a, b, regions = region_counts(bundle, phase, BASE_BIN_BP)
    smoother = window_smoother(regions, WINDOW_RADIUS)
    x, boot = features_from_counts(
        a,
        b,
        bundle.membership,
        len(bundle.barcodes),
        replicates=replicates,
        seed=seed,
        smoother=smoother,
        resampling_seed=resampling_seed,
    )
    membership = bundle.membership
    coverage = smoother @ aggregate_cells(
        a + b, [np.flatnonzero(membership == i) for i in range(len(bundle.barcodes))]
    )
    return dict(
        x=x,
        boot=boot,
        regions=regions,
        coverage=coverage,
        stride_bp=BASE_BIN_BP,
        window_bp=WINDOW_BP,
    )


def partition(x, boot):
    weighted = recursive_groups(x, boot)
    core = recursive_groups(x, boot, core_fraction=0.5)
    return common_refinement(weighted["labels"], core["labels"]), weighted, core


def local_refinement(x, boot, initial, *, return_stability=False):
    """Ask for allele subdivisions inside each currently reported group.

    A failed global HF split must not hide a locally supported difference.
    Existing expression/CN separation is preserved; unresolved members stay so.
    """
    initial = np.asarray(initial)
    if (
        initial.ndim != 1
        or not np.issubdtype(initial.dtype, np.integer)
        or np.any(initial < 0)
    ):
        raise ValueError("group labels must be nonnegative integers")
    if (
        x.ndim != 2
        or boot.ndim != 3
        or x.shape != (boot.shape[1], len(initial))
        or boot.shape[2] != len(initial)
    ):
        raise ValueError("HF feature, bootstrap and group axes disagree")
    labels = np.zeros(len(initial), int)
    next_label = 0
    nodes = []
    stability = np.ones(len(initial))
    for group in sorted(set(initial) - {0}):
        ix = np.flatnonzero(initial == group)
        part, w, c = partition(x[:, ix], boot[:, :, ix])
        stability[ix] = np.minimum(w["stability"], c["stability"])
        for g in sorted(set(part) - {0}):
            next_label += 1
            labels[ix[part == g]] = next_label
        for tag, result in [("weighted", w), ("core", c)]:
            nodes.extend(
                dict(
                    parent_group=int(group),
                    rule=tag,
                    barcode_indices=",".join(map(str, ix)),
                    **row,
                )
                for row in result["nodes"]
            )
    return (labels, nodes, stability) if return_stability else (labels, nodes)
