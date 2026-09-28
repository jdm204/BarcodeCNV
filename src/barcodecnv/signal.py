"""Label-blind regional-count statistic, calibrated by whole-cell permutations.

Port of Julia PBPC's retained fast diagnostic. No HMM or clone calls are used.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class SignalFeatures:
    depth: object
    h1: object
    h2: object
    libraries: np.ndarray
    depth_rate: np.ndarray
    allele_fraction: np.ndarray
    regions: tuple
    width: int

    def score(self, labels):
        labels = np.asarray(labels)
        if (
            labels.shape != self.libraries.shape
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
        ):
            raise ValueError("one nonnegative integer label required per cell")
        weights = csr_matrix(
            (np.ones(len(labels), dtype=np.int64), (np.arange(len(labels)), labels)),
            shape=(len(labels), int(labels.max()) + 1),
        )
        d = (self.depth @ weights).toarray()
        a = (self.h1 @ weights).toarray()
        b = (self.h2 @ weights).toarray()
        exposure = np.asarray(self.libraries @ weights).ravel()
        expected = self.depth_rate[:, None] * exposure
        residual = np.divide(
            (d - expected) ** 2,
            expected,
            out=np.zeros_like(expected),
            where=expected > 0,
        )
        total = a + b
        p = self.allele_fraction[:, None]
        variance = total * p * (1 - p)
        ar = np.divide(
            (a - total * p) ** 2,
            variance,
            out=np.zeros_like(variance),
            where=variance > 0,
        )
        depth, allele = float(residual.sum()), float(ar.sum())
        return np.array([depth + allele, depth, allele])


def signal_features(bundle, bin_width_bp=10_000_000):
    if not isinstance(bin_width_bp, int) or bin_width_bp < 1:
        raise ValueError("bin width must be a positive integer")
    keys = [
        (c, int((p - 1) // bin_width_bp))
        for c, p in zip(bundle.grid.chrom, bundle.grid.start)
    ]
    regions = tuple(dict.fromkeys(keys))
    lookup = {k: i for i, k in enumerate(regions)}

    def projection(markers, values):
        return csr_matrix(
            (values, ([lookup[keys[t]] for t in markers], np.arange(len(markers)))),
            shape=(len(regions), len(markers)),
        )

    genes = projection(
        bundle.gene_markers, np.ones(len(bundle.gene_ids), dtype=np.int64)
    )
    forward = bundle.loci.population_phase >= 0.5
    a = projection(bundle.loci.marker, forward.astype(np.int64))
    b = projection(bundle.loci.marker, (~forward).astype(np.int64))
    depth = genes @ bundle.expression
    h1 = a @ bundle.h1 + b @ bundle.h2
    h2 = a @ bundle.h2 + b @ bundle.h1
    rate = np.asarray(depth.sum(axis=1)).ravel() / bundle.libraries.sum()
    successes = np.asarray(h1.sum(axis=1)).ravel()
    total = successes + np.asarray(h2.sum(axis=1)).ravel()
    fraction = np.divide(
        successes, total, out=np.full(len(regions), 0.5), where=total > 0
    )
    return SignalFeatures(
        depth, h1, h2, bundle.libraries, rate, fraction, regions, bin_width_bp
    )


def barcode_signal_test(bundle, *, permutations=199, bin_width_bp=10_000_000, seed=42):
    if not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be positive")
    features = signal_features(bundle, bin_width_bp)
    labels = bundle.membership
    blocks = [
        np.flatnonzero(np.array(bundle.blocks) == b) for b in sorted(set(bundle.blocks))
    ]
    observed = features.score(labels)
    assessable = any(len(np.unique(labels[ix])) > 1 for ix in blocks)
    rng = np.random.default_rng(seed)
    null = np.empty((permutations if assessable else 0, 3))
    for r in range(len(null)):
        shuffled = labels.copy()
        for ix in blocks:
            shuffled[ix] = rng.permutation(labels[ix])
        null[r] = features.score(shuffled)
    tolerance = 1e-9 * max(1.0, abs(observed[0]))
    p = (
        float((1 + np.sum(null[:, 0] >= observed[0] - tolerance)) / (permutations + 1))
        if assessable
        else None
    )
    return dict(
        status="assessed" if assessable else "unassessable",
        pvalue=p,
        scores=dict(zip(("joint", "depth", "allele"), observed.tolist())),
        null_scores=null,
        excess_over_null_median=float(observed[0] - np.median(null[:, 0]))
        if assessable
        else None,
        bin_width_bp=bin_width_bp,
        permutations=len(null),
        seed=seed,
        barcode_sizes=np.bincount(labels).tolist(),
        block_sizes=[len(ix) for ix in blocks],
    )
