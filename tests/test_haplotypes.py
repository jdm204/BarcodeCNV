"""Mechanistic tests for count-weighted HF refinement."""

import numpy as np
from scipy.sparse import csr_matrix

from barcodecnv.haplotypes import (
    features_from_counts,
    local_refinement,
    partition,
    window_smoother,
)


def test_opposite_retained_haplotypes_separate_at_equal_total_counts():
    membership = np.repeat(np.arange(16), 20)
    n = np.full((40, len(membership)), 10)
    a = n // 2
    a[:12, membership < 8] = 1
    a[:12, membership >= 8] = 9
    x, boot = features_from_counts(
        csr_matrix(a), csr_matrix(n - a), membership, 16, seed=42
    )
    labels, _, _ = partition(x, boot)
    truth = np.arange(16) < 8
    np.testing.assert_array_equal(
        labels[:, None] == labels[None, :], truth[:, None] == truth[None, :]
    )
    assert (labels > 0).all()


def test_no_counts_do_not_become_precise_half_fraction():
    membership = np.repeat(np.arange(8), 10)
    empty = csr_matrix((10, len(membership)))
    x, boot = features_from_counts(empty, empty, membership, 8, seed=5)
    assert np.max(abs(x)) == 0
    assert np.mean(np.var(boot, axis=0)) > 0.2
    _, weighted, core = partition(x, boot)
    assert not any(n["accepted"] for r in (weighted, core) for n in r["nodes"])


def test_overlapping_windows_share_bootstrap_fluctuations():
    membership = np.repeat(np.arange(12), 10)
    a = csr_matrix(np.full((3, len(membership)), 2))
    # Identical output windows must be identical in every draw; separate beta
    # draws at output-window level would violate this covariance requirement.
    smoother = csr_matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    x, boot = features_from_counts(a, a, membership, 12, seed=5, smoother=smoother)
    np.testing.assert_array_equal(boot[:, 0], boot[:, 1])
    assert not np.array_equal(boot[:, 0], boot[:, 2])


def test_smoothing_respects_chromosome_and_physical_gaps():
    regions = [("chr1", 0), ("chr1", 1), ("chr1", 9), ("chr2", 0)]
    np.testing.assert_array_equal(
        window_smoother(regions).toarray(),
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )


def test_refinement_preserves_existing_separations_and_unresolved_members():
    membership = np.repeat(np.arange(17), 20)
    n = np.full((40, len(membership)), 10)
    a = n // 2
    a[:12, membership < 8] = 1
    a[:12, membership >= 8] = 9
    x, boot = features_from_counts(
        csr_matrix(a), csr_matrix(n - a), membership, 17, seed=42
    )
    initial = np.array([1] * 12 + [2] * 4 + [0])
    labels, _, stability = local_refinement(x, boot, initial, return_stability=True)
    assert labels[-1] == 0
    assert (
        len(set(labels[:8])) == len(set(labels[8:12])) == len(set(labels[12:16])) == 1
    )
    assert len({labels[0], labels[8], labels[12]}) == 3
    assert np.min(stability[:16]) >= 0.9
