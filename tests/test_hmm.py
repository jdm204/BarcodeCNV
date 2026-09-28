"""Independent exhaustive-path checks, including underflow and FFBS dependence."""

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from barcodecnv.hmm import forward_backward


def enumerate_paths(e, tr, initial):
    paths = np.array(list(product(range(len(initial)), repeat=len(e))))
    scores = np.array(
        [
            initial[p[0]]
            + sum(e[t, s] for t, s in enumerate(p))
            + sum(tr[t, p[t], p[t + 1]] for t in range(len(p) - 1))
            for p in paths
        ]
    )
    z = logsumexp(scores)
    weights = np.exp(scores - z)
    marginals = np.array(
        [
            [weights[paths[:, t] == s].sum() for s in range(len(initial))]
            for t in range(len(e))
        ]
    )
    return paths, weights, marginals, z


@pytest.mark.parametrize("sparse", [False, True])
def test_dynamic_hmm_against_all_paths(sparse):
    e = np.array([[-0.7, -1.3], [-2.0, -0.1], [-0.3, -0.8], [-3.0, -0.2]])
    tr = np.array(
        [[[0.9, 0.1], [0.3, 0.7]], [[0.5, 0.5], [0.5, 0.5]], [[0.8, 0.2], [0.1, 0.9]]]
    )
    if sparse:
        tr[0] = np.eye(2)
    with np.errstate(divide="ignore"):
        tr = np.log(tr)
    ini = np.log([0.8, 0.2])
    paths, weights, expected, z = enumerate_paths(e, tr, ini)
    got = forward_backward(e, tr, ini)
    np.testing.assert_allclose(got.probabilities, expected, atol=2e-14)
    assert got.log_evidence == pytest.approx(z, abs=2e-14)
    draws = got.sample_paths(40000, seed=82)
    # Whole-path frequencies detect sampling marginals independently by mistake.
    freq = np.array([np.mean(np.all(draws == p, axis=1)) for p in paths])
    assert np.max(np.abs(freq - weights)) < 0.008


def test_flat_map_cannot_erase_recovering_path():
    with np.errstate(divide="ignore"):
        tr = np.log(np.eye(2))[None]
    got = forward_backward(
        np.array([[0.0, -1000.0], [-1000.0, 0.0]]), tr, np.log([0.5, 0.5])
    )
    np.testing.assert_allclose(got.probabilities, 0.5, atol=1e-12)
    assert got.log_evidence == pytest.approx(-1000.0, abs=1e-10)
    draws = got.sample_paths(10000, 49)
    assert np.all(draws[:, 0] == draws[:, 1])
    assert abs(draws[:, 0].mean() - 0.5) < 0.02


def test_single_marker_and_impossible_observations():
    got = forward_backward(
        np.log([[0.2, 0.8]]), np.empty((0, 2, 2)), np.log([0.5, 0.5])
    )
    np.testing.assert_allclose(got.probabilities, [[0.2, 0.8]])
    assert got.log_evidence == pytest.approx(np.log(0.5))
    with pytest.raises(ValueError, match="zero or nonfinite"):
        forward_backward(
            np.full((1, 2), -np.inf), np.empty((0, 2, 2)), np.log([0.5, 0.5])
        )


def test_chromosome_reset_removes_cross_boundary_evidence():
    initial = np.log([0.8, 0.2])
    transitions = np.tile(initial, (1, 2, 1))
    e = np.array([[0.0, -500.0], [-0.3, -0.7]])
    joint = forward_backward(e, transitions, initial)
    single = forward_backward(e[1:], np.empty((0, 2, 2)), initial)
    np.testing.assert_allclose(
        joint.probabilities[1], single.probabilities[0], atol=1e-14
    )


def test_invalid_transition_rows_are_rejected():
    with pytest.raises(ValueError, match="normalized"):
        forward_backward(np.zeros((2, 2)), np.zeros((1, 2, 2)), np.log([0.5, 0.5]))
