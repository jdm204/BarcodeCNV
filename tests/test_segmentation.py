"""Path and segment evidence checks independent of the dynamic recursions."""

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp
from test_hmm import enumerate_paths

from barcodecnv.hmm import HMMEngine
from barcodecnv.segmentation import constant_cn_evidence, filter_segments, segment_runs


@pytest.mark.parametrize("reset", [False, True])
def test_viterbi_against_exhaustive_dynamic_paths(reset):
    e = np.array([[0.0, -1000.0], [-1000.0, 0.0], [-1.0, -2.0]])
    tr = np.array([np.eye(2), [[0.9, 0.1], [0.3, 0.7]]])
    if reset:
        tr[1] = [[0.8, 0.2], [0.8, 0.2]]
    with np.errstate(divide="ignore"):
        tr = np.log(tr)
    ini = np.log([0.6, 0.4])
    paths, weights, _, z = enumerate_paths(e, tr, ini)
    path, score = HMMEngine(tr, ini).viterbi(e)
    np.testing.assert_array_equal(path, paths[weights.argmax()])
    assert score == pytest.approx(z + np.log(weights.max()), abs=1e-10)


def test_viterbi_single_marker_and_impossible_data():
    engine = HMMEngine(np.empty((0, 2, 2)), np.log([0.8, 0.2]))
    path, score = engine.viterbi(np.log([[0.1, 0.9]]))
    assert path[0] == 1
    assert score == pytest.approx(np.log(0.18))
    with pytest.raises(ValueError, match="zero or nonfinite"):
        engine.viterbi(np.full((1, 2), -np.inf))


def test_segment_evidence_integrates_phase_rather_than_selecting_it():
    e = np.array(
        [
            [[-2.0, -0.1], [-0.5, -4.0]],
            [[-0.1, -3.0], [-2.0, -0.1]],
            [[-0.3, -0.7], [-1.0, -0.2]],
        ]
    )
    tr = np.log([[[0.99, 0.01], [0.01, 0.99]], [[0.6, 0.4], [0.4, 0.6]]])
    expected = []
    for c in range(2):
        scores = [
            -np.log(2)
            + sum(e[t, c, z] for t, z in enumerate(path))
            + sum(tr[t, path[t], path[t + 1]] for t in range(2))
            for path in product(range(2), repeat=3)
        ]
        expected.append(logsumexp(scores))
    np.testing.assert_allclose(constant_cn_evidence(e, tr), expected, atol=1e-13)
    # With no evidence, phase multiplicity cannot favor either CN state.
    np.testing.assert_allclose(
        constant_cn_evidence(np.zeros_like(e), tr), 0.0, atol=1e-14
    )


def test_filter_respects_boundaries_and_keeps_strong_short_event():
    calls = np.array([[1, 1, 1, 2, 0]])
    runs = list(segment_runs(calls[0], ["chr1", "chr1", "chr2", "chr2", "chr2"]))
    assert runs == [(0, 2), (2, 3), (3, 4), (4, 5)]
    rows = [
        dict(
            group_index=0,
            start_index=a,
            stop_index=b,
            state=int(calls[0, a]),
            log_bf_vs_diploid=score,
        )
        for (a, b), score in zip(runs, [2.0, 20.0, -1.0, 0.0])
    ]
    np.testing.assert_array_equal(filter_segments(calls, rows, 5), [[0, 0, 1, 0, 0]])
    np.testing.assert_array_equal(calls, [[1, 1, 1, 2, 0]])
