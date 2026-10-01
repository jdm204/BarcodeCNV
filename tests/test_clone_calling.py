"""Clone-calling boundaries that previously hid signal or invented certainty."""

import numpy as np
import pytest

from barcodecnv import api
from barcodecnv.clone_calling import call_clones


def profiles(states):
    probabilities = np.eye(4)[np.asarray(states)]
    distances = np.mean(
        np.asarray(states)[:, None] != np.asarray(states)[None, :], axis=2
    )
    return probabilities, np.repeat(distances[:, :, None], 32, axis=2)


@pytest.mark.parametrize("method", ["mean_profile", "mean_distance", "draw_quantile"])
def test_supported_descendants_survive_outlier_and_expression_veto(method):
    states = np.zeros((9, 60), dtype=int)
    states[:4, :10] = 1
    states[4:8, 10:20] = 2
    states[8] = 3
    p, distances = profiles(states)
    cells = np.array([100, 20, 5, 2] * 2 + [1])
    calls = call_clones(
        distances, p, cells, method=method, expression_groups=np.zeros(9, int)
    )
    assert calls.groups[0] > 0 and len(set(calls.groups[:4])) == 1
    assert calls.groups[4] > 0 and len(set(calls.groups[4:8])) == 1
    assert calls.groups[0] != calls.groups[4] and calls.groups[8] == 0
    assert calls.membership[8]["reason"] == "ambiguous_preference"
    # The previous expression-first rule cannot recover these vetoed barcodes.
    previous = call_clones(
        distances, p, cells, method="expression", expression_groups=np.zeros(9, int)
    )
    assert np.all(previous.groups == 0)


@pytest.mark.parametrize("n", [1, 2])
def test_tiny_dataset_has_no_fabricated_support(n):
    p, distances = profiles(np.zeros((n, 10), int))
    calls = call_clones(distances, p, np.ones(n))
    assert np.all(calls.groups == 0)
    assert np.isnan(calls.stability).all()
    if n == 2:
        assert all(
            row["reason"] == "insufficient_calibration" for row in calls.membership
        )


def test_invalid_method_rejected_before_input_or_output(tmp_path):
    for operation in (api.infer, api.run_pipeline):
        with pytest.raises(ValueError, match="clone_method"):
            operation(out=tmp_path / "out", clone_method="typo")
        assert not (tmp_path / "out").exists()
