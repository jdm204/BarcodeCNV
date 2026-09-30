"""Missing expression blocks should be excluded without noisy reductions."""

import warnings

import numpy as np
import pandas as pd

from barcodecnv.bootstrap import bootstrap_features
from barcodecnv.smoothing import infercnv_features


def test_empty_blocks_and_bootstrap_dropout_preserve_finite_features():
    # chr1 is unexpressed. chr2 can disappear when its only expressing cell is
    # not drawn. chr3 remains expressed, with one permanently missing gene.
    cells = np.zeros((75, 4), dtype=int)
    cells[25:50, 0] = np.arange(1, 26)
    cells[51:] = np.arange(1, 97).reshape(24, 4)
    genes = pd.DataFrame({"chromosome": np.repeat(["chr1", "chr2", "chr3"], 25)})
    captured = {}

    def features(*args, **kwargs):
        captured.update(infercnv_features(*args, **kwargs))
        return captured

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = bootstrap_features(
            cells,
            cells.sum(axis=0) + 100,
            np.array([0, 0, 1, 1]),
            genes,
            ["a", "b"],
            replicates=20,
            seed=42,
            feature_function=features,
        )

    observed = captured["draw_-1"]
    assert np.isnan(observed[:25]).all()
    assert np.isfinite(observed[25:50]).all()
    assert any(np.isnan(captured[f"draw_{r}"][25:50]).all() for r in range(20))
    # The observed-only support is insufficient: a feature must survive every
    # draw. Partial missingness keeps the mean of supported genes, weighted by
    # the original block size (25), as before.
    assert result["x"].shape == (1, 2)
    expected = observed[51:].mean(axis=0) * 5
    np.testing.assert_allclose(result["x"][0], expected)
    for r in range(20):
        np.testing.assert_allclose(
            result["boot"][r, 0], captured[f"draw_{r}"][51:].mean(axis=0) * 5
        )
    np.testing.assert_array_equal(result["self_expression"], observed)
