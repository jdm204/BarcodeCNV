"""Check plotted coordinates, not just whether a Figure is constructed."""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from barcodecnv.data import Grid
from barcodecnv.genomic_plot import (
    genome_layout,
    interval_heatmap,
    label_genome_axis,
    marker_heatmap,
)


def test_gene_rich_chr19_does_not_expand_and_marker_spacing_is_physical():
    positions = [1_000_000, 101_000_000, 1_000_000, 2_000_000, 3_000_000]
    grid = Grid(("chr1", "chr1", "chr19", "chr19", "chr19"), positions, positions)
    layout = genome_layout(grid)
    fig, ax = plt.subplots()
    try:
        values = np.array([[1, 2, 3, 4, 5]])
        marker_heatmap(ax, values, grid, np.arange(5), layout)
        label_genome_axis(ax, layout)
        assert ax.get_xlim() == (0, (248956422 + 58617616) / 1e6)
        np.testing.assert_allclose(
            ax.get_xticks(), [248956422 / 2e6, (248956422 + 58617616 / 2) / 1e6]
        )
        # chr1's two sparse observations span 100 Mb, chr19's three span 2 Mb.
        first, second = [mesh.get_coordinates()[0, :, 0] for mesh in ax.collections]
        np.testing.assert_allclose(first, [0.999999, 50.9999995, 101])
        np.testing.assert_allclose(
            second, 248.956422 + np.array([0.999999, 1.4999995, 2.4999995, 3])
        )
        assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "19"]
        np.testing.assert_array_equal(
            np.concatenate([m.get_array() for m in ax.collections], axis=1), values
        )
    finally:
        plt.close(fig)


def test_coincident_genes_share_one_interval_with_mean_value():
    grid = Grid(("chr1", "chr1"), [1, 101], [1, 101])
    fig, ax = plt.subplots()
    try:
        marker_heatmap(ax, [[2, 4, 8]], grid, [0, 0, 1], genome_layout(grid))
        np.testing.assert_array_equal(ax.collections[0].get_array(), [[3, 8]])
        np.testing.assert_allclose(
            ax.collections[0].get_coordinates()[0, :, 0], np.array([0, 50.5, 101]) / 1e6
        )
    finally:
        plt.close(fig)


def test_haplotype_bins_keep_physical_width_and_uncovered_gaps():
    fig, ax = plt.subplots()
    try:
        mesh = interval_heatmap(
            ax,
            [[0.5, -0.5]],
            ["chr1", "chr1"],
            [0, 20_000_000],
            [2_000_000, 22_000_000],
            {"chr1": (0, 100_000_000)},
        )
        np.testing.assert_array_equal(mesh.get_coordinates()[0, :, 0], [0, 2, 20, 22])
        np.testing.assert_array_equal(mesh.get_array().mask, [[False, True, False]])
        np.testing.assert_array_equal(mesh.get_array().compressed(), [0.5, -0.5])
        assert ax.get_xlim() == (0, 100)
    finally:
        plt.close(fig)


def test_custom_assembly_requires_explicit_valid_lengths():
    grid = Grid(("chr1",), [200], [200])
    with pytest.raises(ValueError, match="provide chromosome_sizes"):
        genome_layout(grid, genome="custom")
    with pytest.raises(ValueError, match="exceed chromosome size"):
        genome_layout(grid, chromosome_sizes={"1": 100})
    assert genome_layout(grid, genome="custom", chromosome_sizes={"1": 1000}) == {
        "chr1": (0, 1000)
    }
