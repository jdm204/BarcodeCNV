from dataclasses import replace

import numpy as np
import pandas as pd

from barcodecnv import AlleleCounts, GeneCounts, Grid, Loci, Model, PreparedCounts
from barcodecnv.cn_reporting import write_cn_tables
from barcodecnv.group_calls import summarize_groups
from barcodecnv.joint import infer_joint


def example():
    from scipy.sparse import csc_matrix

    from barcodecnv.bundle import CellBundle

    return CellBundle(
        Grid(("chr1", "chr1"), [1, 100001], [1, 100001]),
        Loci([0, 1], [1, 100001], [0.9, 0.7], [0.0, 0.2]),
        ("geneA", "geneB"),
        [0, 1],
        [0.1, 0.12],
        tuple(f"c{i}" for i in range(10)),
        ("a",) * 2 + ("b",) * 6 + ("unused",) * 2,
        ("sample",) * 10,
        [100] * 8 + [200000] * 2,
        csc_matrix(
            [[12, 9] + [8] * 6 + [100000] * 2, [16, 25] + [5] * 6 + [100000] * 2]
        ),
        csc_matrix([[9, 8] + [2] * 6 + [999] * 2, [3, 4] + [6] * 6 + [999] * 2]),
        csc_matrix([[1, 2] + [8] * 6 + [1] * 2, [7, 6] + [4] * 6 + [1] * 2]),
        [0.9, 0.8],
    )


def test_pooled_calls_match_hand_summed_cell_counts_and_exclude_unresolved():
    bundle = example()
    m = Model()
    p = np.full((3, 2, 4), 0.25)
    got = summarize_groups(bundle, m, p, [7, 7, 0])
    expected = PreparedCounts(
        bundle.grid,
        ("G7",),
        GeneCounts([0, 0], [0, 1], [69, 71], [80, 96], [1, 1]),
        AlleleCounts([0, 0], [0, 1], [29, 43], [51, 37], [0.9, 0.8]),
        bundle.loci,
    )
    direct = bundle.prepared([7, 7, 0], exclude_unresolved=True)
    np.testing.assert_array_equal(direct.genes.count, expected.genes.count)
    np.testing.assert_allclose(direct.genes.expected, expected.genes.expected)
    np.testing.assert_array_equal(direct.alleles.h1, expected.alleles.h1)
    np.testing.assert_array_equal(direct.alleles.h2, expected.alleles.h2)
    np.testing.assert_array_equal(direct.alleles.het, expected.alleles.het)
    fit = infer_joint(expected, m, got.dispersion.parameters)
    np.testing.assert_allclose(got.pooled, fit.classes, atol=1e-12)
    assert (
        got.groups.tolist() == [7]
        and got.cells.tolist() == [8]
        and got.barcodes.tolist() == [2]
    )
    # A barcode repartition of identical member cells cannot alter pooled calls.
    changed = replace(bundle, barcode_labels=("a",) * 4 + ("b",) * 4 + ("unused",) * 2)
    again = summarize_groups(changed, m, p, [7, 7, 0])
    np.testing.assert_allclose(again.pooled, got.pooled, atol=1e-12)
    assert again.dispersion.parameters == got.dispersion.parameters


def test_descriptive_consensus_separates_uncertainty_from_confident_conflict():
    bundle = example()
    m = Model()
    probabilities = np.repeat(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.25] * 4])[:, None, :], 2, axis=1
    )
    got = summarize_groups(bundle, m, probabilities, [3, 3, 3])
    np.testing.assert_allclose(got.consensus[0], [[0.25, 0.65, 0.05, 0.05]] * 2)
    np.testing.assert_allclose(got.uncertain_cell_fraction, 0.2)
    np.testing.assert_allclose(got.consensus_conflict_cell_fraction, 0.2)


def test_single_member_pseudobulk_matches_barcode_model_at_group_noise():
    bundle = example()
    m = Model()
    got = summarize_groups(bundle, m, np.full((3, 2, 4), 0.25), [5, 0, 0])
    post = infer_joint(bundle.prepared(), m, got.dispersion.parameters)
    np.testing.assert_allclose(got.pooled[0], post.classes[0], atol=1e-12)
    np.testing.assert_allclose(got.phase[0], post.phase[0], atol=1e-12)


def test_gene_exports_and_segment_boundaries_preserve_meanings(tmp_path):
    from test_cli import example_bundle

    bundle = example_bundle()
    data = bundle.prepared()
    B, T = data.shape
    # Both groups have the same marginal MAP on every chromosome; segment
    # export must still break at chromosome boundaries.
    p = np.zeros((B, T, 4))
    p[:, :, 0] = 1
    labels = np.array([2] * 4 + [9] * 4)
    calls = summarize_groups(bundle, Model(), p, labels)
    pooled = np.zeros_like(calls.pooled)
    pooled[:, :, 0] = 0.8
    pooled[:, :, 1] = 0.2
    calls = replace(calls, pooled=pooled)
    write_cn_tables(tmp_path, bundle, p, labels, calls)
    genes = pd.read_csv(tmp_path / "group_cn_genes.csv.gz")
    np.testing.assert_allclose(genes.pooled_p_diploid, 0.8)
    np.testing.assert_allclose(genes.consensus_p_diploid, 1.0)
    assert genes.gene.tolist() == list(bundle.gene_ids) * 2
    segments = pd.read_csv(tmp_path / "group_cn_segments.csv.gz")
    assert (
        segments.groupby("group").chromosome.apply(list).tolist()
        == [["chr1", "chr2", "chr3"]] * 2
    )
    np.testing.assert_allclose(segments.mean_state_probability, 0.8)
    assert (segments.start == 1_000_000).all() and (segments.end == 20_000_000).all()
    barcodes = pd.read_csv(tmp_path / "barcode_cn_genes.csv.gz", dtype={"barcode": str})
    assert barcodes.barcode.unique().tolist() == list(bundle.barcodes)
    assert barcodes.groupby("barcode").group.first().tolist() == labels.tolist()
    # No assigned members means no pooled group, not one pool of unresolved cells.
    empty = summarize_groups(bundle, Model(), p, np.zeros(B, int))
    assert empty.pooled.shape == (0, T, 4)
    write_cn_tables(tmp_path, bundle, p, np.zeros(B, int), empty)
    assert pd.read_csv(tmp_path / "group_cn_genes.csv.gz").empty
    assert pd.read_csv(tmp_path / "group_cn_segments.csv.gz").empty
