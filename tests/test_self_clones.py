"""Reference-free expression inputs, paired resampling, and joint HF evidence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix
from test_cli import example_bundle

from barcodecnv import Loci, api, bootstrap, haplotypes, workflow
from barcodecnv.self_clones import (
    call_self_clones,
    expression_distances,
    joint_features,
)


@pytest.mark.parametrize(
    "method", ["self_expression_hf", "self_expression_hf_debiased"]
)
def test_hf_separates_identical_expression_and_is_scale_invariant(method):
    rng = np.random.default_rng(35)
    x = np.zeros((20, 8))
    boot = rng.normal(0, 0.015, (64, 20, 8))
    hx = np.tile([0.6] * 4 + [-0.6] * 4, (10, 1))
    hb = hx + rng.normal(0, 0.015, (64, 10, 8))
    features = dict(x=x, boot=boot, cells=np.full(8, 20))
    hf = dict(x=hx, boot=hb, coverage=np.ones_like(hx))
    expression = call_self_clones(features, method="self_expression_tree")
    joint = call_self_clones(features, method=method, hf=hf)
    assert len(set(expression.groups) - {0}) == 1
    assert joint.groups[0] > 0 and joint.groups[-1] > 0
    assert len(set(joint.groups[:4])) == len(set(joint.groups[4:])) == 1
    assert joint.groups[0] != joint.groups[-1]
    a, b, _ = joint_features(x, boot, hf)
    scaled = dict(hf, x=hx * 100, boot=hb * 100)
    c, d, _ = joint_features(x, boot, scaled)
    np.testing.assert_allclose(a, c)
    np.testing.assert_allclose(b, d)
    expected, sampled = expression_distances(a, b)
    np.testing.assert_allclose(sampled.mean(axis=2), expected, atol=1e-12)


@pytest.mark.parametrize(
    "method", ["self_expression_correlation", "self_expression_hf_debiased"]
)
def test_no_excess_variation_retains_undivided_group_without_confidence(method):
    rng = np.random.default_rng(0)
    errors = np.tile([0.04, 0.05, 0.07, 0.1, 0.16, 0.25], 2)
    noise = np.repeat(rng.normal(size=(65, 6, 12)) * errors, 10, axis=1)
    x = noise[0] - np.median(noise[0], axis=1, keepdims=True)
    boot = x + noise[1:]
    boot -= np.median(boot, axis=2, keepdims=True)
    calls = call_self_clones(dict(x=x, boot=boot, cells=np.full(12, 20)), method=method)
    # Heterogeneous precision and spurious profile shapes must not alone force
    # discovery. An undivided root must not claim confident clone membership.
    np.testing.assert_array_equal(calls.groups, np.ones(12))
    assert np.isnan(calls.stability).all()
    assert all(
        row["reason"] == "no_excess_profile_variation" for row in calls.membership
    )


def test_whole_cell_bootstraps_are_paired_across_modalities(monkeypatch):
    expression_draws, hf_draws = [], []
    aggregate = bootstrap.aggregate_cells

    def record(target):
        def wrapped(counts, samples):
            target.append([ix.copy() for ix in samples])
            return aggregate(counts, samples)

        return wrapped

    monkeypatch.setattr(bootstrap, "aggregate_cells", record(expression_draws))
    monkeypatch.setattr(haplotypes, "aggregate_cells", record(hf_draws))
    rng = np.random.default_rng(4)
    counts = csc_matrix(rng.poisson(10, (30, 12)))
    membership = np.repeat(np.arange(3), 4)
    genes = pd.DataFrame(dict(chromosome=["chr1"] * 30))
    bootstrap.bootstrap_features(
        counts,
        np.full(12, 1000),
        membership,
        genes,
        ["a", "b", "c"],
        replicates=20,
        seed=17,
    )
    haplotypes.features_from_counts(
        counts, counts, membership, 3, replicates=20, seed=17, resampling_seed=17
    )
    # HF aggregates each draw twice, once for each allele; beta sampling must
    # never advance the cell-selection stream used by expression.
    assert len(expression_draws) == 21 and len(hf_draws) == 42
    for exp, hf in zip(expression_draws, hf_draws[::2], strict=True):
        for a, b in zip(exp, hf, strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "method",
    [
        "self_expression_hf",
        "self_expression_correlation",
        "self_expression_hf_debiased",
    ],
)
def test_self_method_never_samples_cn_paths_and_records_missing_hf(monkeypatch, method):
    def forbidden(*args, **kwargs):
        raise AssertionError("self-centred grouping must not consume CN-path distances")

    monkeypatch.setattr(workflow, "path_distances", forbidden)
    result = api.infer(
        example_bundle(),
        bootstraps=20,
        draws=16,
        skip_signal=True,
        hf_refinement=False,
        clone_method=method,
    )
    assert result.distances is None
    assert result.clone_calls.evidence == "self_expression"
    if "hf" in method:
        assert result.clone_calls.feature_info["hf_status"] == "no_usable_allele_counts"
    assert len(set(result.groups) - {0}) == 2
    assert "clone_membership_support" in result.groups_table()


@pytest.mark.parametrize(
    "method", ["self_expression_hf", "self_expression_hf_debiased"]
)
def test_joint_workflow_retains_hf_without_a_second_refinement(method):
    data = example_bundle()
    first = np.full((60, len(data.cell_ids)), 10)
    first[:20, data.membership < 2] = 18
    first[:20, (data.membership >= 2) & (data.membership < 4)] = 2
    data = replace(
        data,
        loci=Loci(
            np.arange(60),
            data.grid.start,
            np.full(60, 0.99),
            np.tile(np.arange(20, dtype=float), 3),
        ),
        h1=csc_matrix(first),
        h2=csc_matrix(20 - first),
        het=np.ones(60),
    )
    result = api.infer(
        data,
        bootstraps=20,
        draws=16,
        skip_signal=True,
        hf_refinement=False,
        clone_method=method,
    )
    assert result.clone_calls.evidence == "self_expression_and_hf"
    assert result.clone_calls.feature_info["hf_status"] == "fitted_phase"
    assert result.hf_status == "disabled"  # Only the subsequent refinement is off.
    assert result.hf_features is not None and "boot" not in result.hf_features
    assert result.distances is None
    np.testing.assert_array_equal(result.groups, result.pre_hf_groups)
