"""Independent mixture-integral and joint-path checks for the PLN/phase option."""

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import poisson

from barcodecnv import (
    AlleleCounts,
    DepthOptions,
    GeneCounts,
    Grid,
    Loci,
    Model,
    Parameters,
    PreparedCounts,
)
from barcodecnv.depth import poilog_logpmf
from barcodecnv.fitting import infer_cn
from barcodecnv.joint import infer_joint
from barcodecnv.model import STATES, allele_terms, depth_emissions


@pytest.mark.parametrize(
    "count,rate,sigma,expected",
    [
        # Independent high-precision/QUADPACK references retained in Julia's tests.
        (0, 0.1, 0.6, -0.116785798184283),
        (0, 10.0, 2.0, -2.22153883987362),
        (50, 50.0, 0.01, -2.87911051248876),
        (501, 1002.0 * 1.5 * np.exp(0.2), 0.6, -8.960366271751202),
        (1_000_000, 1_000_000.0, 0.6, -14.2236248569750),
        # Direct integration over a standard normal latent variable, without mode
        # rescaling. The former tanh–sinh route stopped early on these low counts.
        (1, 0.23422732534336901, 0.9, -1.638156035061565),
        (1, 8.264928196092937, 2.0, -2.388161531138224),
    ],
)
def test_poilog_independent_integrals(count, rate, sigma, expected):
    assert poilog_logpmf(count, rate, sigma) == pytest.approx(expected, abs=1e-8, rel=0)


def test_poilog_rejects_failed_quadrature(monkeypatch):
    import barcodecnv.depth as depth

    monkeypatch.setattr(depth, "quad", lambda *a, **k: (2.0, 0.1, {}, "limit reached"))
    with pytest.raises(ArithmeticError, match="did not converge"):
        poilog_logpmf(1, 2.0, 0.6)


def test_poilog_normalization_mean_and_poisson_limit():
    y = np.arange(201)
    p = np.exp(poilog_logpmf(y, 2.0, 0.6))
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert (y * p).sum() == pytest.approx(2 * np.exp(0.6**2 / 2), abs=1e-7)
    np.testing.assert_array_equal(poilog_logpmf(y, 2.0, 0.0), poisson.logpmf(y, 2.0))
    np.testing.assert_array_equal(poilog_logpmf([0, 1], 0.0, 0.6), [0.0, -np.inf])


def test_poilog_matches_independent_julia_quadrature():
    table = pd.read_csv(Path(__file__).parent / "fixtures/julia_pln.csv")
    for sigma, rows in table.groupby("sigma"):
        actual = poilog_logpmf(rows["count"], rows.reference * rows.fold, sigma)
        np.testing.assert_allclose(actual, rows.julia, atol=1e-8, rtol=0)


def example():
    return PreparedCounts(
        Grid(("chr1", "chr1"), [1, 1_000_001], [1, 1_000_001]),
        ("b",),
        GeneCounts([0, 0], [0, 1], [4, 6], [10.0, 10.0], [1.0, 1.0]),
        AlleleCounts([0, 0], [0, 1], [9, 2], [1, 8], [1.0, 1.0]),
        Loci([0, 1], [1, 1_000_001], [0.9, 0.7], [0.0, 0.3]),
    )


@pytest.mark.parametrize("epsilon", [0.0, 0.05])
def test_joint_cn_phase_matches_complete_path_enumeration(epsilon):
    d = example()
    m = Model(phase_locus_weight=0.5)
    p = Parameters(0.3)
    depth = depth_emissions(d, p, m)[0]
    if epsilon:
        q = poilog_logpmf(d.genes.count, d.genes.expected * np.exp(p.mu), 3.0)
        depth = np.logaddexp(np.log1p(-epsilon) + depth, np.log(epsilon) + q[:, None])
    alleles = allele_terms(d, m, p)
    ct = m.cn_transitions(d.grid)[0]
    pt = m.phase_transitions(d)[0]
    scores = []
    paths = []
    for c0, c1, z0, z1 in product(
        range(len(STATES)), range(len(STATES)), range(2), range(2)
    ):
        value = np.log(m.initial[c0]) - np.log(2) + ct[c0, c1] + pt[z0, z1]
        for t, c, z in [(0, c0, z0), (1, c1, z1)]:
            pop = d.loci.population_phase[t]
            value += (
                depth[t, c]
                + alleles[t, c, z]
                + m.phase_locus_weight * np.log(pop if z == 0 else 1 - pop)
            )
        scores.append(value)
        paths.append([c0, c1, z0, z1])
    paths = np.array(paths)
    weights = np.exp(scores - logsumexp(scores))
    cn = np.array(
        [
            [weights[paths[:, t] == c].sum() for c in range(len(STATES))]
            for t in range(2)
        ]
    )
    phase = np.array([weights[paths[:, t + 2] == 0].sum() for t in range(2)])
    result = infer_joint(d, m, p, depth_options=DepthOptions(epsilon))
    np.testing.assert_allclose(result.probabilities[0], cn, atol=1e-12)
    np.testing.assert_allclose(result.phase[0], phase, atol=1e-12)
    assert result.log_evidence[0] == pytest.approx(logsumexp(scores), abs=1e-12)
    draws = result.chains[0].sample_paths(40000, 19)
    expected_same = weights[paths[:, 0] == paths[:, 1]].sum()
    assert np.mean(draws[:, 0] == draws[:, 1]) == pytest.approx(expected_same, abs=0.01)


def test_joint_phase_resets_and_unobserved_phase_do_not_change_depth():
    d = example()
    p = Parameters(0.3)
    m = Model(depth_family="pln")
    d = replace(d, alleles=AlleleCounts([], [], [], [], []))
    joint = infer_joint(d, m, p)
    np.testing.assert_allclose(joint.classes, infer_cn(d, m, p).classes, atol=1e-12)
    # Two chromosomes decouple the first chromosome's allele evidence.
    d = replace(
        example(),
        grid=Grid(("chr1", "chr2"), [1, 1], [1, 1]),
        loci=Loci([0, 1], [1, 1], [0.9, 0.7], [0.0, 0.0]),
    )
    original = infer_joint(d, m, p)
    changed = replace(d, alleles=replace(d.alleles, h1=[100, 2], h2=[0, 8]))
    np.testing.assert_allclose(
        infer_joint(changed, m, p).classes[:, 1], original.classes[:, 1], atol=1e-12
    )


def test_joint_gene_only_grid_insertions_preserve_snp_phase_transitions():
    d = example()
    m = Model()
    p = Parameters(0.3)
    inserted = replace(
        d,
        grid=Grid(("chr1",) * 3, [1, 500001, 1000001], [1, 500001, 1000001]),
        genes=replace(d.genes, marker=[0, 2]),
        loci=replace(d.loci, marker=[0, 2]),
    )
    original = infer_joint(d, m, p)
    new = infer_joint(inserted, m, p)
    np.testing.assert_allclose(new.classes[:, [0, 2]], original.classes, atol=1e-12)
    np.testing.assert_allclose(new.phase, original.phase, atol=1e-12)
