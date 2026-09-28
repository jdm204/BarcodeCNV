import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from barcodecnv import (
    AlleleCounts,
    DepthOptions,
    FitOptions,
    GeneCounts,
    Grid,
    Loci,
    Model,
    Parameters,
    PreparedCounts,
    fit_dispersion,
    fit_phase,
    fit_phase_and_dispersion,
    infer_barcodes,
    infer_cn,
)
from barcodecnv.model import (
    STATES,
    allele_emissions,
    allele_terms,
    depth_emissions,
    estimate_parameters,
    nb_logpmf,
)


def example():
    grid = Grid(
        ("chr1",) * 4 + ("chr2",) * 2,
        [1, 1000001, 2000001, 3000001, 1, 1000001],
        [1, 1000001, 2000001, 3000001, 1, 1000001],
    )
    genes = GeneCounts(
        np.repeat([0, 1], 6),
        np.tile(np.arange(6), 2),
        [18, 20, 23, 19, 10, 11, 10, 11, 9, 10, 19, 21],
        np.full(12, 10.0),
        np.ones(12),
    )
    loci = Loci(
        [0, 2, 4, 5],
        [1, 2000001, 1, 1000001],
        [0.9, 0.8, 0.9, 0.7],
        [0.0, 0.2, 0.0, 0.1],
    )
    alleles = AlleleCounts(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 3],
        [14, 16, 5, 6, 5, 14],
        [6, 4, 5, 4, 5, 6],
        np.ones(6),
    )
    return PreparedCounts(grid, ("01", "02"), genes, alleles, loci)


def test_nb_geometric_and_poisson_limits():
    assert nb_logpmf(3, 6.0, 1.0) == pytest.approx(
        -np.log(7) + 3 * np.log(6 / 7), abs=1e-14
    )
    assert nb_logpmf(0, 3.0, 0.0) == pytest.approx(-3.0)
    assert nb_logpmf(0, 0.0, 0.3) == 0.0


def test_prior_mass_belongs_to_configuration_not_orientation():
    model = Model()
    assert model.initial[0] == 0.8
    for label in ("del", "loh", "gain", "amp", "bdel"):
        assert sum(
            p for p, s in zip(model.initial, STATES) if s.label == label
        ) == pytest.approx(0.04)


def test_allele_orientation_is_expected_log_likelihood_not_mixture():
    d = example()
    model = Model()
    p = Parameters(0.2)
    terms = allele_terms(d, model, p)
    q = np.full(4, 0.25)
    got = allele_emissions(d, terms, q)
    np.testing.assert_allclose(got[0, 0], 0.25 * terms[0, :, 0] + 0.75 * terms[0, :, 1])
    mixture = logsumexp(terms[0] + np.log([0.25, 0.75]), axis=-1)
    assert np.max(np.abs(got[0, 0] - mixture)) > 1
    # Swapping H1/H2 and complementing phase preserves CN emissions.
    swapped = replace(d, alleles=replace(d.alleles, h1=d.alleles.h2, h2=d.alleles.h1))
    np.testing.assert_allclose(
        got, allele_emissions(swapped, allele_terms(swapped, model, p), 1 - q)
    )


def test_zero_reference_and_weights_remove_only_their_evidence():
    d = example()
    g = replace(d.genes, expected=np.zeros(12))
    assert np.all(depth_emissions(replace(d, genes=g), Parameters(0.2)) == 0)
    g = replace(d.genes, weight=np.zeros(12))
    assert np.all(depth_emissions(replace(d, genes=g), Parameters(0.2)) == 0)


def test_phase_transition_uses_genetic_distance_and_chromosome_resets():
    d = example()
    tr = np.exp(Model().phase_transitions(d))
    assert tr[0, 0, 1] == pytest.approx((1 - np.exp(-0.4)) / 2)
    np.testing.assert_allclose(tr[1], 0.5)
    flat = replace(d, loci=replace(d.loci, genetic_cm=np.zeros(4)))
    np.testing.assert_array_equal(np.exp(Model().phase_transitions(flat))[0], np.eye(2))


def test_phase_fit_final_marginals_and_reported_nonconvergence():
    d = example()
    m = Model()
    p = Parameters(0.15)
    fit = fit_phase(d, m, p, FitOptions(max_iterations=100))
    assert fit.converged
    np.testing.assert_allclose(
        fit.posterior.probabilities,
        infer_cn(d, m, p, fit.phase).probabilities,
        atol=1e-13,
    )
    stopped = fit_phase(d, m, p, FitOptions(max_iterations=1, tolerance=1e-12))
    assert not stopped.converged


def test_dispersion_fit_improves_fixed_phase_objective_and_exposes_failure():
    d = example()
    m = Model()
    q = d.loci.population_phase
    initial = estimate_parameters(d)
    result = fit_dispersion(d, m, q)
    assert result.converged
    improvement = (
        infer_cn(d, m, result.parameters, q).log_evidence.sum()
        - infer_cn(d, m, initial, q).log_evidence.sum()
    )
    assert improvement >= -1e-9
    assert improvement == pytest.approx(result.log_likelihood_gain, abs=1e-9)
    assert not fit_dispersion(
        d, m, q, FitOptions(dispersion_max_iterations=1)
    ).converged


def test_count_validation_and_owned_arrays():
    d = example()
    original = np.arange(12)
    g = replace(d.genes, count=original)
    original[:] = 0
    assert g.count[-1] == 11
    with pytest.raises(ValueError):
        replace(d.genes, count=np.full(12, 0.5))
    with pytest.raises(ValueError):
        replace(d, group_ids=("a", "a"))
    with pytest.raises(ValueError):
        replace(d, alleles=replace(d.alleles, locus=np.full(6, 4)))


def test_independently_generated_julia_nb_fixture():
    # Saved expectations were generated independently by the Julia implementation,
    # not by the Python implementation under test.
    with (Path(__file__).parent / "fixtures/julia_nb.toml").open("rb") as f:
        expected = tomllib.load(f)
    d = example()
    m = Model(depth_family="nb")
    p = Parameters(0.15)
    np.testing.assert_allclose(
        depth_emissions(d, p, m)[0], expected["depth_first"], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        allele_emissions(d, allele_terms(d, m, p), d.loci.population_phase)[0],
        expected["allele_first"],
        atol=1e-12,
        rtol=0,
    )
    fit = fit_phase(
        d, m, p, FitOptions(max_iterations=100), depth_options=DepthOptions(0.0)
    )
    np.testing.assert_allclose(fit.phase, expected["phase"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        fit.posterior.probabilities,
        [expected["posterior_first"], expected["posterior_second"]],
        atol=1e-10,
        rtol=0,
    )
    assert fit.variational_bound == pytest.approx(expected["bound"], abs=1e-10)


def test_barcode_inference_rejects_unconverged_or_misaligned_fit():
    d = example()
    m = Model()
    fit = fit_phase_and_dispersion(d, m, options=FitOptions(max_iterations=100))
    assert fit.converged
    bad = replace(fit, final_phase=replace(fit.final_phase, converged=False))
    with pytest.raises(ValueError, match="converged"):
        infer_barcodes(d, m, bad)
    with pytest.raises(ValueError, match="model and genomic grid"):
        infer_barcodes(d, replace(m, diploid_probability=0.9), fit)
    shifted = replace(d, loci=replace(d.loci, genetic_cm=d.loci.genetic_cm + 0.1))
    with pytest.raises(ValueError, match="catalogues"):
        infer_barcodes(shifted, m, fit)


@pytest.mark.parametrize("with_snp", [False, True])
def test_single_marker_fit_with_optional_phase_chain(with_snp):
    loci = Loci([0], [1], [0.9], [0.0]) if with_snp else Loci([], [], [], [])
    alleles = (
        AlleleCounts([0], [0], [15], [5], [1.0])
        if with_snp
        else AlleleCounts([], [], [], [], [])
    )
    d = PreparedCounts(
        Grid(("chr1",), [1], [1]),
        ("barcode",),
        GeneCounts([0], [0], [15], [10.0], [1.0]),
        alleles,
        loci,
    )
    m = Model()
    p = Parameters(0.2)
    fit = fit_phase(d, m, p, depth_options=DepthOptions(0.0))
    assert fit.converged
    # A single-marker HMM reduces to direct Bayes updating of state weights.
    emission = depth_emissions(d, p) + allele_emissions(
        d, allele_terms(d, m, p), fit.phase
    )
    log_joint = np.log(m.initial) + emission[0, 0]
    np.testing.assert_allclose(
        fit.posterior.probabilities[0, 0],
        np.exp(log_joint - logsumexp(log_joint)),
        atol=1e-12,
    )
