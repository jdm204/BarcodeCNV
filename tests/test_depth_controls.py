"""Independent depth-mixture probabilities and preservation of HF evidence."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import poisson
from test_joint_depth import example

from barcodecnv import Model, Parameters
from barcodecnv.depth_controls import DepthOptions
from barcodecnv.joint import JointWorkspace, infer_joint
from barcodecnv.model import STATES, depth_emissions


def test_depth_mixture_matches_independent_poisson_calculation():
    d = example()
    parameters = Parameters(0.0)
    model = Model(depth_family="nb")
    q = poisson.logpmf(d.genes.count, 3.0)
    got = depth_emissions(
        d, parameters, model, outlier_probability=0.1, outlier_log_likelihood=q
    )[0]
    for c, state in enumerate(STATES):
        expected = np.log(
            0.9 * poisson.pmf(d.genes.count, 10 * state.fold) + 0.1 * np.exp(q)
        )
        np.testing.assert_allclose(got[:, c], expected, atol=1e-13)


def test_extreme_depth_outlier_loses_cn_evidence_but_not_allele_evidence():
    d = example()
    d = replace(d, genes=replace(d.genes, count=[0, 0], expected=[10000.0, 10000.0]))
    model = Model()
    p = Parameters(0.2)
    raw = JointWorkspace(d, model, p, depth_options=DepthOptions(0.0))
    robust = JointWorkspace(
        d, model, p, depth_options=DepthOptions(outlier_probability=0.05)
    )
    a = next(raw.emissions(p))
    b = next(robust.emissions(p))
    # The two phase likelihoods differ only through allele counts, exactly as
    # before. Count outliers must not temper haplotype evidence as a side effect.
    np.testing.assert_allclose(
        a[:, :, 0] - a[:, :, 1], b[:, :, 0] - b[:, :, 1], atol=1e-12
    )
    np.testing.assert_array_equal(raw.engine.transitions, robust.engine.transitions)
    # Removing alleles isolates the loss of extreme expression-driven evidence.
    neutralized = replace(d, alleles=replace(d.alleles, h1=[0, 0], h2=[0, 0]))
    old = next(
        JointWorkspace(
            neutralized, model, p, depth_options=DepthOptions(0.0)
        ).emissions(p)
    )
    new = next(
        JointWorkspace(
            neutralized, model, p, depth_options=DepthOptions(outlier_probability=0.05)
        ).emissions(p)
    )
    assert np.ptp(old[0, :, 0]) > 10
    assert np.ptp(new[0, :, 0]) < 0.01


def test_depth_controls_cannot_change_an_allele_only_posterior():
    d = example()
    d = replace(d, genes=replace(d.genes, expected=[0.0, 0.0]))
    model = Model()
    p = Parameters(0.2)
    expected = infer_joint(d, model, p).classes
    option = DepthOptions(outlier_probability=0.1)
    np.testing.assert_allclose(
        infer_joint(d, model, p, depth_options=option).classes, expected, atol=1e-13
    )


@pytest.mark.parametrize("kwargs", [{"outlier_probability": 1}, {"outlier_log_sd": 0}])
def test_invalid_depth_controls(kwargs):
    with pytest.raises(ValueError):
        DepthOptions(**kwargs)


@pytest.mark.parametrize("epsilon,phase", [(0.0, "conditional"), (0.12, "joint")])
def test_workflow_uses_requested_mixture_in_all_noise_fits(epsilon, phase):
    from test_cli import example_bundle

    from barcodecnv import FitOptions, infer_cn
    from barcodecnv.workflow import run_pbpc

    bundle = example_bundle()
    control = DepthOptions(epsilon)
    result = run_pbpc(
        bundle,
        replicates=20,
        cn_refinement=False,
        hf_refinement=False,
        options=FitOptions(barcode_phase=phase),
        depth_options=control,
    )
    fit = result["fit"]
    pooled = fit.pooled_fit
    groups = result["group_calls"]
    for post in (
        pooled.initial_phase.posterior,
        pooled.final_phase.posterior,
        fit.posterior,
    ):
        assert post.depth_options == control
    assert groups.depth_options == control
    # Each optimizer's initial objective must use the requested mixture, not
    # merely record it or apply it only after fitting inlier dispersion.
    post = pooled.initial_phase.posterior
    expected = infer_cn(
        post.data,
        post.model,
        pooled.dispersion.initial_parameters,
        pooled.initial_phase.phase,
        depth_options=control,
    ).log_evidence.sum()
    assert pooled.dispersion.trace[0][1] == pytest.approx(-expected)
    post = fit.posterior
    if phase == "joint":
        check = infer_joint(
            post.data,
            post.model,
            fit.dispersion.initial_parameters,
            depth_options=control,
        )
    else:
        check = infer_cn(
            post.data,
            post.model,
            fit.dispersion.initial_parameters,
            pooled.final_phase.phase,
            depth_options=control,
        )
    assert fit.dispersion.trace[0][1] == pytest.approx(-check.log_evidence.sum())
    check = infer_joint(
        bundle.prepared(result["groups"], exclude_unresolved=True),
        post.model,
        groups.dispersion.initial_parameters,
        depth_options=control,
    )
    assert groups.dispersion.trace[0][1] == pytest.approx(-check.log_evidence.sum())


@pytest.mark.parametrize("value", ["-0.1", "1", "nan", "inf"])
def test_cli_rejects_invalid_outlier_probability_before_loading(value):
    from barcodecnv.cli import parser

    with pytest.raises(SystemExit) as exc:
        parser().parse_args(
            [
                "run",
                "absent.h5",
                "--out",
                "unused",
                "--depth-outlier-probability",
                value,
            ]
        )
    assert exc.value.code == 2
