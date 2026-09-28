"""Shared-phase coordinate updates and conditional empirical dispersion fits."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize_scalar

from .data import PreparedCounts, owned
from .depth_controls import DepthOptions, DepthWorkspace
from .hmm import HMMEngine
from .model import (
    Model,
    Parameters,
    allele_emissions,
    allele_terms,
    class_probabilities,
    estimate_parameters,
)


@dataclass(frozen=True)
class FitOptions:
    initialization: str = "multistart"
    max_iterations: int = 60
    tolerance: float = 1e-4
    dispersion_max_iterations: int = 35
    dispersion_tolerance: float = 1e-3
    # Preserve the Julia benchmark search domain, expressed in native NB alpha.
    alpha_bounds: tuple[float, float] = (
        float(np.expm1(0.02**2)),
        float(np.expm1(2.0**2)),
    )
    barcode_phase: str = "joint"

    def __post_init__(self):
        if self.barcode_phase not in ("conditional", "joint"):
            raise ValueError("barcode_phase must be conditional or joint")
        if self.initialization not in ("multistart", "population", "pooled_imbalance"):
            raise ValueError("unknown phase initialization")
        if (
            not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
            or not isinstance(self.dispersion_max_iterations, int)
            or self.dispersion_max_iterations < 1
        ):
            raise ValueError("iteration limits must be positive integers")
        if (
            not 0 < self.tolerance < np.inf
            or not 0 < self.dispersion_tolerance < np.inf
        ):
            raise ValueError("tolerances must be finite and positive")
        lo, hi = self.alpha_bounds
        if not 0 < lo < hi < np.inf:
            raise ValueError("invalid dispersion bounds")


@dataclass(frozen=True)
class CNPosterior:
    """CN chains and phase probabilities.

    Phase has locus axes for the shared conditional fit, or group × locus axes
    for exact per-group joint chains. JointChain projects sampled whole paths
    onto their CN component; it never samples position marginals independently.
    """

    data: PreparedCounts
    model: Model
    parameters: Parameters
    phase: np.ndarray
    chains: tuple
    depth_options: DepthOptions = DepthOptions()

    @property
    def probabilities(self):
        """Axes: group × genomic position × detailed oriented CN state."""
        return np.stack([p.probabilities for p in self.chains])

    @property
    def classes(self):
        """Axes: group × position × (diploid, gain, loss, cnloh)."""
        return class_probabilities(self.probabilities)

    @property
    def log_evidence(self):
        return np.array([p.log_evidence for p in self.chains])

    def sample_paths(self, draws=256, seed=None):
        """Zero-based state indices; axes draw × group × genomic position."""
        seeds = np.random.SeedSequence(seed).spawn(len(self.chains))
        return np.stack(
            [p.sample_paths(draws, s) for p, s in zip(self.chains, seeds)], axis=1
        )


class _Workspace:
    def __init__(self, data, model, parameters, depth_options):
        self.data, self.model, self.parameters = data, model, parameters
        self.engine = HMMEngine(model.cn_transitions(data.grid), np.log(model.initial))
        self.depth_options = depth_options
        self.depth = DepthWorkspace(data, model, depth_options).emissions(parameters)
        self.terms = allele_terms(data, model, parameters)

    def infer(self, phase):
        emissions = self.depth + allele_emissions(self.data, self.terms, phase)
        chains = tuple(self.engine.infer(e) for e in emissions)
        return CNPosterior(
            self.data,
            self.model,
            self.parameters,
            owned(phase, float),
            chains,
            self.depth_options,
        )


def infer_cn(data, model, parameters, phase=None, *, depth_options=DepthOptions()):
    if phase is None:
        phase = np.clip(data.loci.population_phase, 1e-6, 1 - 1e-6)
    return _Workspace(data, model, parameters, depth_options).infer(phase)


@dataclass(frozen=True)
class PhaseFit:
    posterior: CNPosterior
    converged: bool
    iterations: int
    initialization: str
    variational_bound: float
    changes: tuple[float, ...]

    @property
    def phase(self):
        return self.posterior.phase


def fit_phase(
    data, model, parameters, options=FitOptions(), *, depth_options=DepthOptions()
):
    workspace = _Workspace(data, model, parameters, depth_options)
    population = np.clip(data.loci.population_phase, 1e-6, 1 - 1e-6)
    starts = (
        ("population", "pooled_imbalance")
        if options.initialization == "multistart"
        else (options.initialization,)
    )
    if len(population) == 0:
        post = workspace.infer(population)
        return PhaseFit(post, True, 1, starts[0], float(post.log_evidence.sum()), ())
    phase_engine = HMMEngine(model.phase_transitions(data), np.log([0.5, 0.5]))
    unary = model.phase_locus_weight * np.stack(
        (np.log(population), np.log1p(-population)), axis=-1
    )
    best = None
    a = data.alleles
    for start in starts:
        phase = population.copy()
        if start == "pooled_imbalance":
            imbalance = np.zeros(len(population))
            np.add.at(imbalance, a.locus, a.het * (a.h1 - a.h2))
            phase[imbalance > 0] = 0.99
            phase[imbalance < 0] = 0.01
        changes = []
        converged = False
        for _iteration in range(1, options.max_iterations + 1):
            cn = workspace.infer(phase)
            marginals = cn.probabilities[a.group, data.loci.marker[a.locus]]
            potentials = unary.copy()
            values = np.einsum("rs,rso->ro", marginals, workspace.terms)
            np.add.at(potentials, a.locus, values)
            updated = phase_engine.infer(potentials)
            difference = float(np.max(np.abs(updated.probabilities[:, 0] - phase)))
            changes.append(difference)
            phase = updated.probabilities[:, 0]
            prior_entropy = updated.log_evidence - np.sum(
                updated.probabilities * (potentials - unary)
            )
            if difference < options.tolerance:
                converged = True
                break
        final = workspace.infer(phase)
        candidate = PhaseFit(
            final,
            converged,
            _iteration,
            start,
            float(final.log_evidence.sum() + prior_entropy),
            tuple(changes),
        )
        if best is None or candidate.variational_bound > best.variational_bound:
            best = candidate
    return best


@dataclass(frozen=True)
class DispersionFit:
    parameters: Parameters
    initial_parameters: Parameters
    converged: bool
    at_boundary: bool
    log_likelihood_gain: float
    trace: tuple[tuple[float, float], ...]


def fit_dispersion(
    data, model, phase, options=FitOptions(), *, depth_options=DepthOptions()
):
    initial = estimate_parameters(data)
    engine = HMMEngine(model.cn_transitions(data.grid), np.log(model.initial))
    alleles = allele_emissions(data, allele_terms(data, model, initial), phase)

    depth = DepthWorkspace(data, model, depth_options)

    def score(parameters):
        emissions = depth.emissions(parameters) + alleles
        return sum(engine.infer(e, marginals=False) for e in emissions)

    return _fit_dispersion_objective(initial, score, options)


def _fit_dispersion_objective(initial, score, options):
    """One bounded noise search, reused by conditional and joint HMMs."""
    trace = []

    def objective(log_sigma):
        # Same search coordinate as Julia; alpha is the public parameter.
        parameters = replace(initial, alpha=float(np.expm1(np.exp(2 * log_sigma))))
        objective_value = -score(parameters)
        trace.append((parameters.alpha, objective_value))
        return objective_value

    low, high = (0.5 * np.log(np.log1p(a)) for a in options.alpha_bounds)
    baseline = objective(0.5 * np.log(np.log1p(initial.alpha)))
    result = minimize_scalar(
        objective,
        bounds=(low, high),
        method="bounded",
        options={
            "xatol": options.dispersion_tolerance,
            "maxiter": options.dispersion_max_iterations,
        },
    )
    chosen = (
        float(np.expm1(np.exp(2 * result.x)))
        if result.fun <= baseline
        else initial.alpha
    )
    coordinate = 0.5 * np.log(np.log1p(chosen))
    boundary = coordinate <= low + 0.011 or coordinate >= high - 0.011
    return DispersionFit(
        replace(initial, alpha=chosen),
        initial,
        bool(result.success),
        bool(boundary),
        float(baseline - min(baseline, result.fun)),
        tuple(trace),
    )


@dataclass(frozen=True)
class PooledFit:
    initial_phase: PhaseFit
    dispersion: DispersionFit
    final_phase: PhaseFit

    @property
    def converged(self):
        return (
            self.initial_phase.converged
            and self.dispersion.converged
            and self.final_phase.converged
        )


def fit_phase_and_dispersion(
    pooled, model=Model(), *, options=FitOptions(), depth_options=DepthOptions()
):
    initial = fit_phase(
        pooled, model, estimate_parameters(pooled), options, depth_options=depth_options
    )
    if not initial.converged:
        raise RuntimeError(
            "initial pooled phase did not converge; inspect fit_phase before continuing"
        )
    noise = fit_dispersion(
        pooled, model, initial.phase, options, depth_options=depth_options
    )
    if not noise.converged:
        raise RuntimeError("pooled dispersion search did not converge")
    final = fit_phase(
        pooled, model, noise.parameters, options, depth_options=depth_options
    )
    return PooledFit(initial, noise, final)


@dataclass(frozen=True)
class BarcodeFit:
    posterior: CNPosterior
    dispersion: DispersionFit
    pooled_fit: PooledFit

    @property
    def classes(self):
        return self.posterior.classes

    def sample_paths(self, draws=256, seed=None):
        return self.posterior.sample_paths(draws, seed)


def infer_barcodes(barcodes, model, fit, *, options=FitOptions()):
    if not fit.converged:
        raise ValueError("barcode inference requires converged pooled fitting")
    depth_options = fit.final_phase.posterior.depth_options
    pooled = fit.final_phase.posterior.data
    if model != fit.final_phase.posterior.model or not barcodes.grid.same_as(
        pooled.grid
    ):
        raise ValueError("model and genomic grid must agree with pooled fit")
    for name in ("marker", "position", "population_phase", "genetic_cm"):
        if not np.array_equal(getattr(barcodes.loci, name), getattr(pooled.loci, name)):
            raise ValueError("barcode and pooled SNP catalogues must agree")
    if options.barcode_phase == "joint":
        from .joint import fit_joint_dispersion, infer_joint

        noise = fit_joint_dispersion(
            barcodes, model, options=options, depth_options=depth_options
        )
        if not noise.converged:
            raise RuntimeError("joint barcode dispersion search did not converge")
        post = infer_joint(
            barcodes, model, noise.parameters, depth_options=depth_options
        )
        return BarcodeFit(post, noise, fit)
    noise = fit_dispersion(
        barcodes, model, fit.final_phase.phase, options, depth_options=depth_options
    )
    if not noise.converged:
        raise RuntimeError("barcode dispersion search did not converge")
    post = infer_cn(
        barcodes,
        model,
        noise.parameters,
        fit.final_phase.phase,
        depth_options=depth_options,
    )
    return BarcodeFit(post, noise, fit)
