"""Julia CNVModel1 catalogue/priors with selectable NB or PLN depth emissions."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import betabinom, nbinom, poisson

from .data import PreparedCounts


@dataclass(frozen=True)
class State:
    label: str
    h1: int
    h2: int
    fold: float
    broad: int


STATES = tuple(
    State(label, a, b, fold, broad)
    for label, major, minor, fold, broad in (
        ("diploid", 1, 1, 1.0, 0),
        ("del", 1, 0, 0.5, 2),
        ("loh", 2, 0, 1.0, 3),
        ("gain", 2, 1, 1.5, 1),
        ("amp", 5, 1, 3.0, 1),
        ("bdel", 0, 0, 0.05, 2),
    )
    for a, b in (
        ((major, minor),) if major == minor else ((major, minor), (minor, major))
    )
)
CLASS_NAMES = ("diploid", "gain", "loss", "cnloh")


@dataclass(frozen=True)
class Model:
    correlation_length_bp: float = 30e6
    diploid_probability: float = 0.8
    phase_switch_rate_per_cm: float = 1.0
    phase_locus_weight: float = 0.0
    allele_concentration: float = 20.0
    allele_background: float = 0.1
    depth_family: str = "pln"

    def __post_init__(self):
        if self.depth_family not in ("nb", "pln"):
            raise ValueError("depth_family must be nb or pln")
        if not all(
            np.isfinite(v) for k, v in vars(self).items() if k != "depth_family"
        ):
            raise ValueError("model parameters must be finite")
        if (
            self.correlation_length_bp <= 0
            or not 0 < self.diploid_probability < 1
            or min(self.allele_concentration, self.allele_background) <= 0
            or min(self.phase_switch_rate_per_cm, self.phase_locus_weight) < 0
        ):
            raise ValueError("invalid model parameters")

    @property
    def initial(self):
        return np.array(
            [
                self.diploid_probability
                if s.label == "diploid"
                else (1 - self.diploid_probability) / 5 / (1 if s.h1 == s.h2 else 2)
                for s in STATES
            ]
        )

    def cn_transitions(self, grid):
        positions = (grid.start.astype(float) + grid.end) / 2
        distances = np.diff(positions)
        boundary = np.array(
            [a != b for a, b in zip(grid.chrom[:-1], grid.chrom[1:])], dtype=bool
        )
        distances[boundary] = np.inf
        persistence = np.exp(-distances / self.correlation_length_bp)
        reset = -np.expm1(-distances / self.correlation_length_bp)
        transitions = (
            reset[:, None, None]
            * np.broadcast_to(
                self.initial, (len(distances), len(STATES), len(STATES))
            ).copy()
        )
        transitions += persistence[:, None, None] * np.eye(len(STATES))
        with np.errstate(divide="ignore"):
            return np.log(transitions)

    def phase_transitions(self, data):
        loc = data.loci
        distance = np.diff(loc.genetic_cm)
        boundary = np.array(
            [
                data.grid.chrom[a] != data.grid.chrom[b]
                for a, b in zip(loc.marker[:-1], loc.marker[1:])
            ],
            dtype=bool,
        )
        distance[boundary] = 0
        switch = -np.expm1(-2 * self.phase_switch_rate_per_cm * distance) / 2
        switch[boundary] = 0.5
        tr = np.empty((len(switch), 2, 2))
        tr[:, 0, 0] = tr[:, 1, 1] = 1 - switch
        tr[:, 0, 1] = tr[:, 1, 0] = switch
        with np.errstate(divide="ignore"):
            return np.log(tr)


@dataclass(frozen=True)
class Parameters:
    alpha: float
    mu: float = 0.0
    allele_logit_bias: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()) or self.alpha < 0:
            raise ValueError("dispersion must be nonnegative and all parameters finite")


def estimate_parameters(data: PreparedCounts):
    """Julia heuristic initialization only; fitted dispersion is a separate step."""
    g, a = data.genes, data.alleles
    keep = (g.count > 0) & (g.expected > 0)
    ratios = np.log(g.count[keep] / g.expected[keep])
    sigma = (
        0.5
        if len(ratios) < 2
        else max(0.05, 1.4826 * np.median(np.abs(ratios - np.median(ratios))))
    )
    total = float(a.h1.sum()) + float(a.h2.sum())
    fraction = 0.5 if total == 0 else np.clip(a.h1.sum() / total, 0.45, 0.55)
    return Parameters(
        float(np.expm1(sigma * sigma)), allele_logit_bias=float(logit(fraction))
    )


def nb_logpmf(count, mean, alpha):
    """NB2 parameterization; alpha=0 is its exact Poisson limit."""
    if alpha == 0:
        return poisson.logpmf(count, mean)
    return nbinom.logpmf(count, 1 / alpha, 1 / (1 + alpha * mean))


def depth_emissions(
    data,
    parameters,
    model=Model(),
    *,
    outlier_probability=0.0,
    outlier_log_likelihood=None,
):
    g = data.genes
    # No usable reference means no depth evidence, matching the Julia adapter.
    keep = (g.expected > 0) & (g.weight > 0)
    folds, inverse = np.unique([s.fold for s in STATES], return_inverse=True)
    values = np.zeros((*data.shape, len(folds)))
    for j, fold in enumerate(folds):
        rate = g.expected[keep] * fold * np.exp(parameters.mu)
        if model.depth_family == "pln":
            from .depth import poilog_logpmf

            likelihood = poilog_logpmf(
                g.count[keep], rate, np.sqrt(np.log1p(parameters.alpha))
            )
        else:
            likelihood = nb_logpmf(g.count[keep], rate, parameters.alpha)
        if outlier_probability:
            # A shared, CN-independent contamination distribution: an outlier
            # may explain the observation without preferring a copy-number state.
            if not 0 < outlier_probability < 1 or outlier_log_likelihood is None:
                raise ValueError(
                    "outlier mixture requires a probability and count log likelihoods"
                )
            outlier = np.asarray(outlier_log_likelihood)
            if outlier.shape != likelihood.shape or np.any(~np.isfinite(outlier)):
                raise ValueError("invalid outlier log likelihoods")
            likelihood = np.logaddexp(
                np.log1p(-outlier_probability) + likelihood,
                np.log(outlier_probability) + outlier,
            )
        np.add.at(
            values[:, :, j],
            (g.group[keep], g.marker[keep]),
            g.weight[keep] * likelihood,
        )
    return np.ascontiguousarray(values[:, :, inverse])


def allele_terms(data, model, parameters):
    """Record × state × orientation log likelihoods, weighted by heterozygosity."""
    a = data.alleles
    fraction = np.array(
        [
            0.5
            if s.label == "bdel"
            else (s.h1 + model.allele_background)
            / (s.h1 + s.h2 + 2 * model.allele_background)
            for s in STATES
        ]
    )
    fraction = np.clip(
        expit(logit(fraction) + parameters.allele_logit_bias), 1e-6, 1 - 1e-6
    )
    aa, bb = (
        fraction * model.allele_concentration,
        (1 - fraction) * model.allele_concentration,
    )
    terms = np.empty((len(a.h1), len(STATES), 2))
    for i, counts in enumerate((a.h1, a.h2)):
        terms[:, :, i] = a.het[:, None] * betabinom.logpmf(
            counts[:, None], (a.h1 + a.h2)[:, None], aa, bb
        )
    return terms


def allele_emissions(data, terms, phase):
    q = np.asarray(phase, dtype=float)
    if (
        q.shape != (len(data.loci),)
        or np.any(~np.isfinite(q))
        or np.any((q < 0) | (q > 1))
    ):
        raise ValueError("phase must cover every prepared SNP locus with a probability")
    a = data.alleles
    values = q[a.locus, None] * terms[:, :, 0] + (1 - q[a.locus, None]) * terms[:, :, 1]
    out = np.zeros((*data.shape, len(STATES)))
    np.add.at(out, (a.group, data.loci.marker[a.locus]), values)
    return out


def class_probabilities(probabilities):
    return np.stack(
        [
            probabilities[..., [s.broad == c for s in STATES]].sum(axis=-1)
            for c in range(4)
        ],
        axis=-1,
    )
