"""Exact CN × phase HMMs within barcodes, using the existing HMM engine.

This changes the model: each barcode has its own phase path with the existing
genetic-distance prior. It does not approximate shared phase as an independent
marginal at each SNP. CN and phase paths are summed and sampled jointly. The
separate pooled phase fit can still align cross-barcode HF grouping features.
"""

from dataclasses import dataclass

import numpy as np

from .data import owned
from .depth_controls import DepthOptions, DepthWorkspace
from .fitting import CNPosterior, FitOptions, _fit_dispersion_objective
from .hmm import HMMEngine
from .model import STATES, allele_terms, estimate_parameters


@dataclass(frozen=True)
class JointChain:
    joint: object

    @property
    def probabilities(self):
        return self.joint.probabilities.reshape(-1, len(STATES), 2).sum(axis=2)

    @property
    def log_evidence(self):
        return self.joint.log_evidence

    def sample_paths(self, draws=256, seed=None):
        return self.joint.sample_paths(draws, seed) // 2


class JointWorkspace:
    def __init__(self, data, model, parameters, *, depth_options=DepthOptions()):
        if np.any(np.diff(data.loci.marker) == 0):
            raise ValueError(
                "joint inference requires each SNP to occupy a distinct grid marker"
            )
        self.data, self.model = data, model
        self.depth_options = depth_options
        self.depth = DepthWorkspace(data, model, depth_options)
        n, k = len(data.grid), len(STATES)
        phase = np.full((n - 1, 2, 2), -np.inf)
        phase[:, 0, 0] = phase[:, 1, 1] = 0.0
        boundaries = np.array(
            [a != b for a, b in zip(data.grid.chrom[:-1], data.grid.chrom[1:])],
            dtype=bool,
        )
        phase[boundaries] = np.log(0.5)
        # A phase step occurs on entry to the next SNP; gene-only positions
        # simply carry phase forward. No artificial SNP is introduced.
        for locus, transition in enumerate(model.phase_transitions(data), 1):
            phase[data.loci.marker[locus] - 1] = transition
        self.phase_transitions = phase
        cn = model.cn_transitions(data.grid)
        transitions = (cn[:, :, None, :, None] + phase[:, None, :, None, :]).reshape(
            n - 1, 2 * k, 2 * k
        )
        self.engine = HMMEngine(
            transitions, np.repeat(np.log(model.initial), 2) - np.log(2)
        )
        self.terms = allele_terms(data, model, parameters)
        self.indices = [
            np.flatnonzero(data.alleles.group == b) for b in range(len(data.group_ids))
        ]

    def emissions(self, parameters):
        """Yield each group's position × CN × phase count log likelihoods."""
        data, model = self.data, self.model
        depth = self.depth.emissions(parameters)
        for b, indices in enumerate(self.indices):
            emission = np.repeat(depth[b, :, :, None], 2, axis=2)
            markers = data.loci.marker[data.alleles.locus[indices]]
            np.add.at(emission, markers, self.terms[indices])
            if model.phase_locus_weight:
                pop = np.clip(data.loci.population_phase, 1e-6, 1 - 1e-6)
                emission[data.loci.marker] += (
                    model.phase_locus_weight
                    * np.stack((np.log(pop), np.log1p(-pop)), axis=1)[:, None, :]
                )
            yield emission

    def infer(self, parameters, *, marginals=True):
        data, model = self.data, self.model
        chains, phase, scores = [], [], []
        for emission in self.emissions(parameters):
            result = self.engine.infer(
                emission.reshape(len(data.grid), 2 * len(STATES)), marginals=marginals
            )
            if marginals:
                chains.append(JointChain(result))
                phase.append(
                    result.probabilities[data.loci.marker]
                    .reshape(-1, len(STATES), 2)
                    .sum(1)[:, 0]
                )
            else:
                scores.append(result)
        if not marginals:
            return float(sum(scores))
        return CNPosterior(
            data,
            model,
            parameters,
            owned(phase, float, ndim=2),
            tuple(chains),
            self.depth_options,
        )


def infer_joint(data, model, parameters, *, depth_options=DepthOptions()):
    return JointWorkspace(data, model, parameters, depth_options=depth_options).infer(
        parameters
    )


def fit_joint_dispersion(
    data, model, *, options=FitOptions(), depth_options=DepthOptions()
):
    initial = estimate_parameters(data)
    workspace = JointWorkspace(data, model, initial, depth_options=depth_options)
    return _fit_dispersion_objective(
        initial,
        lambda parameters: workspace.infer(parameters, marginals=False),
        options,
    )
