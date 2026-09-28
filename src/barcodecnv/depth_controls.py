"""Optional CN-independent depth contamination; allele evidence is unchanged."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthOptions:
    outlier_probability: float = 0.05
    outlier_log_sd: float = 3.0

    def __post_init__(self):
        if (
            not np.isfinite(self.outlier_probability)
            or not 0 <= self.outlier_probability < 1
        ):
            raise ValueError("depth outlier probability must be in [0, 1)")
        if not np.isfinite(self.outlier_log_sd) or self.outlier_log_sd <= 0:
            raise ValueError("depth outlier log standard deviation must be positive")


class DepthWorkspace:
    """Cache the CN-independent component across inlier dispersion evaluations."""

    def __init__(self, data, model, options=DepthOptions()):
        self.data, self.model, self.options = data, model, options
        self._mu, self._outlier = None, None

    def emissions(self, parameters):
        from .depth import poilog_logpmf
        from .model import depth_emissions

        if self.options.outlier_probability and self._mu != parameters.mu:
            genes = self.data.genes
            keep = (genes.expected > 0) & (genes.weight > 0)
            self._outlier = poilog_logpmf(
                genes.count[keep],
                genes.expected[keep] * np.exp(parameters.mu),
                self.options.outlier_log_sd,
            )
            self._mu = parameters.mu
        return depth_emissions(
            self.data,
            parameters,
            self.model,
            outlier_probability=self.options.outlier_probability,
            outlier_log_likelihood=self._outlier,
        )
