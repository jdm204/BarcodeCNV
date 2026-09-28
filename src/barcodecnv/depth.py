"""Poisson–lognormal count probabilities using SciPy numerical specialists.

Adapts the mode/curvature rescaling used by the Julia BCNV emission (itself
following Numbat's Poisson–lognormal likelihood). SciPy Wright omega locates
the unique mode analytically; SciPy QUADPACK owns adaptive integration and its
error estimate. A Numba C callback avoids Python calls at each quadrature node.
The supplied rate is the latent median, not its mean.
"""

import numpy as np
from numba import carray, cfunc, types
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.special import wrightomega
from scipy.stats import poisson


@cfunc(types.double(types.intc, types.CPointer(types.double)), cache=True)
def _integrand(n, pointer):
    z, mode_rate, scale, variance = carray(pointer, n)
    delta = scale * z
    return np.exp(
        -mode_rate * (np.expm1(delta) - delta) - delta * delta / (2 * variance)
    )


_callback = LowLevelCallable(_integrand.ctypes)


def poilog_logpmf(count, median, sigma):
    y, rate = np.broadcast_arrays(np.asarray(count, float), np.asarray(median, float))
    if (
        not np.isfinite(sigma)
        or sigma < 0
        or np.any(~np.isfinite(y))
        or np.any(~np.isfinite(rate))
        or np.any(y < 0)
        or np.any(y != np.floor(y))
        or np.any(rate < 0)
    ):
        raise ValueError(
            "Poisson–lognormal requires nonnegative integer counts, rates and sigma"
        )
    if sigma == 0:
        return poisson.logpmf(y, rate)
    shape = y.shape
    y, rate = y.ravel(), rate.ravel()
    result = np.where(y == 0, 0.0, -np.inf)
    keep = np.flatnonzero(rate > 0)
    variance = sigma * sigma

    for begin in range(0, len(keep), 4096):
        ix = keep[begin : begin + 4096]
        log_median = np.log(rate[ix])
        mode_rate = (
            wrightomega(np.log(variance) + log_median + variance * y[ix]).real
            / variance
        )
        scale = 1 / np.sqrt(mode_rate + 1 / variance)
        integrals = np.empty(len(ix))
        for j, (mode, width) in enumerate(zip(mode_rate, scale)):
            integral = quad(
                _callback,
                -np.inf,
                np.inf,
                args=(mode, width, variance),
                epsabs=0.0,
                epsrel=1e-9,
                full_output=1,
            )
            # With full_output, QUADPACK appends an explanation on failure
            # instead of emitting IntegrationWarning. Never silently accept it.
            if len(integral) != 3 or not np.isfinite(integral[0]) or integral[0] <= 0:
                raise ArithmeticError("Poisson–lognormal quadrature did not converge")
            integrals[j] = integral[0]
        displacement = np.log(mode_rate) - log_median
        result[ix] = (
            poisson.logpmf(y[ix], mode_rate)
            - displacement**2 / (2 * variance)
            + np.log(scale / sigma)
            - 0.5 * np.log(2 * np.pi)
            + np.log(integrals)
        )
    return result.reshape(shape)
