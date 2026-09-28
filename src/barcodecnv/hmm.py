"""Small numerical adapter for dynamic HMMs, independent of the CN model.

SciPy supplies logsumexp for validation; Numba compiles the sequential recursions.
Well-connected chains use scaled probability arithmetic. Chains containing tiny
or zero transitions use log arithmetic, preserving paths that recover later.
No probability floor or transition regularisation is introduced.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import logsumexp


@njit(cache=True)
def _logsum(a):
    m = np.max(a)
    if m == -np.inf:
        return m
    return m + np.log(np.exp(a - m).sum())


@njit(cache=True)
def _viterbi(e, tr, initial):
    """Max-product counterpart of the existing dynamic log-space recursion."""
    n, k = e.shape
    back = np.empty((n, k), np.int32)
    score = initial + e[0]
    for t in range(1, n):
        next_score = np.empty(k)
        for j in range(k):
            candidates = score + tr[t - 1, :, j]
            i = np.argmax(candidates)
            back[t, j] = i
            next_score[j] = candidates[i] + e[t, j]
        score = next_score
    path = np.empty(n, np.int32)
    path[-1] = np.argmax(score)
    best = score[path[-1]]
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, best


@njit(cache=True)
def _forward(e, tr, initial, logarithmic):
    tmax, k = e.shape
    f = np.empty_like(e)
    pred = initial.copy()
    norm = 0.0
    for t in range(tmax):
        row = pred + e[t]
        z = _logsum(row)
        norm += z
        f[t] = row - z
        if t + 1 < tmax:
            for j in range(k):
                if logarithmic:
                    pred[j] = _logsum(f[t] + tr[t, :, j])
                else:
                    pred[j] = np.log(np.sum(np.exp(f[t]) * np.exp(tr[t, :, j])))
    return f, norm


@njit(cache=True)
def _smooth(f, e, tr, logarithmic):
    tmax, k = e.shape
    p = np.empty_like(e)
    beta = np.zeros(k)
    for t in range(tmax - 1, -1, -1):
        row = f[t] + beta
        p[t] = np.exp(row - _logsum(row))
        if t:
            potential = e[t] + beta
            potential -= np.max(potential)
            for i in range(k):
                if logarithmic:
                    beta[i] = _logsum(tr[t - 1, i, :] + potential)
                else:
                    beta[i] = np.log(
                        np.sum(np.exp(tr[t - 1, i, :]) * np.exp(potential))
                    )
            beta -= np.max(beta)
    return p


@njit(cache=True)
def _sample(f, tr, uniforms):
    draws, tmax = uniforms.shape
    paths = np.empty((draws, tmax), np.int32)
    k = f.shape[1]
    # All independent draws reuse the K backward conditionals at a position.
    # Keep only one position's table instead of a T × K × K cache.
    cdf = np.empty((k, k))
    for t in range(tmax - 1, -1, -1):
        for successor in range(k if t + 1 < tmax else 1):
            row = f[t].copy()
            if t + 1 < tmax:
                row += tr[t, :, successor]
            z = _logsum(row)
            cumulative = 0.0
            for j in range(k):
                cumulative += np.exp(row[j] - z) if z != -np.inf else 0.0
                cdf[successor, j] = cumulative
        for d in range(draws):
            successor = paths[d, t + 1] if t + 1 < tmax else 0
            state = k - 1
            for j in range(k):
                if uniforms[d, t] < cdf[successor, j]:
                    state = j
                    break
            paths[d, t] = state
    return paths


def _validate(emissions, transitions, initial):
    e, tr, ini = (
        np.asarray(x, dtype=np.float64) for x in (emissions, transitions, initial)
    )
    if e.ndim != 2 or not e.shape[0] or e.shape[1] < 2:
        raise ValueError("emissions must be a nonempty position × state matrix")
    t, k = e.shape
    if tr.shape != (t - 1, k, k) or ini.shape != (k,):
        raise ValueError("HMM axes disagree")
    for a in (e, tr, ini):
        if np.any(np.isnan(a)) or np.any(a == np.inf):
            raise ValueError("log potentials may be finite or -inf")
    if not np.isclose(logsumexp(ini), 0, atol=1e-10) or not np.allclose(
        logsumexp(tr, axis=2), 0, atol=1e-10
    ):
        raise ValueError("initial and transition probabilities must be normalized")
    return e, tr, ini


@dataclass(frozen=True)
class HMMPosterior:
    probabilities: np.ndarray
    log_evidence: float
    _filtered: np.ndarray
    _transitions: np.ndarray

    def sample_paths(self, draws=256, seed=None):
        if not isinstance(draws, (int, np.integer)) or draws < 0:
            raise ValueError("draws must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        return _sample(
            self._filtered, self._transitions, rng.random((draws, len(self._filtered)))
        )


class HMMEngine:
    """Own fixed numerical priors, reusable for different emission matrices."""

    def __init__(self, log_transitions, log_initial):
        tr = np.array(log_transitions, dtype=float, copy=True)
        initial = np.array(log_initial, dtype=float, copy=True)
        _validate(np.zeros((len(tr) + 1, len(initial))), tr, initial)
        self.transitions, self.initial = tr, initial
        self.transitions.flags.writeable = self.initial.flags.writeable = False
        self.logarithmic = bool(
            np.any(tr < np.log(np.finfo(float).eps))
            or np.any(initial < np.log(np.finfo(float).eps))
        )

    def infer(self, emissions, *, marginals=True):
        e = np.asarray(emissions, dtype=float)
        if (
            e.shape != (len(self.transitions) + 1, len(self.initial))
            or np.any(np.isnan(e))
            or np.any(e == np.inf)
        ):
            raise ValueError("invalid emissions")
        filtered, evidence = _forward(
            e, self.transitions, self.initial, self.logarithmic
        )
        if not np.isfinite(evidence):
            raise ValueError(
                "observations have zero or nonfinite probability under this HMM"
            )
        if not marginals:
            return float(evidence)
        p = _smooth(filtered, e, self.transitions, self.logarithmic)
        if not np.all(np.isfinite(p)):
            raise ArithmeticError("nonfinite HMM marginals")
        p.flags.writeable = filtered.flags.writeable = False
        return HMMPosterior(p, float(evidence), filtered, self.transitions)

    def viterbi(self, emissions):
        """Return the joint MAP state path and its unnormalized log probability."""
        e = np.asarray(emissions, dtype=float)
        if (
            e.shape != (len(self.transitions) + 1, len(self.initial))
            or np.any(np.isnan(e))
            or np.any(e == np.inf)
        ):
            raise ValueError("invalid emissions")
        path, score = _viterbi(e, self.transitions, self.initial)
        if not np.isfinite(score):
            raise ValueError(
                "observations have zero or nonfinite probability under this HMM"
            )
        path.flags.writeable = False
        return path, float(score)


def forward_backward(log_emissions, log_transitions, log_initial):
    return HMMEngine(log_transitions, log_initial).infer(log_emissions)
