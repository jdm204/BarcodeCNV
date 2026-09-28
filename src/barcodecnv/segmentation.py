"""Experimental Viterbi summaries and fixed-boundary segment evidence.

This does not replace posterior marginals or change grouping. Unlike Numbat's
continuous-effect retest, alternatives here use the existing discrete CN states.
Scores integrate phase paths and detailed states within a broad class, keeping
one detailed CN state throughout the segment. Fitted noise/reference/boundaries
are conditional; selection and multiple testing are not accounted for.
"""

import numpy as np
from numba import njit
from scipy.special import logsumexp

from .depth_controls import DepthOptions
from .hmm import _logsum
from .joint import JointWorkspace
from .model import STATES, class_probabilities


@njit(cache=True)
def constant_cn_evidence(emission, phase_transitions):
    """Log likelihood for each fixed detailed CN state; sum all phase paths."""
    n, k, _ = emission.shape
    out = np.empty(k)
    for c in range(k):
        score = emission[0, c].copy() - np.log(2.0)
        for t in range(1, n):
            next_score = np.empty(2)
            for z in range(2):
                next_score[z] = (
                    _logsum(score + phase_transitions[t - 1, :, z]) + emission[t, c, z]
                )
            score = next_score
        out[c] = _logsum(score)
    return out


def segment_runs(calls, chromosomes):
    chromosomes = np.asarray(chromosomes)
    breaks = np.r_[
        0,
        np.flatnonzero(
            (calls[1:] != calls[:-1]) | (chromosomes[1:] != chromosomes[:-1])
        )
        + 1,
        len(calls),
    ]
    return zip(breaks[:-1], breaks[1:])


def summarize_segments(data, model, parameters, *, depth_options=DepthOptions()):
    """Return raw marginals, joint-Viterbi broad calls and segment score records.

    Example: p, calls, segments = summarize_segments(data, model, parameters)
    filtered = filter_segments(calls, segments, minimum_log_bf=5.)
    No threshold is chosen here and no minimum event length is imposed.
    """
    workspace = JointWorkspace(data, model, parameters, depth_options=depth_options)
    broad = np.array([s.broad for s in STATES])
    probabilities, calls, rows = [], [], []
    for g, emission in enumerate(workspace.emissions(parameters)):
        flat = emission.reshape(len(data.grid), -1)
        posterior = workspace.engine.infer(flat)
        probabilities.append(
            class_probabilities(
                posterior.probabilities.reshape(-1, len(STATES), 2).sum(-1)
            )
        )
        path, _ = workspace.engine.viterbi(flat)
        call = broad[path // 2]
        calls.append(call)
        for a, b in segment_runs(call, data.grid.chrom):
            ll = constant_cn_evidence(
                emission[a:b], workspace.phase_transitions[a : b - 1]
            )
            class_ll = np.array(
                [
                    logsumexp(
                        ll[broad == c],
                        b=model.initial[broad == c] / model.initial[broad == c].sum(),
                    )
                    for c in range(4)
                ]
            )
            state = int(call[a])
            genes = (
                (data.genes.group == g)
                & (data.genes.marker >= a)
                & (data.genes.marker < b)
            )
            rows.append(
                dict(
                    group_index=g,
                    group=data.group_ids[g],
                    start_index=int(a),
                    stop_index=int(b),
                    chromosome=data.grid.chrom[a],
                    start_bp=int(data.grid.start[a]),
                    end_bp=int(data.grid.end[b - 1]),
                    state=state,
                    genes=int(genes.sum()),
                    log_bf_vs_diploid=float(class_ll[state] - class_ll[0]),
                    best_constant_class=int(class_ll.argmax()),
                )
            )
    return np.array(probabilities), np.array(calls), rows


def filter_segments(calls, segments, minimum_log_bf=5.0):
    """Replace altered segments below the chosen support threshold by diploid.

    This is a decision rule, not a recalculated probability tensor. Raw
    probabilities must remain available separately.
    """
    result = calls.copy()
    for r in segments:
        if r["state"] != 0 and r["log_bf_vs_diploid"] < minimum_log_bf:
            result[r["group_index"], r["start_index"] : r["stop_index"]] = 0
    return result
