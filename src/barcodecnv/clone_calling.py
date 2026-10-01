"""Broad-CN clone calling with shared discovery and interchangeable membership rules.

Support and adaptive distance thresholds are not clone posterior probabilities.
Expression-first calling is retained as an explicit alternative.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

SELF_METHODS = (
    "self_expression",
    "self_expression_tree",
    "self_expression_hf",
    "self_expression_correlation",
    "self_expression_hf_debiased",
)
CLONE_METHODS = (
    "mean_profile",
    "mean_distance",
    "draw_quantile",
    "expression",
    *SELF_METHODS,
)
DEFAULT_CLONE_METHOD = "mean_profile"


@dataclass
class CloneCalls:
    """Pre-HF group calls and their evidence, in input barcode order.

    stability is relative CN membership support, NaN when unavailable. Membership
    records describe the pre-HF decision; expression calling has no such records.
    Node and reference indices use the original barcode axis. Group IDs may have
    gaps after rejection; zero always means unresolved.
    """

    method: str
    groups: np.ndarray
    discovered: np.ndarray
    anchors: np.ndarray
    stability: np.ndarray
    nodes: list
    membership: list
    evidence: str = "cn"
    feature_info: dict = field(default_factory=dict)


def validate_clone_method(method):
    if method not in CLONE_METHODS:
        raise ValueError(
            f"clone_method must be one of {', '.join(CLONE_METHODS)}; got {method!r}"
        )


def call_clones(
    distances,
    probabilities,
    cells,
    *,
    method=DEFAULT_CLONE_METHOD,
    expression_groups=None,
):
    """Call groups from barcode × barcode × draw distances and gene probabilities.

    probabilities has barcode × gene × four broad-CN-class axes. No fitted CN
    values are modified. Mean-profile calling uses a fixed reference-member
    envelope; tiny groups borrow within-group scores from other anchored groups.
    If neither local nor borrowed calibration is possible, they stay unresolved.
    """
    validate_clone_method(method)
    if method in SELF_METHODS:
        raise ValueError(
            "self-centred methods require call_self_clones with expression bootstrap features"
        )
    distances = np.asarray(distances)
    probabilities = np.asarray(probabilities)
    cells = np.asarray(cells)
    n = len(cells) if cells.ndim == 1 else 0
    if n == 0 or not np.isfinite(cells).all() or np.any(cells <= 0):
        raise ValueError(
            "cells must be a nonempty vector of positive barcode cell counts"
        )
    if (
        distances.ndim != 3
        or distances.shape[:2] != (n, n)
        or distances.shape[2] < 1
        or not np.isfinite(distances).all()
    ):
        raise ValueError(
            "distances must be finite barcode × barcode × draw values with positive draws"
        )
    if (
        probabilities.ndim != 3
        or probabilities.shape[0] != n
        or probabilities.shape[1] < 1
        or probabilities.shape[2] != 4
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=2), 1, atol=1e-6)
    ):
        raise ValueError(
            "probabilities must be normalized barcode × gene × four-class values"
        )
    if method == "expression":
        initial = np.asarray(expression_groups)
        if (
            initial.shape != (n,)
            or not np.issubdtype(initial.dtype, np.integer)
            or np.any(initial < 0)
        ):
            raise ValueError(
                "expression calling requires nonnegative expression group labels"
            )
        labels, nodes = refine_groups(distances, initial, cells)
        return CloneCalls(
            method,
            labels,
            initial.copy(),
            initial.copy(),
            np.full(n, np.nan),
            nodes,
            [],
            evidence="expression_then_cn",
        )
    discovered, nodes = _split_tree(distances, cells)
    anchors, _ = _membership(distances, cells, discovered)
    baseline = {
        "mean_profile": "profile_q95",
        "mean_distance": "mean",
        "draw_quantile": "member_q95",
    }[method]
    labels, stability, audit = _assess_membership(
        distances, probabilities, cells, anchors, baseline=baseline
    )
    return CloneCalls(method, labels, discovered, anchors, stability, nodes, audit)


def _uncertainty(probabilities):
    """Expected disagreement of two independent paths from one barcode."""
    return np.mean(1 - np.square(probabilities).sum(axis=2), axis=1)


def _probability_distance(probabilities):
    """Half mean squared distance between class-probability profiles.

    Equals expected inter-barcode disagreement minus half of each barcode's
    expected within-posterior disagreement. Used for the mean-profile envelope.
    """
    flattened = probabilities.reshape(len(probabilities), -1)
    gram = flattened @ flattened.T / probabilities.shape[1]
    squared = np.diag(gram)
    distance = np.maximum((squared[:, None] + squared[None, :]) / 2 - gram, 0)
    np.fill_diagonal(distance, 0)
    return distance


def _split_tree(distances, cells, *, separation=0.25, proposal=None):
    """Broad-CN contrast test with search below rejected parents.

    Rescue searches only when a supported descendant exists. Branches skipped
    on the route to a supported split remain unresolved; it does not invent
    groups for every discarded branch. A supported split's terminal children
    are kept intact when they have no supported subdivision.
    """
    size = len(cells)
    if size < 2:
        return np.zeros(size, dtype=int), []
    proposal = distances.mean(axis=2) if proposal is None else proposal
    tree = linkage(squareform(proposal, checks=False), method="average")
    members = {i: [i] for i in range(size)}
    records = {}
    descendant_support = {i: False for i in range(size)}

    def average(a, b=None):
        pairs = (
            [(i, j) for i in a for j in b]
            if b is not None
            else [(i, j) for k, j in enumerate(a) for i in a[:k]]
        )
        weights = np.array([cells[i] * cells[j] for i, j in pairs], dtype=float)
        return np.average(
            np.array([distances[i, j] for i, j in pairs]), axis=0, weights=weights
        )

    for offset, row in enumerate(tree):
        node = size + offset
        left, right = map(int, row[:2])
        a, b = members[left], members[right]
        members[node] = a + b
        lower = relative = None
        accepted = False
        if min(len(a), len(b)) >= 2:
            between = average(a, b)
            delta = between - (average(a) + average(b)) / 2
            lower = float(np.quantile(delta, 0.05))
            relative = float(delta.mean() / max(between.mean(), np.finfo(float).eps))
            accepted = lower > 0 and relative >= separation
        records[node] = dict(
            node=node,
            left=left,
            right=right,
            members=members[node],
            sizes=[len(a), len(b)],
            lower95=lower,
            relative=relative,
            accepted=accepted,
            visited=False,
        )
        descendant_support[node] = (
            accepted or descendant_support[left] or descendant_support[right]
        )
    labels = np.zeros(size, dtype=int)
    next_group = 0

    def visit(node):
        nonlocal next_group
        if node < size:
            return
        record = records[node]
        record["visited"] = True
        children = [record["left"], record["right"]]
        if record["accepted"]:
            for child in children:
                visit(child)
        elif descendant_support[node]:
            for child in children:
                if descendant_support[child]:
                    visit(child)
        else:
            next_group += 1
            labels[members[node]] = next_group

    visit(2 * size - 2)
    return labels, list(records.values())


def _membership(distances, cells, discovered, *, support=0.9):
    """Leave-one-out posterior nearest-group agreement; not clone probability.

    Membership is assessed against fixed discovered groups, not iteratively
    updated using newly assigned barcodes. This discriminates candidate groups;
    it does not establish absolute compatibility with any group.
    """
    groups = sorted(set(discovered) - {0})
    labels = discovered.copy()
    stability = np.full(len(cells), np.nan)
    if len(groups) < 2:
        return labels, stability
    for i in range(len(cells)):
        if discovered[i] == 0:
            continue
        costs = []
        for group in groups:
            others = np.flatnonzero(
                (discovered == group) & (np.arange(len(cells)) != i)
            )
            costs.append(
                np.average(distances[i, others], axis=0, weights=cells[others])
                if len(others)
                else np.full(distances.shape[2], np.inf)
            )
        costs = np.array(costs)
        target = groups.index(discovered[i])
        margin = np.min(np.delete(costs, target, axis=0), axis=0) - costs[target]
        stability[i] = np.mean(margin > 0)
        labels[i] = groups[target] if stability[i] >= support else 0
    # A lone retained barcode cannot establish a replicated group.
    for group in groups:
        if np.sum(labels == group) < 2:
            labels[labels == group] = 0
    return labels, stability


def _member_distances(corrected, cells, members):
    """Each member's cell-weighted distance to the remaining members, per draw."""
    return np.array(
        [
            np.average(
                corrected[j, members[members != j]],
                axis=0,
                weights=cells[members[members != j]],
            )
            for j in members
        ]
    )


def _reference_envelope(corrected, cells, anchors, query, references):
    """95th percentile over member scores on each draw; minimum three scores.

    The query is excluded from all calibration profiles and their references.
    A small target borrows scores from other groups with at least three anchors.
    Scores have equal barcode weight; their component distances have cell weight.
    This empirical envelope is not a calibrated 95% prediction interval.
    """
    groups = sorted(set(anchors) - {0})
    if len(references) >= 3:
        sets = [references]
        source = "target_group"
    else:
        sets = []
        for group in groups:
            members = np.flatnonzero(
                (anchors == group) & (np.arange(len(cells)) != query)
            )
            if len(members) >= 3:
                sets.append(members)
        source = "pooled_other_groups" if sets else "unavailable"
    audit = dict(
        baseline_source=source,
        baseline_members=[members.tolist() for members in sets],
        baseline_scores=sum(len(members) for members in sets),
    )
    if not sets:
        return None, audit
    scores = np.concatenate(
        [_member_distances(corrected, cells, members) for members in sets]
    )
    return np.quantile(scores, 0.95, axis=0), audit


def _assess_membership(distances, probabilities, cells, anchors, *, baseline):
    """Relative agreement plus a posterior excess-disagreement rejection rule.

    Expected path mismatch includes independent posterior uncertainty. Subtract
    half each endpoint's expected self-disagreement before comparing candidate
    distances to the within-anchor baseline. A positive fifth percentile of
    excess disagreement vetoes membership. Failing to reject is not proof of
    clone identity, and these adaptive summaries are not calibrated p-values.
    """
    if baseline not in ("mean", "member_q95", "profile_q95"):
        raise ValueError(f"Unknown compatibility baseline: {baseline}")
    impurity = _uncertainty(probabilities)
    corrected = distances - (impurity[:, None] + impurity[None, :])[:, :, None] / 2
    expected = (
        _probability_distance(probabilities)[:, :, None]
        if baseline == "profile_q95"
        else None
    )
    return _assess_distances(
        distances, corrected, expected, cells, anchors, baseline=baseline
    )


def _assess_distances(distances, corrected, expected, cells, anchors, *, baseline):
    """Shared fixed-anchor membership for CN or bootstrap expression distances."""
    groups = sorted(set(anchors) - {0})
    labels = np.zeros(len(cells), dtype=int)
    support = np.full(len(cells), np.nan)
    audit = []
    for i in range(len(cells)):
        record = dict(barcode_index=i, anchor_group=int(anchors[i]))
        if not groups:
            audit.append(dict(record, reason="no_discovered_groups"))
            continue
        costs = []
        peers = []
        for group in groups:
            ix = np.flatnonzero((anchors == group) & (np.arange(len(cells)) != i))
            peers.append(ix)
            costs.append(np.average(distances[i, ix], axis=0, weights=cells[ix]))
        costs = np.array(costs)
        target = (
            groups.index(anchors[i])
            if anchors[i] > 0
            else int(np.argmin(costs.mean(axis=1)))
        )
        preferred = True
        if len(groups) > 1:
            margin = np.min(np.delete(costs, target, axis=0), axis=0) - costs[target]
            support[i] = float(np.mean(margin > 0))
            preferred = support[i] >= 0.9
        ix = peers[target]
        to_group = np.average(corrected[i, ix], axis=0, weights=cells[ix])
        calibration = {}
        if baseline in ("member_q95", "profile_q95"):
            within, calibration = _reference_envelope(
                expected if baseline == "profile_q95" else corrected,
                cells,
                anchors,
                i,
                ix,
            )
        elif len(ix) >= 2:
            a, b = np.triu_indices(len(ix), 1)
            a, b = ix[a], ix[b]
            within = np.average(
                corrected[a, b],
                axis=0,
                weights=cells[a] * cells[b],
            )
        else:
            # No distinct-peer pair remains for a two-anchor group's own member.
            # A profile's uncertainty-corrected expected distance to itself is 0.
            within = np.zeros(distances.shape[2])
        excess = (
            to_group - within if within is not None else np.full_like(to_group, np.nan)
        )
        lower = float(np.quantile(excess, 0.05))
        compatible = within is not None and lower <= 0
        accepted = preferred and compatible
        if accepted:
            labels[i] = groups[target]
        audit.append(
            dict(
                record,
                proposed_group=groups[target],
                reference_members=ix.tolist(),
                preference_support=support[i],
                excess_mean=float(excess.mean()),
                excess_lower05=lower,
                baseline=baseline,
                baseline_mean=float(np.mean(within)) if within is not None else np.nan,
                **calibration,
                accepted=accepted,
                reason=(
                    "ambiguous_preference"
                    if not preferred
                    else "insufficient_calibration"
                    if within is None
                    else "excess_disagreement"
                    if not compatible
                    else "assigned"
                ),
            )
        )
    for group in groups:
        members = np.flatnonzero(labels == group)
        if len(members) == 1:
            labels[members] = 0
            audit[members[0]].update(accepted=False, reason="unreplicated_group")
    return labels, support, audit


def refine_groups(distances, initial, cells, minimum_relative_separation=0.25):
    """Existing cell-weighted broad-CN contrast rule; only subdivides groups."""
    labels = np.zeros(len(initial), dtype=int)
    audit = []
    next_label = 0
    for group in sorted(set(initial) - {0}):
        ix = np.flatnonzero(initial == group)
        if len(ix) < 2:
            continue
        d = distances[ix][:, ix]
        n = cells[ix]
        tree = linkage(squareform(d.mean(axis=2), checks=False), method="average")
        members = {i: [i] for i in range(len(ix))}
        records = {}

        def average(pairs, n=n, d=d):
            pairs = list(pairs)
            return sum(n[a] * n[b] * d[a, b] for a, b in pairs) / sum(
                n[a] * n[b] for a, b in pairs
            )

        def within(m):
            return average((a, b) for j, b in enumerate(m) for a in m[:j])

        for j, row in enumerate(tree):
            node = j + len(ix)
            left, right = map(int, row[:2])
            a, b = members[left], members[right]
            members[node] = a + b
            eligible = min(len(a), len(b)) >= 2
            lower = relative = None
            if eligible:
                between = average((aa, bb) for aa in a for bb in b)
                delta = between - (within(a) + within(b)) / 2
                lower = float(np.quantile(delta, 0.05))
                relative = float(
                    delta.mean() / max(between.mean(), np.finfo(float).eps)
                )
            supported = bool(
                eligible and lower > 0 and relative >= minimum_relative_separation
            )
            records[node] = dict(
                expression_group=int(group),
                node=node,
                left=left,
                right=right,
                n=len(a + b),
                lower95=lower,
                relative=relative,
                accepted=supported,
            )

        def visit(node, ix=ix, records=records, members=members):
            nonlocal next_label
            if node >= len(ix):
                audit.append(records[node])
            if node >= len(ix) and records[node]["accepted"]:
                visit(records[node]["left"])
                visit(records[node]["right"])
            else:
                next_label += 1
                labels[ix[members[node]]] = next_label

        visit(2 * len(ix) - 2)
    return labels, audit
