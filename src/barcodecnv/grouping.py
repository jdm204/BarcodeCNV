"""Expression grouping rules shared by PBPC2 and its benchmark adapters.

Bootstrap stability and adaptive split statistics are not calibrated posterior
clone probabilities. Simulation truth and filesystem paths are not inputs.
"""

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform


def silhouette(distance, labels):
    """Mean silhouette, including zero contribution for singleton clusters."""
    values = []
    for i, group in enumerate(labels):
        own = np.flatnonzero(labels == group)
        if len(own) == 1:
            values.append(0.0)
            continue
        a = distance[i, own].sum() / (len(own) - 1)
        b = min(
            distance[i, labels == g].mean() for g in np.unique(labels) if g != group
        )
        values.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(values))


def cluster(x, k=None):
    # Rows of x are genes; every selected gene has equal weight as a feature.
    # Genes absent from all samples/reference are blank in the shared signal.
    valid = np.isfinite(x).all(axis=1)
    assert np.all(valid | np.isnan(x).all(axis=1)), "partially missing feature row"
    if not valid.any():
        raise ValueError("no complete expression feature rows")
    if x.shape[1] < 3 or np.all(x[valid] == x[valid, :1]):
        return (
            np.ones(x.shape[1], dtype=int),
            None,
            {1: 0.0},
            np.zeros((x.shape[1], x.shape[1])),
        )
    distance = squareform(pdist(x[valid].T, metric="correlation"))
    if not np.isfinite(distance).all():
        raise ValueError(
            "correlation grouping requires variable profiles for each barcode"
        )
    tree = linkage(
        squareform(distance, checks=False), method="average", optimal_ordering=True
    )
    labels = {
        j: cut_tree(tree, n_clusters=j).ravel() + 1
        for j in range(2, min(5, x.shape[1]) + 1)
    }
    scores = {j: silhouette(distance, z) for j, z in labels.items()}
    chosen = max(scores, key=scores.get) if k is None else k
    return labels[chosen], tree, scores, distance


def leading(x, weights):
    centred = x - np.average(x, axis=1, weights=weights)[:, None]
    weighted = centred * np.sqrt(weights)[None, :]
    gram = weighted.T @ weighted
    values, vectors = eigh(gram, subset_by_index=[len(weights) - 1, len(weights) - 1])
    direction = weighted @ vectors[:, 0]
    return float(max(values[0], 0)), direction


def recursive_groups(
    x,
    validation_features,
    *,
    alpha=0.05,
    membership_support=0.9,
    minimum_coherence=0.1,
    core_fraction=None,
    coherence_rule="both",
):
    """Return labels and node audit, conditional on the InferCNV feature universe.

    Centered bootstrap residuals approximate a no-difference null. The leading
    eigenvalue is recomputed under each null draw, accounting for direction
    selection within a node. Adaptive recursion/multiple nodes are not calibrated.
    """
    boot = np.asarray(validation_features)
    R, F, B = boot.shape
    assert coherence_rule in ["both", "either"]
    assert (
        x.shape == (F, B)
        and R >= 20
        and np.isfinite(x).all()
        and np.isfinite(boot).all()
    )
    residuals = boot - boot.mean(axis=0)
    variance = np.mean(np.var(boot, axis=0, ddof=1), axis=0)
    weights = 1 / np.maximum(variance, np.finfo(float).eps)
    weights /= weights.mean()
    labels = np.zeros(B, dtype=int)
    confidence = np.ones(B)
    nodes = []
    leaves = []

    def visit(ix, parent=-1):
        node = len(nodes)
        record = dict(
            node=node,
            parent=parent,
            members=",".join(map(str, ix)),
            n=len(ix),
            accepted=False,
        )
        nodes.append(record)
        if len(ix) < 4:
            record["reason"] = "fewer_than_four_barcodes"
        else:
            w = weights[ix]
            v = x[:, ix]
            stat, direction = leading(v, w)
            if stat <= np.finfo(float).eps:
                record["reason"] = "no_profile_variation"
                leaves.append(ix)
                labels[ix] = len(leaves)
                return
            # Binary partition from the first weighted component, followed by
            # deterministic nearest-centroid assignment along its separating axis.
            projection = (v - np.average(v, axis=1, weights=w)[:, None]).T @ direction
            z = projection > 0
            core = np.ones(len(ix), dtype=bool)
            if core_fraction is not None:
                core[:] = False
                core[
                    np.argsort(w)[-max(4, int(np.ceil(len(ix) * core_fraction))) :]
                ] = True
                vc = v[:, core] - np.median(v[:, core], axis=1, keepdims=True)
                distances = pdist(vc.T, "correlation")
                # A precision core can contain only identical profiles even
                # when the complete node varies. Correlation then cannot
                # propose a split; retain this node rather than feed NaNs to SciPy.
                if not np.isfinite(distances).all():
                    record["reason"] = "undefined_core_correlation"
                    leaves.append(ix)
                    labels[ix] = len(leaves)
                    return
                zz = (
                    cut_tree(linkage(distances, method="average"), n_clusters=2)
                    .ravel()
                    .astype(bool)
                )
                z[core] = zz
                means = [
                    np.average(v[:, core][:, zz == k], axis=1, weights=w[core][zz == k])
                    for k in [0, 1]
                ]
                z[~core] = ((v[:, ~core] - means[1][:, None]) ** 2).sum(0) < (
                    (v[:, ~core] - means[0][:, None]) ** 2
                ).sum(0)
                record["core_members"] = ",".join(map(str, ix[core]))
            else:
                for _ in range(30):
                    if min(z.sum(), (~z).sum()) < 2:
                        break
                    means = [
                        np.average(v[:, z == k], axis=1, weights=w[z == k])
                        for k in [0, 1]
                    ]
                    new = ((v - means[1][:, None]) ** 2).sum(0) < (
                        (v - means[0][:, None]) ** 2
                    ).sum(0)
                    if np.array_equal(new, z):
                        break
                    z = new
            null = np.array([leading(e[:, ix], w)[0] for e in residuals])
            p = (1 + (null >= stat).sum()) / (R + 1)
            centred = v - np.average(v[:, core], axis=1, weights=w[core])[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.nan_to_num(np.corrcoef(centred.T), nan=0.0)
            coherence = []
            for k in [0, 1]:
                child = np.flatnonzero((z == k) & core)
                a, b = np.tril_indices(len(child), -1)
                coherence.append(
                    np.average(
                        corr[child[a], child[b]], weights=w[child[a]] * w[child[b]]
                    )
                    if len(a)
                    else -1.0
                )
            record.update(
                statistic=stat,
                null95=np.quantile(null, 0.95),
                bootstrap_tail=p,
                left=int((~z).sum()),
                right=int(z.sum()),
            )
            record.update(left_coherence=coherence[0], right_coherence=coherence[1])
            if min(z.sum(), (~z).sum()) < 2:
                record["reason"] = "unsupported_singleton_child"
            elif p > alpha:
                record["reason"] = "no_excess_over_bootstrap_noise"
            elif (
                min(coherence) if coherence_rule == "both" else max(coherence)
            ) < minimum_coherence:
                record["reason"] = "insufficient_replicated_profile_coherence"
            else:
                # Leave the queried barcode out of its own bootstrap centroid.
                # This avoids rewarding self-inclusion in a small candidate group.
                stable = []
                for j in range(len(ix)):
                    losses = []
                    for k in [0, 1]:
                        other = (z == k) & (np.arange(len(ix)) != j) & core
                        centre = np.average(
                            boot[:, :, ix[other]], axis=2, weights=w[other]
                        )
                        losses.append(np.sum((boot[:, :, ix[j]] - centre) ** 2, axis=1))
                    assignments = losses[1] < losses[0]
                    stable.append(np.mean(assignments == z[j]))
                stable = np.array(stable)
                confident = stable >= membership_support
                record["min_membership_stability"] = float(stable.min())
                record["uncertain_barcodes"] = int((~confident).sum())
                if min(np.sum(confident & z), np.sum(confident & ~z)) < 2:
                    record["reason"] = "fewer_than_two_stable_members_per_child"
                else:
                    record.update(accepted=True, reason="supported_split")
                    confidence[ix] = np.minimum(confidence[ix], stable)
                    # Ambiguous members stay unresolved; they do not force a
                    # hard branch choice that downstream recursion cannot undo.
                    visit(ix[confident & ~z], node)
                    visit(ix[confident & z], node)
                    return
        if len(ix) >= 2:
            leaves.append(ix)
            labels[ix] = len(leaves)

    visit(np.arange(B))
    return dict(labels=labels, stability=confidence, nodes=nodes, weights=weights)


def common_refinement(*partitions):
    if not partitions or len({len(p) for p in partitions}) != 1:
        raise ValueError("partitions must have matching barcode axes")
    if any(
        not np.issubdtype(np.asarray(p).dtype, np.integer) or np.any(np.asarray(p) < 0)
        for p in partitions
    ):
        raise ValueError("partitions require nonnegative integer labels")
    pairs = list(zip(*partitions))
    labels = np.zeros(len(pairs), dtype=int)
    next_group = 0
    for pair in sorted(set(pairs)):
        ix = np.array([p == pair for p in pairs])
        if 0 not in pair and ix.sum() >= 2:
            next_group += 1
            labels[ix] = next_group
    return labels
