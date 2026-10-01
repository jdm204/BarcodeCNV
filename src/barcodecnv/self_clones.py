"""Clone grouping from cohort-relative expression and phase-aligned allele counts.

No external-reference expression or fitted CN probabilities enter these rules.
HF features may condition on the existing fitted phase alignment.
Feature construction and bootstrap validity are conditional on the input cells
and gene universe. Expression programmes can resemble CN differences.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .clone_calling import (
    SELF_METHODS,
    CloneCalls,
    _assess_distances,
    _membership,
    _split_tree,
)
from .grouping import common_refinement, leading, recursive_groups


def expression_distances(x, boot):
    """Paired whole-cell residual distances, corrected for resampling noise.

    Residual centering preserves within-draw covariance between barcodes/features.
    Averaging corrected draws recovers the observed squared profile distance.
    """
    expected = squareform(pdist(x.T, metric="sqeuclidean")) / len(x)
    residuals = boot - boot.mean(axis=0)
    draws = np.array(
        [
            squareform(pdist((x + residual).T, metric="sqeuclidean")) / len(x)
            for residual in residuals
        ]
    )
    noise = np.mean(
        [
            squareform(pdist(residual.T, metric="sqeuclidean")) / len(x)
            for residual in residuals
        ],
        axis=0,
    )
    return expected, (draws - noise).transpose(1, 2, 0)


def correlation_distance(x):
    """Profile shape distance, with a shared direction for constant profiles."""
    centered = x - x.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    variable = norms > np.finfo(float).eps
    unit = np.divide(centered, norms, out=np.zeros_like(centered), where=variable)
    result = np.clip(1 - unit.T @ unit, 0, 2)
    result[np.ix_(~variable, ~variable)] = 0
    np.fill_diagonal(result, 0)
    return result


def profile_distances(x, boot, metric):
    """Experimental distances; neither correction guarantees calibrated support.

    Debiasing also subtracts estimated measurement noise from the observed
    distance, unlike expression_distances, which removes added resampling noise.
    Correlation discards amplitude and centres bootstrap distances on the
    observed shape distance. Both preserve paired whole-cell residual draws.
    """
    residuals = boot - boot.mean(axis=0)
    if metric == "debiased":
        observed, draws = expression_distances(x, boot)
        noise = sum(
            squareform(pdist(r.T, metric="sqeuclidean")) / len(x) for r in residuals
        ) / (len(boot) - 1)
        return np.maximum(observed - noise, 0), draws - noise[:, :, None]
    if metric == "correlation":
        expected = correlation_distance(x)
        samples = np.array([correlation_distance(x + r) for r in residuals])
        return expected, (samples - samples.mean(axis=0) + expected).transpose(1, 2, 0)
    raise ValueError(f"Unknown profile distance: {metric}")


def excess_variation(x, boot):
    """Root heterogeneity diagnostic, using the existing splitter's statistic."""
    variance = np.mean(np.var(boot, axis=0, ddof=1), axis=0)
    weights = 1 / np.maximum(variance, np.finfo(float).eps)
    weights /= weights.mean()
    statistic = leading(x, weights)[0]
    residuals = boot - boot.mean(axis=0)
    null = np.array([leading(r, weights)[0] for r in residuals])
    tail = float((1 + np.sum(null >= statistic)) / (len(null) + 1))
    return dict(statistic=statistic, bootstrap_tail=tail, alpha=0.05)


def joint_features(x, boot, hf):
    """Give each modality equal total bootstrap-noise energy, not equal row weight."""
    matrices, resamples, info = [], [], {}
    modalities = [("expression", x, boot)]
    if hf is not None:
        valid = (hf["coverage"] > 0).any(axis=1)
        if valid.any():
            modalities.append(("haplotype", hf["x"][valid], hf["boot"][:, valid]))
    for name, values, replicates in modalities:
        variance = float(np.mean(np.var(replicates, axis=0, ddof=1)))
        scale = float(np.sqrt(len(values) * max(variance, np.finfo(float).eps)))
        matrices.append(values / scale)
        resamples.append(replicates / scale)
        info[name] = dict(
            features=len(values), bootstrap_variance=variance, scale=scale
        )
    info["hf_status"] = (
        "fitted_phase" if len(modalities) == 2 else "no_usable_allele_counts"
    )
    return np.concatenate(matrices), np.concatenate(resamples, axis=1), info


def call_self_clones(features, *, method, hf=None, weighted=None, core=None):
    """Return pre-refinement CloneCalls without external-reference inputs.

    hf, when used, must be phase-aligned features with whole-cell resamples
    aligned to the expression bootstrap. See rolling_features(resampling_seed).
    """
    if method not in SELF_METHODS:
        raise ValueError(f"self-centred clone method must be one of {SELF_METHODS}")
    x, boot, cells = features["x"], features["boot"], features["cells"]
    if (
        x.ndim != 2
        or boot.ndim != 3
        or boot.shape[1:] != x.shape
        or boot.shape[0] < 20
        or x.shape[1] != len(cells)
        or x.shape[0] == 0
        or x.shape[1] == 0
        or not np.isfinite(x).all()
        or not np.isfinite(boot).all()
    ):
        raise ValueError(
            "finite expression features and at least 20 matching bootstrap draws are required"
        )
    if method == "self_expression":
        weighted = recursive_groups(x, boot) if weighted is None else weighted
        core = recursive_groups(x, boot, core_fraction=0.5) if core is None else core
        labels = common_refinement(weighted["labels"], core["labels"])
        support = np.minimum(weighted["stability"], core["stability"])
        if not any(
            node["accepted"] for result in (weighted, core) for node in result["nodes"]
        ):
            support[:] = np.nan
        nodes = [
            dict(rule=rule, **node)
            for rule, result in (("weighted", weighted), ("core", core))
            for node in result["nodes"]
        ]
        audit = [
            dict(
                barcode_index=i,
                reason="expression_partition"
                if group
                else "unresolved_expression_partition",
                accepted=bool(group),
                preference_support=support[i],
            )
            for i, group in enumerate(labels)
        ]
        return CloneCalls(
            method,
            labels,
            labels.copy(),
            labels.copy(),
            support,
            nodes,
            audit,
            evidence="self_expression",
        )
    info = {}
    evidence = "self_expression"
    if method in ("self_expression_hf", "self_expression_hf_debiased"):
        x, boot, info = joint_features(x, boot, hf)
        if info["hf_status"] == "fitted_phase":
            evidence = "self_expression_and_hf"
    metric = {
        "self_expression_correlation": "correlation",
        "self_expression_hf_debiased": "debiased",
    }.get(method)
    if metric is not None:
        info["distance"] = metric
        diagnostic = excess_variation(x, boot)
        info["excess_variation"] = diagnostic
        if diagnostic["bootstrap_tail"] > diagnostic["alpha"]:
            # An undivided group means no detected heterogeneity, not positive
            # evidence of one clone. Do not assign a membership confidence.
            labels = np.full(len(cells), int(len(cells) >= 2))
            return CloneCalls(
                method,
                labels,
                labels.copy(),
                labels.copy(),
                np.full(len(cells), np.nan),
                [],
                [
                    dict(
                        barcode_index=i,
                        reason="no_excess_profile_variation",
                        accepted=bool(label),
                    )
                    for i, label in enumerate(labels)
                ],
                evidence=evidence,
                feature_info=info,
            )
        expected, distances = profile_distances(x, boot, metric)
    else:
        expected, distances = expression_distances(x, boot)
    discovered, nodes = _split_tree(
        distances, cells, proposal=expected, separation=0.25
    )
    anchors, _ = _membership(distances, cells, discovered)
    labels, support, audit = _assess_distances(
        distances,
        distances,
        expected[:, :, None],
        cells,
        anchors,
        baseline="profile_q95",
    )
    return CloneCalls(
        method,
        labels,
        discovered,
        anchors,
        support,
        nodes,
        audit,
        evidence=evidence,
        feature_info=info,
    )
