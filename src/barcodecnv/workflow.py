"""Count-based CN/phase inference with selectable clone calling."""

import numpy as np
import pandas as pd
from numba import njit
from scipy.cluster.hierarchy import leaves_list

from .bootstrap import bootstrap_features
from .clone_calling import (
    DEFAULT_CLONE_METHOD,
    SELF_METHODS,
    call_clones,
    validate_clone_method,
)
from .depth_controls import DepthOptions
from .fitting import FitOptions, fit_phase_and_dispersion, infer_barcodes
from .group_calls import summarize_groups
from .grouping import cluster, common_refinement, recursive_groups
from .haplotypes import local_refinement, rolling_features
from .model import STATES, Model
from .self_clones import call_self_clones
from .smoothing import infercnv_smoothing


@njit(cache=True)
def path_distances(paths):
    # Input draw × barcode × gene, output barcode × barcode × draw.
    d, b, g = paths.shape
    result = np.zeros((b, b, d), np.float32)
    for draw in range(d):
        for a in range(b):
            for z in range(a):
                n = 0
                for j in range(g):
                    n += paths[draw, a, j] != paths[draw, z, j]
                result[a, z, draw] = result[z, a, draw] = n / g
    return result


def run_pbpc(
    bundle,
    *,
    replicates=64,
    draws=256,
    seed=42,
    cn_refinement=True,
    clone_method=DEFAULT_CLONE_METHOD,
    hf_refinement=True,
    options=FitOptions(),
    model=Model(),
    depth_options=DepthOptions(),
    progress=lambda message: None,
):
    validate_clone_method(clone_method)
    if replicates < 20:
        raise ValueError("at least 20 cell bootstraps are required")
    if draws < 0 or (cn_refinement and draws < 1):
        raise ValueError("CN refinement needs positive draws")
    if bundle.reference_fractions.sum() <= 0:
        raise ValueError("full inference requires an external reference")
    progress("Expression smoothing and whole-cell bootstrap grouping")
    genes = pd.DataFrame(
        dict(
            gene=bundle.gene_ids,
            chromosome=[bundle.grid.chrom[t] for t in bundle.gene_markers],
        )
    )
    features = bootstrap_features(
        bundle.expression,
        bundle.libraries,
        bundle.membership,
        genes,
        bundle.barcodes,
        replicates=replicates,
        seed=seed,
    )
    phase_groups, tree, _, _ = cluster(features["self_expression"])
    weighted = recursive_groups(features["x"], features["boot"])
    core = recursive_groups(features["x"], features["boot"], core_fraction=0.5)
    expression_groups = common_refinement(weighted["labels"], core["labels"])
    progress(f"{model.depth_family.upper()} pooled phase and dispersion fitting")
    fit = fit_phase_and_dispersion(
        bundle.prepared(phase_groups),
        model,
        options=options,
        depth_options=depth_options,
    )
    progress(
        f"{options.barcode_phase.capitalize()} barcode CN/phase inference and dispersion fitting"
    )
    barcodes = infer_barcodes(bundle.prepared(), model, fit, options=options)
    labels = expression_groups.copy()
    audit = []
    distances = None
    clone_calls = None
    grouping_hf = None
    usable_hf = (
        len(bundle.loci) and float(((bundle.h1 + bundle.h2).T @ bundle.het).sum()) > 0
    )
    if cn_refinement:
        if clone_method in SELF_METHODS:
            progress("Self-centred expression clone calling")
            if (
                clone_method in ("self_expression_hf", "self_expression_hf_debiased")
                and usable_hf
            ):
                grouping_hf = rolling_features(
                    bundle,
                    barcodes.pooled_fit.final_phase.phase,
                    replicates=replicates,
                    seed=seed,
                    resampling_seed=seed,
                )
            clone_calls = call_self_clones(
                features,
                method=clone_method,
                hf=grouping_hf,
                weighted=weighted,
                core=core,
            )
        else:
            progress("Independent posterior CN paths and group refinement")
            seeds = np.random.SeedSequence(seed).spawn(len(bundle.barcodes))
            paths = np.empty(
                (draws, len(bundle.barcodes), len(bundle.gene_ids)), dtype=np.uint8
            )
            classes = np.array([s.broad for s in STATES], dtype=np.uint8)
            for b, (chain, key) in enumerate(zip(barcodes.posterior.chains, seeds)):
                paths[:, b] = classes[
                    chain.sample_paths(draws, key)[:, bundle.gene_markers]
                ]
            distances = path_distances(paths)
            clone_calls = call_clones(
                distances,
                barcodes.classes[:, bundle.gene_markers],
                features["cells"],
                method=clone_method,
                expression_groups=expression_groups,
            )
        labels, audit = clone_calls.groups, clone_calls.nodes
        progress(
            f"Clone calling ({clone_method}): {len(set(labels) - {0})} groups, {np.sum(labels == 0)} unresolved barcodes"
        )
    pre_hf_groups = labels.copy()
    hf_features = grouping_hf
    hf_audit = []
    hf_stability = None
    hf_status = "disabled" if not hf_refinement else "no_usable_allele_counts"
    if hf_refinement and usable_hf:
        progress("Rolling haplotype fractions and local group refinement")
        if hf_features is None:
            hf_features = rolling_features(
                bundle,
                barcodes.pooled_fit.final_phase.phase,
                replicates=replicates,
                seed=seed,
            )
        labels, hf_audit, hf_stability = local_refinement(
            hf_features["x"], hf_features["boot"], labels, return_stability=True
        )
        hf_status = "applied"
    if hf_features is not None:
        # Resampling arrays are temporary; retain only summaries for reporting.
        hf_features.pop("boot")
    # Use observed expression hierarchy within each reported group, with group
    # boundaries contiguous. No simulated truth enters ordering or grouping.
    order = np.arange(len(labels)) if tree is None else leaves_list(tree)
    rank = {int(b): i for i, b in enumerate(order)}
    order = np.array(
        sorted(order, key=lambda b: (labels[b] == 0, labels[b], rank[int(b)]))
    )
    ref = bundle.reference_fractions.copy()
    pooled_counts = (
        bundle.prepared()
        .genes.count.reshape(len(bundle.barcodes), len(bundle.gene_ids))
        .T
    )
    ref *= np.median(pooled_counts.sum(axis=0)) / ref.sum()
    external = infercnv_smoothing(pooled_counts, genes.chromosome, reference=ref)
    progress("Final group consensus, member disagreement and pseudobulk CN/phase calls")
    group_calls = summarize_groups(
        bundle,
        model,
        barcodes.classes,
        labels,
        options=options,
        depth_options=depth_options,
    )
    return dict(
        bundle=bundle,
        fit=barcodes,
        groups=labels,
        expression_groups=expression_groups,
        phase_groups=phase_groups,
        order=order,
        cells=features["cells"],
        self_expression=features["self_expression"],
        external_expression=external,
        weighted=weighted,
        core=core,
        cn_audit=audit,
        clone_calls=clone_calls,
        distances=distances,
        pre_hf_groups=pre_hf_groups,
        hf_features=hf_features,
        hf_audit=hf_audit,
        hf_stability=hf_stability,
        hf_status=hf_status,
        group_calls=group_calls,
    )
