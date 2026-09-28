"""Expression-first PBPC2 grouping with count-based CN/phase inference."""

import numpy as np
import pandas as pd
from numba import njit
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .bootstrap import bootstrap_features
from .depth_controls import DepthOptions
from .fitting import FitOptions, fit_phase_and_dispersion, infer_barcodes
from .group_calls import summarize_groups
from .grouping import cluster, common_refinement, recursive_groups
from .haplotypes import local_refinement, rolling_features
from .model import STATES, Model
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


def run_pbpc(
    bundle,
    *,
    replicates=64,
    draws=256,
    seed=42,
    cn_refinement=True,
    hf_refinement=True,
    options=FitOptions(),
    model=Model(),
    depth_options=DepthOptions(),
    progress=lambda message: None,
):
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
    if cn_refinement:
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
        labels, audit = refine_groups(distances, expression_groups, features["cells"])
    pre_hf_groups = labels.copy()
    hf_features = None
    hf_audit = []
    hf_stability = None
    hf_status = "disabled" if not hf_refinement else "no_usable_allele_counts"
    if (
        hf_refinement
        and len(bundle.loci)
        and float(((bundle.h1 + bundle.h2).T @ bundle.het).sum()) > 0
    ):
        progress("Rolling haplotype fractions and local group refinement")
        hf_features = rolling_features(
            bundle,
            barcodes.pooled_fit.final_phase.phase,
            replicates=replicates,
            seed=seed,
        )
        labels, hf_audit, hf_stability = local_refinement(
            hf_features["x"], hf_features["boot"], labels, return_stability=True
        )
        # Resampling arrays are temporary; retain only summaries for reporting.
        hf_features.pop("boot")
        hf_status = "applied"
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
        distances=distances,
        pre_hf_groups=pre_hf_groups,
        hf_features=hf_features,
        hf_audit=hf_audit,
        hf_stability=hf_stability,
        hf_status=hf_status,
        group_calls=group_calls,
    )
