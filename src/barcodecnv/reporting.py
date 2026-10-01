"""Shared report serialization for the Python API and CLI."""

import json
from dataclasses import asdict

import h5py
import numpy as np
import pandas as pd

from .cn_reporting import write_cn_tables
from .haplotypes import BASE_BIN_BP, WINDOW_BP
from .model import CLASS_NAMES


def write_signal(out, result):
    from .plotting import plot_signal

    values = result.to_dict()
    null = values.pop("null_scores")
    (out / "signal.json").write_text(
        json.dumps(values, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame(null, columns=["joint", "depth", "allele"]).to_csv(
        out / "signal_null.csv", index=False
    )
    plot_signal(result, out / "signal.png")


def write_run(out, result):
    bundle = result.bundle
    fit = result.fit
    calls = result.group_calls
    probabilities = fit.classes
    groups = result.groups_table()
    groups.to_csv(out / "groups.csv", index=False)
    membership = result.clone_membership_table()
    if not membership.empty:
        membership.to_csv(out / "clone_membership.csv", index=False)
    pd.DataFrame(dict(barcode=np.array(bundle.barcodes)[result.order])).to_csv(
        out / "order.csv", index=False
    )
    for name, rows in (
        ("weighted", result.weighted["nodes"]),
        ("core", result.core["nodes"]),
        ("cn", result.cn_audit),
        ("hf", result.hf_audit),
    ):
        pd.DataFrame(rows).to_csv(out / f"{name}_nodes.csv", index=False)
    write_cn_tables(out, bundle, probabilities, result.groups, calls)
    with h5py.File(out / "result.h5", "w") as f:
        f.attrs["barcodecnv_schema"] = "result-v1"
        f.attrs["clone_method"] = (
            result.clone_calls.method if result.clone_calls is not None else "disabled"
        )
        if result.clone_calls is not None:
            for name in ("groups", "discovered", "anchors", "stability"):
                f.create_dataset(
                    "clone_calling/" + name, data=getattr(result.clone_calls, name)
                )
            f["clone_calling"].attrs["stage"] = "before optional HF refinement"
            f["clone_calling"].attrs["evidence"] = result.clone_calls.evidence
            f["clone_calling"].attrs["feature_info"] = json.dumps(
                result.clone_calls.feature_info
            )
        for name, value in asdict(fit.posterior.depth_options).items():
            f.attrs["depth_" + name] = value
        for key, values in (
            ("barcodes", bundle.barcodes),
            ("genes", bundle.gene_ids),
            ("classes", CLASS_NAMES),
        ):
            f.create_dataset(key, data=np.array(values, dtype=h5py.string_dtype()))
        f.create_dataset("gene_markers", data=bundle.gene_markers)
        f.create_dataset(
            "grid/chrom", data=np.array(bundle.grid.chrom, dtype=h5py.string_dtype())
        )
        f.create_dataset("grid/start", data=bundle.grid.start)
        f.create_dataset("grid/end", data=bundle.grid.end)
        f.create_dataset("order", data=result.order)
        f.create_dataset("probabilities", data=probabilities, compression="gzip")
        f["probabilities"].attrs["axes"] = "barcode,marker,class"
        f.create_dataset("phase", data=fit.pooled_fit.final_phase.phase)
        f["phase"].attrs["meaning"] = "pooled phase for cross-barcode HF alignment"
        if fit.posterior.phase.ndim == 2:
            f.create_dataset(
                "barcode_phase", data=fit.posterior.phase, compression="gzip"
            )
            f["barcode_phase"].attrs["axes"] = "barcode,locus"
        hf = result.hf_features
        if hf is not None:
            group = f.create_group("haplotype_features")
            group.create_dataset(
                "chromosome",
                data=np.array([c for c, _ in hf["regions"]], dtype=h5py.string_dtype()),
            )
            group.create_dataset(
                "center_bp",
                data=np.array(
                    [
                        i * hf["stride_bp"] + hf["stride_bp"] // 2
                        for _, i in hf["regions"]
                    ]
                ),
            )
            for name, value in (
                ("centered_signed_fraction", hf["x"]),
                ("coverage", hf["coverage"]),
            ):
                group.create_dataset(name, data=value, compression="gzip")
                group[name].attrs["axes"] = "region,barcode"
            group.attrs["window_bp"] = hf["window_bp"]
            group.attrs["stride_bp"] = hf["stride_bp"]
        for name in ("self_expression", "external_expression"):
            f.create_dataset(name, data=getattr(result, name), compression="gzip")
            f[name].attrs["axes"] = "gene,barcode"
        if result.distances is not None:
            f.create_dataset(
                "broad_cn_distances", data=result.distances, compression="gzip"
            )
            f["broad_cn_distances"].attrs["axes"] = "barcode,barcode,draw"
        group = f.create_group("group_calls")
        for name in (
            "groups",
            "cells",
            "barcodes",
            "consensus",
            "pooled",
            "phase",
            "uncertain_cell_fraction",
            "consensus_conflict_cell_fraction",
            "pooled_conflict_cell_fraction",
        ):
            group.create_dataset(name, data=getattr(calls, name), compression="gzip")
        for name in ("consensus", "pooled"):
            group[name].attrs["axes"] = "group,marker,class"
        group["phase"].attrs["axes"] = "group,locus"
        for name in (
            "uncertain_cell_fraction",
            "consensus_conflict_cell_fraction",
            "pooled_conflict_cell_fraction",
        ):
            group[name].attrs["axes"] = "group,marker"
        group.attrs["consensus_meaning"] = (
            "cell-weighted barcode class probabilities; not a pooled-state posterior"
        )
        group.attrs["pooled_meaning"] = (
            "summed cell counts per group; joint CN/phase paths; fixed membership; noise refitted to pseudobulks"
        )
        group.attrs["confident_threshold"] = calls.confident_threshold
        for name, value in asdict(calls.depth_options).items():
            group.attrs["depth_" + name] = value
        if calls.dispersion is not None:
            group.attrs["alpha"] = calls.dispersion.parameters.alpha
            group.attrs["allele_logit_bias"] = (
                calls.dispersion.parameters.allele_logit_bias
            )
            group.attrs["dispersion_converged"] = calls.dispersion.converged
            group.attrs["dispersion_at_boundary"] = calls.dispersion.at_boundary
    pooled = fit.pooled_fit
    return dict(
        phase_converged=pooled.converged,
        initial_phase_iterations=pooled.initial_phase.iterations,
        final_phase_iterations=pooled.final_phase.iterations,
        pooled_alpha=pooled.dispersion.parameters.alpha,
        barcode_alpha=fit.dispersion.parameters.alpha,
        dispersion_at_boundary=fit.dispersion.at_boundary,
        groups=len(set(result.groups) - {0}),
        unresolved_barcodes=int(np.sum(result.groups == 0)),
        clone_calling=dict(
            method=result.clone_calls.method
            if result.clone_calls is not None
            else None,
            status="applied" if result.clone_calls is not None else "disabled",
            evidence=result.clone_calls.evidence
            if result.clone_calls is not None
            else "expression",
            feature_info=result.clone_calls.feature_info
            if result.clone_calls is not None
            else {},
            membership="pre-HF decisions; support is relative preference, not a clone probability",
        ),
        model=asdict(fit.posterior.model),
        depth_options=asdict(fit.posterior.depth_options),
        hf_refinement=dict(
            status=result.hf_status,
            window_bp=WINDOW_BP,
            stride_bp=BASE_BIN_BP,
            phase="conditional MAP",
            groups_before=len(set(result.pre_hf_groups) - {0}),
        ),
        barcode_phase_method="joint"
        if fit.posterior.phase.ndim == 2
        else "conditional",
        group_reporting=dict(
            pooled_model="one CN and phase path per final reporting-group pseudobulk",
            likelihood="summed expression and allele counts; summed library exposure",
            depth_options=asdict(calls.depth_options),
            noise="fitted to group pseudobulks",
            dispersion=None if calls.dispersion is None else asdict(calls.dispersion),
            membership="fixed; group zero excluded",
            consensus="cell-weighted barcode class probabilities",
            confident_threshold=calls.confident_threshold,
            conflict="cell fraction with a confident barcode MAP class different from the indicated group call",
            segments="runs of marginal MAP classes; bounds are first/last observed markers; probability summaries are not whole-segment probabilities",
        ),
        depth_sigma=float(np.sqrt(np.log1p(fit.dispersion.parameters.alpha))),
        reference_parameterization="latent lognormal median"
        if fit.posterior.model.depth_family == "pln"
        else "NB mean",
        uncertainty=(
            "CN and phase integrated jointly within barcodes, conditional on fitted noise; "
            if fit.posterior.phase.ndim == 2
            else "CN probabilities conditional on fitted phase/noise; "
        )
        + "CN membership support and expression/HF bootstrap stability are not clone posterior probabilities",
    )
