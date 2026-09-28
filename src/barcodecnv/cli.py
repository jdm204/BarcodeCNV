"""Command-line entry points; numerical/statistical functions perform no I/O."""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pandas as pd

from .bundle import read_bundle, write_bundle
from .cn_reporting import write_cn_tables
from .depth_controls import DepthOptions
from .fitting import FitOptions
from .haplotypes import BASE_BIN_BP, WINDOW_BP
from .loading import load_cells
from .model import CLASS_NAMES, Model
from .preprocessing import add_arguments as add_preprocessing_arguments
from .preprocessing import preprocess, write_json
from .reference import add_reference_arguments, fit_for_args, write_fit
from .signal import barcode_signal_test
from .workflow import run_pbpc


def write_signal(out, result):
    from .plotting import plot_signal

    values = dict(result)
    null = values.pop("null_scores")
    (out / "signal.json").write_text(
        json.dumps(values, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame(null, columns=["joint", "depth", "allele"]).to_csv(
        out / "signal_null.csv", index=False
    )
    plot_signal(result, out / "signal.png")


def write_run(out, result):
    bundle = result["bundle"]
    fit = result["fit"]
    calls = result["group_calls"]
    probabilities = fit.classes
    groups = pd.DataFrame(
        dict(
            barcode=bundle.barcodes,
            cells=result["cells"],
            group=result["groups"],
            expression_group=result["expression_groups"],
            phase_group=result["phase_groups"],
            pre_hf_group=result["pre_hf_groups"],
            hf_stability=result["hf_stability"],
            weighted_stability=result["weighted"]["stability"],
            core_stability=result["core"]["stability"],
        )
    )
    groups.to_csv(out / "groups.csv", index=False)
    pd.DataFrame(dict(barcode=np.array(bundle.barcodes)[result["order"]])).to_csv(
        out / "order.csv", index=False
    )
    for name, rows in (
        ("weighted", result["weighted"]["nodes"]),
        ("core", result["core"]["nodes"]),
        ("cn", result["cn_audit"]),
        ("hf", result["hf_audit"]),
    ):
        pd.DataFrame(rows).to_csv(out / f"{name}_nodes.csv", index=False)
    write_cn_tables(out, bundle, probabilities, result["groups"], calls)
    with h5py.File(out / "result.h5", "w") as f:
        f.attrs["barcodecnv_schema"] = "result-v1"
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
        f.create_dataset("order", data=result["order"])
        f.create_dataset("probabilities", data=probabilities, compression="gzip")
        f["probabilities"].attrs["axes"] = "barcode,marker,class"
        f.create_dataset("phase", data=fit.pooled_fit.final_phase.phase)
        f["phase"].attrs["meaning"] = "pooled phase for cross-barcode HF alignment"
        if fit.posterior.phase.ndim == 2:
            f.create_dataset(
                "barcode_phase", data=fit.posterior.phase, compression="gzip"
            )
            f["barcode_phase"].attrs["axes"] = "barcode,locus"
        hf = result["hf_features"]
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
            f.create_dataset(name, data=result[name], compression="gzip")
            f[name].attrs["axes"] = "gene,barcode"
        if result["distances"] is not None:
            f.create_dataset(
                "broad_cn_distances", data=result["distances"], compression="gzip"
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
        groups=int(result["groups"].max()),
        unresolved_barcodes=int(np.sum(result["groups"] == 0)),
        model=asdict(fit.posterior.model),
        depth_options=asdict(fit.posterior.depth_options),
        hf_refinement=dict(
            status=result["hf_status"],
            window_bp=WINDOW_BP,
            stride_bp=BASE_BIN_BP,
            phase="conditional MAP",
            groups_before=int(result["pre_hf_groups"].max()),
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
        + "membership values are bootstrap stability, not clone posterior probabilities",
    )


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def outlier_probability(value):
    number = float(value)
    if not np.isfinite(number) or not 0 <= number < 1:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1)")
    return number


def add_count_inputs(command, *, required=False):
    """One loader interface shared by prepare, run and signal."""
    command.add_argument(
        "--matrix",
        required=required,
        type=Path,
        help="10x matrix directory or filtered_feature_bc_matrix.h5",
    )
    command.add_argument(
        "--cells", required=required, type=Path, help="cell/barcode/block CSV or TSV"
    )
    command.add_argument(
        "--genes",
        required=required,
        type=Path,
        help="gene coordinates CSV/TSV or GTF[.gz]",
    )
    add_reference_arguments(command)
    command.add_argument(
        "--alleles",
        type=Path,
        help="phased per-cell h1/h2 table or Numbat count TSV[.gz]",
    )
    command.add_argument(
        "--genetic-map",
        type=Path,
        help="PLINK map file/directory; otherwise use per-SNP genetic_cm/cM",
    )
    command.add_argument(
        "--one-block",
        action="store_true",
        help="explicitly declare one exchangeable cohort if cells lack a block column",
    )


def load_count_inputs(args):
    fitted = fit_for_args(args, args.matrix)
    bundle = load_cells(
        args.matrix,
        args.cells,
        args.genes,
        reference=fitted.profile if fitted else args.reference,
        alleles=args.alleles,
        genetic_map=args.genetic_map,
        one_block=args.one_block,
    )
    return bundle, fitted


def validate_source(args, command_parser):
    """Reject ambiguous/incomplete input modes before reading or writing files."""
    if args.command == "prepare":
        return
    raw = (
        any(
            getattr(args, key) is not None
            for key in (
                "matrix",
                "cells",
                "genes",
                "reference",
                "reference_panel",
                "alleles",
                "genetic_map",
            )
        )
        or args.one_block
    )
    if args.input is not None:
        if raw:
            command_parser.error(
                "use either a prepared bundle or count-input options, not both"
            )
        return
    if args.matrix is None:
        command_parser.error(
            "provide a prepared bundle or --matrix with --cells and --genes"
        )
    needed = ["cells", "genes"] + (
        ["reference"]
        if args.command == "infer" and args.reference_panel is None
        else []
    )
    missing = ["--" + key for key in needed if getattr(args, key) is None]
    if missing:
        command_parser.error("count-input mode requires " + ", ".join(missing))


def add_analysis_options(command, *, inference, seed=True):
    if seed:
        command.add_argument("--seed", type=int, default=42)
    command.add_argument("--permutations", type=positive, default=199)
    command.add_argument("--bin-width", type=positive, default=10_000_000, metavar="BP")
    if not inference:
        return
    command.add_argument(
        "--bootstraps",
        type=positive,
        default=64,
        help="whole-cell grouping replicates (at least 20)",
    )
    command.add_argument(
        "--draws",
        type=positive,
        default=256,
        help="independent CN paths for refinement",
    )
    command.add_argument("--phase-iterations", type=positive, default=60)
    command.add_argument(
        "--depth-family",
        choices=["nb", "pln"],
        default="pln",
        help="NB reference mean or Poisson–lognormal latent reference median (default: pln)",
    )
    command.add_argument(
        "--depth-outlier-probability",
        type=outlier_probability,
        default=DepthOptions().outlier_probability,
        metavar="P",
        help="CN-independent depth outlier probability (default: 0.05); lower for a well-matched reference; 0 disables",
    )
    command.add_argument(
        "--barcode-phase",
        choices=["conditional", "joint"],
        default="joint",
        help="shared fitted phase approximation or exact per-barcode CN/phase HMM (default: joint)",
    )
    command.add_argument(
        "--skip-signal",
        action="store_true",
        help="omit the separate barcode association diagnostic",
    )
    command.add_argument(
        "--no-cn-refinement", action="store_true", help="omit CN-path group refinement"
    )
    command.add_argument(
        "--no-hf-refinement",
        action="store_true",
        help="omit rolling haplotype-fraction group refinement",
    )


def parser():
    root = argparse.ArgumentParser(
        prog="barcodecnv",
        description="Barcode expression grouping and count-based CN inference.",
    )
    root.add_argument("--version", action="version", version=version("barcodecnv"))
    commands = root.add_subparsers(dest="command", required=True)
    from .resources import add_arguments as add_setup_arguments

    add_setup_arguments(
        commands.add_parser(
            "setup",
            help="Download hg38/B-cell resources and install preprocessing tools",
        )
    )
    for name, description in (
        ("run", "End-to-end: Cell Ranger BAM/counts to CN inference, tables and plots"),
        ("preprocess", "Cell Ranger BAM/counts to a phased cell-count bundle"),
    ):
        command = commands.add_parser(name, help=description, description=description)
        add_preprocessing_arguments(command)
        if name == "run":
            add_analysis_options(command, inference=True, seed=False)
    reference = commands.add_parser(
        "fit-reference",
        help="Fit a normal expression panel and save a reusable profile",
    )
    for option in ("matrix", "cells", "genes", "out"):
        reference.add_argument("--" + option, type=Path, required=True)
    add_reference_arguments(reference, required=True, panel_only=True)
    prepare = commands.add_parser(
        "prepare",
        help="Load existing count and annotation tables into a cell-count bundle (no BAM processing)",
    )
    add_count_inputs(prepare, required=True)
    prepare.add_argument("--out", required=True, type=Path, help="new HDF5 bundle file")
    for name, description in (
        ("signal", "Test barcode-associated regional signal; no CN inference"),
        (
            "infer",
            "Infer CN and groups from a prepared bundle or existing count tables",
        ),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument(
            "input",
            nargs="?",
            type=Path,
            help="prepared cell-counts-v1 HDF5 bundle; omit when using --matrix",
        )
        add_count_inputs(command)
        command.add_argument(
            "--out",
            required=True,
            type=Path,
            help="new output directory (existing directories are not overwritten)",
        )
        add_analysis_options(command, inference=name == "infer")
    return root


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.command not in ("run", "preprocess", "fit-reference", "setup"):
        validate_source(args, command_parser)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("barcodecnv")
    if args.command == "setup":
        import subprocess

        from .resources import setup

        try:
            setup(args)
            return 0
        except (
            ValueError,
            OSError,
            RuntimeError,
            KeyError,
            subprocess.SubprocessError,
        ) as error:
            log.error("%s", error)
            return 1
    if args.command == "fit-reference":
        try:
            if args.out.exists():
                raise FileExistsError(f"output directory already exists: {args.out}")
            fitted = fit_for_args(args, args.matrix)
            write_fit(args.out, fitted)
            log.info("Fitted reference: %s", args.out / "reference.tsv")
            return 0
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            log.error("%s", error)
            return 1
    if args.command == "run":
        return run_end_to_end(args)
    if args.command == "preprocess":
        try:
            preprocess(args)
            return 0
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            log.error("%s", error)
            return 1
    if args.command == "prepare":
        try:
            if args.out.exists():
                raise FileExistsError(f"output file already exists: {args.out}")
            bundle, fitted = load_count_inputs(args)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if fitted:
                write_fit(args.out.with_name(args.out.name + ".reference_fit"), fitted)
            write_bundle(args.out, bundle)
            log.info(
                "Prepared %d cells, %d barcodes, %d genes and %d SNP loci: %s",
                len(bundle.cell_ids),
                len(bundle.barcodes),
                len(bundle.gene_ids),
                len(bundle.loci),
                args.out,
            )
            return 0
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            log.error("%s", error)
            return 1
    return analyse(args)


def validate_analysis(args):
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.command in ("run", "infer") and args.bootstraps < 20:
        raise ValueError("at least 20 bootstraps are required")


def run_end_to_end(args):
    """Compose the standalone stages, retaining the bundle if inference fails."""
    log = logging.getLogger("barcodecnv")
    out = args.out.resolve()
    started = perf_counter()
    created = False
    metadata = dict(
        command="run",
        version=version("barcodecnv"),
        status="running",
        stage="preprocessing",
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        preprocessing=str(out / "preprocessing"),
        inference=str(out / "inference"),
    )
    try:
        validate_analysis(
            args
        )  # Reject bad inference options before any expensive preprocessing.
        out.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(out / "pipeline.json", metadata)
        log.info("[run 1/2] Preprocessing Cell Ranger data")
        preparation = argparse.Namespace(**{**vars(args), "out": out / "preprocessing"})
        bundle_path = preprocess(preparation, inference_out=out / "inference")
        metadata.update(stage="inference", prepared=str(bundle_path))
        write_json(out / "pipeline.json", metadata)
        log.info("[run 2/2] Inferring CN and barcode groups")
        inference = argparse.Namespace(
            **{
                **vars(args),
                "command": "infer",
                "input": bundle_path,
                "out": out / "inference",
            }
        )
        result = analyse(inference)
        if result:
            path = out / "inference/run.json"
            reason = (
                json.loads(path.read_text()).get("error", "inference failed")
                if path.is_file()
                else "inference failed"
            )
            raise RuntimeError(reason)
        metadata.update(status="complete", stage="complete")
        log.info("End-to-end run complete: %s", out / "inference")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, ArithmeticError) as error:
        metadata.update(status="failed", error=str(error))
        log.error("%s", error)
        return 1
    finally:
        if created:
            metadata["seconds"] = perf_counter() - started
            write_json(out / "pipeline.json", metadata)


def analyse(args):
    log = logging.getLogger("barcodecnv")
    started = perf_counter()
    created = False
    raw = args.input is None
    input_path = args.out / "prepared.h5" if raw else args.input
    metadata = dict(
        command=args.command,
        input=str(input_path.resolve()),
        input_kind="counts" if raw else "bundle",
        version=version("barcodecnv"),
        seed=args.seed,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )
    try:
        validate_analysis(args)
        if args.out.exists():
            raise FileExistsError(f"output directory already exists: {args.out}")
        log.info(
            "[infer 1/4] Loading inputs"
            if args.command == "infer"
            else "Loading inputs"
        )
        if raw:
            log.info("Loading count matrix and annotations")
            metadata["source_paths"] = {
                key: str(getattr(args, key).resolve())
                for key in (
                    "matrix",
                    "cells",
                    "genes",
                    "reference",
                    "reference_panel",
                    "alleles",
                    "genetic_map",
                )
                if getattr(args, key) is not None
            }
            bundle, fitted = load_count_inputs(args)
        else:
            bundle = read_bundle(input_path)
        args.out.mkdir(parents=True, exist_ok=False)
        created = True
        if raw:
            if fitted:
                write_fit(args.out / "reference_fit", fitted)
                metadata["reference_fit"] = str((args.out / "reference_fit").resolve())
            write_bundle(input_path, bundle)
            log.info("Prepared input snapshot: %s", input_path)
        with input_path.open("rb") as handle:
            metadata["input_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        signal = None
        if args.command == "infer":
            log.info(
                "[infer 2/4] Barcode association diagnostic%s",
                " (skipped)" if args.skip_signal else "",
            )
        if args.command == "signal" or not args.skip_signal:
            log.info(
                "Barcode association: %s permutations within declared exchangeability blocks",
                args.permutations,
            )
            signal = barcode_signal_test(
                bundle,
                permutations=args.permutations,
                bin_width_bp=args.bin_width,
                seed=args.seed,
            )
            write_signal(args.out, signal)
            metadata["barcode_signal_pvalue"] = signal["pvalue"]
            log.info("Barcode signal: %s; p=%s", signal["status"], signal["pvalue"])
        if args.command == "infer":
            log.info("[infer 3/4] Fitting CN profiles and barcode groups")
            result = run_pbpc(
                bundle,
                replicates=args.bootstraps,
                draws=args.draws,
                seed=args.seed,
                cn_refinement=not args.no_cn_refinement,
                hf_refinement=not args.no_hf_refinement,
                model=Model(depth_family=args.depth_family),
                depth_options=DepthOptions(
                    outlier_probability=args.depth_outlier_probability
                ),
                options=FitOptions(
                    max_iterations=args.phase_iterations,
                    barcode_phase=args.barcode_phase,
                ),
                progress=log.info,
            )
            log.info("[infer 4/4] Writing tables and plots")
            metadata.update(write_run(args.out, result))
            from .plotting import plot_group_calls, plot_run

            plot_run(result, args.out / "summary.png", signal)
            plot_group_calls(bundle, result["group_calls"], args.out / "group_cn.png")
        metadata.update(status="complete", seconds=perf_counter() - started)
        (args.out / "run.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        log.info("Complete: %s", args.out)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, ArithmeticError) as error:
        log.error("%s", error)
        if created:
            metadata.update(
                status="failed", error=str(error), seconds=perf_counter() - started
            )
            (args.out / "run.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
