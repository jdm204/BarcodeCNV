"""CLI argument parsing and exit codes; orchestration lives in barcodecnv.api."""

import argparse
import logging
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import api
from .depth_controls import DepthOptions
from .preprocessing import add_arguments as add_preprocessing_arguments
from .reference import add_reference_arguments


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
        "--cells", required=required, type=Path, help="cell/barcode CSV or TSV"
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


def validate_source(args, command_parser):
    """Reject ambiguous/incomplete input modes before reading or writing files."""
    if args.command == "prepare":
        return
    raw = any(
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
    prepare.add_argument(
        "--out",
        required=True,
        type=Path,
        help="new directory for prepared counts, reference fit and provenance",
    )
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
            help="prepared cell-counts HDF5 bundle; omit when using --matrix",
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
    values = vars(args).copy()
    command = values.pop("command")
    for flag in ("cn_refinement", "hf_refinement"):
        if "no_" + flag in values:
            values[flag] = not values.pop("no_" + flag)
    try:
        if command == "setup" and values.pop("dry_run", False):
            from .resources import setup

            setup(**values, dry_run=True)
        else:
            operation = (
                "run_pipeline" if command == "run" else command.replace("-", "_")
            )
            getattr(api, operation)(**values)
    except (
        ValueError,
        OSError,
        RuntimeError,
        KeyError,
        ArithmeticError,
        subprocess.SubprocessError,
    ) as error:
        logging.getLogger("barcodecnv").error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
