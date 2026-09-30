"""High-level Python workflows shared by notebooks and the command line.

All paths accept strings or pathlib.Path. Functions raise on failure and never
configure global logging. Analysis output paths must be new; completed input
snapshots and failure manifests are retained. setup reuses its resource cache.
The CLI only parses and calls these functions.
"""

import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pandas as pd

from .bundle import CellBundle, read_bundle, write_bundle
from .depth_controls import DepthOptions
from .fitting import FitOptions
from .loading import load_cells
from .model import Model
from .preprocessing import preprocess as preprocess
from .preprocessing import write_json
from .reference import fit_from_files, write_fit
from .reporting import write_run, write_signal
from .resources import resolve_expression_panel
from .resources import setup as setup
from .signal import barcode_signal_test
from .workflow import run_pbpc

LOG = logging.getLogger("barcodecnv")


def _arguments(values):
    """JSON metadata for workflow inputs, without copying in-memory counts."""
    return {
        k: str(v)
        if isinstance(v, Path)
        else "<AnnData>"
        if k == "adata" and v is not None
        else "<DataFrame>"
        if isinstance(v, pd.DataFrame)
        else "<CellBundle>"
        if isinstance(v, CellBundle)
        else v
        for k, v in values.items()
    }


def fit_reference(
    *,
    matrix,
    cells,
    genes,
    reference_panel,
    out=None,
    genome="hg38",
    reference_method="global",
    reference_min_cpm=2.0,
    reference_iterations=2000,
    reference_bin_genes=200,
    config=None,
):
    """Return a FittedReference; optionally write its profile, weights and audit.

    reference_panel is a directory or installed panel name (e.g. b-cells-v1).
    The numerical fit uses only the cells named in cells. No resources are fetched.
    """
    if out is not None:
        out = Path(out)
        if out.exists():
            raise FileExistsError(f"output directory already exists: {out}")
    panel = resolve_expression_panel(reference_panel, config)
    LOG.info("Fitting normal expression reference (%s)", reference_method)
    fitted = fit_from_files(
        matrix,
        cells,
        genes,
        panel,
        genome=genome,
        method=reference_method,
        min_cpm=reference_min_cpm,
        max_iter=reference_iterations,
        bin_genes=reference_bin_genes,
    )
    if out is not None:
        write_fit(out, fitted)
        LOG.info("Fitted reference: %s", out / "reference.tsv")
    return fitted


def _load_counts(
    *,
    matrix,
    cells,
    genes,
    reference=None,
    reference_panel=None,
    alleles=None,
    genetic_map=None,
    genome="hg38",
    reference_method="global",
    reference_min_cpm=2.0,
    reference_iterations=2000,
    reference_bin_genes=200,
    config=None,
):
    if any(value is None for value in (matrix, cells, genes)):
        raise ValueError("count-input mode requires matrix, cells and genes")
    if reference is not None and reference_panel is not None:
        raise ValueError("use either reference or reference_panel, not both")
    fitted = None
    if reference_panel is not None:
        fitted = fit_reference(
            matrix=matrix,
            cells=cells,
            genes=genes,
            reference_panel=reference_panel,
            genome=genome,
            reference_method=reference_method,
            reference_min_cpm=reference_min_cpm,
            reference_iterations=reference_iterations,
            reference_bin_genes=reference_bin_genes,
            config=config,
        )
    bundle = load_cells(
        matrix,
        cells,
        genes,
        reference=fitted.profile if fitted else reference,
        alleles=alleles,
        genetic_map=genetic_map,
    )
    return bundle, fitted


def prepare(
    *,
    matrix,
    cells,
    genes,
    out,
    reference=None,
    reference_panel=None,
    alleles=None,
    genetic_map=None,
    genome="hg38",
    reference_method="global",
    reference_min_cpm=2.0,
    reference_iterations=2000,
    reference_bin_genes=200,
    config=None,
):
    """Load count tables, save a reusable bundle and return the CellBundle.

    No BAM processing occurs. A supplied reference panel is fitted and audited
    beside the bundle, in <out>.reference_fit. Reference is optional for signal.
    """
    inputs = dict(locals())
    inputs.pop("out")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output file already exists: {out}")
    bundle, fitted = _load_counts(**inputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fitted:
        write_fit(out.with_name(out.name + ".reference_fit"), fitted)
    write_bundle(out, bundle)
    LOG.info(
        "Prepared %d cells, %d barcodes, %d genes and %d SNP loci: %s",
        len(bundle.cell_ids),
        len(bundle.barcodes),
        len(bundle.gene_ids),
        len(bundle.loci),
        out,
    )
    return bundle


def _validate_analysis(
    *,
    seed,
    permutations,
    bin_width,
    bootstraps=None,
    draws=256,
    phase_iterations=60,
    depth_family="pln",
    barcode_phase="joint",
    depth_outlier_probability=0.05,
):
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(not isinstance(x, int) or x < 1 for x in (permutations, bin_width, draws)):
        raise ValueError("permutations, bin_width and draws must be positive integers")
    if bootstraps is not None and (not isinstance(bootstraps, int) or bootstraps < 20):
        raise ValueError("at least 20 bootstraps are required")
    Model(depth_family=depth_family)
    FitOptions(max_iterations=phase_iterations, barcode_phase=barcode_phase)
    DepthOptions(outlier_probability=depth_outlier_probability)


def infer(
    input=None,
    *,
    out,
    matrix=None,
    cells=None,
    genes=None,
    reference=None,
    reference_panel=None,
    alleles=None,
    genetic_map=None,
    genome="hg38",
    reference_method="global",
    reference_min_cpm=2.0,
    reference_iterations=2000,
    reference_bin_genes=200,
    config=None,
    seed=42,
    permutations=199,
    bin_width=10_000_000,
    bootstraps=64,
    draws=256,
    phase_iterations=60,
    depth_family="pln",
    depth_outlier_probability=0.05,
    barcode_phase="joint",
    skip_signal=False,
    cn_refinement=True,
    hf_refinement=True,
):
    """Infer from a bundle path, CellBundle, or count inputs and write a full report.

    Returns run_pbpc's result dictionary, plus signal and output_directory.
    A CellBundle or count inputs are snapshotted to out/prepared.h5. Failures
    after output creation are recorded in out/run.json and re-raised.
    """
    return _analyse("infer", **locals())


def signal(
    input=None,
    *,
    out,
    matrix=None,
    cells=None,
    genes=None,
    reference=None,
    reference_panel=None,
    alleles=None,
    genetic_map=None,
    genome="hg38",
    reference_method="global",
    reference_min_cpm=2.0,
    reference_iterations=2000,
    reference_bin_genes=200,
    config=None,
    seed=42,
    permutations=199,
    bin_width=10_000_000,
):
    """Return the barcode-signal diagnostic and write its JSON, null table and plot.

    input may be a bundle path or CellBundle; alternatively supply count inputs.
    No reference or HMM fitting is required. Failure handling matches infer.
    """
    return _analyse("signal", **locals())


def _analyse(
    command,
    *,
    input,
    out,
    matrix,
    cells,
    genes,
    reference,
    reference_panel,
    alleles,
    genetic_map,
    genome,
    reference_method,
    reference_min_cpm,
    reference_iterations,
    reference_bin_genes,
    config,
    seed,
    permutations,
    bin_width,
    bootstraps=None,
    draws=256,
    phase_iterations=60,
    depth_family="pln",
    depth_outlier_probability=0.05,
    barcode_phase="joint",
    skip_signal=False,
    cn_refinement=True,
    hf_refinement=True,
):
    arguments = _arguments(locals())
    _validate_analysis(
        seed=seed,
        permutations=permutations,
        bin_width=bin_width,
        bootstraps=bootstraps,
        draws=draws,
        phase_iterations=phase_iterations,
        depth_family=depth_family,
        barcode_phase=barcode_phase,
        depth_outlier_probability=depth_outlier_probability,
    )
    source_paths = dict(
        matrix=matrix,
        cells=cells,
        genes=genes,
        reference=reference,
        reference_panel=reference_panel,
        alleles=alleles,
        genetic_map=genetic_map,
    )
    if input is not None and any(v is not None for v in source_paths.values()):
        raise ValueError(
            "use either a prepared bundle or count-input options, not both"
        )
    if (
        input is None
        and command == "infer"
        and reference is None
        and reference_panel is None
    ):
        raise ValueError("count-input inference requires reference or reference_panel")
    out = Path(out)
    raw = input is None
    memory = isinstance(input, CellBundle)
    input_path = out / "prepared.h5" if raw or memory else Path(input)
    started = perf_counter()
    created = False
    metadata = dict(
        command=command,
        input=str(input_path.resolve()),
        input_kind="counts" if raw else "memory" if memory else "bundle",
        version=version("barcodecnv"),
        seed=seed,
        arguments=arguments,
    )
    try:
        if out.exists():
            raise FileExistsError(f"output directory already exists: {out}")
        LOG.info(
            "[infer 1/4] Loading inputs" if command == "infer" else "Loading inputs"
        )
        fitted = None
        if raw:
            LOG.info("Loading count matrix and annotations")
            metadata["source_paths"] = {
                k: str(Path(v).resolve())
                for k, v in source_paths.items()
                if v is not None
            }
            bundle, fitted = _load_counts(
                **source_paths,
                genome=genome,
                reference_method=reference_method,
                reference_min_cpm=reference_min_cpm,
                reference_iterations=reference_iterations,
                reference_bin_genes=reference_bin_genes,
                config=config,
            )
        else:
            bundle = input if memory else read_bundle(input_path)
        out.mkdir(parents=True, exist_ok=False)
        created = True
        if raw or memory:
            if fitted:
                write_fit(out / "reference_fit", fitted)
                metadata["reference_fit"] = str((out / "reference_fit").resolve())
            write_bundle(input_path, bundle)
            LOG.info("Prepared input snapshot: %s", input_path)
        with input_path.open("rb") as handle:
            metadata["input_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        diagnostic = None
        if command == "infer":
            LOG.info(
                "[infer 2/4] Barcode association diagnostic%s",
                " (skipped)" if skip_signal else "",
            )
        if command == "signal" or not skip_signal:
            LOG.info(
                "Barcode association: %s permutations across all selected cells",
                permutations,
            )
            diagnostic = barcode_signal_test(
                bundle, permutations=permutations, bin_width_bp=bin_width, seed=seed
            )
            write_signal(out, diagnostic)
            metadata["barcode_signal_pvalue"] = diagnostic["pvalue"]
            LOG.info(
                "Barcode signal: %s; p=%s", diagnostic["status"], diagnostic["pvalue"]
            )
        if command == "infer":
            LOG.info("[infer 3/4] Fitting CN profiles and barcode groups")
            result = run_pbpc(
                bundle,
                replicates=bootstraps,
                draws=draws,
                seed=seed,
                cn_refinement=cn_refinement,
                hf_refinement=hf_refinement,
                model=Model(depth_family=depth_family),
                depth_options=DepthOptions(
                    outlier_probability=depth_outlier_probability
                ),
                options=FitOptions(
                    max_iterations=phase_iterations, barcode_phase=barcode_phase
                ),
                progress=LOG.info,
            )
            LOG.info("[infer 4/4] Writing tables and plots")
            metadata.update(write_run(out, result))
            from .plotting import plot_group_calls, plot_run

            plot_run(result, out / "summary.png", diagnostic)
            plot_group_calls(bundle, result["group_calls"], out / "group_cn.png")
            result.update(signal=diagnostic, output_directory=out)
        else:
            result = diagnostic
        metadata.update(status="complete", seconds=perf_counter() - started)
        (out / "run.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        LOG.info("Complete: %s", out)
        return result
    except Exception as error:
        if created:
            metadata.update(
                status="failed", error=str(error), seconds=perf_counter() - started
            )
            (out / "run.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
        raise


def run(
    *,
    outs=None,
    adata=None,
    bam=None,
    layer=None,
    use_raw=False,
    gene_id_key=None,
    library_size_key=None,
    cells,
    out,
    reference=None,
    reference_panel=None,
    config=None,
    genes=None,
    snp_vcf=None,
    genetic_map=None,
    phasing_panel=None,
    beagle_jar=None,
    genome="hg38",
    chromosomes=None,
    threads=8,
    memory_gb=12,
    seed=42,
    cellsnp_dir=None,
    phased_vcf=None,
    reference_method="global",
    reference_min_cpm=2.0,
    reference_iterations=2000,
    reference_bin_genes=200,
    permutations=199,
    bin_width=10_000_000,
    bootstraps=64,
    draws=256,
    phase_iterations=60,
    depth_family="pln",
    depth_outlier_probability=0.05,
    barcode_phase="joint",
    skip_signal=False,
    cn_refinement=True,
    hf_refinement=True,
):
    """Run preprocessing then inference; return the same result as infer.

    Supply outs or adata plus bam. AnnData count selection, cell intersection,
    gene IDs and whole-assay library sizes follow preprocess (see its docstring).

    Writes preprocessing/, inference/ and pipeline.json under a new out directory.
    Failed inference retains prepared.h5. Errors are recorded and re-raised.
    """
    arguments = _arguments(locals())
    _validate_analysis(
        seed=seed,
        permutations=permutations,
        bin_width=bin_width,
        bootstraps=bootstraps,
        draws=draws,
        phase_iterations=phase_iterations,
        depth_family=depth_family,
        barcode_phase=barcode_phase,
        depth_outlier_probability=depth_outlier_probability,
    )
    out = Path(out).resolve()
    started = perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        command="run",
        version=version("barcodecnv"),
        status="running",
        stage="preprocessing",
        arguments=arguments,
        preprocessing=str(out / "preprocessing"),
        inference=str(out / "inference"),
    )
    try:
        write_json(out / "pipeline.json", metadata)
        LOG.info("[run 1/2] Preprocessing Cell Ranger data")
        prepared = preprocess(
            outs=outs,
            adata=adata,
            bam=bam,
            layer=layer,
            use_raw=use_raw,
            gene_id_key=gene_id_key,
            library_size_key=library_size_key,
            cells=cells,
            out=out / "preprocessing",
            reference=reference,
            reference_panel=reference_panel,
            config=config,
            genes=genes,
            snp_vcf=snp_vcf,
            genetic_map=genetic_map,
            phasing_panel=phasing_panel,
            beagle_jar=beagle_jar,
            genome=genome,
            chromosomes=chromosomes,
            threads=threads,
            memory_gb=memory_gb,
            seed=seed,
            cellsnp_dir=cellsnp_dir,
            phased_vcf=phased_vcf,
            reference_method=reference_method,
            reference_min_cpm=reference_min_cpm,
            reference_iterations=reference_iterations,
            reference_bin_genes=reference_bin_genes,
            inference_out=out / "inference",
        )
        metadata.update(stage="inference", prepared=str(prepared))
        write_json(out / "pipeline.json", metadata)
        LOG.info("[run 2/2] Inferring CN and barcode groups")
        result = infer(
            prepared,
            out=out / "inference",
            seed=seed,
            permutations=permutations,
            bin_width=bin_width,
            bootstraps=bootstraps,
            draws=draws,
            phase_iterations=phase_iterations,
            depth_family=depth_family,
            depth_outlier_probability=depth_outlier_probability,
            barcode_phase=barcode_phase,
            skip_signal=skip_signal,
            cn_refinement=cn_refinement,
            hf_refinement=hf_refinement,
        )
        metadata.update(status="complete", stage="complete")
        LOG.info("End-to-end run complete: %s", out / "inference")
        return result
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        metadata["seconds"] = perf_counter() - started
        write_json(out / "pipeline.json", metadata)
