"""High-level Python workflows shared by notebooks and the command line.

The recommended route is setup() -> from_anndata()/from_10x() ->
fit_reference() -> preprocess() -> infer(). run_pipeline() composes preprocess
and infer; prepare() is the alternative for existing allele tables. Both
preparation routes return PreparedResult. signal() also runs inside infer().

ExpressionInput and PreparedResult objects connect the operations. AnnData and file
readers are adapters to that boundary. File outputs are optional for fitting,
preparation and analysis; BAM preprocessing uses a temporary work directory
when out is omitted. Functions raise on failure and never configure global logging. Requested output paths must
be new, except setup's reusable cache. The CLI parses and calls these functions.
"""

import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pandas as pd

from .bundle import CellBundle, read_bundle, write_bundle
from .clone_calling import DEFAULT_CLONE_METHOD, validate_clone_method
from .depth_controls import DepthOptions
from .expression import ExpressionInput, resolve_expression
from .expression import from_10x as from_10x
from .expression import from_anndata as from_anndata
from .fitting import FitOptions
from .loading import load_cells
from .model import Model
from .preprocessing import preprocess as preprocess
from .preprocessing import provenance, write_json
from .reference import FittedReference, ReferencePanel, resolve_reference, write_fit
from .reporting import write_signal
from .resource_set import Resources, load_resources
from .resources import CHROMOSOMES
from .results import InferenceResult, PreparedResult, SignalResult
from .signal import barcode_signal_test
from .workflow import run_pbpc

LOG = logging.getLogger("barcodecnv")

__all__ = [
    "setup",
    "load_resources",
    "from_anndata",
    "from_10x",
    "fit_reference",
    "prepare",
    "preprocess",
    "signal",
    "infer",
    "run_pipeline",
    "ExpressionInput",
    "Resources",
    "ReferencePanel",
    "FittedReference",
    "PreparedResult",
    "SignalResult",
    "InferenceResult",
    "CellBundle",
]


def __dir__():
    """Keep notebook completion focused on the supported workflow interface."""
    return sorted(__all__)


def setup(
    *,
    resource_dir=None,
    genome="hg38",
    chromosomes=CHROMOSOMES,
    tools="managed",
    expression_panel="b-cells-v1",
) -> Resources:
    """Provision/cache resources once, separately from run_pipeline.

    Returns Resources with genes, reference_panel and config_path attributes.
    For an existing installation, load_resources(config_path) needs no downloads.
    The CLI's setup --dry-run prints a download plan without provisioning.
    """
    from .resources import setup as provision

    return load_resources(
        provision(
            resource_dir=resource_dir,
            genome=genome,
            chromosomes=chromosomes,
            tools=tools,
            expression_panel=expression_panel,
        )
    )


def _arguments(values):
    """JSON metadata for workflow inputs, without copying in-memory counts."""
    return {
        k: str(v)
        if isinstance(v, Path)
        else str(v.config_path)
        if isinstance(v, Resources)
        else v.provenance()
        if isinstance(v, ExpressionInput)
        else {"kind": "fitted_reference", "audit": v.audit}
        if isinstance(v, FittedReference)
        else {"kind": "reference_panel", "id": v.id}
        if isinstance(v, ReferencePanel)
        else "<AnnData>"
        if k == "adata" and v is not None
        else "<DataFrame>"
        if isinstance(v, pd.DataFrame)
        else "<PreparedResult>"
        if isinstance(v, PreparedResult)
        else "<CellBundle>"
        if isinstance(v, CellBundle)
        else v
        for k, v in values.items()
    }


def fit_reference(
    input=None,
    *,
    matrix=None,
    cells=None,
    adata=None,
    layer=None,
    use_raw=False,
    gene_id_key=None,
    library_size_key=None,
    genes,
    reference_panel,
    out=None,
    genome="hg38",
    reference_method="global",
    reference_min_cpm=2.0,
    reference_iterations=2000,
    reference_bin_genes=200,
    config=None,
) -> FittedReference:
    """Fit an ExpressionInput (or adapt AnnData/10x inputs); return FittedReference.

    genes accepts an annotation DataFrame or file. reference_panel accepts a
    ReferencePanel, directory or installed name. out optionally saves the fit.
    Without out, no files are written. Lineage annotations are not required.
    """
    if out is not None and Path(out).exists():
        raise FileExistsError(f"output directory already exists: {out}")
    expression = resolve_expression(
        input,
        matrix=matrix,
        cells=cells,
        adata=adata,
        layer=layer,
        use_raw=use_raw,
        gene_id_key=gene_id_key,
        library_size_key=library_size_key,
    )
    _, fitted = resolve_reference(
        expression,
        genes,
        reference_panel=reference_panel,
        config=config,
        genome=genome,
        reference_method=reference_method,
        reference_min_cpm=reference_min_cpm,
        reference_iterations=reference_iterations,
        reference_bin_genes=reference_bin_genes,
    )
    if out is not None:
        write_fit(out, fitted)
    return fitted


def _load_counts(
    input=None,
    *,
    matrix=None,
    cells=None,
    adata=None,
    layer=None,
    use_raw=False,
    gene_id_key=None,
    library_size_key=None,
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
    arguments = _arguments(locals())
    if genes is None:
        raise ValueError("count-input mode requires genes")
    expression = resolve_expression(
        input,
        matrix=matrix,
        cells=cells,
        adata=adata,
        layer=layer,
        use_raw=use_raw,
        gene_id_key=gene_id_key,
        library_size_key=library_size_key,
    )
    labels = expression.cell_table(require_barcodes=True)
    profile, fitted = resolve_reference(
        expression,
        genes,
        reference=reference,
        reference_panel=reference_panel,
        config=config,
        genome=genome,
        reference_method=reference_method,
        reference_min_cpm=reference_min_cpm,
        reference_iterations=reference_iterations,
        reference_bin_genes=reference_bin_genes,
    )
    bundle = load_cells(
        expression,
        labels,
        genes,
        reference=profile,
        alleles=alleles,
        genetic_map=genetic_map,
    )
    return PreparedResult(
        bundle,
        fitted,
        dict(
            command="prepare",
            status="complete",
            version=version("barcodecnv"),
            arguments=arguments,
            inputs=dict(
                expression=expression.provenance(),
                genes=provenance(genes, digest=True),
                alleles=provenance(alleles, digest=True)
                if alleles is not None
                else None,
                reference=provenance(
                    fitted if fitted is not None else profile, digest=True
                )
                if fitted is not None or profile is not None
                else None,
            ),
        ),
    )


def prepare(
    input=None,
    *,
    matrix=None,
    cells=None,
    adata=None,
    layer=None,
    use_raw=False,
    gene_id_key=None,
    library_size_key=None,
    genes,
    out=None,
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
) -> PreparedResult:
    """Prepare existing count tables; return counts, reference fit and provenance.

    input is an ExpressionInput; adata or matrix are convenience adapters.
    reference accepts a FittedReference, gene-to-fraction mapping or profile file.
    genes and alleles accept DataFrames or files. No BAM processing occurs.
    This is the alternative to preprocess() when allele counts already exist.
    Pass the returned PreparedResult to infer(). out optionally exports to a new
    directory, just like result.save(out); omitted means no output files.
    """
    inputs = dict(locals())
    inputs.pop("out")
    if out is not None and Path(out).exists():
        raise FileExistsError(f"output directory already exists: {out}")
    prepared = _load_counts(**inputs)
    if out is not None:
        prepared.save(out)
    return prepared


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
    clone_method=DEFAULT_CLONE_METHOD,
):
    validate_clone_method(clone_method)
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
    out=None,
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
    clone_method=DEFAULT_CLONE_METHOD,
    hf_refinement=True,
) -> InferenceResult:
    """Infer from a PreparedResult, CellBundle, expression or count-file inputs.

    Returns an InferenceResult with arrays, fits, diagnostics and provenance.
    Use result.plot_summary(), plot_groups() and plot_signal() for Figures,
    and result.save(out) to export a report later without recomputing inference.
    clone_method selects mean_profile (default), mean_distance, draw_quantile,
    or the original expression-first method, expression. self_expression and
    self_expression_tree group self-centred expression; self_expression_hf
    jointly groups it with phase-aligned HF. Their decisions do not use fitted
    CN calls. self_expression_correlation compares profile shapes;
    self_expression_hf_debiased corrects joint distances for measurement noise.
    These experimental alternatives test for excess variation before splitting.
    All methods share CN fitting for reports. Optional HF refinement
    follows clone calling (the joint method already uses HF before that step). cn_refinement=False
    bypasses the selected method and uses expression groups.
    A PreparedResult carries its fit/provenance into result.prepared.
    Omit out for in-memory results only (output_directory is None).
    With out, write the full report and completion/failure metadata.
    A CellBundle or count inputs are snapshotted to out/prepared.h5. Failures
    after output creation are recorded in out/run.json and re-raised.
    """
    return _analyse("infer", **locals())


def signal(
    input=None,
    *,
    out=None,
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
) -> SignalResult:
    """Return a SignalResult; inspect .pvalue/.scores or call .plot()/.save(out).

    input may be a PreparedResult, CellBundle, ExpressionInput or bundle path.
    No files are written when out is omitted; count-file inputs are also accepted.
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
    clone_method=DEFAULT_CLONE_METHOD,
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
        clone_method=clone_method,
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
    expression_input = isinstance(input, ExpressionInput)
    if (
        input is not None
        and not expression_input
        and any(v is not None for v in source_paths.values())
    ):
        raise ValueError(
            "use either a prepared bundle or count-input options, not both"
        )
    if (
        (input is None or expression_input)
        and command == "infer"
        and reference is None
        and reference_panel is None
    ):
        raise ValueError("count-input inference requires reference or reference_panel")
    out = Path(out) if out is not None else None
    raw = input is None or expression_input
    preprocessing = input if isinstance(input, PreparedResult) else None
    memory = isinstance(input, (CellBundle, PreparedResult))
    input_path = (
        (out / "prepared.h5" if out is not None else None)
        if raw or memory
        else Path(input)
    )
    started = perf_counter()
    created = False
    metadata = dict(
        command=command,
        input=str(input_path.resolve()) if input_path is not None else None,
        input_kind="counts" if raw else "memory" if memory else "bundle",
        version=version("barcodecnv"),
        seed=seed,
        arguments=arguments,
    )
    try:
        if out is not None and out.exists():
            raise FileExistsError(f"output directory already exists: {out}")
        LOG.info(
            "[infer 1/4] Loading inputs" if command == "infer" else "Loading inputs"
        )
        fitted = None
        if raw:
            LOG.info("Loading count matrix and annotations")
            metadata["source_paths"] = {
                k: str(Path(v).resolve())
                if isinstance(v, (str, Path))
                else _arguments({k: v})[k]
                for k, v in source_paths.items()
                if v is not None
            }
            preprocessing = _load_counts(
                input=input if expression_input else None,
                **source_paths,
                genome=genome,
                reference_method=reference_method,
                reference_min_cpm=reference_min_cpm,
                reference_iterations=reference_iterations,
                reference_bin_genes=reference_bin_genes,
                config=config,
            )
            bundle, fitted = preprocessing.bundle, preprocessing.reference_fit
        else:
            if preprocessing is not None:
                bundle = preprocessing.bundle
                fitted = preprocessing.reference_fit
            else:
                bundle = input if memory else read_bundle(input_path)
        if out is not None:
            out.mkdir(parents=True, exist_ok=False)
            created = True
            if preprocessing is not None:
                preprocessing._write_provenance(out / "preprocessing.json")
                metadata["preprocessing"] = str((out / "preprocessing.json").resolve())
            if raw or memory:
                if fitted:
                    write_fit(out / "reference_fit", fitted)
                    metadata["reference_fit"] = str((out / "reference_fit").resolve())
                write_bundle(input_path, bundle)
                LOG.info("Prepared input snapshot: %s", input_path)
            with input_path.open("rb") as handle:
                metadata["input_sha256"] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
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
            if out is not None and command == "signal":
                write_signal(out, diagnostic)
            metadata["barcode_signal_pvalue"] = diagnostic.pvalue
            LOG.info("Barcode signal: %s; p=%s", diagnostic.status, diagnostic.pvalue)
        if command == "infer":
            LOG.info("[infer 3/4] Fitting CN profiles and barcode groups")
            result = run_pbpc(
                bundle,
                replicates=bootstraps,
                draws=draws,
                seed=seed,
                cn_refinement=cn_refinement,
                clone_method=clone_method,
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
            result = InferenceResult(
                **result,
                signal=diagnostic,
                metadata=metadata.copy(),
                reference_fit=fitted,
                prepared=preprocessing,
            )
            if out is not None:
                LOG.info("[infer 4/4] Writing tables and plots")
                metadata.update(result._write_report(out))
                result.output_directory = out
        else:
            result = diagnostic
        metadata.update(status="complete", seconds=perf_counter() - started)
        result.metadata = metadata.copy()
        if command == "signal":
            result.output_directory = out
        if out is not None:
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


def run_pipeline(
    input=None,
    *,
    outs=None,
    adata=None,
    bam=None,
    layer=None,
    use_raw=False,
    gene_id_key=None,
    library_size_key=None,
    cells=None,
    out=None,
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
    clone_method=DEFAULT_CLONE_METHOD,
    hf_refinement=True,
) -> InferenceResult:
    """Compose preprocess() then infer(); setup() is a separate, explicit step.

    preprocess() fits a reference when given reference_panel, or reuses reference.
    infer() includes signal() unless skip_signal=True. prepare() is an alternative
    to BAM preprocessing for existing allele tables, not an extra pipeline step.

    Supply an ExpressionInput plus bam, outs, or adata plus bam.
    AnnData count selection, cell intersection,
    gene IDs and whole-assay library sizes follow preprocess (see its docstring).

    Without out, preprocessing uses temporary files and inference stays in
    memory. The result retains preprocessing provenance and can be saved later.
    With out, write preprocessing/, inference/ and pipeline.json under a new
    directory. Failed inference then retains prepared.h5 and failure records.
    Exceptions propagate in either mode.
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
        clone_method=clone_method,
    )
    out = Path(out).resolve() if out is not None else None
    preprocessing_out = out / "preprocessing" if out is not None else None
    inference_out = out / "inference" if out is not None else None
    started = perf_counter()
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        command="run",
        version=version("barcodecnv"),
        status="running",
        stage="preprocessing",
        arguments=arguments,
        preprocessing=str(preprocessing_out) if preprocessing_out is not None else None,
        inference=str(inference_out) if inference_out is not None else None,
    )
    try:
        if out is not None:
            write_json(out / "pipeline.json", metadata)

        LOG.info("[run 1/2] Preprocessing Cell Ranger data")
        prepared = preprocess(
            input,
            outs=outs,
            adata=adata,
            bam=bam,
            layer=layer,
            use_raw=use_raw,
            gene_id_key=gene_id_key,
            library_size_key=library_size_key,
            cells=cells,
            out=preprocessing_out,
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
            inference_out=inference_out,
        )
        metadata.update(
            stage="inference",
            prepared=str(preprocessing_out / "prepared.h5")
            if preprocessing_out is not None
            else None,
        )
        if out is not None:
            write_json(out / "pipeline.json", metadata)
        LOG.info("[run 2/2] Inferring CN and barcode groups")
        result = infer(
            prepared,
            out=inference_out,
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
            clone_method=clone_method,
            hf_refinement=hf_refinement,
        )
        metadata.update(status="complete", stage="complete")
        LOG.info(
            "End-to-end run complete%s",
            f": {inference_out}" if inference_out is not None else " (in memory)",
        )
        return result
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        metadata["seconds"] = perf_counter() - started
        if out is not None:
            write_json(out / "pipeline.json", metadata)
