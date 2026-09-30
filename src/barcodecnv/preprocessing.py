"""Cell Ranger outputs -> cell-count bundle, using cellSNP-lite and Beagle.

Orchestration and pooled genotype rules are adapted from the Julia BarcodeCNV
GermlineCalling module. No Julia, R, somatic-SNV or mitochondrial stages run.
"""

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .anndata_input import ExpressionInput, from_anndata
from .bundle import write_bundle
from .loading import chromosome, chromosome_key, load_cells, table
from .preprocessing_io import export_alleles, phased_sites, pooled_vcf, vcf_records
from .reference import add_reference_arguments, fit_from_files, write_fit

LOG = logging.getLogger(__name__)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def resolve_resource(directory, chrom, kind):
    directory = Path(directory)
    bare = chrom.removeprefix("chr")
    names = (
        [
            f"1kGP_HC_{chrom}.bref3",
            f"1kGP_HC_{bare}.bref3",
            f"{chrom}.bref3",
            f"{bare}.bref3",
        ]
        if kind == "panel"
        else [
            f"plink.chr{chrom}.GRCh38.map",
            f"plink.{chrom}.GRCh38.map",
            f"{chrom}.map",
        ]
    )
    if kind == "panel" and chrom == "chrX":
        names.append("1kGP_HC_chrX_nonPAR.bref3")
    for name in names:
        if (directory / name).is_file():
            return directory / name
    # Deliberately no substring fallback: chr1 must never accidentally select chr10.
    raise FileNotFoundError(f"no {kind} for {chrom} in {directory}")


def provenance(path, *, digest=False):
    if isinstance(path, ExpressionInput):
        return path.provenance()
    if isinstance(path, pd.DataFrame):
        return dict(
            kind="table",
            rows=len(path),
            sha256=hashlib.sha256(path.to_csv(index=False).encode()).hexdigest(),
        )
    path = Path(path).resolve()
    stat = path.stat()
    result = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if digest and path.is_file():
        with path.open("rb") as handle:
            result["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Commands:
    def __init__(self, tools, out):
        self.tools = tools
        self.out = out
        self.history = []

    def __call__(self, name, *args):
        argv = [self.tools[name], *map(str, args)]
        log = self.out / f"{len(self.history):02d}-{name}.log"
        self.history.append(dict(argv=argv, log=str(log)))
        LOG.info("%s", shlex.join(argv))
        with log.open("wb") as handle:
            try:
                subprocess.run(
                    argv, stdout=handle, stderr=subprocess.STDOUT, check=True
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"{name} failed; see {log}") from exc


def add_arguments(root):
    """Shared options for preprocess, end-to-end run and the legacy entry point."""
    root.add_argument(
        "--outs",
        type=Path,
        required=True,
        help="Cell Ranger outs directory (one donor)",
    )
    root.add_argument(
        "--cells",
        type=Path,
        required=True,
        help="cell/barcode CSV or TSV; aliases cell_barcode/lineage_barcode",
    )
    add_reference_arguments(root, required=False, genome=False)
    root.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new output directory (never overwritten)",
    )
    root.add_argument(
        "--config",
        type=Path,
        help="tools/resources JSON; default discovered from barcodecnv setup",
    )
    root.add_argument(
        "--genes", type=Path, help="GTF or gene coordinates; overrides config"
    )
    root.add_argument("--snp-vcf", type=Path, help="common SNP sites; overrides config")
    root.add_argument(
        "--genetic-map",
        type=Path,
        help="directory of chromosome PLINK maps; overrides config",
    )
    root.add_argument(
        "--phasing-panel",
        type=Path,
        help="directory of chromosome bref3 files; overrides config",
    )
    root.add_argument("--beagle-jar", type=Path, help="Beagle jar; overrides config")
    root.add_argument(
        "--genome",
        choices=["hg38"],
        default="hg38",
        help="currently provisioned build (must match BAM and resources)",
    )
    root.add_argument(
        "--chromosomes",
        default=None,
        help="comma-separated chromosomes; default configured set or 1..22,X",
    )
    root.add_argument("--threads", type=positive, default=8)
    root.add_argument(
        "--memory-gb",
        type=positive,
        default=12,
        help="Beagle Java heap; chromosomes run sequentially",
    )
    root.add_argument("--seed", type=int, default=42)
    root.add_argument(
        "--cellsnp-dir",
        type=Path,
        help="reuse existing cellSNP counts for EXACTLY these cells and BAM; caller asserts provenance",
    )
    root.add_argument(
        "--phased-vcf",
        type=Path,
        help="reuse a single-donor phased VCF for this donor/build",
    )
    return root


def parser():
    return add_arguments(argparse.ArgumentParser(description=__doc__))


def discover_config(explicit=None):
    """Explicit path, environment, working directory, then source checkout."""
    if explicit is not None:
        return Path(explicit)
    if value := os.environ.get("BARCODECNV_PREPROCESS_CONFIG"):
        return Path(value)
    from .resources import default_config

    checkout = Path(__file__).resolve().parents[2]
    for candidate in (
        Path.cwd() / ".preprocessing/config.json",
        checkout / ".preprocessing/config.json",
        default_config(),
    ):
        if candidate.is_file():
            return candidate
    return None


def preprocess(
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
    inference_out=None,
):
    """Preprocess one donor's counts and BAM; return the prepared bundle Path.

    Supply either outs, or adata plus bam. For AnnData, cells is a table path or
    DataFrame with cell/barcode columns; the intersection with obs_names is used.
    layer selects raw counts from a layer; use_raw selects raw.X and raw.var.
    Otherwise X is used. gene_id_key selects a var column instead of var_names.
    After gene filtering, library_size_key must name an obs column of original
    whole-assay totals, unless using a full-gene count source in raw. Otherwise
    totals are computed from the selected source. Inputs are not modified.

    Paths accept strings or Path objects. Resources are discovered from config;
    explicit resource arguments take precedence. Reference and reference_panel
    are alternatives. Existing output directories are never overwritten.
    """
    arguments = dict(locals())
    if (outs is None) == (adata is None):
        raise ValueError("provide either outs or adata with bam, not both")
    if outs is not None and (
        bam is not None
        or layer is not None
        or use_raw
        or gene_id_key is not None
        or library_size_key is not None
    ):
        raise ValueError("bam and AnnData options require adata")
    if adata is not None:
        if bam is None:
            raise ValueError("AnnData preprocessing requires bam")
        matrix, cells = from_anndata(
            adata,
            cells,
            layer=layer,
            use_raw=use_raw,
            gene_id_key=gene_id_key,
            library_size_key=library_size_key,
        )
        bam = Path(bam)
        arguments["adata"] = matrix.provenance()
        arguments["cells"] = provenance(cells)
        LOG.info("AnnData selection: %s", matrix.metadata)
    else:
        outs = Path(outs)
        matrix = outs / "filtered_feature_bc_matrix.h5"
        if not matrix.is_file():
            matrix = outs / "filtered_feature_bc_matrix"
        bam = outs / "possorted_genome_bam.bam"
        if isinstance(cells, pd.DataFrame):
            arguments["cells"] = provenance(cells)
    if genome != "hg38":
        raise ValueError("preprocessing currently supports hg38 only")
    if reference is not None and reference_panel is not None:
        raise ValueError("use either reference or reference_panel, not both")
    if any(not isinstance(v, int) or v < 1 for v in (threads, memory_gb)):
        raise ValueError("threads and memory_gb must be positive integers")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cells = cells if isinstance(cells, pd.DataFrame) else Path(cells)
    out = Path(out)
    reference = Path(reference) if reference is not None else None
    cellsnp_dir = Path(cellsnp_dir) if cellsnp_dir is not None else None
    phased_vcf = Path(phased_vcf) if phased_vcf is not None else None
    LOG.info("[preprocess 1/5] Validating inputs and resolving tools/resources")
    config_path = discover_config(config)
    settings = json.loads(config_path.read_text()) if config_path else {}
    if settings.get("genome", genome) != genome:
        raise ValueError("configured resource build differs from genome")
    resources = settings.get("resources", {})
    resolved = {
        key: Path(value or resources[key]).resolve()
        if value or resources.get(key)
        else None
        for key, value in dict(
            genes=genes,
            snp_vcf=snp_vcf,
            genetic_map=genetic_map,
            phasing_panel=phasing_panel,
            beagle_jar=beagle_jar,
        ).items()
    }
    genes, snp_vcf, genetic_map, phasing_panel, beagle_jar = resolved.values()
    if reference is None and reference_panel is None:
        reference_panel = settings.get("default_reference_panel", "b-cells-v1")
        LOG.info("Using default B-cell expression panel: %s", reference_panel)
    if reference_panel is not None:
        from .resources import resolve_expression_panel

        reference_panel = resolve_expression_panel(reference_panel, config_path)
    if chromosomes is None:
        chromosomes = settings.get("chromosomes", [*map(str, range(1, 23)), "X"])
    if isinstance(chromosomes, str):
        chromosomes = chromosomes.split(",")
    chroms = [chromosome(str(x).strip()) for x in chromosomes]
    supported = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
    if not chroms or len(set(chroms)) != len(chroms) or not set(chroms) <= supported:
        raise ValueError("chromosomes must be unique and drawn from 1..22,X")
    chroms.sort(key=chromosome_key)
    LOG.info("Allele chromosomes: %s", ",".join(chroms))
    required = ["genes", "genetic_map"]
    if not cellsnp_dir:
        required.append("snp_vcf")
    if not phased_vcf:
        required += ["phasing_panel", "beagle_jar"]
    for key in required:
        if resolved[key] is None or not resolved[key].exists():
            raise ValueError(f"provide {key} or run barcodecnv setup --genome hg38")
    arguments.update(
        resolved,
        config=config_path,
        reference_panel=reference_panel,
        chromosomes=chroms,
    )
    # Validate every requested chromosome before expensive pileup, even if no sites survive.
    maps = {c: resolve_resource(genetic_map, c, "map") for c in chroms}
    panels = (
        {}
        if phased_vcf
        else {c: resolve_resource(phasing_panel, c, "panel") for c in chroms}
    )
    if not cellsnp_dir:
        if not bam.is_file() or not any(
            p.is_file()
            for p in (
                Path(str(bam) + ".bai"),
                bam.with_suffix(".bai"),
                Path(str(bam) + ".csi"),
            )
        ):
            raise ValueError(
                "provide an indexed BAM (Cell Ranger outs must contain possorted_genome_bam.bam)"
            )
    if phased_vcf:
        phased_sites(phased_vcf)
    labels = table(cells).rename(
        columns={"cell_barcode": "cell", "lineage_barcode": "barcode"}
    )
    # Use the production loader for input validation, including cells in 10x and library sizes.
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    fitted_reference = (
        fit_from_files(
            matrix,
            cells,
            genes,
            reference_panel,
            genome=genome,
            method=reference_method,
            min_cpm=reference_min_cpm,
            max_iter=reference_iterations,
            bin_genes=reference_bin_genes,
        )
        if reference_panel is not None
        else None
    )
    reference_profile = fitted_reference.profile if fitted_reference else reference
    initial = load_cells(
        matrix,
        cells,
        genes,
        reference=reference_profile,
    )
    if not len(initial.cell_ids):
        raise ValueError("empty cell map")
    del initial
    if labels[["cell", "barcode"]].eq("").any().any():
        raise ValueError("cell and barcode labels must be nonempty")
    tools = {}
    needed = (["cellsnp-lite", "bcftools"] if not cellsnp_dir else []) + (
        [] if phased_vcf else ["bcftools", "java"]
    )
    for name in set(needed):
        executable = settings.get("tools", {}).get(name) or shutil.which(name)
        if not executable or not os.access(executable, os.X_OK):
            raise ValueError(
                f"{name} unavailable; configure preprocessing tools or add to PATH"
            )
        tools[name] = str(Path(executable).absolute())
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = Commands(tools, out)
    status = dict(
        status="running",
        genome=genome,
        chromosomes=chroms,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()
        },
        tools={k: provenance(v, digest=True) for k, v in tools.items()},
        inputs={
            k: provenance(v, digest=k in ("cells", "reference", "beagle_jar"))
            for k, v in dict(
                cells=cells,
                genes=genes,
                matrix=matrix,
                **(
                    {"reference": reference}
                    if reference
                    else {"reference_panel": reference_panel}
                ),
                **({"bam": bam} if not cellsnp_dir else {}),
                **({"beagle_jar": beagle_jar} if not phased_vcf else {}),
            ).items()
        },
        commands=run.history,
        confidence="GT-based heterozygosity=1 without GP; phase_prob=0.99 without PQ; these are assumptions, not estimated calibration",
    )
    manifest = out / "preprocessing.json"
    write_json(manifest, status)
    try:
        if fitted_reference:
            write_fit(out / "reference_fit", fitted_reference)
            status["reference_fit"] = str(out / "reference_fit")
        labels[["cell", "barcode"]].to_csv(out / "cells.tsv", sep="\t", index=False)
        (out / "cell_barcodes.txt").write_text("\n".join(labels.cell) + "\n")
        cellsnp = cellsnp_dir.resolve() if cellsnp_dir else out / "cellsnp"
        LOG.info(
            "[preprocess 2/5] %s per-cell allele counts",
            "Reusing" if cellsnp_dir else "Counting",
        )
        if not cellsnp_dir:
            # Restrict the catalogue too, making chromosome-specific smoke tests inexpensive.
            sites = out / "sites.vcf.gz"
            targets = ",".join(chroms + [c.removeprefix("chr") for c in chroms])
            run(
                "bcftools",
                "view",
                "-t",
                targets,
                "-m2",
                "-M2",
                "-v",
                "snps",
                "-Oz",
                "-o",
                sites,
                snp_vcf,
            )
            cellsnp.mkdir()
            run(
                "cellsnp-lite",
                "-s",
                bam,
                "-b",
                out / "cell_barcodes.txt",
                "-O",
                cellsnp,
                "-R",
                sites,
                "-p",
                threads,
                "--minMAF",
                "0",
                "--minCOUNT",
                "2",
                "--gzip",
                "--cellTAG",
                "CB",
                "--UMItag",
                "UB",
            )
            # No --genotype: only the AD/DP matrices are consumed. Donor genotypes are pooled below.
        phased = phased_vcf.resolve() if phased_vcf else out / "phased.vcf.gz"
        LOG.info(
            "[preprocess 3/5] %s donor haplotypes",
            "Reusing" if phased_vcf else "Phasing",
        )
        if not phased_vcf:
            present, n = pooled_vcf(
                cellsnp / "cellSNP.base.vcf.gz", out / "pooled.vcf", set(chroms)
            )
            status.update(
                pooled_genotypes=n,
                skipped_empty_chromosomes=[c for c in chroms if c not in present],
            )
            run(
                "bcftools",
                "view",
                "-Oz",
                "-o",
                out / "pooled.vcf.gz",
                out / "pooled.vcf",
            )
            run("bcftools", "index", "-t", out / "pooled.vcf.gz")
            products = []
            for chrom in present:
                LOG.info("Phasing %s", chrom)
                prefix = out / chrom
                run(
                    "java",
                    f"-Xmx{memory_gb}g",
                    "-jar",
                    beagle_jar,
                    f"gt={out / 'pooled.vcf.gz'}",
                    f"map={maps[chrom]}",
                    f"ref={panels[chrom]}",
                    f"out={prefix}",
                    "impute=false",
                    f"chrom={chrom}",
                    f"nthreads={threads}",
                    f"seed={seed}",
                )
                product = Path(str(prefix) + ".vcf.gz")
                seen = {chromosome(row[0]) for row in vcf_records(product)}
                if seen != {chrom}:
                    raise ValueError(
                        f"Beagle output for {chrom} is empty or contains another chromosome"
                    )
                run("bcftools", "index", "-t", product)
                products.append(product)
            run("bcftools", "concat", "-Oz", "-o", phased, *products)
            run("bcftools", "index", "-t", phased)
        LOG.info("[preprocess 4/5] Orienting per-cell counts into phased haplotypes")
        allele_file = out / "alleles.tsv.gz"
        status.update(
            export_alleles(cellsnp, phased, allele_file, labels.cell, set(chroms))
        )
        LOG.info("[preprocess 5/5] Building the reusable cell-count bundle")
        data = load_cells(
            matrix,
            out / "cells.tsv",
            genes,
            reference=reference_profile,
            alleles=allele_file,
            genetic_map=genetic_map,
        )
        temporary = out / "prepared.tmp.h5"
        write_bundle(temporary, data)
        temporary.replace(out / "prepared.h5")
        status.update(
            status="complete",
            cells=len(data.cell_ids),
            lineage_barcodes=len(set(data.barcode_labels)),
            phased_vcf=provenance(phased, digest=True),
            prepared=provenance(out / "prepared.h5", digest=True),
        )
        command = [
            "barcodecnv",
            "infer",
            str(out / "prepared.h5"),
            "--out",
            str(inference_out if inference_out is not None else out / "results"),
        ]
        status["run_command"] = command
        (out / "infer.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nexec "
            + shlex.join(command)
            + ' "$@"\n'
        )
        (out / "infer.sh").chmod(0o755)
        LOG.info("Prepared %s. Next: %s", out / "prepared.h5", shlex.join(command))
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(manifest, status)
    return out / "prepared.h5"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parser().parse_args(argv)
    try:
        preprocess(**vars(args))
    except (ValueError, OSError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
