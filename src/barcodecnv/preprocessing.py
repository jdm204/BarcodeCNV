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

from .bundle import write_bundle
from .loading import chromosome, chromosome_key, load_cells, table
from .preprocessing_io import export_alleles, phased_sites, pooled_vcf, vcf_records
from .reference import add_reference_arguments, fit_for_args, write_fit

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


def preprocess(args, *, inference_out=None):
    LOG.info("[preprocess 1/5] Validating inputs and resolving tools/resources")
    args.config = discover_config(args.config)
    config = json.loads(args.config.read_text()) if args.config else {}
    if config.get("genome", args.genome) != args.genome:
        raise ValueError("configured resource build differs from --genome")
    resources = config.get("resources", {})
    for key in ("genes", "snp_vcf", "genetic_map", "phasing_panel", "beagle_jar"):
        value = getattr(args, key) or resources.get(key)
        setattr(args, key, Path(value).resolve() if value else None)
    if args.reference is None and args.reference_panel is None:
        args.reference_panel = Path(config.get("default_reference_panel", "b-cells-v1"))
        LOG.info("Using default B-cell expression panel: %s", args.reference_panel)
    if args.chromosomes is None:
        args.chromosomes = ",".join(
            config.get("chromosomes", [*map(str, range(1, 23)), "X"])
        )
    chroms = [chromosome(x.strip()) for x in args.chromosomes.split(",")]
    supported = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
    if len(set(chroms)) != len(chroms) or not set(chroms) <= supported:
        raise ValueError("chromosomes must be unique and drawn from 1..22,X")
    chroms.sort(key=chromosome_key)
    LOG.info("Allele chromosomes: %s", ",".join(chroms))
    required = ["genes", "genetic_map"]
    if not args.cellsnp_dir:
        required.append("snp_vcf")
    if not args.phased_vcf:
        required += ["phasing_panel", "beagle_jar"]
    for key in required:
        if getattr(args, key) is None or not getattr(args, key).exists():
            raise ValueError(
                f"provide --{key.replace('_', '-')} or run barcodecnv setup --genome hg38"
            )
    # Validate every requested chromosome before expensive pileup, even if no sites survive.
    maps = {c: resolve_resource(args.genetic_map, c, "map") for c in chroms}
    panels = (
        {}
        if args.phased_vcf
        else {c: resolve_resource(args.phasing_panel, c, "panel") for c in chroms}
    )
    matrix = args.outs / "filtered_feature_bc_matrix.h5"
    if not matrix.is_file():
        matrix = args.outs / "filtered_feature_bc_matrix"
    bam = args.outs / "possorted_genome_bam.bam"
    if not args.cellsnp_dir:
        if not bam.is_file() or not any(
            p.is_file()
            for p in (
                Path(str(bam) + ".bai"),
                bam.with_suffix(".bai"),
                Path(str(bam) + ".csi"),
            )
        ):
            raise ValueError(
                "Cell Ranger outs must contain an indexed possorted_genome_bam.bam"
            )
    if args.phased_vcf:
        phased_sites(args.phased_vcf)
    labels = table(args.cells).rename(
        columns={"cell_barcode": "cell", "lineage_barcode": "barcode"}
    )
    # Use the production loader for input validation, including cells in 10x and library sizes.
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    fitted_reference = fit_for_args(args, matrix)
    reference_profile = fitted_reference.profile if fitted_reference else args.reference
    initial = load_cells(
        matrix,
        args.cells,
        args.genes,
        reference=reference_profile,
    )
    if not len(initial.cell_ids):
        raise ValueError("empty cell map")
    del initial
    if labels[["cell", "barcode"]].eq("").any().any():
        raise ValueError("cell and barcode labels must be nonempty")
    tools = {}
    needed = (["cellsnp-lite", "bcftools"] if not args.cellsnp_dir else []) + (
        [] if args.phased_vcf else ["bcftools", "java"]
    )
    for name in set(needed):
        executable = config.get("tools", {}).get(name) or shutil.which(name)
        if not executable or not os.access(executable, os.X_OK):
            raise ValueError(
                f"{name} unavailable; configure preprocessing tools or add to PATH"
            )
        tools[name] = str(Path(executable).absolute())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = Commands(tools, out)
    status = dict(
        status="running",
        genome=args.genome,
        chromosomes=chroms,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        tools={k: provenance(v, digest=True) for k, v in tools.items()},
        inputs={
            k: provenance(v, digest=k in ("cells", "reference", "beagle_jar"))
            for k, v in dict(
                cells=args.cells,
                genes=args.genes,
                matrix=matrix,
                **(
                    {"reference": args.reference}
                    if args.reference
                    else {"reference_panel": args.reference_panel}
                ),
                **({"bam": bam} if not args.cellsnp_dir else {}),
                **({"beagle_jar": args.beagle_jar} if not args.phased_vcf else {}),
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
        cellsnp = args.cellsnp_dir.resolve() if args.cellsnp_dir else out / "cellsnp"
        LOG.info(
            "[preprocess 2/5] %s per-cell allele counts",
            "Reusing" if args.cellsnp_dir else "Counting",
        )
        if not args.cellsnp_dir:
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
                args.snp_vcf,
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
                args.threads,
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
        phased = args.phased_vcf.resolve() if args.phased_vcf else out / "phased.vcf.gz"
        LOG.info(
            "[preprocess 3/5] %s donor haplotypes",
            "Reusing" if args.phased_vcf else "Phasing",
        )
        if not args.phased_vcf:
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
                    f"-Xmx{args.memory_gb}g",
                    "-jar",
                    args.beagle_jar,
                    f"gt={out / 'pooled.vcf.gz'}",
                    f"map={maps[chrom]}",
                    f"ref={panels[chrom]}",
                    f"out={prefix}",
                    "impute=false",
                    f"chrom={chrom}",
                    f"nthreads={args.threads}",
                    f"seed={args.seed}",
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
            args.genes,
            reference=reference_profile,
            alleles=allele_file,
            genetic_map=args.genetic_map,
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
        preprocess(args)
    except (ValueError, OSError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
