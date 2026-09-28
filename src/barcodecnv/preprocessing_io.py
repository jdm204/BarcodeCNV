"""cellSNP-lite/Beagle adapters, adapted from BarcodeCNV's Julia preprocessing.

Genotype selection follows GermlineCalling.create_pseudobulk_vcf. Matrix rows
are joined by chromosome, position AND alleles; columns always use samples.tsv.
"""

import csv
import gzip
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

from .loading import chromosome, chromosome_key


def open_text(path, mode="rt"):
    path = Path(path)
    return gzip.open(path, mode) if path.suffix == ".gz" else path.open(mode)


def vcf_records(path):
    with open_text(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8:
                    raise ValueError(f"malformed VCF record in {path}")
                yield fields


def variant_key(fields):
    return chromosome(fields[0]), int(fields[1]), fields[3], fields[4]


def pooled_vcf(base_vcf, destination, chromosomes):
    """Julia filters: OTH=0, AF 0.1..0.9 or AF=1 with DP>=10; SNVs only."""
    records = []
    seen = set()
    for fields in vcf_records(base_vcf):
        chrom, pos, ref, alt = variant_key(fields)
        if chrom not in chromosomes or len(ref) != 1 or len(alt) != 1:
            continue
        if (chrom, pos) in seen:
            raise ValueError(f"duplicate SNP coordinate: {chrom}:{pos}")
        seen.add((chrom, pos))
        info = dict(item.split("=", 1) for item in fields[7].split(";") if "=" in item)
        ad, dp, oth = (int(info.get(k, 0)) for k in ("AD", "DP", "OTH"))
        if ad < 0 or dp < ad or oth < 0:
            raise ValueError(f"invalid pooled counts at {chrom}:{pos}")
        if dp == 0 or oth > 0:
            continue
        af = ad / dp
        gt = "0/1" if 0.1 <= af <= 0.9 else "1/1" if ad == dp and dp >= 10 else None
        if gt is not None:
            records.append((chrom, pos, ref, alt, gt, ad, dp))
    if not records:
        raise ValueError("no usable germline genotypes after pooled AF/OTH filters")
    records.sort(key=lambda r: (chromosome_key(r[0]), r[1]))
    present = sorted({r[0] for r in records}, key=chromosome_key)
    with Path(destination).open("w") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        for chrom in present:
            handle.write(f"##contig=<ID={chrom}>\n")
        handle.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="UMI depth">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tBULK\n"
        )
        for chrom, pos, ref, alt, gt, ad, dp in records:
            handle.write(
                f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT:AD:DP\t{gt}:{dp - ad},{ad}:{dp}\n"
            )
    return present, len(records)


def phased_sites(path, default_phase_probability=0.99):
    """Read single-donor phased hets. Absent confidence fields stay assumptions.

    GQ is genotype quality, not phase quality, so unlike the old Julia fallback it
    is not used as a phase probability. AP1/AP2 are not independent phase posteriors.
    """
    sites = {}
    coordinates = set()
    for fields in vcf_records(path):
        if len(fields) != 10:
            raise ValueError("phased VCF must contain exactly one donor sample")
        key = variant_key(fields)
        if len(key[2]) != 1 or len(key[3]) != 1:
            continue
        values = dict(zip(fields[8].split(":"), fields[9].split(":")))
        gt = values.get("GT")
        if gt not in ("0|1", "1|0"):
            continue
        if key[:2] in coordinates:
            raise ValueError(f"duplicate phased SNP coordinate: {key[:2]}")
        coordinates.add(key[:2])
        gp = values.get("GP", ".")
        het = float(gp.split(",")[1]) if gp != "." else 1.0
        pq = values.get("PQ", ".")
        phase = 1 - 10 ** (-float(pq) / 10) if pq != "." else default_phase_probability
        if not (
            np.isfinite(het)
            and np.isfinite(phase)
            and 0 <= het <= 1
            and 0 <= phase <= 1
        ):
            raise ValueError(f"invalid SNP probabilities: {key}")
        sites[key] = (gt, het, phase)
    if not sites:
        raise ValueError("phasing produced no biallelic heterozygous SNPs")
    return sites


def matrix_file(directory, name):
    for suffix in (".mtx", ".mtx.gz"):
        path = Path(directory) / (name + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing {name}.mtx[.gz] in {directory}")


def read_counts(path):
    matrix = csr_matrix(mmread(path, spmatrix=False))
    matrix.sum_duplicates()
    if (
        np.any(~np.isfinite(matrix.data))
        or np.any(matrix.data < 0)
        or np.any(matrix.data != np.floor(matrix.data))
    ):
        raise ValueError(f"non-integer or negative allele counts: {path}")
    return matrix.astype(np.int64)


def export_alleles(directory, phased_vcf, output, selected_cells, chromosomes):
    """Retain cell-level UMI counts, rather than collapsing lineage barcodes."""
    directory = Path(directory)
    samples = (directory / "cellSNP.samples.tsv").read_text().splitlines()
    if len(set(samples)) != len(samples):
        raise ValueError("duplicate cellSNP sample IDs")
    if set(samples) != set(selected_cells):
        raise ValueError("cellSNP sample IDs must match the selected cell map exactly")
    base = list(vcf_records(directory / "cellSNP.base.vcf.gz"))
    ad = read_counts(matrix_file(directory, "cellSNP.tag.AD"))
    dp = read_counts(matrix_file(directory, "cellSNP.tag.DP"))
    if ad.shape != dp.shape or ad.shape != (len(base), len(samples)):
        raise ValueError("cellSNP VCF, matrix and sample dimensions disagree")
    difference = dp - ad
    if np.any(difference.data < 0):
        raise ValueError("cellSNP AD exceeds DP")
    sites = phased_sites(phased_vcf)
    rows = loci = 0
    used_coordinates = set()
    with gzip.open(output, "wt") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["cell", "chromosome", "position", "h1", "h2", "het", "phase_prob"]
        )
        for i, fields in enumerate(base):
            key = variant_key(fields)
            if key not in sites or key[0] not in chromosomes:
                continue
            if key[:2] in used_coordinates:
                raise ValueError(f"duplicate count SNP coordinate: {key[:2]}")
            used_coordinates.add(key[:2])
            gt, het, phase = sites[key]
            alt = dict(zip(ad[i].indices, ad[i].data))
            observed = False
            for cell, total in zip(dp[i].indices, dp[i].data):
                if total == 0:
                    continue
                a = int(alt.get(cell, 0))
                total = int(total)
                h1 = a if gt == "1|0" else total - a
                writer.writerow(
                    [samples[cell], key[0], key[1], h1, total - h1, het, phase]
                )
                rows += 1
                observed = True
            loci += int(observed)
    if not rows:
        raise ValueError(
            "no phased alleles matched the cellSNP counts (check build and REF/ALT)"
        )
    return dict(phased_count_loci=loci, nonzero_cell_locus_pairs=rows)
