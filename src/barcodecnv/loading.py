"""10x matrices and explicit annotation tables to the validated cell bundle."""

import gzip
import re
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix

from .bundle import CellBundle
from .data import Grid, Loci


def table(path):
    name = str(path).lower()
    sep = (
        "\t"
        if name.endswith((".tsv", ".tsv.gz"))
        else ","
        if name.endswith((".csv", ".csv.gz"))
        else None
    )
    return pd.read_csv(path, sep=sep, engine="python", dtype=str, keep_default_na=False)


def chromosome(value):
    value = str(value).removeprefix("chr")
    return "chr" + ("MT" if value in ("M", "MT") else value)


def chromosome_key(value):
    bare = value.removeprefix("chr")
    return (
        int(bare) if bare.isdigit() else {"X": 23, "Y": 24, "MT": 25}.get(bare, 26),
        bare,
    )


def canonical_genes(values):
    values = [
        re.sub(r"^(ENS[A-Z]*G[0-9]+)\.[0-9]+(_PAR_Y)?$", r"\1\2", str(v))
        for v in values
    ]
    if len(set(values)) != len(values):
        raise ValueError("gene IDs ambiguous after removing version suffixes")
    return values


def read_10x(path):
    path = Path(path)
    if path.is_dir():

        def find(name):
            for candidate in (path / name, path / (name + ".gz")):
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"missing {name} in {path}")

        features = pd.read_csv(
            find("features.tsv"),
            sep="\t",
            header=None,
            dtype=str,
            keep_default_na=False,
        )
        ids = pd.read_csv(find("barcodes.tsv"), sep="\t", header=None, dtype=str)[
            0
        ].tolist()
        source = find("matrix.mtx")
        with (
            gzip.open(source, "rb") if source.suffix == ".gz" else source.open("rb")
        ) as f:
            counts = csc_matrix(mmread(f, spmatrix=False))
        keep = (
            features[2].eq("Gene Expression").to_numpy()
            if features.shape[1] >= 3
            else np.ones(len(features), bool)
        )
        genes = features[0][keep].tolist()
        counts = counts[keep]
    else:
        with h5py.File(path) as f:
            m = f["matrix"]
            counts = csc_matrix(
                (m["data"][:], m["indices"][:], m["indptr"][:]),
                shape=tuple(m["shape"][:]),
            )
            ids = m["barcodes"].asstr()[:].tolist()
            features = m["features"]
            keep = (
                features["feature_type"].asstr()[:] == "Gene Expression"
                if "feature_type" in features
                else np.ones(counts.shape[0], bool)
            )
            genes = features["id"].asstr()[:][keep].tolist()
            counts = counts[keep]
    if len(set(ids)) != len(ids) or counts.shape != (len(genes), len(ids)):
        raise ValueError("10x axes disagree or cell IDs repeat")
    if (
        np.any(~np.isfinite(counts.data))
        or np.any(counts.data < 0)
        or np.any(counts.data != np.floor(counts.data))
    ):
        raise ValueError("10x counts must be nonnegative integers")
    return counts, canonical_genes(genes), ids


def read_gene_coordinates(path):
    path = Path(path)
    if path.name.endswith((".gtf", ".gtf.gz")):
        import re

        rows = []
        with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) != 9 or fields[2] != "gene":
                    continue
                match = re.search(r'gene_id "([^"]+)"', fields[8])
                if match:
                    rows.append((match[1], fields[0], int(fields[3]), int(fields[4])))
        result = pd.DataFrame(rows, columns=["gene", "chromosome", "start", "end"])
    else:
        result = table(path).rename(columns={"gene_id": "gene", "chrom": "chromosome"})
    if not {"gene", "chromosome"}.issubset(result):
        raise ValueError("genes need gene and chromosome columns")
    result["gene"] = canonical_genes(result.gene)
    if "position" not in result:
        result["position"] = (
            pd.to_numeric(result.start) + pd.to_numeric(result.end)
        ) // 2
    result["position"] = pd.to_numeric(result.position).astype(np.int64)
    if (result.position < 1).any():
        raise ValueError("gene positions must be positive")
    result["chromosome"] = result.chromosome.map(chromosome)
    return result


def read_map(path):
    path = Path(path)
    files = sorted(path.rglob("*.map")) if path.is_dir() else [path]
    maps = {}
    for file in files:
        t = pd.read_csv(file, sep=r"\s+", header=None, dtype={0: str})
        for chrom, rows in t.groupby(0):
            key = chromosome(chrom)
            rows = rows.sort_values(3)
            positions = rows[3].to_numpy()
            cms = rows[2].to_numpy(dtype=float)
            if (
                np.any(np.diff(positions) <= 0)
                or np.any(np.diff(cms) < 0)
                or np.any(~np.isfinite(cms))
            ):
                raise ValueError("invalid genetic map")
            if key in maps and (
                not np.array_equal(maps[key][0], positions)
                or not np.array_equal(maps[key][1], cms)
            ):
                raise ValueError("conflicting chromosome maps")
            maps[key] = (positions, cms)
    if not maps:
        raise ValueError("empty genetic map")
    return maps


def load_cells(
    matrix,
    cells,
    genes,
    *,
    reference=None,
    alleles=None,
    genetic_map=None,
):
    counts, gene_ids, cell_ids = read_10x(matrix)
    labels = table(cells).rename(
        columns={"cell_barcode": "cell", "lineage_barcode": "barcode"}
    )
    if not {"cell", "barcode"}.issubset(labels):
        raise ValueError("cell table needs cell and barcode columns")
    if labels.cell.duplicated().any():
        raise ValueError("duplicate cell annotations")
    ix = pd.Index(cell_ids).get_indexer(labels.cell)
    if np.any(ix < 0):
        raise ValueError("annotated cells are missing from expression matrix")
    counts = counts[:, ix]
    libraries = np.asarray(counts.sum(axis=0)).ravel()
    gene_table = read_gene_coordinates(genes)
    fractions = {}
    if reference is not None:
        r = (
            pd.DataFrame(
                {"gene": list(reference), "fraction": list(reference.values())}
            )
            if isinstance(reference, Mapping)
            else table(reference)
        )
        r["gene"] = canonical_genes(r.gene)
        key = "fraction" if "fraction" in r else "expression"
        values = pd.to_numeric(r[key]).to_numpy(float)
        if np.any(~np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("invalid reference profile")
        if key == "expression":
            values = values / values.sum()
        if values.sum() > 1 + 1e-8:
            raise ValueError("reference fractions sum above one")
        fractions = dict(zip(r.gene, values))
        gene_table = gene_table[
            gene_table.gene.map(lambda g: fractions.get(g, 0) > 0)
        ].copy()
    gene_table = gene_table[gene_table.gene.isin(gene_ids)].copy()
    if gene_table.empty:
        raise ValueError("no shared annotated genes with usable reference")
    order = sorted(
        range(len(gene_table)),
        key=lambda i: (
            chromosome_key(gene_table.iloc[i].chromosome),
            gene_table.iloc[i].position,
            gene_table.iloc[i].gene,
        ),
    )
    gene_table = gene_table.iloc[order]
    counts = counts[pd.Index(gene_ids).get_indexer(gene_table.gene)]
    gene_keys = list(zip(gene_table.chromosome, gene_table.position))
    snp_keys = []
    snp_meta = []
    at = None
    if alleles is not None:
        at = table(alleles).rename(
            columns={
                "CHROM": "chromosome",
                "chrom": "chromosome",
                "POS": "position",
                "cM": "genetic_cm",
            }
        )
        if not {"cell", "chromosome", "position"}.issubset(at):
            raise ValueError("alleles need cell/chromosome/position columns")
        at = at[at.cell.isin(labels.cell)].copy()
        at["chromosome"] = at.chromosome.map(chromosome)
        at["position"] = pd.to_numeric(at.position).astype(np.int64)
        if not {"h1", "h2"}.issubset(at):
            if (
                not {"AD", "DP", "GT"}.issubset(at)
                or not at.GT.isin(["0|1", "1|0"]).all()
            ):
                raise ValueError("alleles need h1/h2 counts or phased Numbat AD/DP/GT")
            ad = pd.to_numeric(at.AD)
            dp = pd.to_numeric(at.DP)
            at["h1"] = np.where(at.GT == "1|0", ad, dp - ad)
            at["h2"] = dp - at.h1
        for column, default in (("het", 1.0), ("phase_prob", 0.99)):
            if column not in at:
                at[column] = default
            at[column] = pd.to_numeric(at[column])
        maps = read_map(genetic_map) if genetic_map else None
        metadata = at.groupby(["chromosome", "position"], sort=False)
        for key, rows in metadata:
            if rows.het.nunique() != 1 or rows.phase_prob.nunique() != 1:
                raise ValueError("inconsistent SNP metadata across cells")
            if maps is not None:
                if key[0] not in maps:
                    raise ValueError(f"missing map for {key[0]}")
                cm = float(np.interp(key[1], *maps[key[0]]))
            else:
                if "genetic_cm" not in rows or rows.genetic_cm.nunique() != 1:
                    raise ValueError(
                        "supply a genetic map or consistent per-SNP genetic_cm values"
                    )
                cm = float(rows.genetic_cm.iloc[0])
            snp_keys.append(key)
            snp_meta.append(
                (float(rows.het.iloc[0]), float(rows.phase_prob.iloc[0]), cm)
            )
    keys = sorted(set(gene_keys + snp_keys), key=lambda k: (chromosome_key(k[0]), k[1]))
    index = {k: i for i, k in enumerate(keys)}
    grid = Grid(tuple(k[0] for k in keys), [k[1] for k in keys], [k[1] for k in keys])
    so = sorted(range(len(snp_keys)), key=lambda i: index[snp_keys[i]])
    snp_keys = [snp_keys[i] for i in so]
    snp_meta = [snp_meta[i] for i in so]
    loci = Loci(
        [index[k] for k in snp_keys],
        [k[1] for k in snp_keys],
        [x[1] for x in snp_meta],
        [x[2] for x in snp_meta],
    )
    shape = (len(loci), len(labels))
    a = b = csc_matrix(shape, dtype=np.int64)
    if at is not None and len(at):
        lookup = {k: i for i, k in enumerate(snp_keys)}
        rows = [lookup[k] for k in zip(at.chromosome, at.position)]
        cols = pd.Index(labels.cell).get_indexer(at.cell)
        a = csc_matrix((pd.to_numeric(at.h1), (rows, cols)), shape=shape)
        b = csc_matrix((pd.to_numeric(at.h2), (rows, cols)), shape=shape)
    return CellBundle(
        grid,
        loci,
        tuple(gene_table.gene),
        np.array([index[k] for k in gene_keys]),
        np.array([fractions.get(g, 0.0) for g in gene_table.gene]),
        tuple(labels.cell),
        tuple(labels.barcode),
        libraries,
        counts,
        a,
        b,
        np.array([x[0] for x in snp_meta]),
    )
