"""Versioned cell-count boundary for both CLI commands; no BAM/reference fitting."""

from dataclasses import dataclass

import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from .data import AlleleCounts, GeneCounts, Grid, Loci, PreparedCounts, owned


def sparse_counts(value, shape):
    a = csc_matrix(value, copy=True)
    if (
        a.shape != shape
        or np.any(~np.isfinite(a.data))
        or np.any(a.data < 0)
        or np.any(a.data != np.floor(a.data))
    ):
        raise ValueError("sparse count axes disagree or counts are invalid")
    a = a.astype(np.int64)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    for array in (a.data, a.indices, a.indptr):
        array.flags.writeable = False
    return a


@dataclass(frozen=True)
class CellBundle:
    grid: Grid
    loci: Loci
    gene_ids: tuple[str, ...]
    gene_markers: np.ndarray
    reference_fractions: np.ndarray
    cell_ids: tuple[str, ...]
    barcode_labels: tuple[str, ...]
    libraries: np.ndarray
    expression: csc_matrix
    h1: csc_matrix
    h2: csc_matrix
    het: np.ndarray

    def __post_init__(self):
        for name in ("gene_ids", "cell_ids", "barcode_labels"):
            labels = tuple(getattr(self, name))
            if any(not isinstance(x, str) or not x for x in labels):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, labels)
        c = len(self.cell_ids)
        g = len(self.gene_ids)
        n_loci = len(self.loci)
        if (
            not c
            or not g
            or len(set(self.cell_ids)) != c
            or len(set(self.gene_ids)) != g
        ):
            raise ValueError("cell and gene identifiers must be nonempty and unique")
        for name, dtype, size in (
            ("gene_markers", np.int64, g),
            ("reference_fractions", float, g),
            ("libraries", float, c),
            ("het", float, n_loci),
        ):
            a = owned(getattr(self, name), dtype)
            if len(a) != size or np.any(~np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, a)
        if (
            len(self.barcode_labels) != c
            or np.any(self.gene_markers >= len(self.grid))
            or np.any(np.diff(self.gene_markers) < 0)
        ):
            raise ValueError("bundle label or gene-position axes disagree")
        if np.any(self.het > 1) or np.any(self.libraries <= 0):
            raise ValueError("invalid heterozygosity or library exposure")
        if self.reference_fractions.sum() > 1 + 1e-8:
            raise ValueError("reference fractions exceed the whole assay")
        for name, shape in (
            ("expression", (g, c)),
            ("h1", (n_loci, c)),
            ("h2", (n_loci, c)),
        ):
            object.__setattr__(self, name, sparse_counts(getattr(self, name), shape))
        if np.any(np.asarray(self.expression.sum(axis=0)).ravel() > self.libraries):
            raise ValueError(
                "whole-assay library sizes must cover selected gene counts"
            )
        # Reuse the inference boundary's genomic/SNP validation without pooling.
        PreparedCounts(
            self.grid,
            ("validation",),
            GeneCounts([], [], [], [], []),
            AlleleCounts([], [], [], [], []),
            self.loci,
        )

    @property
    def barcodes(self):
        return tuple(sorted(set(self.barcode_labels)))

    @property
    def membership(self):
        index = {b: i for i, b in enumerate(self.barcodes)}
        return np.array([index[b] for b in self.barcode_labels])

    def prepared(self, labels=None, *, exclude_unresolved=False):
        """Pseudobulk cells directly, preserving individual gene/SNP likelihood terms."""
        if labels is None:
            ids = self.barcodes
            membership = self.membership
        else:
            labels = np.asarray(labels)
            if (
                labels.shape != (len(self.barcodes),)
                or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < (0 if exclude_unresolved else 1))
            ):
                raise ValueError(
                    "groups must be integer labels per barcode; zero requires exclude_unresolved"
                )
            unique = sorted(set(labels) - {0})
            lookup = {x: i for i, x in enumerate(unique)}
            if not unique:
                raise ValueError("no resolved groups to pseudobulk")
            ids = tuple(f"G{x}" for x in unique)
            membership = np.array([lookup.get(labels[i], -1) for i in self.membership])
        selected = np.flatnonzero(membership >= 0)
        weights = csr_matrix(
            (np.ones(len(selected), dtype=np.int64), (selected, membership[selected])),
            shape=(len(membership), len(ids)),
        )
        counts = (self.expression @ weights).toarray().T
        exposure = np.asarray(self.libraries @ weights).ravel()
        g = len(self.gene_ids)
        genes = GeneCounts(
            np.repeat(np.arange(len(ids)), g),
            np.tile(self.gene_markers, len(ids)),
            counts.ravel(),
            (exposure[:, None] * self.reference_fractions).ravel(),
            np.ones(counts.size),
        )
        a = (self.h1 @ weights).toarray().T
        b = (self.h2 @ weights).toarray().T
        groups, loci = np.nonzero(a + b)
        alleles = AlleleCounts(
            groups, loci, a[groups, loci], b[groups, loci], self.het[loci]
        )
        return PreparedCounts(self.grid, ids, genes, alleles, self.loci)


def write_sparse(f, name, a):
    group = f.create_group(name)
    for key, value in (
        ("data", a.data),
        ("indices", a.indices),
        ("indptr", a.indptr),
        ("shape", a.shape),
    ):
        group.create_dataset(key, data=value)


def read_sparse(f):
    return csc_matrix(
        (f["data"][:], f["indices"][:], f["indptr"][:]), shape=tuple(f["shape"][:])
    )


def write_bundle(path, data):
    """Write a fresh bundle. IDs remain strings, including numeric-looking IDs."""
    with h5py.File(path, "x") as f:
        f.attrs["pbpc_schema"] = "cell-counts-v2"

        def strings(key, x):
            f.create_dataset(key, data=np.array(x, dtype=h5py.string_dtype()))

        strings("grid/chrom", data.grid.chrom)
        for key, value in (
            ("grid/start", data.grid.start),
            ("grid/end", data.grid.end),
            ("genes/marker", data.gene_markers),
            ("genes/reference_fraction", data.reference_fractions),
            ("cells/library", data.libraries),
            ("snps/het", data.het),
        ):
            f.create_dataset(key, data=value)
        for name in ("marker", "position", "population_phase", "genetic_cm"):
            f.create_dataset("snps/" + name, data=getattr(data.loci, name))
        for key, value in (
            ("genes/id", data.gene_ids),
            ("cells/id", data.cell_ids),
            ("cells/barcode", data.barcode_labels),
        ):
            strings(key, value)
        for name in ("expression", "h1", "h2"):
            write_sparse(f, "counts/" + name, getattr(data, name))


def read_bundle(path):
    with h5py.File(path, "r") as f:
        if f.attrs.get("pbpc_schema") not in ("cell-counts-v1", "cell-counts-v2"):
            raise ValueError("expected PBPC cell-counts-v1 or cell-counts-v2 bundle")

        def strings(key):
            return tuple(f[key].asstr()[:])

        return CellBundle(
            Grid(strings("grid/chrom"), f["grid/start"][:], f["grid/end"][:]),
            Loci(
                *(
                    f["snps/" + k][:]
                    for k in ("marker", "position", "population_phase", "genetic_cm")
                )
            ),
            strings("genes/id"),
            f["genes/marker"][:],
            f["genes/reference_fraction"][:],
            strings("cells/id"),
            strings("cells/barcode"),
            f["cells/library"][:],
            *(read_sparse(f["counts/" + k]) for k in ("expression", "h1", "h2")),
            f["snps/het"][:],
        )
