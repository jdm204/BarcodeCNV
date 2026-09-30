"""Owned expression inputs shared by Python workflows and file adapters."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from .bundle import sparse_counts
from .data import owned


@dataclass(frozen=True)
class ExpressionInput:
    """Gene-by-cell raw counts, whole-library totals and optional lineage labels.

    Arrays are copied and made read-only. Gene/cell IDs must be unique. Lineage
    labels are optional for reference fitting, required for bundle preparation.
    """

    counts: csc_matrix
    genes: tuple[str, ...]
    cells: tuple[str, ...]
    libraries: np.ndarray
    barcode_labels: tuple[str, ...] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        from .loading import canonical_genes

        for name in ("genes", "cells"):
            values = tuple(getattr(self, name))
            if (
                not values
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{name} must be unique nonempty string IDs")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "genes", tuple(canonical_genes(self.genes)))
        counts = sparse_counts(self.counts, (len(self.genes), len(self.cells)))
        libraries = owned(self.libraries, float)
        if (
            libraries.shape != (len(self.cells),)
            or not np.isfinite(libraries).all()
            or np.any(libraries <= 0)
            or np.any(libraries != np.floor(libraries))
            or np.any(libraries < np.asarray(counts.sum(axis=0)).ravel())
        ):
            raise ValueError(
                "library sizes must be positive integer whole-assay totals covering retained counts"
            )
        if self.barcode_labels is not None:
            labels = tuple(self.barcode_labels)
            if len(labels) != len(self.cells) or any(
                not isinstance(v, str) or not v for v in labels
            ):
                raise ValueError(
                    "barcode labels must be nonempty and match the cell axis"
                )
            object.__setattr__(self, "barcode_labels", labels)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "libraries", libraries)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def cell_table(self, *, require_barcodes=False):
        if require_barcodes and self.barcode_labels is None:
            raise ValueError(
                "lineage barcode labels are required for preparation/inference"
            )
        result = pd.DataFrame({"cell": self.cells})
        if self.barcode_labels is not None:
            result["barcode"] = self.barcode_labels
        return result

    def provenance(self):
        digest = hashlib.sha256()
        digest.update(
            json.dumps([self.genes, self.cells, self.barcode_labels]).encode()
        )
        for values, dtype in (
            (self.counts.data, "<f8"),
            (self.counts.indices, "<i8"),
            (self.counts.indptr, "<i8"),
            (self.libraries, "<f8"),
        ):
            digest.update(np.asarray(values, dtype=dtype).tobytes())
        return dict(self.metadata, sha256=digest.hexdigest())


def from_10x(matrix, *, cells=None):
    """Read counts once; select a cell table strictly, or retain all cells.

    cells accepts a DataFrame or CSV/TSV. Library totals include all assayed genes.
    Requested cells missing from the matrix raise, matching the CLI contract.
    """
    from .loading import read_10x, table

    counts, genes, ids = read_10x(matrix)
    labels = (
        pd.DataFrame({"cell": ids})
        if cells is None
        else table(cells).rename(
            columns={"cell_barcode": "cell", "lineage_barcode": "barcode"}
        )
    )
    if (
        "cell" not in labels
        or labels.empty
        or labels.cell.isna().any()
        or labels.cell.eq("").any()
        or labels.cell.duplicated().any()
    ):
        raise ValueError("expression input requires nonempty unique selected cells")
    selected = pd.Index(ids).get_indexer(labels.cell)
    if np.any(selected < 0):
        raise ValueError("annotated cells are missing from expression matrix")
    counts = counts[:, selected]
    return ExpressionInput(
        counts,
        tuple(genes),
        tuple(labels.cell),
        np.asarray(counts.sum(axis=0)).ravel(),
        tuple(labels.barcode) if "barcode" in labels else None,
        dict(
            kind="10x", path=str(Path(matrix).resolve()), selected_cells=len(selected)
        ),
    )


def from_anndata(
    adata,
    *,
    cells=None,
    layer=None,
    use_raw=False,
    gene_id_key=None,
    library_size_key=None,
):
    """Copy selected AnnData counts without changing the source object.

    Cells are the intersection with the optional annotation table in AnnData
    order. Without a table all observations are retained, without lineage labels.
    Preserve whole-assay totals in library_size_key before gene filtering.
    """
    from .anndata_input import from_anndata as adapt

    return adapt(
        adata,
        cells,
        layer=layer,
        use_raw=use_raw,
        gene_id_key=gene_id_key,
        library_size_key=library_size_key,
    )


def resolve_expression(
    input=None,
    *,
    matrix=None,
    cells=None,
    adata=None,
    layer=None,
    use_raw=False,
    gene_id_key=None,
    library_size_key=None,
):
    """One source-selection boundary for all workflows."""
    if sum(v is not None for v in (input, matrix, adata)) != 1:
        raise ValueError("provide exactly one ExpressionInput, matrix or adata")
    options = dict(
        layer=layer,
        use_raw=use_raw,
        gene_id_key=gene_id_key,
        library_size_key=library_size_key,
    )
    if input is not None:
        if not isinstance(input, ExpressionInput):
            raise TypeError(
                "input must be an ExpressionInput; use from_anndata or from_10x"
            )
        if (
            cells is not None
            or layer is not None
            or use_raw
            or gene_id_key is not None
            or library_size_key is not None
        ):
            raise ValueError(
                "ExpressionInput already defines cell selection and count source"
            )
        return input
    if adata is not None:
        return from_anndata(adata, cells=cells, **options)
    if (
        layer is not None
        or use_raw
        or gene_id_key is not None
        or library_size_key is not None
    ):
        raise ValueError("AnnData source options require adata")
    return from_10x(matrix, cells=cells)
