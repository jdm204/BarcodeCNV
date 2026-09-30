"""Adapt AnnData counts to the shared loaders without importing Scanpy."""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, issparse

from .expression import ExpressionInput


def from_anndata(
    adata, cells, *, layer=None, use_raw=False, gene_id_key=None, library_size_key=None
):
    """Select mapped observations and copy integer counts, retaining full exposures."""
    from .loading import canonical_genes, table

    if layer is not None and use_raw:
        raise ValueError("choose either layer or use_raw, not both")
    if getattr(adata, "obs_names", None) is None:
        raise TypeError("adata must be an AnnData object")
    ids = pd.Index(adata.obs_names)
    if (
        ids.has_duplicates
        or ids.hasnans
        or any(not isinstance(x, str) or not x for x in ids)
    ):
        raise ValueError(
            "AnnData cell IDs must be unique, nonempty strings matching BAM CB tags"
        )
    labels = (pd.DataFrame({"cell": ids}) if cells is None else table(cells)).rename(
        columns={"cell_barcode": "cell", "lineage_barcode": "barcode"}
    )
    if "cell" not in labels:
        raise ValueError("cell table needs a cell column")
    labels = labels[["cell", "barcode"] if "barcode" in labels else ["cell"]]
    if (
        labels.isna().any().any()
        or labels.eq("").any().any()
        or labels.cell.duplicated().any()
    ):
        raise ValueError(
            "cell table must have unique cells and nonempty cell/barcode labels"
        )
    table_cells = len(labels)
    selected = np.flatnonzero(ids.isin(labels.cell))
    if not len(selected):
        raise ValueError("no shared cells between AnnData and barcode table")
    selected_ids = ids[selected]
    labels = (
        labels.set_index("cell").loc[selected_ids].rename_axis("cell").reset_index()
    )
    source = adata.raw if use_raw else adata
    if source is None:
        raise ValueError("use_raw requires adata.raw containing unnormalized counts")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"AnnData has no layer {layer!r}")
    values = source.X if layer is None else adata.layers[layer]
    if values is None:
        raise ValueError("the selected AnnData count matrix is empty")
    values = values[selected, :]
    if not isinstance(values, np.ndarray) and not issparse(values):
        raise TypeError(
            "AnnData counts must be NumPy or SciPy sparse arrays; load lazy data into memory first"
        )
    counts = csc_matrix(values.T, dtype=np.float64, copy=True)
    if (
        np.any(~np.isfinite(counts.data))
        or np.any(counts.data < 0)
        or np.any(counts.data != np.floor(counts.data))
    ):
        raise ValueError(
            "AnnData expression must contain raw nonnegative integer counts, not normalized/log counts"
        )
    counts.sum_duplicates()
    counts.sort_indices()
    if gene_id_key is not None and gene_id_key not in source.var:
        raise ValueError(f"selected AnnData var has no gene ID column {gene_id_key!r}")
    gene_ids = source.var_names if gene_id_key is None else source.var[gene_id_key]
    if pd.isna(gene_ids).any() or any(
        not isinstance(x, str) or not x for x in gene_ids
    ):
        raise ValueError("AnnData gene IDs must be nonempty strings")
    genes = tuple(canonical_genes(gene_ids))
    if counts.shape != (len(genes), len(selected)) or not len(genes):
        raise ValueError("AnnData count axes disagree or contain no genes")
    totals = np.asarray(counts.sum(axis=0)).ravel()
    if library_size_key is None:
        libraries = totals.copy()
    else:
        if library_size_key not in adata.obs:
            raise ValueError(
                f"AnnData obs has no library-size column {library_size_key!r}"
            )
        libraries = np.asarray(
            adata.obs[library_size_key].iloc[selected], dtype=float
        ).copy()
    if (
        not np.isfinite(libraries).all()
        or np.any(libraries <= 0)
        or np.any(libraries != np.floor(libraries))
        or np.any(libraries < totals)
    ):
        raise ValueError(
            "library sizes must be positive integer whole-assay totals, at least the sum of retained counts"
        )
    metadata = dict(
        kind="anndata",
        source="raw.X"
        if use_raw
        else f"layers[{layer!r}]"
        if layer is not None
        else "X",
        gene_id_key=gene_id_key,
        library_size_key=library_size_key,
        observations=len(ids),
        selected_cells=len(selected),
        unmapped_observations=len(ids) - len(selected),
        excluded_table_cells=table_cells - len(selected),
        genes=len(genes),
    )
    return ExpressionInput(
        counts,
        genes,
        tuple(selected_ids),
        libraries,
        tuple(labels.barcode) if "barcode" in labels else None,
        metadata,
    )
