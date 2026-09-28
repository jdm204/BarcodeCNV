"""Flat CN exports; no fitting, grouping, or filesystem policy decisions."""

import gzip

import numpy as np
import pandas as pd

from .model import CLASS_NAMES


def _probability_columns(p, prefix=""):
    return {
        prefix + "call": np.asarray(CLASS_NAMES)[p.argmax(-1)],
        **{prefix + "p_" + name: p[:, i] for i, name in enumerate(CLASS_NAMES)},
    }


def _write_frames(path, frames, empty_columns):
    with gzip.open(path, "wt") as handle:
        first = True
        for frame in frames:
            frame.to_csv(handle, index=False, header=first)
            first = False
        if first:
            pd.DataFrame(columns=empty_columns).to_csv(handle, index=False)


def write_cn_tables(out, bundle, probabilities, labels, calls):
    markers = bundle.gene_markers
    grid = bundle.grid
    genes = dict(
        gene=bundle.gene_ids,
        chromosome=np.asarray(grid.chrom)[markers],
        start=grid.start[markers],
        end=grid.end[markers],
    )
    base = list(genes)
    state_columns = list(_probability_columns(np.zeros((0, 4))))

    def barcode_frames():
        for i, barcode in enumerate(bundle.barcodes):
            yield pd.DataFrame(
                dict(
                    barcode=barcode,
                    group=int(labels[i]),
                    **genes,
                    **_probability_columns(probabilities[i, markers]),
                )
            )

    _write_frames(
        out / "barcode_cn_genes.csv.gz",
        barcode_frames(),
        ["barcode", "group"] + base + state_columns,
    )
    metrics = (
        "uncertain_cell_fraction",
        "consensus_conflict_cell_fraction",
        "pooled_conflict_cell_fraction",
    )
    group_columns = ["group", "cells", "barcodes"] + base
    for prefix in ("consensus_", "pooled_"):
        group_columns += [prefix + c for c in state_columns]
    group_columns += list(metrics)

    def group_frames():
        for i, group in enumerate(calls.groups):
            yield pd.DataFrame(
                dict(
                    group=group,
                    cells=calls.cells[i],
                    barcodes=calls.barcodes[i],
                    **genes,
                    **_probability_columns(calls.consensus[i, markers], "consensus_"),
                    **_probability_columns(calls.pooled[i, markers], "pooled_"),
                    **{name: getattr(calls, name)[i, markers] for name in metrics},
                )
            )

    _write_frames(out / "group_cn_genes.csv.gz", group_frames(), group_columns)
    segment_columns = [
        "group",
        "chromosome",
        "start",
        "end",
        "markers",
        "call",
        "mean_state_probability",
        "min_state_probability",
        "mean_uncertain_cell_fraction",
        "mean_pooled_conflict_cell_fraction",
    ]

    def segment_frames():
        chrom = np.asarray(grid.chrom)
        for i, group in enumerate(calls.groups):
            states = calls.pooled[i].argmax(-1)
            breaks = np.r_[
                0,
                np.flatnonzero((states[1:] != states[:-1]) | (chrom[1:] != chrom[:-1]))
                + 1,
                len(grid),
            ]
            rows = []
            for start, stop in zip(breaks[:-1], breaks[1:]):
                p = calls.pooled[i, start:stop, states[start]]
                rows.append(
                    dict(
                        group=group,
                        chromosome=chrom[start],
                        start=grid.start[start],
                        end=grid.end[stop - 1],
                        markers=stop - start,
                        call=CLASS_NAMES[states[start]],
                        mean_state_probability=p.mean(),
                        min_state_probability=p.min(),
                        mean_uncertain_cell_fraction=calls.uncertain_cell_fraction[
                            i, start:stop
                        ].mean(),
                        mean_pooled_conflict_cell_fraction=calls.pooled_conflict_cell_fraction[
                            i, start:stop
                        ].mean(),
                    )
                )
            yield pd.DataFrame(rows, columns=segment_columns)

    _write_frames(out / "group_cn_segments.csv.gz", segment_frames(), segment_columns)
