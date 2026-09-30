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


def collect_frames(frames, columns):
    """Materialize fresh tables for Python; disk exports stream the same frames."""
    frames = list(frames)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=columns)
    )


def _indices(values, selected, name):
    if selected is None:
        return range(len(values))
    matches = np.flatnonzero(np.asarray(values) == selected)
    if not len(matches):
        raise ValueError(f"unknown {name}: {selected!r}")
    return matches


def _genes(bundle):
    markers = bundle.gene_markers
    return dict(
        gene=bundle.gene_ids,
        chromosome=np.asarray(bundle.grid.chrom)[markers],
        start=bundle.grid.start[markers],
        end=bundle.grid.end[markers],
    )


def barcode_cn_frames(bundle, probabilities, labels, *, barcode=None):
    markers = bundle.gene_markers
    genes = _genes(bundle)
    base = list(genes)
    state_columns = list(_probability_columns(np.zeros((0, 4))))
    indices = _indices(bundle.barcodes, barcode, "barcode")

    def barcode_frames():
        for i in indices:
            barcode = bundle.barcodes[i]
            yield pd.DataFrame(
                dict(
                    barcode=barcode,
                    group=int(labels[i]),
                    **genes,
                    **_probability_columns(probabilities[i, markers]),
                )
            )

    return barcode_frames(), ["barcode", "group"] + base + state_columns


def group_cn_frames(bundle, calls, *, group=None):
    markers = bundle.gene_markers
    genes = _genes(bundle)
    base = list(genes)
    state_columns = list(_probability_columns(np.zeros((0, 4))))
    indices = _indices(calls.groups, group, "group")
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
        for i in indices:
            group = calls.groups[i]
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

    return group_frames(), group_columns


def group_segment_frames(bundle, calls, *, group=None):
    grid = bundle.grid
    indices = _indices(calls.groups, group, "group")
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
        for i in indices:
            group = calls.groups[i]
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

    return segment_frames(), segment_columns


def write_cn_tables(out, bundle, probabilities, labels, calls):
    """Stream the same labelled tables exposed by InferenceResult to disk."""
    for name, (frames, columns) in (
        ("barcode_cn_genes", barcode_cn_frames(bundle, probabilities, labels)),
        ("group_cn_genes", group_cn_frames(bundle, calls)),
        ("group_cn_segments", group_segment_frames(bundle, calls)),
    ):
        _write_frames(out / f"{name}.csv.gz", frames, columns)
