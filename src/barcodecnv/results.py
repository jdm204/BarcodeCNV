"""Python inference results: inspect, plot interactively, or export without refitting."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import CellBundle, write_bundle
from .fitting import BarcodeFit
from .group_calls import GroupCalls
from .reference import FittedReference, write_fit


@dataclass(repr=False)
class PreparedResult:
    """Prepared counts plus the reference fit and execution provenance.

    All numerical data and audit records live in memory. output_directory is
    None for temporary runs. Paths inside provenance are historical execution
    records; provenance["workspace"]["retained"] says whether native working
    files were kept. Saving exports the scientific result and audit, not native
    scratch files or the contents of tool logs.
    """

    bundle: CellBundle
    reference_fit: FittedReference | None
    provenance: dict
    output_directory: Path | None = None

    def __post_init__(self):
        self.provenance = deepcopy(self.provenance)

    def __repr__(self):
        return (
            f"PreparedResult(cells={len(self.bundle.cell_ids)}, "
            f"genes={len(self.bundle.gene_ids)}, "
            f"output_directory={self.output_directory!r})"
        )

    def _write_provenance(self, path):
        path.write_text(json.dumps(self.provenance, indent=2, allow_nan=False) + "\n")

    def save(self, out):
        """Save prepared.h5, reference_fit/ and preprocessing.json without rerunning tools.

        out must be new. Original command paths are retained as provenance, not
        rewritten to suggest that tool logs or native outputs were recreated.
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        write_bundle(out / "prepared.h5", self.bundle)
        if self.reference_fit is not None:
            write_fit(out / "reference_fit", self.reference_fit)
        self._write_provenance(out / "preprocessing.json")
        self.output_directory = out
        return out


@dataclass(frozen=True)
class SignalScores:
    """Observed statistics; only joint receives a calibrated permutation p-value."""

    joint: float
    depth: float
    allele: float


@dataclass(repr=False)
class SignalResult:
    """Stored barcode association test, with inspectable fields, plot and save.

    null_scores columns are joint, depth, allele; null_table() labels them.
    pvalue is None when there is only one barcode and the test is unassessable.
    """

    status: str
    pvalue: float | None
    scores: SignalScores
    null_scores: np.ndarray
    excess_over_null_median: float | None
    bin_width_bp: int
    permutations: int
    seed: int
    barcode_sizes: list[int]
    metadata: dict = field(default_factory=dict)
    output_directory: Path | None = None

    def __post_init__(self):
        self.null_scores = np.array(self.null_scores, copy=True)
        self.null_scores.flags.writeable = False
        self.barcode_sizes = list(self.barcode_sizes)
        self.metadata = deepcopy(self.metadata)

    def __repr__(self):
        return f"SignalResult(status={self.status!r}, pvalue={self.pvalue!r}, permutations={self.permutations})"

    def to_dict(self):
        """Return a copy of diagnostic values for serialization."""
        values = asdict(self)
        values.pop("metadata")
        values.pop("output_directory")
        return values

    def null_table(self) -> pd.DataFrame:
        """Permutation statistics with named columns, in permutation order."""
        return pd.DataFrame(
            self.null_scores.copy(), columns=["joint", "depth", "allele"]
        )

    def plot(self):
        """Return an open diagnostic Figure without rerunning permutations."""
        from .plotting import plot_signal

        return plot_signal(self)

    def save(self, out) -> Path:
        """Save the diagnostic, null table and plot to a new directory."""
        from .reporting import write_signal

        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        metadata = dict(self.metadata)
        try:
            write_signal(out, self)
            metadata["status"] = "complete"
        except Exception as error:
            metadata.update(status="failed", error=str(error))
            raise
        finally:
            (out / "run.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
        self.output_directory = out
        return out


@dataclass(repr=False)
class InferenceResult:
    """Stored inference results with plotting and report-export methods.

    Access stored values through attributes such as result.fit. Plot methods create
    and return open Matplotlib Figures without writing files. No method refits
    the model or reruns permutations. Figures are not cached: edits to one do
    not affect later plots or exports.
    """

    bundle: CellBundle
    fit: BarcodeFit
    groups: np.ndarray
    expression_groups: np.ndarray
    phase_groups: np.ndarray
    order: np.ndarray
    cells: np.ndarray
    self_expression: np.ndarray
    external_expression: np.ndarray
    weighted: dict
    core: dict
    cn_audit: list
    distances: np.ndarray | None
    pre_hf_groups: np.ndarray
    hf_features: dict | None
    hf_audit: list
    hf_stability: np.ndarray | None
    hf_status: str
    group_calls: GroupCalls
    signal: SignalResult | None = None
    output_directory: Path | None = None
    metadata: dict = field(default_factory=dict)
    reference_fit: FittedReference | None = None
    prepared: PreparedResult | None = None

    def __repr__(self):
        return (
            f"InferenceResult(barcodes={len(self.bundle.barcodes)}, "
            f"genes={len(self.bundle.gene_ids)}, "
            f"groups={len(self.group_calls.groups)}, "
            f"output_directory={self.output_directory!r})"
        )

    @property
    def probabilities(self):
        """Barcode × genomic marker × CN class probabilities."""
        return self.fit.classes

    @property
    def class_names(self) -> tuple[str, ...]:
        """Labels for the last axis of probabilities, in matching order."""
        from .model import CLASS_NAMES

        return CLASS_NAMES

    def marker_table(self) -> pd.DataFrame:
        """Coordinates in exactly the order of the probabilities marker axis."""
        grid = self.bundle.grid
        return pd.DataFrame(
            dict(chromosome=grid.chrom, start=grid.start.copy(), end=grid.end.copy())
        ).rename_axis("marker")

    def groups_table(self) -> pd.DataFrame:
        """Barcode membership and stability, in probabilities barcode-axis order."""
        return pd.DataFrame(
            dict(
                barcode=self.bundle.barcodes,
                cells=self.cells,
                group=self.groups,
                expression_group=self.expression_groups,
                phase_group=self.phase_groups,
                pre_hf_group=self.pre_hf_groups,
                hf_stability=self.hf_stability,
                weighted_stability=self.weighted["stability"],
                core_stability=self.core["stability"],
            )
        )

    def barcode_cn_table(self, barcode=None) -> pd.DataFrame:
        """Gene-level CN probabilities; optionally select one lineage barcode ID."""
        from .cn_reporting import barcode_cn_frames, collect_frames

        return collect_frames(
            *barcode_cn_frames(
                self.bundle, self.probabilities, self.groups, barcode=barcode
            )
        )

    def group_cn_table(self, group=None) -> pd.DataFrame:
        """Gene-level consensus, pooled CN and conflict metrics for a group or all groups."""
        from .cn_reporting import collect_frames, group_cn_frames

        return collect_frames(
            *group_cn_frames(self.bundle, self.group_calls, group=group)
        )

    def group_segments_table(self, group=None) -> pd.DataFrame:
        """Runs of marginal pooled MAP calls; probabilities summarize markers, not segments."""
        from .cn_reporting import collect_frames, group_segment_frames

        return collect_frames(
            *group_segment_frames(self.bundle, self.group_calls, group=group)
        )

    def plot_summary(self):
        """Return the expression/HF/CN summary Figure for display or customization."""
        from .plotting import plot_run

        return plot_run(self, signal=self.signal)

    def plot_groups(self):
        """Return the group consensus, pooled CN and disagreement Figure."""
        from .plotting import plot_group_calls

        return plot_group_calls(self.bundle, self.group_calls)

    def plot_signal(self):
        """Return the stored permutation diagnostic Figure; raise if it was skipped."""
        if self.signal is None:
            raise ValueError(
                "signal diagnostic was skipped; no stored signal results to plot"
            )
        return self.signal.plot()

    def _write_report(self, out):
        """Shared export implementation, also used by the CLI's active run directory."""
        from matplotlib import pyplot as plt

        from .reporting import write_run, write_signal

        metadata = write_run(out, self)
        if self.prepared is not None:
            self.prepared._write_provenance(out / "preprocessing.json")
            metadata["preprocessing"] = str((out / "preprocessing.json").resolve())
        with plt.ioff():
            if self.signal is not None:
                write_signal(out, self.signal)
            for make_figure, name in (
                (self.plot_summary, "summary.png"),
                (self.plot_groups, "group_cn.png"),
            ):
                fig = make_figure()
                try:
                    fig.savefig(
                        out / name, dpi=220, bbox_inches="tight", facecolor="white"
                    )
                finally:
                    plt.close(fig)
        return metadata

    def save(self, out):
        """Export the complete report to a new directory and return its Path.

        Includes a prepared-input snapshot, fitted-reference audit when present,
        tables, HDF5 results, PNG plots and run.json. Uses only stored results;
        original input files need not remain available. Existing directories are
        refused. Export errors retain partial files and a failed run.json.
        Interactive figures already open in the caller are left alone.
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        metadata = dict(self.metadata)
        try:
            prepared = out / "prepared.h5"
            write_bundle(prepared, self.bundle)
            with prepared.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            metadata.update(
                input=str(prepared.resolve()), input_kind="memory", input_sha256=digest
            )
            if self.reference_fit is not None:
                write_fit(out / "reference_fit", self.reference_fit)
                metadata["reference_fit"] = str((out / "reference_fit").resolve())
            metadata.update(self._write_report(out), status="complete")
            (out / "run.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
        except Exception as error:
            metadata.update(status="failed", error=str(error))
            (out / "run.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
            raise
        self.metadata = metadata
        self.output_directory = out
        return out
