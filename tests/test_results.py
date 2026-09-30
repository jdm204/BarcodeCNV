"""Interactive figures and exports must derive from an existing fitted result."""

import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from test_cli import example_bundle

from barcodecnv import api
from barcodecnv.results import InferenceResult


@pytest.fixture(scope="module")
def fitted():
    return api.infer(example_bundle(), bootstraps=20, draws=16, permutations=19)


def test_plots_and_export_reuse_results_and_leave_caller_figures_open(
    tmp_path, monkeypatch, fitted
):
    from barcodecnv import signal, workflow

    def no_recomputation(*args, **kwargs):
        pytest.fail("plotting/export must not run inference or permutations")

    monkeypatch.setattr(api, "run_pbpc", no_recomputation)
    monkeypatch.setattr(workflow, "run_pbpc", no_recomputation)
    monkeypatch.setattr(api, "barcode_signal_test", no_recomputation)
    monkeypatch.setattr(signal, "barcode_signal_test", no_recomputation)
    monkeypatch.chdir(tmp_path)
    assert isinstance(fitted, InferenceResult)
    figures = [fitted.plot_summary(), fitted.plot_groups(), fitted.plot_signal()]
    try:
        assert all(
            isinstance(fig, Figure) and plt.fignum_exists(fig.number) for fig in figures
        )
        assert list(tmp_path.iterdir()) == []
        # Inspect actual plotted summaries rather than merely testing figure creation.
        summary_expression = np.concatenate(
            [
                mesh.get_array().filled(np.nan)
                for mesh in figures[0].axes[0].collections
            ],
            axis=1,
        )
        np.testing.assert_array_equal(
            summary_expression, fitted.self_expression[:, fitted.order].T
        )
        observed_line = figures[2].axes[0].lines[0]
        np.testing.assert_array_equal(
            observed_line.get_xdata(), [fitted.signal.scores.joint] * 2
        )
        figures[0].axes[0].set_title("Caller customization")
        before_export = set(plt.get_fignums())
        out = fitted.save(tmp_path / "saved")
        assert set(plt.get_fignums()) == before_export
        assert figures[0].axes[0].get_title() == "Caller customization"
        assert fitted.output_directory == out
        for name in (
            "prepared.h5",
            "result.h5",
            "groups.csv",
            "summary.png",
            "group_cn.png",
            "signal.png",
            "signal.json",
        ):
            assert (out / name).is_file()
        with h5py.File(out / "result.h5") as saved:
            np.testing.assert_array_equal(
                saved["probabilities"][:], fitted.probabilities
            )
        metadata = json.loads((out / "run.json").read_text())
        assert metadata["status"] == "complete"
        assert metadata["arguments"]["bootstraps"] == 20
        sentinel = (out / "run.json").read_bytes()
        with pytest.raises(FileExistsError):
            fitted.save(out)
        assert (out / "run.json").read_bytes() == sentinel
    finally:
        for fig in figures:
            plt.close(fig)


def test_failed_export_closes_only_its_figures_and_preserves_result(
    tmp_path, monkeypatch, fitted
):
    original_directory = fitted.output_directory
    before = set(plt.get_fignums())
    original_savefig = Figure.savefig

    def fail_summary(self, path, *args, **kwargs):
        if str(path).endswith("summary.png"):
            raise OSError("controlled plot export failure")
        return original_savefig(self, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail_summary)
    with pytest.raises(OSError, match="controlled plot export failure"):
        fitted.save(tmp_path / "failed")
    assert set(plt.get_fignums()) == before
    assert fitted.output_directory == original_directory
    assert json.loads((tmp_path / "failed/run.json").read_text())["status"] == "failed"
    assert fitted.probabilities.shape[0] == len(fitted.bundle.barcodes)


def test_skipped_signal_has_no_plot():
    result = api.infer(example_bundle(), bootstraps=20, draws=16, skip_signal=True)
    with pytest.raises(ValueError, match="skipped"):
        result.plot_signal()


def test_import_preserves_configured_matplotlib_backend():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; import barcodecnv.plotting; print(matplotlib.get_backend())",
        ],
        env={**os.environ, "MPLBACKEND": "svg"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip().lower() == "svg"
