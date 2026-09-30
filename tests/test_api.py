"""Exercise public Python workflows without CLI argument objects."""

import hashlib
import json

import h5py
import numpy as np
import pytest
from test_cli import preprocessing_inputs as preprocessing_inputs
from test_cli import raw_inputs as raw_inputs
from test_reference import write_panel

from barcodecnv import api
from barcodecnv.bundle import read_bundle
from barcodecnv.cli import main
from barcodecnv.results import PreparedResult


def count_inputs(root):
    return dict(
        matrix=str(root / "10x"),
        cells=root / "cells.tsv",
        genes=root / "genes.tsv",
        reference=root / "reference.tsv",
    )


def test_prepare_signal_infer_match_cli(tmp_path, raw_inputs):
    directory = tmp_path / "prepared"
    prepared = directory / "prepared.h5"
    bundle = api.prepare(**count_inputs(tmp_path), out=directory)
    assert isinstance(bundle, PreparedResult)
    np.testing.assert_array_equal(
        read_bundle(prepared).libraries, bundle.bundle.libraries
    )
    diagnostic = api.signal(bundle, out=tmp_path / "signal", permutations=19)
    result = api.infer(
        bundle,
        out=tmp_path / "python",
        bootstraps=20,
        draws=32,
        permutations=19,
        depth_outlier_probability=0.01,
        cn_refinement=False,
        hf_refinement=False,
    )
    assert (
        main(
            [
                "infer",
                str(prepared),
                "--out",
                str(tmp_path / "cli"),
                "--bootstraps",
                "20",
                "--draws",
                "32",
                "--permutations",
                "19",
                "--depth-outlier-probability",
                "0.01",
                "--no-cn-refinement",
                "--no-hf-refinement",
            ]
        )
        == 0
    )
    with h5py.File(tmp_path / "cli/result.h5") as saved:
        np.testing.assert_array_equal(saved["probabilities"][:], result.fit.classes)
        np.testing.assert_array_equal(
            saved["group_calls/pooled"][:], result.group_calls.pooled
        )
    assert result.output_directory == tmp_path / "python"
    assert diagnostic.pvalue == result.signal.pvalue
    assert json.loads((tmp_path / "python/signal.json").read_text()) == json.loads(
        (tmp_path / "cli/signal.json").read_text()
    )
    metadata = json.loads((tmp_path / "python/run.json").read_text())
    assert metadata["input_kind"] == "memory"
    assert (
        metadata["input_sha256"]
        == hashlib.sha256((tmp_path / "python/prepared.h5").read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        api.prepare(**count_inputs(tmp_path), out=directory)


def test_run_matches_python_stages_without_parser(
    tmp_path, preprocessing_inputs, monkeypatch
):
    import barcodecnv.cli as cli

    def no_parser():
        pytest.fail("Python workflows must not parse CLI arguments")

    monkeypatch.setattr(cli, "parser", no_parser)
    inputs = dict(
        outs=str(tmp_path / "outs"),
        cells=tmp_path / "cells.tsv",
        reference=tmp_path / "reference.tsv",
        genes=tmp_path / "genes.tsv",
        genetic_map=tmp_path / "maps",
        chromosomes=("1", "2", "3"),
        cellsnp_dir=tmp_path / "cellsnp",
        phased_vcf=tmp_path / "phased.vcf.gz",
    )
    settings = dict(bootstraps=20, draws=32, skip_signal=True)
    combined = api.run_pipeline(**inputs, out=tmp_path / "combined", **settings)
    prepared = api.preprocess(**inputs, out=tmp_path / "preprocessed")
    staged = api.infer(prepared, out=tmp_path / "staged", **settings)
    np.testing.assert_array_equal(combined.fit.classes, staged.fit.classes)
    np.testing.assert_array_equal(combined.groups, staged.groups)
    assert (
        json.loads((tmp_path / "combined/pipeline.json").read_text())["status"]
        == "complete"
    )


def test_fit_reference_returns_fit_and_optional_report(tmp_path, raw_inputs):
    panel = tmp_path / "panel"
    fractions = np.r_[np.full(60, 0.002), 0.88]
    write_panel(
        panel, [*[f"g{i}" for i in range(60)], "outside"], np.c_[fractions, fractions]
    )
    inputs = count_inputs(tmp_path)
    inputs.pop("reference")
    fitted = api.fit_reference(**inputs, reference_panel=panel)
    saved = api.fit_reference(**inputs, reference_panel=panel, out=tmp_path / "fit")
    assert fitted.weights == saved.weights
    assert fitted.profile == saved.profile
    assert (tmp_path / "fit/reference.tsv").is_file()
    assert (
        main(
            [
                "fit-reference",
                "--matrix",
                str(inputs["matrix"]),
                "--cells",
                str(inputs["cells"]),
                "--genes",
                str(inputs["genes"]),
                "--reference-panel",
                str(panel),
                "--out",
                str(tmp_path / "cli-fit"),
            ]
        )
        == 0
    )
    assert (tmp_path / "fit/reference.tsv").read_bytes() == (
        tmp_path / "cli-fit/reference.tsv"
    ).read_bytes()


def test_api_failure_raises_and_preserves_snapshot(tmp_path, raw_inputs, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(api, "run_pbpc", fail)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="controlled failure"):
        api.infer(**count_inputs(tmp_path), out=out, skip_signal=True)
    assert (out / "prepared.h5").is_file()
    assert json.loads((out / "run.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="not both"):
        api.infer(out / "prepared.h5", matrix="unused", out=tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()


def test_setup_plan_without_writes(tmp_path, capsys):
    from barcodecnv.resources import setup

    config = setup(resource_dir=str(tmp_path), chromosomes=("chr22",), dry_run=True)
    plan = json.loads(capsys.readouterr().out)
    assert list(plan["panels"]) == ["22"]
    assert config == tmp_path / "hg38/config.json"
    assert not config.parent.exists()
    with pytest.raises(ValueError, match="unique"):
        setup(resource_dir=tmp_path, chromosomes=("1", "chr1"), dry_run=True)
