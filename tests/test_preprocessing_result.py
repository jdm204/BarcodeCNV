"""Preprocessing audit must survive temporary cleanup and later inference export."""

import json
from pathlib import Path

import pandas as pd
import pytest
from test_cli import preprocessing_inputs as preprocessing_inputs
from test_cli import raw_inputs as raw_inputs
from test_expression_api import objects

from barcodecnv import api
from barcodecnv.results import PreparedResult


@pytest.mark.parametrize("persistent", [False, True])
def test_context_survives_preprocessing_inference_and_delayed_export(
    tmp_path, preprocessing_inputs, monkeypatch, persistent
):
    from barcodecnv import preprocessing, reference

    adata, labels, genes, panel = objects()
    expression = api.from_anndata(adata, cells=labels, library_size_key="total")
    prepared = api.preprocess(
        expression,
        bam=tmp_path / "cached.bam",
        genes=genes,
        reference_panel=panel,
        genetic_map=tmp_path / "maps",
        chromosomes="1,2,3",
        cellsnp_dir=tmp_path / "cellsnp",
        phased_vcf=tmp_path / "phased.vcf.gz",
        out=tmp_path / "work" if persistent else None,
    )
    assert isinstance(prepared, PreparedResult)
    original = prepared.provenance
    workspace = Path(original["workspace"]["path"])
    assert workspace.exists() == persistent
    assert original["workspace"]["retained"] == persistent
    assert prepared.output_directory == (workspace if persistent else None)
    assert prepared.reference_fit.weights == {"normal": 1.0}
    assert original["inputs"]["matrix"]["selected_cells"] == 160
    assert original["arguments"]["seed"] == 42
    assert original["status"] == "complete"

    def forbidden(*args, **kwargs):
        pytest.fail(
            "retaining or exporting provenance must not rerun preprocessing/reference fitting"
        )

    monkeypatch.setattr(preprocessing, "preprocess", forbidden)
    monkeypatch.setattr(reference, "fit_from_expression", forbidden)
    result = api.infer(
        prepared,
        bootstraps=20,
        draws=16,
        skip_signal=True,
        out=tmp_path / "immediate" if persistent else None,
    )
    assert result.prepared is prepared
    assert result.bundle is prepared.bundle
    assert result.reference_fit is prepared.reference_fit
    if persistent:
        assert (
            json.loads((tmp_path / "immediate/preprocessing.json").read_text())
            == original
        )
        assert (tmp_path / "immediate/reference_fit/weights.tsv").is_file()

    monkeypatch.setattr(api, "run_pbpc", forbidden)
    exported = result.save(tmp_path / "exported")
    assert json.loads((exported / "preprocessing.json").read_text()) == original
    audit = json.loads((exported / "reference_fit/fit.json").read_text())
    assert audit["assay_inputs"] == prepared.reference_fit.audit["assay_inputs"]
    weights = pd.read_csv(exported / "reference_fit/weights.tsv", sep="\t")
    assert weights["weight"].tolist() == [1.0]
    manifest = json.loads((exported / "run.json").read_text())
    assert manifest["preprocessing"] == str(exported / "preprocessing.json")
    assert manifest["reference_fit"] == str(exported / "reference_fit")
    saved = prepared.save(tmp_path / "prepared-export")
    assert (saved / "prepared.h5").is_file()
    assert json.loads((saved / "preprocessing.json").read_text()) == original
    assert prepared.provenance["workspace"]["retained"] == persistent
    assert not (
        saved / "infer.sh"
    ).exists()  # No fake claim that native processing was rerun.
    with pytest.raises(FileExistsError):
        prepared.save(saved)


@pytest.mark.parametrize("fail_inference", [False, True])
def test_run_without_out_cleans_workspace_before_inference(
    tmp_path, preprocessing_inputs, monkeypatch, fail_inference
):
    adata, labels, genes, panel = objects()
    expression = api.from_anndata(adata, cells=labels, library_size_key="total")
    original_infer = api.infer
    observed = []
    before = set(tmp_path.rglob("*"))
    monkeypatch.chdir(tmp_path)

    def inspect_infer(prepared, **kwargs):
        observed.append(prepared)
        assert kwargs["out"] is None
        assert prepared.output_directory is None
        assert not Path(prepared.provenance["workspace"]["path"]).exists()
        assert prepared.provenance["workspace"]["retained"] is False
        if fail_inference:
            raise RuntimeError("controlled inference failure after cleanup")
        return original_infer(prepared, **kwargs)

    monkeypatch.setattr(api, "infer", inspect_infer)
    options = dict(
        bam=tmp_path / "cached.bam",
        genes=genes,
        reference_panel=panel,
        genetic_map=tmp_path / "maps",
        chromosomes="1,2,3",
        cellsnp_dir=tmp_path / "cellsnp",
        phased_vcf=tmp_path / "phased.vcf.gz",
        bootstraps=20,
        draws=16,
        skip_signal=True,
    )
    if fail_inference:
        with pytest.raises(RuntimeError, match="controlled inference failure"):
            api.run_pipeline(expression, **options)
    else:
        result = api.run_pipeline(expression, **options)
        assert result.output_directory is None
        assert result.prepared is observed[0]
        assert result.reference_fit.weights == {"normal": 1.0}
        assert result.probabilities.shape[0] == 8
    assert len(observed) == 1
    assert set(tmp_path.rglob("*")) == before
