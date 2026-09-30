"""Object workflow contracts, ownership and persistence boundaries."""

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from test_cli import example_bundle
from test_cli import preprocessing_inputs as preprocessing_inputs
from test_cli import raw_inputs as raw_inputs

from barcodecnv import api
from barcodecnv.bundle import CellBundle
from barcodecnv.reference import ReferencePanel
from barcodecnv.results import PreparedResult


def objects():
    bundle = example_bundle()
    adata = ad.AnnData(
        bundle.expression.T.tocsr(copy=True),
        obs=pd.DataFrame({"total": bundle.libraries}, index=bundle.cell_ids),
        var=pd.DataFrame(index=bundle.gene_ids),
    )
    labels = pd.DataFrame({"cell": bundle.cell_ids, "barcode": bundle.barcode_labels})
    genes = pd.DataFrame(
        {
            "gene": bundle.gene_ids,
            "chromosome": bundle.grid.chrom,
            "position": bundle.grid.start,
        }
    )
    panel = ReferencePanel(
        "single",
        "hg38",
        "test",
        (*bundle.gene_ids, "outside"),
        ("normal",),
        np.r_[np.full(60, 0.002), 0.88][:, None],
        pd.DataFrame({"id": ["normal"]}),
        {},
    )
    return adata, labels, genes, panel


def test_object_chain_needs_no_intermediate_files(tmp_path, monkeypatch):
    from barcodecnv import loading, reference, reporting

    adata, labels, genes, panel = objects()
    expected = adata.X.toarray().T.copy()
    expression = api.from_anndata(adata, cells=labels, library_size_key="total")
    # Adapters own the data, including lineage labels, rather than retaining mutable input views.
    adata.X.data[:] = 0
    labels.loc[:, "barcode"] = "changed"
    np.testing.assert_array_equal(expression.counts.toarray(), expected)
    assert expression.barcode_labels[0] == "00"

    def forbidden(*args, **kwargs):
        pytest.fail("object-only workflow tried to read/write a file")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loading, "read_10x", forbidden)
    monkeypatch.setattr(reference, "read_panel", forbidden)
    monkeypatch.setattr(api, "write_bundle", forbidden)
    monkeypatch.setattr(api, "write_fit", forbidden)
    monkeypatch.setattr(reporting, "write_run", forbidden)
    monkeypatch.setattr(api, "write_signal", forbidden)
    monkeypatch.setattr(reporting, "write_cn_tables", forbidden)
    fitted = api.fit_reference(expression, genes=genes, reference_panel=panel)
    assert fitted.weights == {
        "normal": 1.0
    }  # Single-profile fit has an exact expected answer.
    assert fitted.profile["outside"] == 0.88
    prepared = api.prepare(expression, genes=genes, reference=fitted)
    bundle = prepared.bundle
    np.testing.assert_array_equal(bundle.expression.toarray(), expected)
    np.testing.assert_array_equal(bundle.libraries, np.full(160, 10000))
    np.testing.assert_allclose(bundle.reference_fractions, 0.002)
    diagnostic = api.signal(prepared, permutations=19)
    assert diagnostic.status == "assessed"
    result = api.infer(prepared, bootstraps=20, draws=16, skip_signal=True)
    assert result.prepared is prepared
    assert result.reference_fit is fitted
    assert result.bundle is bundle
    assert result.output_directory is None
    assert list(tmp_path.iterdir()) == []


def test_anndata_convenience_adapters_and_missing_lineage_labels():
    adata, labels, genes, panel = objects()
    fitted = api.fit_reference(
        adata=adata, genes=genes, reference_panel=panel, library_size_key="total"
    )
    bundle = api.prepare(
        adata=adata,
        cells=labels,
        genes=genes,
        reference=fitted,
        library_size_key="total",
    )
    assert isinstance(bundle, PreparedResult)
    expression = api.from_anndata(adata, library_size_key="total")
    with pytest.raises(ValueError, match="lineage barcode"):
        api.prepare(expression, genes=genes, reference=fitted)
    with pytest.raises(ValueError, match="exactly one"):
        api.prepare(expression, adata=adata, genes=genes, reference=fitted)
    with pytest.raises(ValueError, match="already defines"):
        api.fit_reference(expression, cells=labels, genes=genes, reference_panel=panel)


def test_preprocessing_reuses_fit_object_and_returns_bundle(
    tmp_path, preprocessing_inputs, monkeypatch
):
    from barcodecnv import reference

    adata, labels, genes, panel = objects()
    expression = api.from_anndata(adata, cells=labels, library_size_key="total")
    fitted = api.fit_reference(expression, genes=genes, reference_panel=panel)

    def no_refit(*args, **kwargs):
        pytest.fail("a supplied FittedReference must not be fitted again")

    monkeypatch.setattr(reference, "fit_from_expression", no_refit)
    prepared = api.preprocess(
        expression,
        bam=tmp_path / "reused.bam",
        reference=fitted,
        genes=genes,
        genetic_map=tmp_path / "maps",
        chromosomes="1,2,3",
        cellsnp_dir=tmp_path / "cellsnp",
        phased_vcf=tmp_path / "phased.vcf.gz",
        out=tmp_path / "prep",
    )
    bundle = prepared.bundle
    assert isinstance(bundle, CellBundle)
    assert prepared.reference_fit is fitted
    np.testing.assert_array_equal(bundle.libraries, expression.libraries)
    np.testing.assert_allclose(bundle.reference_fractions, 0.002)
    assert (tmp_path / "prep/prepared.h5").is_file()
    audit = json.loads((tmp_path / "prep/reference_fit/fit.json").read_text())
    assert audit["assay_inputs"] == fitted.audit["assay_inputs"]
    assert audit["mean_squared_log_error"] == fitted.audit["mean_squared_log_error"]


def test_file_adapter_preserves_whole_assay_and_strict_selection(tmp_path, raw_inputs):
    labels = pd.read_csv(tmp_path / "cells.tsv", sep="\t", dtype=str).iloc[[42, 2, 87]]
    expression = api.from_10x(tmp_path / "10x", cells=labels)
    assert expression.cells == tuple(labels.cell)
    assert expression.barcode_labels == tuple(labels.barcode)
    np.testing.assert_array_equal(expression.libraries, [10000, 10000, 10000])
    with pytest.raises(ValueError, match="missing from expression"):
        api.from_10x(
            tmp_path / "10x", cells=labels.assign(cell=["missing", "cell2", "cell87"])
        )


@pytest.mark.parametrize("fail", [False, True])
def test_preprocessing_temporary_workspace_is_cleaned(
    tmp_path, preprocessing_inputs, monkeypatch, fail
):
    from barcodecnv import preprocessing

    adata, labels, genes, panel = objects()
    expression = api.from_anndata(adata, cells=labels, library_size_key="total")
    fitted = api.fit_reference(expression, genes=genes, reference_panel=panel)
    workspaces = []
    original_init = preprocessing.Commands.__init__

    def record_workspace(self, tools, out):
        workspaces.append(out)
        original_init(self, tools, out)

    monkeypatch.setattr(preprocessing.Commands, "__init__", record_workspace)
    if fail:

        def failed_export(*args, **kwargs):
            assert workspaces[0].exists()
            raise RuntimeError("controlled allele export failure")

        monkeypatch.setattr(preprocessing, "export_alleles", failed_export)

    options = dict(
        bam=tmp_path / "reused.bam",
        reference=fitted,
        genes=genes,
        genetic_map=tmp_path / "maps",
        chromosomes="1,2,3",
        cellsnp_dir=tmp_path / "cellsnp",
        phased_vcf=tmp_path / "phased.vcf.gz",
    )
    if fail:
        with pytest.raises(RuntimeError, match="controlled allele export failure"):
            api.preprocess(expression, **options)
    else:
        prepared = api.preprocess(expression, **options)
        bundle = prepared.bundle
        assert prepared.output_directory is None
        assert prepared.provenance["workspace"]["retained"] is False
        # Access actual arrays after the workspace has gone: no lazy file dependency.
        np.testing.assert_array_equal(
            bundle.expression.toarray(), expression.counts.toarray()
        )
        np.testing.assert_array_equal(bundle.libraries, expression.libraries)
        np.testing.assert_array_equal(bundle.h1.toarray(), np.full((3, 160), 5))
    assert len(workspaces) == 1
    assert not workspaces[0].parent.exists()
    assert (tmp_path / "cellsnp/cellSNP.samples.tsv").is_file()
    assert (tmp_path / "phased.vcf.gz").is_file()
