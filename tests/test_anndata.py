"""Adapter checks with explicit count/exposure expectations and real AnnData views."""

import json
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, csr_matrix
from test_cli import example_bundle
from test_cli import preprocessing_inputs as preprocessing_inputs
from test_cli import raw_inputs as raw_inputs
from test_reference import write_panel

from barcodecnv import api
from barcodecnv.api import from_anndata
from barcodecnv.bundle import read_bundle


@pytest.fixture
def annotated():
    return ad.AnnData(
        csr_matrix([[2, 3, 7], [11, 13, 17], [19, 23, 29]]),
        obs=pd.DataFrame({"total_counts": [12, 41, 71]}, index=["c0", "c1", "c2"]),
        var=pd.DataFrame({"gene_ids": ["g0", "g1", "g2"]}, index=["A", "B", "C"]),
    )


def test_filtered_view_aligns_cells_genes_and_whole_library_totals(annotated):
    annotated.layers["counts"] = annotated.X.copy()
    annotated.X = np.log1p(annotated.X.toarray())
    # Gene filtering and cell reordering must not redefine library sizes or align by row number.
    view = annotated[[2, 0], [1, 0]]
    labels = pd.DataFrame(
        {"cell": ["c0", "filtered-out", "c2"], "barcode": ["01", "02", "03"]}
    )
    expression = from_anndata(
        view,
        cells=labels,
        layer="counts",
        gene_id_key="gene_ids",
        library_size_key="total_counts",
    )
    assert expression.cells == ("c2", "c0")
    assert expression.genes == ("g1", "g0")
    np.testing.assert_array_equal(expression.counts.toarray(), [[23, 3], [19, 2]])
    np.testing.assert_array_equal(expression.libraries, [71, 12])
    assert expression.barcode_labels == ("03", "01")
    assert expression.metadata["excluded_table_cells"] == 1
    with pytest.raises(ValueError, match="read-only"):
        expression.counts.data[:] = 0
    np.testing.assert_array_equal(view.layers["counts"].toarray(), [[23, 19], [3, 2]])
    assert view.is_view


def test_raw_preserves_its_gene_axis_after_filtering(annotated):
    annotated.raw = annotated.copy()
    annotated.X = np.log1p(annotated.X.toarray())
    view = annotated[[2, 0], [1]]
    labels = pd.DataFrame({"cell": ["c0", "c2"], "barcode": ["b", "a"]})
    counts = from_anndata(view, cells=labels, use_raw=True, gene_id_key="gene_ids")
    assert counts.genes == ("g0", "g1", "g2")
    np.testing.assert_array_equal(counts.counts.toarray(), [[19, 2], [23, 3], [29, 7]])
    np.testing.assert_array_equal(counts.libraries, [71, 12])
    with pytest.raises(ValueError, match="raw nonnegative integer"):
        from_anndata(view, cells=labels)
    with pytest.raises(ValueError, match="either layer or use_raw"):
        from_anndata(view, cells=labels, layer="counts", use_raw=True)


def test_rejects_invalid_exposures_and_ambiguous_ids(annotated):
    labels = pd.DataFrame({"cell": ["c0"], "barcode": ["a"]})
    annotated.obs["bad_totals"] = [11, 41, 71]
    with pytest.raises(ValueError, match="whole-assay totals"):
        from_anndata(annotated, cells=labels, library_size_key="bad_totals")
    with pytest.raises(ValueError, match="unique cells"):
        from_anndata(annotated, cells=pd.concat([labels, labels]))
    with pytest.raises(ValueError, match="nonempty"):
        from_anndata(annotated, cells=labels.assign(barcode=None))
    with pytest.raises(ValueError, match="no shared cells"):
        from_anndata(annotated, cells=labels.assign(cell="absent"))
    annotated.var["gene_ids"] = ["ENSG0001.1", "ENSG0001.2", "g2"]
    with pytest.raises(ValueError, match="ambiguous"):
        from_anndata(annotated, cells=labels, gene_id_key="gene_ids")


def fixture_anndata():
    data = example_bundle()
    return ad.AnnData(
        data.expression.T.tocsr(),
        obs=pd.DataFrame({"total_counts": data.libraries}, index=data.cell_ids),
        var=pd.DataFrame(index=data.gene_ids),
    )


def subset_cellsnp(source, target, names):
    existing = (source / "cellSNP.samples.tsv").read_text().splitlines()
    indices = pd.Index(existing).get_indexer(names)
    target.mkdir(exist_ok=True)
    (target / "cellSNP.samples.tsv").write_text("\n".join(names) + "\n")
    shutil.copyfile(source / "cellSNP.base.vcf.gz", target / "cellSNP.base.vcf.gz")
    for name in ("cellSNP.tag.AD.mtx", "cellSNP.tag.DP.mtx"):
        mmwrite(
            target / name, csc_matrix(mmread(source / name, spmatrix=False))[:, indices]
        )


def test_preprocess_uses_selected_cells_and_supplied_bam(
    tmp_path, preprocessing_inputs, monkeypatch
):
    from barcodecnv import preprocessing

    adata = fixture_anndata()[[42, 2, 87], [4, 1, 50]]
    original = adata.X.toarray().copy()
    bam = tmp_path / "custom.bam"
    bam.touch()
    bam.with_suffix(".bam.bai").touch()
    sites = tmp_path / "sites.vcf.gz"
    sites.touch()
    commands = []
    selected_names = ["cell42", "cell2", "cell87"]

    def fake_native(self, tool, *args):
        commands.append((tool, args))
        if tool == "cellsnp-lite":
            assert args[args.index("-s") + 1] == bam
            cell_file = args[args.index("-b") + 1]
            assert cell_file.read_text().splitlines() == selected_names
            target = args[args.index("-O") + 1]
            subset_cellsnp(tmp_path / "cellsnp", target, selected_names)

    monkeypatch.setattr(preprocessing.Commands, "__call__", fake_native)
    monkeypatch.setattr(preprocessing.shutil, "which", lambda name: "/bin/true")
    labels = pd.read_csv(tmp_path / "cells.tsv", sep="\t", dtype=str).iloc[::-1]
    prepared = api.preprocess(
        adata=adata,
        bam=bam,
        cells=labels,
        out=tmp_path / "prepared",
        library_size_key="total_counts",
        reference=tmp_path / "reference.tsv",
        genes=tmp_path / "genes.tsv",
        genetic_map=tmp_path / "maps",
        chromosomes="1,2,3",
        phased_vcf=tmp_path / "phased.vcf.gz",
        snp_vcf=sites,
    )
    bundle = prepared.bundle
    saved = read_bundle(tmp_path / "prepared/prepared.h5")
    np.testing.assert_array_equal(
        saved.expression.toarray(), bundle.expression.toarray()
    )
    assert bundle.cell_ids == tuple(selected_names)
    assert bundle.barcode_labels == ("02", "00", "04")
    assert bundle.gene_ids == ("g1", "g4", "g50")
    np.testing.assert_array_equal(bundle.expression.toarray(), original[:, [1, 0, 2]].T)
    np.testing.assert_array_equal(bundle.libraries, [10000, 10000, 10000])
    np.testing.assert_array_equal(bundle.h1.toarray(), np.full((3, 3), 5))
    np.testing.assert_array_equal(adata.X.toarray(), original)
    metadata = json.loads((tmp_path / "prepared/preprocessing.json").read_text())
    assert metadata["inputs"]["matrix"]["selected_cells"] == 3
    assert metadata["inputs"]["matrix"]["excluded_table_cells"] == 157
    assert metadata["inputs"]["bam"]["path"] == str(bam)
    assert [tool for tool, _ in commands] == ["bcftools", "cellsnp-lite"]


def test_anndata_run_with_fitted_reference(tmp_path, preprocessing_inputs):
    adata = fixture_anndata()
    # Simulate user filtering observations; keep all genes, and fit the reference from these cells.
    adata = adata[:80]
    subset_cellsnp(
        tmp_path / "cellsnp", tmp_path / "selected-cellsnp", list(adata.obs_names)
    )
    values = np.r_[np.full(60, 0.002), 0.88]
    panel = tmp_path / "panel"
    write_panel(
        panel, [*[f"g{i}" for i in range(60)], "outside"], np.c_[values, values]
    )
    result = api.run_pipeline(
        adata=adata,
        bam=tmp_path / "cached.bam",
        cells=tmp_path / "cells.tsv",
        library_size_key="total_counts",
        reference_panel=panel,
        genes=tmp_path / "genes.tsv",
        genetic_map=tmp_path / "maps",
        chromosomes="1,2,3",
        cellsnp_dir=tmp_path / "selected-cellsnp",
        phased_vcf=tmp_path / "phased.vcf.gz",
        out=tmp_path / "run",
        bootstraps=20,
        draws=16,
        skip_signal=True,
    )
    assert result.bundle.cell_ids == tuple(adata.obs_names)
    assert result.fit.classes.shape[0] == 4
    audit = json.loads(
        (tmp_path / "run/preprocessing/reference_fit/fit.json").read_text()
    )
    assert audit["assay_inputs"]["selected_cells"] == 80
    assert audit["assay_inputs"]["matrix"]["kind"] == "anndata"
    assert (
        json.loads((tmp_path / "run/pipeline.json").read_text())["status"] == "complete"
    )
