"""User-facing handoffs: resource handles, retained fits and labelled tables."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from test_cli import example_bundle
from test_cli import preprocessing_inputs as preprocessing_inputs
from test_cli import raw_inputs as raw_inputs
from test_expression_api import objects
from test_reference import write_panel

from barcodecnv import api
from barcodecnv.results import SignalResult


def test_setup_handles_feed_staged_workflow(
    tmp_path, preprocessing_inputs, monkeypatch
):
    from barcodecnv import resources as provisioning

    config = tmp_path / "config.json"
    panel_path = tmp_path / "panel"
    adata, labels, _, panel = objects()
    write_panel(panel_path, list(panel.genes), np.repeat(panel.expression, 2, axis=1))
    config.write_text(
        json.dumps(
            dict(
                genome="hg38",
                chromosomes=["1", "2", "3"],
                resources={
                    "genes": str(tmp_path / "genes.tsv"),
                    "genetic_map": str(tmp_path / "maps"),
                },
                expression_panels={"normal": str(panel_path)},
                default_reference_panel="normal",
            )
        )
    )
    # Exercise the setup return boundary with an already provisioned fixture cache.
    # This does not test installation or network downloads.
    monkeypatch.setattr(provisioning, "setup", lambda **kwargs: config)
    resources = api.setup(resource_dir=tmp_path)
    assert resources.config_path == config
    assert resources.genes == tmp_path / "genes.tsv"
    assert api.load_resources(config) == resources
    assert len(resources.load_genes()) == 60
    np.testing.assert_array_equal(
        resources.load_reference_panel().expression,
        np.repeat(panel.expression, 2, axis=1),
    )
    expression = api.from_anndata(adata, cells=labels, library_size_key="total")
    fitted = api.fit_reference(
        expression, genes=resources.genes, reference_panel=resources.reference_panel
    )
    assert list(fitted.weights.values()) == [0.5, 0.5]
    prepared = api.preprocess(
        expression,
        bam=tmp_path / "cached.bam",
        config=resources,
        reference=fitted,
        cellsnp_dir=tmp_path / "cellsnp",
        phased_vcf=tmp_path / "phased.vcf.gz",
    )
    assert prepared.reference_fit is fitted
    assert prepared.provenance["workspace"]["retained"] is False
    fitted.save(tmp_path / "fit")
    saved = pd.read_csv(tmp_path / "fit/weights.tsv", sep="\t")
    assert saved.weight.tolist() == [0.5, 0.5]


def test_count_preparation_retains_new_fit_and_can_export_after_inference(
    tmp_path, monkeypatch
):
    from barcodecnv import reference

    adata, labels, genes, panel = objects()
    expression = api.from_anndata(adata, cells=labels, library_size_key="total")
    prepared = api.prepare(expression, genes=genes, reference_panel=panel)
    assert prepared.reference_fit.weights == {"normal": 1.0}
    assert prepared.provenance["command"] == "prepare"
    assert prepared.output_directory is None

    def forbidden(*args, **kwargs):
        pytest.fail("consuming/exporting prepared results must not refit the reference")

    monkeypatch.setattr(reference, "fit_from_expression", forbidden)
    result = api.infer(prepared, bootstraps=20, draws=16, skip_signal=True)
    assert result.prepared is prepared
    assert result.reference_fit is prepared.reference_fit
    prepared.save(tmp_path / "prepared")
    result.save(tmp_path / "inferred")
    for folder in ("prepared", "inferred"):
        weights = pd.read_csv(tmp_path / folder / "reference_fit/weights.tsv", sep="\t")
        assert weights.weight.tolist() == [1.0]
        assert (
            json.loads((tmp_path / folder / "preprocessing.json").read_text())
            == prepared.provenance
        )


@pytest.mark.parametrize("one_barcode", [False, True])
def test_signal_has_inspectable_results_and_exports_without_recomputing(
    tmp_path, monkeypatch, one_barcode
):
    bundle = example_bundle()
    if one_barcode:
        bundle = replace(bundle, barcode_labels=("one",) * len(bundle.cell_ids))
    result = api.signal(bundle, permutations=19)
    assert isinstance(result, SignalResult)
    assert (result.pvalue is None) == one_barcode
    assert result.scores.joint == result.scores.depth + result.scores.allele
    assert result.null_table().columns.tolist() == ["joint", "depth", "allele"]
    assert len(result.null_table()) == (0 if one_barcode else 19)

    def forbidden(*args, **kwargs):
        pytest.fail("plot/save must consume stored permutations")

    monkeypatch.setattr(api, "barcode_signal_test", forbidden)
    figure = result.plot()
    try:
        result.save(tmp_path / "signal")
        assert plt.fignum_exists(figure.number)
        saved = json.loads((tmp_path / "signal/signal.json").read_text())
        assert saved["pvalue"] == result.pvalue
        assert saved["scores"]["joint"] == result.scores.joint
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "signal/signal_null.csv"),
            result.null_table(),
            check_dtype=False,
        )
    finally:
        plt.close(figure)


def test_labelled_tables_align_with_probability_axes_and_are_owned(tmp_path):
    result = api.infer(example_bundle(), bootstraps=20, draws=16, skip_signal=True)
    members = result.groups_table()
    assert members.barcode.tolist() == list(result.bundle.barcodes)
    assert result.class_names == ("diploid", "gain", "loss", "cnloh")
    markers = result.marker_table()
    np.testing.assert_array_equal(markers.start, result.bundle.grid.start)
    barcode = members.barcode.iloc[-1]
    table = result.barcode_cn_table(barcode)
    assert table.barcode.unique().tolist() == [barcode]
    assert table.gene.tolist() == list(result.bundle.gene_ids)
    np.testing.assert_allclose(
        table[["p_" + x for x in result.class_names]],
        result.probabilities[-1, result.bundle.gene_markers],
    )
    with pytest.raises(ValueError, match="unknown barcode"):
        result.barcode_cn_table("missing")
    with pytest.raises(ValueError, match="unknown group"):
        result.group_cn_table(-123)
    members.loc[:, "group"] = -123
    markers.loc[:, "start"] = -123
    assert (result.groups >= 0).all()
    assert (result.bundle.grid.start >= 0).all()
    if len(result.group_calls.groups):
        group = result.group_calls.groups[-1]
        group_table = result.group_cn_table(group)
        assert group_table.group.unique().tolist() == [group]
        np.testing.assert_allclose(
            group_table[["pooled_p_" + x for x in result.class_names]],
            result.group_calls.pooled[-1, result.bundle.gene_markers],
        )
    result.save(tmp_path / "saved")
    for name, expected in (
        ("barcode_cn_genes", result.barcode_cn_table()),
        ("group_cn_genes", result.group_cn_table()),
        ("group_cn_segments", result.group_segments_table()),
    ):
        saved = pd.read_csv(
            tmp_path / "saved" / f"{name}.csv.gz", dtype={"barcode": str}
        )
        pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
