import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix

from barcodecnv import Grid, Loci
from barcodecnv.bundle import CellBundle, read_bundle, write_bundle
from barcodecnv.cli import main
from barcodecnv.loading import load_cells
from barcodecnv.signal import barcode_signal_test
from barcodecnv.smoothing import infercnv_smoothing
from barcodecnv.workflow import refine_groups


def example_bundle():
    rng = np.random.default_rng(782)
    genes = 60
    cells = 160
    membership = np.repeat(np.arange(8), 20)
    means = np.full((genes, cells), 20.0)
    means[:20, membership < 4] = 30
    means[20:40, membership >= 4] = 30
    counts = rng.poisson(means)
    pos = np.tile(np.arange(1, 21) * 1_000_000, 3)
    return CellBundle(
        Grid(tuple(np.repeat(["chr1", "chr2", "chr3"], 20)), pos, pos),
        Loci([], [], [], []),
        tuple(f"g{i}" for i in range(genes)),
        np.arange(genes),
        np.full(genes, 0.002),
        tuple(f"cell{i}" for i in range(cells)),
        tuple(f"0{i}" for i in membership),
        np.full(cells, 10000.0),
        csc_matrix(counts),
        csc_matrix((0, cells)),
        csc_matrix((0, cells)),
        np.array([]),
    )


@pytest.mark.parametrize("mode", ["external", "self", "none"])
@pytest.mark.parametrize("width", [1, 3, 7, 101])
def test_infercnv_matches_r_reference(mode, width):
    root = Path(__file__).parent / "fixtures/infercnv"
    df = pd.read_csv(root / "counts.tsv", sep="\t")
    counts = df.iloc[:, 3:].to_numpy()
    ref = None
    if mode == "external":
        ref = df.reference.to_numpy()
    if mode == "self":
        ref = np.median(counts / counts.sum(axis=0), axis=1)
        ref *= np.median(counts.sum(axis=0)) / ref.sum()
    expected = (
        pd.read_csv(root / f"{mode}_{width}.tsv", sep="\t").iloc[:, 1:].to_numpy()
    )
    np.testing.assert_allclose(
        infercnv_smoothing(counts, df.chrom, reference=ref, window_length=width),
        expected,
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )


def test_permutation_signal_and_single_barcode():
    data = example_bundle()
    result = barcode_signal_test(data, permutations=39, seed=20)
    assert result.pvalue == 1 / 40
    restricted = replace(data, barcode_labels=("only",) * len(data.cell_ids))
    result = barcode_signal_test(restricted, permutations=39)
    assert result.status == "unassessable" and result.pvalue is None
    # Identical cells have exactly the same score under every permutation.
    same = replace(data, expression=csc_matrix(np.full(data.expression.shape, 20)))
    assert barcode_signal_test(same, permutations=19).pvalue == 1.0


def test_bundle_v2_and_legacy_v1_share_single_cohort_behavior(tmp_path):
    data = example_bundle()
    path = tmp_path / "cells.h5"
    write_bundle(path, data)
    with h5py.File(path, "r+") as f:
        assert f.attrs["pbpc_schema"] == "cell-counts-v2"
        assert set(f["cells"]) == {"id", "barcode", "library"}
        # Mimic a previously saved v1 input without retaining a runtime field.
        f.attrs["pbpc_schema"] = "cell-counts-v1"
        f.create_dataset(
            "cells/block", data=np.array(["cohort"] * len(data.cell_ids), dtype="S")
        )
    loaded = read_bundle(path)
    np.testing.assert_array_equal(loaded.libraries, data.libraries)
    np.testing.assert_array_equal(
        loaded.expression.toarray(), data.expression.toarray()
    )
    np.testing.assert_array_equal(
        barcode_signal_test(loaded, permutations=19).null_scores,
        barcode_signal_test(data, permutations=19).null_scores,
    )


def test_gene_identity_preserved_when_multiple_genes_share_marker():
    data = example_bundle()
    markers = data.gene_markers.copy()
    markers[1] = 0
    data = replace(data, gene_markers=markers)
    pooled = data.prepared(np.ones(8, dtype=int))
    # Gene terms must remain separate; summing different genes changes NB noise.
    assert len(pooled.genes.count) == 60
    np.testing.assert_array_equal(
        pooled.genes.count, np.asarray(data.expression.sum(axis=1)).ravel()
    )
    assert np.sum(pooled.genes.marker == 0) == 2


@pytest.mark.parametrize("family,phase", [("nb", "conditional"), ("pln", "joint")])
def test_cli_signal_and_full_pipeline(tmp_path, family, phase):
    data = example_bundle()
    input = tmp_path / "cells.h5"
    write_bundle(input, data)
    read = read_bundle(input)
    assert read.barcodes == data.barcodes
    np.testing.assert_array_equal(
        read.prepared().genes.count, data.prepared().genes.count
    )
    out = tmp_path / "signal"
    assert main(["signal", str(input), "--out", str(out), "--permutations", "19"]) == 0
    assert json.loads((out / "signal.json").read_text())["pvalue"] == 0.05
    sentinel = (out / "run.json").read_bytes()
    assert main(["signal", str(input), "--out", str(out)]) == 1
    assert (out / "run.json").read_bytes() == sentinel
    out = tmp_path / "full"
    # Exercise the production defaults as well as the explicit legacy override.
    overrides = (
        [
            "--depth-family",
            family,
            "--barcode-phase",
            phase,
            "--depth-outlier-probability",
            "0.01",
        ]
        if family == "nb"
        else []
    )
    assert (
        main(
            [
                "infer",
                str(input),
                "--out",
                str(out),
                "--bootstraps",
                "20",
                "--draws",
                "32",
                "--permutations",
                "19",
            ]
            + overrides
        )
        == 0
    )
    groups = pd.read_csv(out / "groups.csv", dtype={"barcode": str})
    assert groups.barcode.tolist() == list(data.barcodes)
    assert groups.group.iloc[:4].nunique() == groups.group.iloc[4:].nunique() == 1
    assert groups.group.iloc[0] != groups.group.iloc[-1]
    metadata = json.loads((out / "run.json").read_text())
    epsilon = 0.01 if family == "nb" else 0.05
    assert metadata["depth_options"]["outlier_probability"] == epsilon
    assert metadata["group_reporting"]["depth_options"] == metadata["depth_options"]
    assert (
        metadata["model"]["depth_family"] == family
        and metadata["barcode_phase_method"] == phase
    )
    assert metadata["phase_converged"] and metadata["status"] == "complete"
    assert metadata["hf_refinement"]["status"] == "no_usable_allele_counts"
    with h5py.File(out / "result.h5") as f:
        assert f.attrs["depth_outlier_probability"] == epsilon
        assert f["group_calls"].attrs["depth_outlier_probability"] == epsilon
        p = f["probabilities"][:]
        assert p[:4, :20, 1].mean() > 0.9 and p[4:, 20:40, 1].mean() > 0.9
        assert p[:, :, 0][:, 40:].mean() > 0.9
        pooled = f["group_calls/pooled"][:]
        assert pooled.shape[0] == 2
        assert pooled[0, :20, 1].mean() > 0.9 and pooled[1, 20:40, 1].mean() > 0.9
        exported = pd.read_csv(out / "group_cn_genes.csv.gz")
        np.testing.assert_allclose(
            exported.pooled_p_gain.to_numpy(), pooled[:, :, 1].ravel()
        )
    assert metadata["group_reporting"]["noise"] == "fitted to group pseudobulks"


def test_hf_default_adds_allele_split_and_can_be_disabled():
    from barcodecnv.cli import parser
    from barcodecnv.workflow import run_pbpc

    data = example_bundle()
    membership = data.membership
    # The first expression group contains two opposite retained haplotypes.
    # Depth profiles stay unchanged, so expression cannot make this split.
    total = np.full((60, len(membership)), 20)
    first = np.full(total.shape, 10)
    first[:20, membership < 2] = 18
    first[:20, (membership >= 2) & (membership < 4)] = 2
    data = replace(
        data,
        loci=Loci(
            np.arange(60),
            data.grid.start,
            np.full(60, 0.99),
            np.tile(np.arange(20, dtype=float), 3),
        ),
        h1=csc_matrix(first),
        h2=csc_matrix(total - first),
        het=np.ones(60),
    )
    baseline = run_pbpc(
        data, replicates=20, draws=0, cn_refinement=False, hf_refinement=False
    )
    refined = run_pbpc(data, replicates=20, draws=0, cn_refinement=False)
    assert baseline["groups"].max() == 2 and baseline["hf_status"] == "disabled"
    labels = refined["groups"]
    assert refined["hf_status"] == "applied"
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    assert len(set(labels[4:])) == 1 and len(set(labels)) == 3
    np.testing.assert_array_equal(refined["pre_hf_groups"], baseline["groups"])
    # Reporting refinement must not silently change fitted CN probabilities.
    np.testing.assert_array_equal(refined["fit"].classes, baseline["fit"].classes)
    args = parser().parse_args(
        ["infer", "sample.h5", "--out", "results", "--no-hf-refinement"]
    )
    assert args.no_hf_refinement


def test_cn_refinement_preserves_unresolved_and_only_splits_supported_contrasts():
    distances = np.full((9, 9, 32), 0.4, dtype=float)
    distances[:4, :4] = 0
    distances[4:8, 4:8] = 0
    for i in range(9):
        distances[i, i] = 0
    initial = np.array([1] * 8 + [0])
    cells = np.arange(1, 10) * 10
    labels, _ = refine_groups(distances, initial, cells)
    assert len(set(labels[:4])) == len(set(labels[4:8])) == 1
    assert labels[0] != labels[4] and labels[-1] == 0
    distances[:8, :8] = 0.4
    for i in range(9):
        distances[i, i] = 0
    labels, _ = refine_groups(distances, initial, cells)
    assert len(set(labels[:8])) == 1 and labels[-1] == 0


@pytest.mark.parametrize("hdf5", [False, True])
def test_10x_loading_preserves_library_universe_and_phased_alleles(tmp_path, hdf5):
    from scipy.io import mmwrite

    raw = csc_matrix(np.array([[10, 20], [30, 40], [50, 60], [1000, 2000]]))
    features = ["ENSG2.1", "ENSG1.2", "unused.1", "antibody"]
    cell_ids = ["c2", "c1"]
    if hdf5:
        source = tmp_path / "matrix.h5"
        with h5py.File(source, "w") as f:
            for key, values in (
                ("data", raw.data),
                ("indices", raw.indices),
                ("indptr", raw.indptr),
                ("shape", raw.shape),
            ):
                f["matrix/" + key] = values
            for key, values in (
                ("barcodes", cell_ids),
                ("features/id", features),
                (
                    "features/feature_type",
                    ["Gene Expression"] * 3 + ["Antibody Capture"],
                ),
            ):
                f.create_dataset(
                    "matrix/" + key, data=np.array(values, dtype=h5py.string_dtype())
                )
    else:
        source = tmp_path / "matrix"
        source.mkdir()
        mmwrite(source / "matrix.mtx", raw)
        pd.DataFrame(
            {
                0: features,
                1: features,
                2: ["Gene Expression"] * 3 + ["Antibody Capture"],
            }
        ).to_csv(source / "features.tsv", sep="\t", header=False, index=False)
        (source / "barcodes.tsv").write_text("c2\nc1\n")
    (tmp_path / "cells.tsv").write_text("cell\tbarcode\nc1\t01\nc2\t02\n")
    (tmp_path / "genes.tsv").write_text(
        "gene\tchromosome\tposition\nENSG1\t1\t10\nENSG2\t1\t20\n"
    )
    (tmp_path / "ref.tsv").write_text("gene\tfraction\nENSG1\t0.2\nENSG2\t0.1\n")
    (tmp_path / "alleles.tsv").write_text(
        "cell\tCHROM\tPOS\tAD\tDP\tGT\tcM\nc1\t1\t15\t7\t10\t1|0\t0.1\nc2\t1\t15\t2\t8\t0|1\t0.1\n"
    )
    bundle = load_cells(
        source,
        tmp_path / "cells.tsv",
        tmp_path / "genes.tsv",
        reference=tmp_path / "ref.tsv",
        alleles=tmp_path / "alleles.tsv",
    )
    assert bundle.cell_ids == ("c1", "c2") and bundle.barcodes == ("01", "02")
    np.testing.assert_array_equal(bundle.libraries, [120, 90])
    np.testing.assert_array_equal(bundle.expression.toarray(), [[40, 30], [20, 10]])
    np.testing.assert_array_equal(bundle.h1.toarray(), [[7, 6]])
    np.testing.assert_array_equal(bundle.h2.toarray(), [[3, 2]])
    np.testing.assert_allclose(bundle.prepared().genes.expected, [24, 12, 18, 9])
    assert bundle.grid.start.tolist() == [10, 15, 20]


@pytest.fixture
def raw_inputs(tmp_path):
    """10x counts with unselected RNA, annotations and phased per-cell SNPs."""
    from scipy.io import mmwrite
    from scipy.sparse import vstack

    data = example_bundle()
    source = tmp_path / "10x"
    source.mkdir()
    unused = data.libraries - np.asarray(data.expression.sum(axis=0)).ravel()
    raw = vstack([data.expression, csc_matrix(unused[None, :])], format="csc")
    mmwrite(source / "matrix.mtx", raw)
    genes = [*data.gene_ids, "unannotated"]
    pd.DataFrame({0: genes, 1: genes, 2: ["Gene Expression"] * len(genes)}).to_csv(
        source / "features.tsv", sep="\t", header=False, index=False
    )
    (source / "barcodes.tsv").write_text("\n".join(data.cell_ids) + "\n")
    pd.DataFrame(dict(cell=data.cell_ids, barcode=data.barcode_labels)).to_csv(
        tmp_path / "cells.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        dict(gene=data.gene_ids, chromosome=data.grid.chrom, position=data.grid.start)
    ).to_csv(tmp_path / "genes.tsv", sep="\t", index=False)
    pd.DataFrame(dict(gene=data.gene_ids, fraction=data.reference_fractions)).to_csv(
        tmp_path / "reference.tsv", sep="\t", index=False
    )
    # Explicit per-locus genetic positions exercise allele loading without a map.
    rows = [
        dict(cell=cell, chromosome=c, position=1_500_000, h1=5, h2=5, genetic_cm=0.1)
        for cell in data.cell_ids
        for c in ("chr1", "chr2", "chr3")
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "alleles.tsv", sep="\t", index=False)
    return [
        "--matrix",
        str(source),
        "--cells",
        str(tmp_path / "cells.tsv"),
        "--genes",
        str(tmp_path / "genes.tsv"),
        "--reference",
        str(tmp_path / "reference.tsv"),
        "--alleles",
        str(tmp_path / "alleles.tsv"),
    ]


def test_one_command_matches_prepare_then_run(tmp_path, raw_inputs):
    import hashlib

    directory = tmp_path / "prepared"
    prepared = directory / "prepared.h5"
    assert main(["prepare", *raw_inputs, "--out", str(directory)]) == 0
    direct = tmp_path / "direct"
    staged = tmp_path / "staged"
    settings = [
        "--bootstraps",
        "20",
        "--draws",
        "32",
        "--permutations",
        "19",
        "--depth-outlier-probability",
        "0.01",
    ]
    assert main(["infer", *raw_inputs, "--out", str(direct), *settings]) == 0
    assert main(["infer", str(prepared), "--out", str(staged), *settings]) == 0
    a = read_bundle(direct / "prepared.h5")
    b = read_bundle(prepared)
    # Loading must preserve original exposure, not recompute it over chosen genes.
    np.testing.assert_array_equal(a.libraries, example_bundle().libraries)
    np.testing.assert_array_equal(
        a.prepared().genes.expected, b.prepared().genes.expected
    )
    assert a.barcodes == b.barcodes
    pd.testing.assert_frame_equal(
        pd.read_csv(direct / "groups.csv"), pd.read_csv(staged / "groups.csv")
    )
    with h5py.File(direct / "result.h5") as f, h5py.File(staged / "result.h5") as g:
        for key in (
            "probabilities",
            "phase",
            "barcode_phase",
            "group_calls/pooled",
            "group_calls/consensus",
        ):
            np.testing.assert_allclose(f[key][:], g[key][:], atol=1e-12, rtol=0)
    assert json.loads((direct / "signal.json").read_text()) == json.loads(
        (staged / "signal.json").read_text()
    )
    meta = json.loads((direct / "run.json").read_text())
    assert meta["input_kind"] == "counts" and meta["input"] == str(
        (direct / "prepared.h5").resolve()
    )
    assert (
        meta["input_sha256"]
        == hashlib.sha256((direct / "prepared.h5").read_bytes()).hexdigest()
    )
    assert meta["source_paths"]["matrix"] == str((tmp_path / "10x").resolve())
    # The snapshot is directly reusable; direct mode must not overwrite results.
    sentinel = (direct / "run.json").read_bytes()
    assert main(["infer", *raw_inputs, "--out", str(direct), *settings]) == 1
    assert (direct / "run.json").read_bytes() == sentinel


def test_signal_accepts_raw_counts_without_reference(tmp_path, raw_inputs):
    args = raw_inputs.copy()
    i = args.index("--reference")
    del args[i : i + 2]
    out = tmp_path / "signal_raw"
    assert main(["signal", *args, "--out", str(out), "--permutations", "19"]) == 0
    assert json.loads((out / "signal.json").read_text())["pvalue"] == 0.05
    assert read_bundle(out / "prepared.h5").reference_fractions.sum() == 0


@pytest.mark.parametrize(
    "inputs,message",
    [
        ([], "provide a prepared bundle"),
        (["--matrix", "counts"], "--cells, --genes, --reference"),
        (["--matrix", "counts", "--cells", "cells", "--genes", "genes"], "--reference"),
        (["bundle.h5", "--matrix", "counts"], "not both"),
        (["bundle.h5", "--alleles", "alleles"], "not both"),
    ],
)
def test_cli_rejects_incomplete_or_mixed_input_modes(tmp_path, capsys, inputs, message):
    out = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        main(["infer", *inputs, "--out", str(out)])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_raw_input_snapshot_survives_inference_failure(
    tmp_path, raw_inputs, monkeypatch
):
    from barcodecnv import api

    def fail(*args, **kwargs):
        raise RuntimeError("controlled fitting failure")

    monkeypatch.setattr(api, "run_pbpc", fail)
    out = tmp_path / "failed"
    assert main(["infer", *raw_inputs, "--out", str(out), "--skip-signal"]) == 1
    meta = json.loads((out / "run.json").read_text())
    assert meta["status"] == "failed" and meta["error"] == "controlled fitting failure"
    np.testing.assert_array_equal(
        read_bundle(meta["input"]).libraries, example_bundle().libraries
    )


@pytest.fixture
def preprocessing_inputs(tmp_path, raw_inputs):
    """Real adapters with supplied cellSNP/phase outputs; no external tools needed."""
    import gzip

    from scipy.io import mmwrite

    outs = tmp_path / "outs"
    outs.mkdir()
    (outs / "filtered_feature_bc_matrix").symlink_to(
        tmp_path / "10x", target_is_directory=True
    )
    cellsnp = tmp_path / "cellsnp"
    cellsnp.mkdir()
    data = example_bundle()
    (cellsnp / "cellSNP.samples.tsv").write_text("\n".join(data.cell_ids) + "\n")
    mmwrite(
        cellsnp / "cellSNP.tag.AD.mtx", csc_matrix(np.full((3, len(data.cell_ids)), 5))
    )
    mmwrite(
        cellsnp / "cellSNP.tag.DP.mtx", csc_matrix(np.full((3, len(data.cell_ids)), 10))
    )
    phased = tmp_path / "phased.vcf.gz"
    with (
        gzip.open(cellsnp / "cellSNP.base.vcf.gz", "wt") as a,
        gzip.open(phased, "wt") as b,
    ):
        for chrom in ("chr1", "chr2", "chr3"):
            prefix = f"{chrom}\t1500000\t.\tA\tG\t.\tPASS\t"
            a.write(prefix + "AD=800;DP=1600;OTH=0\n")
            b.write(prefix + ".\tGT\t0|1\n")
    maps = tmp_path / "maps"
    maps.mkdir()
    for chrom in ("chr1", "chr2", "chr3"):
        (maps / f"{chrom}.map").write_text(
            f"{chrom} a 0 500000\n{chrom} b 0.2 2500000\n"
        )
    return [
        "--outs",
        str(outs),
        "--cells",
        str(tmp_path / "cells.tsv"),
        "--reference",
        str(tmp_path / "reference.tsv"),
        "--genes",
        str(tmp_path / "genes.tsv"),
        "--genetic-map",
        str(maps),
        "--chromosomes",
        "1,2,3",
        "--cellsnp-dir",
        str(cellsnp),
        "--phased-vcf",
        str(phased),
    ]


def test_end_to_end_matches_separate_stages(tmp_path, preprocessing_inputs, caplog):
    caplog.set_level("INFO")
    settings = [
        "--bootstraps",
        "20",
        "--draws",
        "32",
        "--permutations",
        "19",
        "--depth-outlier-probability",
        "0.01",
    ]
    combined = tmp_path / "combined"
    prep = tmp_path / "prep"
    separate = tmp_path / "separate"
    assert main(["run", *preprocessing_inputs, "--out", str(combined), *settings]) == 0
    messages = caplog.text
    markers = [
        "[run 1/2]",
        "[preprocess 1/5]",
        "[preprocess 2/5]",
        "[preprocess 3/5]",
        "[preprocess 4/5]",
        "[preprocess 5/5]",
        "[run 2/2]",
        "[infer 1/4]",
        "[infer 2/4]",
        "[infer 3/4]",
        "[infer 4/4]",
    ]
    indices = [messages.index(marker) for marker in markers]
    assert indices == sorted(indices)
    assert main(["preprocess", *preprocessing_inputs, "--out", str(prep)]) == 0
    assert (
        main(["infer", str(prep / "prepared.h5"), "--out", str(separate), *settings])
        == 0
    )
    with (
        h5py.File(combined / "inference/result.h5") as a,
        h5py.File(separate / "result.h5") as b,
    ):
        for key in ("probabilities", "barcode_phase", "group_calls/pooled"):
            np.testing.assert_array_equal(a[key][:], b[key][:])
    pd.testing.assert_frame_equal(
        pd.read_csv(combined / "inference/groups.csv"),
        pd.read_csv(separate / "groups.csv"),
    )
    assert json.loads((combined / "inference/signal.json").read_text()) == json.loads(
        (separate / "signal.json").read_text()
    )
    assert json.loads((combined / "pipeline.json").read_text())["status"] == "complete"
    command = json.loads((combined / "preprocessing/preprocessing.json").read_text())[
        "run_command"
    ]
    assert command[-1] == str(combined / "inference")
    sentinel = (combined / "pipeline.json").read_bytes()
    assert main(["run", *preprocessing_inputs, "--out", str(combined), *settings]) == 1
    assert (combined / "pipeline.json").read_bytes() == sentinel
    assert "barcodecnv infer" in (prep / "infer.sh").read_text()


@pytest.mark.parametrize("stage", ["preprocessing", "inference"])
def test_end_to_end_stops_at_failed_stage(
    tmp_path, preprocessing_inputs, monkeypatch, stage
):
    from barcodecnv import api

    def fail(*args, **kwargs):
        raise RuntimeError("controlled " + stage + " failure")

    monkeypatch.setattr(
        api, "preprocess" if stage == "preprocessing" else "run_pbpc", fail
    )
    out = tmp_path / "failed_pipeline"
    assert main(["run", *preprocessing_inputs, "--out", str(out), "--skip-signal"]) == 1
    metadata = json.loads((out / "pipeline.json").read_text())
    assert metadata["status"] == "failed" and metadata["stage"] == stage
    assert metadata["error"] == "controlled " + stage + " failure"
    if stage == "preprocessing":
        assert not (out / "inference").exists()
    else:
        data = read_bundle(out / "preprocessing/prepared.h5")
        np.testing.assert_array_equal(data.libraries, example_bundle().libraries)
        assert (
            json.loads((out / "inference/run.json").read_text())["status"] == "failed"
        )


def test_run_validates_inference_settings_before_preprocessing(
    tmp_path, preprocessing_inputs, monkeypatch
):
    from barcodecnv import api

    def unexpected(*args):
        pytest.fail("preprocessing should not start")

    monkeypatch.setattr(api, "preprocess", unexpected)
    out = tmp_path / "invalid"
    assert (
        main(["run", *preprocessing_inputs, "--out", str(out), "--bootstraps", "1"])
        == 1
    )
    assert not out.exists()
