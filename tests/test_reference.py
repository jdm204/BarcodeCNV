# Explicit re-exports register these shared pytest fixtures in this module.
import hashlib
import json
import shutil
import tomllib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from test_cli import preprocessing_inputs as preprocessing_inputs
from test_cli import raw_inputs as raw_inputs

from barcodecnv.cli import main
from barcodecnv.loading import canonical_genes
from barcodecnv.reference import (
    contiguous_bins,
    fit_mixture,
    fit_reference,
    global_weights,
    read_panel,
    regional_weights,
)

FIXTURES = Path(__file__).parent / "fixtures/reference"


def test_adam_matches_julia_at_intermediate_and_final_updates():
    fixtures = tomllib.loads((FIXTURES / "julia_reference.toml").read_text())
    for case in fixtures["global"]:
        actual = global_weights(
            np.array(case["matrix"]),
            np.array(case["observed"]),
            max_iter=case["iterations"],
        )
        np.testing.assert_allclose(
            actual, case["weights"], atol=2e-11, rtol=2e-11, err_msg=case["name"]
        )
    case = fixtures["global"][0]
    matrix = np.array(case["matrix"])
    observed = np.array(case["observed"])
    weights, converged = fit_mixture(matrix, 7 * observed)
    assert converged
    np.testing.assert_allclose(weights, [0.2, 0.5, 0.3], atol=1e-5)
    assert not fit_mixture(matrix, observed, max_iter=1)[1]


def test_regional_matches_julia_and_refuses_unidentified_or_unfinished_fits():
    case = tomllib.loads((FIXTURES / "julia_reference.toml").read_text())["regional"]
    matrix = np.array(case["matrix"])
    observed = np.array(case["counts"])
    observed /= observed.sum()
    weights, audit = regional_weights(
        matrix, observed, case["chromosomes"], bin_genes=12
    )
    # BFGS line searches differ; use the 1e-5 mixture tolerance of the Julia tests.
    np.testing.assert_allclose(weights, case["weights"], atol=1e-5)
    np.testing.assert_allclose(matrix @ weights, matrix @ case["weights"], atol=1e-7)
    assert weights[0] > 0.99 and audit["failed_bins"] == 0
    with pytest.raises(RuntimeError, match="did not converge"):
        regional_weights(
            matrix, observed, case["chromosomes"], bin_genes=12, max_iter=1
        )
    with pytest.raises(ValueError, match="unidentified"):
        regional_weights(matrix[:, [0, 0]], observed, case["chromosomes"], bin_genes=12)
    # Small tails merge within chromosomes; no gene is lost or counted twice.
    bins = contiguous_bins(["chr1"] * 11 + ["chr2"] * 9, 4, 2)
    assert [b.tolist() for b in bins] == [
        list(range(6)),
        list(range(6, 11)),
        list(range(11, 20)),
    ]
    np.testing.assert_array_equal(np.concatenate(bins), np.arange(20))
    assert all(len(set(np.array(["chr1"] * 11 + ["chr2"] * 9)[b])) == 1 for b in bins)


def test_julia_panel_identity_normalization_and_integrity(tmp_path):
    panel = read_panel(FIXTURES / "panel")
    assert panel.genes == ("ENSG1", "ENSG2", "ENSG3", "ENSG4")
    np.testing.assert_array_equal(
        panel.expression, [[0.7, 0.1], [0.2, 0.7], [0.1, 0.2], [0, 0]]
    )
    assert canonical_genes(["ENSG1.5_PAR_Y", "HLA.DRA", "custom.1"]) == [
        "ENSG1_PAR_Y",
        "HLA.DRA",
        "custom.1",
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        canonical_genes(["ENSG1.2", "ENSG1.3"])
    copied = tmp_path / "panel"
    shutil.copytree(FIXTURES / "panel", copied)
    with (copied / "profiles.tsv").open("a") as f:
        f.write("corrupt\n")
    with pytest.raises(ValueError, match="checksum"):
        read_panel(copied)


def test_gene_join_selection_and_full_panel_normalization():
    panel = read_panel(FIXTURES / "panel")
    ids = ["ENSG3", "ENSG1.7", "ENSG5", "ENSG2", "ENSG4"]
    genes = pd.DataFrame(
        dict(
            gene=["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"],
            chromosome=["chr1"] * 5,
            position=np.arange(1, 6),
        )
    )
    fitted = fit_reference(
        panel, ids, [16, 34, 50, 50, 10], genes, genome="hg38", min_cpm=0
    )
    assert fitted.weights["naive"] == pytest.approx(0.4, abs=1e-5)
    assert fitted.profile["ENSG1"] == pytest.approx(0.34, abs=1e-5)
    assert fitted.profile["ENSG4"] == 0 and "ENSG5" not in fitted.profile
    assert fitted.audit["absent_gene_ids"] == ["ENSG5"]
    assert fitted.audit["fitting_gene_ids"] == ["ENSG3", "ENSG1", "ENSG2"]
    subset = fit_reference(
        panel, ["ENSG1", "ENSG2"], [34, 50], genes, genome="hg38", min_cpm=0
    )
    assert sum(subset.profile.values()) == pytest.approx(1)
    assert (
        subset.profile["ENSG3"] > 0
    )  # never normalize the output on the fitting subset
    with pytest.raises(ValueError, match="genome"):
        fit_reference(panel, ids, [16, 34, 50, 50, 10], genes, genome="hg19")
    with pytest.raises(ValueError, match="absent from annotation"):
        fit_reference(panel, ids, [16, 34, 50, 50, 10], genes.iloc[1:], genome="hg38")


def write_panel(directory, genes, matrix):
    """Test-only writer of the documented Julia on-disk format."""
    directory.mkdir()
    with h5py.File(directory / "expression.h5", "w") as f:
        f["expression"] = np.asarray(matrix).T
        for key, values in [
            ("gene_ids", genes),
            ("gene_symbols", genes),
            ("profile_ids", ["a", "b"]),
        ]:
            f.create_dataset(key, data=values, dtype=h5py.string_dtype())
    metadata = pd.DataFrame({"id": ["a", "b"]})
    for key in (
        "study",
        "donor",
        "tissue",
        "cell_type",
        "cell_state",
        "capture",
        "platform",
        "processing",
        "n_cells",
    ):
        metadata[key] = ""
    metadata.to_csv(directory / "profiles.tsv", sep="\t", index=False)
    (directory / "panel.toml").write_text(
        'schema_version=1\nunits="fraction"\nid="fixture"\ngenome="hg38"\nannotation="fixture"\n'
    )
    lines = ["complete=true", "version=1", "[files]"]
    for path in sorted(directory.iterdir()):
        lines.append(f'"{path.name}"="{hashlib.sha256(path.read_bytes()).hexdigest()}"')
    (directory / "dataset.toml").write_text("\n".join(lines) + "\n")


def test_fit_reference_cli_and_selected_cells(tmp_path):
    from scipy.io import mmwrite
    from scipy.sparse import coo_matrix

    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "features.tsv").write_text(
        "ENSG1.2\tA\tGene Expression\nENSG2\tB\tGene Expression\nENSG3\tC\tGene Expression\n"
    )
    (matrix / "barcodes.tsv").write_text("selected\nexcluded\n")
    mmwrite(matrix / "matrix.mtx", coo_matrix([[34, 100000], [50, 0], [16, 0]]))
    (tmp_path / "cells.tsv").write_text("cell\tbarcode\nselected\tbc1\n")
    (tmp_path / "genes.tsv").write_text(
        "gene\tchromosome\tposition\nENSG1\t1\t1\nENSG2\t1\t2\nENSG3\t1\t3\n"
    )
    args = [
        "fit-reference",
        "--matrix",
        str(matrix),
        "--cells",
        str(tmp_path / "cells.tsv"),
        "--genes",
        str(tmp_path / "genes.tsv"),
        "--reference-panel",
        str(FIXTURES / "panel"),
        "--out",
        str(tmp_path / "fit"),
    ]
    assert main(args) == 0
    weights = pd.read_csv(tmp_path / "fit/weights.tsv", sep="\t").set_index("id")
    assert weights.loc["naive", "weight"] == pytest.approx(0.4, abs=1e-5)
    assert main(args) == 1
    bad = ["prepare", *args[1:]]
    bad[bad.index("--out") + 1] = str(tmp_path / "unfinished.h5")
    assert (
        main(
            [
                *bad,
                "--one-block",
                "--reference-method",
                "regional_consensus",
                "--reference-iterations",
                "1",
            ]
        )
        == 1
    )
    assert not (tmp_path / "unfinished.h5").exists()


def test_run_with_panel_matches_pre_fitted_profile(
    tmp_path, preprocessing_inputs, raw_inputs
):
    panel = tmp_path / "panel"
    values = np.r_[np.full(60, 0.002), 0.88]
    write_panel(
        panel, [*[f"g{i}" for i in range(60)], "outside_assay"], np.c_[values, values]
    )
    panel_args = preprocessing_inputs.copy()
    i = panel_args.index("--reference")
    panel_args[i : i + 2] = ["--reference-panel", str(panel)]
    settings = ["--bootstraps", "20", "--draws", "32", "--skip-signal"]
    assert (
        main(["run", *panel_args, "--out", str(tmp_path / "panel_run"), *settings]) == 0
    )
    assert (
        main(
            [
                "run",
                *preprocessing_inputs,
                "--out",
                str(tmp_path / "fixed_run"),
                *settings,
            ]
        )
        == 0
    )
    fit = tmp_path / "panel_run/preprocessing/reference_fit"
    assert json.loads((fit / "fit.json").read_text())["normalization"].startswith(
        "observed normalized"
    )
    assert pd.read_csv(fit / "reference.tsv", sep="\t").fraction.sum() == pytest.approx(
        1
    )
    with (
        h5py.File(tmp_path / "panel_run/inference/result.h5") as a,
        h5py.File(tmp_path / "fixed_run/inference/result.h5") as b,
    ):
        np.testing.assert_allclose(
            a["probabilities"][:], b["probabilities"][:], atol=1e-12
        )
    raw_panel = raw_inputs.copy()
    i = raw_panel.index("--reference")
    raw_panel[i : i + 2] = ["--reference-panel", str(panel)]
    assert (
        main(["infer", *raw_panel, "--out", str(tmp_path / "raw_infer"), *settings])
        == 0
    )
    assert main(["prepare", *raw_panel, "--out", str(tmp_path / "prepared.h5")]) == 0
    assert (tmp_path / "prepared.h5.reference_fit/reference.tsv").is_file()
    with (
        h5py.File(tmp_path / "raw_infer/result.h5") as a,
        h5py.File(tmp_path / "panel_run/inference/result.h5") as b,
    ):
        np.testing.assert_allclose(
            a["probabilities"][:], b["probabilities"][:], atol=1e-12
        )
    # Raw infer also owns and exports its fit; reuse one end-to-end prepared bundle
    # only after fitting, never silently override a bundle's existing reference.
    with pytest.raises(SystemExit):
        main(
            [
                "infer",
                str(tmp_path / "panel_run/preprocessing/prepared.h5"),
                "--reference-panel",
                str(panel),
                "--out",
                str(tmp_path / "bad"),
            ]
        )
