import gzip
import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

from barcodecnv.bundle import read_bundle
from barcodecnv.preprocessing import main, resolve_resource
from barcodecnv.preprocessing_io import export_alleles, pooled_vcf


def write_vcf(path, records, samples=False):
    with gzip.open(path, "wt") as f:
        f.write(
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
            + ("\tFORMAT\tDONOR" if samples else "")
            + "\n"
        )
        f.write("\n".join(records) + "\n")


def fixtures(root):
    cs = root / "cellsnp"
    cs.mkdir()
    # Matrix columns deliberately differ from the cell map order.
    (cs / "cellSNP.samples.tsv").write_text("c2\nc1\n")
    write_vcf(
        cs / "cellSNP.base.vcf.gz",
        [
            "1\t100\t.\tA\tG\t.\tPASS\tAD=4;DP=10;OTH=0",
            "1\t200\t.\tC\tT\t.\tPASS\tAD=7;DP=10;OTH=0",
            "1\t300\t.\tG\tA\t.\tPASS\tAD=4;DP=10;OTH=0",
        ],
    )
    mmwrite(cs / "cellSNP.tag.AD.mtx", coo_matrix([[1, 3], [5, 2], [2, 2]]))
    mmwrite(cs / "cellSNP.tag.DP.mtx", coo_matrix([[4, 6], [7, 3], [4, 6]]))
    phased = root / "phased.vcf.gz"
    write_vcf(
        phased,
        [
            "chr1\t100\t.\tA\tG\t.\tPASS\t.\tGT:GQ\t0|1:60",
            "chr1\t200\t.\tC\tT\t.\tPASS\t.\tGT:GP:PQ\t1|0:0.1,0.8,0.1:20",
            # Same coordinate but different ALT must NOT match.
            "chr1\t300\t.\tG\tC\t.\tPASS\t.\tGT\t0|1",
        ],
        True,
    )
    return cs, phased


def test_pooled_genotypes_preserve_julia_filters(tmp_path):
    source = tmp_path / "base.vcf.gz"
    destination = tmp_path / "pooled.vcf"
    cases = [
        (100, 1, 10, 0),
        (200, 9, 10, 0),
        (300, 10, 10, 0),
        (400, 9, 9, 0),
        (500, 1, 11, 0),
        (600, 5, 10, 1),
        (700, 0, 0, 0),
    ]
    write_vcf(
        source,
        [f"1\t{p}\t.\tA\tG\t.\tPASS\tAD={a};DP={d};OTH={o}" for p, a, d, o in cases],
    )
    assert pooled_vcf(source, destination, {"chr1"}) == (["chr1"], 3)
    records = [
        line.split("\t")
        for line in destination.read_text().splitlines()
        if not line.startswith("#")
    ]
    assert [(int(r[1]), r[-1]) for r in records] == [
        (100, "0/1:9,1:10"),
        (200, "0/1:1,9:10"),
        (300, "1/1:0,10:10"),
    ]


def test_export_preserves_cells_and_haplotype_orientation(tmp_path):
    cs, phased = fixtures(tmp_path)
    out = tmp_path / "alleles.tsv.gz"
    audit = export_alleles(cs, phased, out, ["c1", "c2"], {"chr1"})
    assert audit == dict(phased_count_loci=2, nonzero_cell_locus_pairs=4)
    t = pd.read_csv(out, sep="\t").set_index(["position", "cell"])
    assert t.loc[(100, "c2"), ["h1", "h2"]].tolist() == [3, 1]
    assert t.loc[(100, "c1"), ["h1", "h2"]].tolist() == [3, 3]
    assert t.loc[(200, "c2"), ["h1", "h2"]].tolist() == [5, 2]
    assert t.loc[(200, "c1"), ["h1", "h2"]].tolist() == [2, 1]
    # GQ must not be mistaken for phase quality.
    assert t.loc[(100, "c2"), "phase_prob"] == 0.99
    assert t.loc[(200, "c2"), "het"] == 0.8
    with pytest.raises(ValueError, match="sample IDs"):
        export_alleles(cs, phased, out, ["c1", "c3"], {"chr1"})
    mmwrite(cs / "cellSNP.tag.AD.mtx", coo_matrix([[5, 3], [5, 2], [2, 2]]))
    with pytest.raises(ValueError, match="AD exceeds DP"):
        export_alleles(cs, phased, out, ["c1", "c2"], {"chr1"})


def test_resource_resolution_does_not_match_chr10_for_chr1(tmp_path):
    (tmp_path / "1kGP_HC_chr10.bref3").touch()
    with pytest.raises(FileNotFoundError):
        resolve_resource(tmp_path, "chr1", "panel")


def prepared_args(root, cs, phased):
    outs = root / "outs"
    matrix = outs / "filtered_feature_bc_matrix"
    matrix.mkdir(parents=True)
    (matrix / "features.tsv").write_text(
        "g1\tG1\tGene Expression\ng2\tG2\tGene Expression\n"
    )
    (matrix / "barcodes.tsv").write_text("c1\nc2\n")
    mmwrite(matrix / "matrix.mtx", coo_matrix([[10, 20], [30, 40]]))
    (root / "cells.tsv").write_text(
        "cell_barcode\tlineage_barcode\n" + "c1\t001\nc2\t002\n"
    )
    (root / "genes.tsv").write_text(
        "gene\tchromosome\tposition\ng1\t1\t110\ng2\t1\t210\n"
    )
    (root / "reference.tsv").write_text("gene\tfraction\ng1\t0.25\ng2\t0.75\n")
    maps = root / "maps"
    maps.mkdir()
    (maps / "chr1.map").write_text("chr1 a 0 1\nchr1 b 1 1001\n")
    return [
        "--outs",
        str(outs),
        "--cells",
        str(root / "cells.tsv"),
        "--reference",
        str(root / "reference.tsv"),
        "--genes",
        str(root / "genes.tsv"),
        "--genetic-map",
        str(maps),
        "--cellsnp-dir",
        str(cs),
        "--phased-vcf",
        str(phased),
        "--chromosomes",
        "1",
        "--out",
        str(root / "output"),
    ]


def test_preprocessed_counts_to_production_bundle(tmp_path):
    cs, phased = fixtures(tmp_path)
    args = prepared_args(tmp_path, cs, phased)
    assert main(args) == 0
    data = read_bundle(tmp_path / "output/prepared.h5")
    assert data.cell_ids == ("c1", "c2") and data.barcode_labels == ("001", "002")
    np.testing.assert_array_equal(data.h1.toarray(), [[3, 3], [2, 5]])
    np.testing.assert_array_equal(data.h2.toarray(), [[3, 1], [1, 2]])
    np.testing.assert_array_equal(data.libraries, [40, 60])
    np.testing.assert_allclose(data.loci.genetic_cm, [0.099, 0.199])
    manifest = tmp_path / "output/preprocessing.json"
    previous = manifest.read_bytes()
    assert json.loads(previous)["status"] == "complete"
    assert main(args) == 1
    assert manifest.read_bytes() == previous


def test_failed_conversion_cannot_publish_complete_bundle(tmp_path):
    cs, phased = fixtures(tmp_path)
    args = prepared_args(tmp_path, cs, phased)
    mmwrite(cs / "cellSNP.tag.DP.mtx", coo_matrix([[0, 0], [0, 0], [0, 0]]))
    assert main(args) == 1
    assert (
        json.loads((tmp_path / "output/preprocessing.json").read_text())["status"]
        == "failed"
    )
    assert not (tmp_path / "output/prepared.h5").exists()


def test_cli_preprocess_config_discovery_and_precedence(tmp_path, monkeypatch):
    from barcodecnv.cli import main as cli_main
    from barcodecnv.preprocessing import discover_config

    cs, phased = fixtures(tmp_path)
    args = prepared_args(tmp_path, cs, phased)
    resources = {}
    for option, key in [("--genes", "genes"), ("--genetic-map", "genetic_map")]:
        i = args.index(option)
        resources[key] = args[i + 1]
        del args[i : i + 2]
    local = tmp_path / ".preprocessing"
    local.mkdir()
    config = local / "config.json"
    config.write_text(json.dumps(dict(resources=resources)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BARCODECNV_PREPROCESS_CONFIG", raising=False)
    assert discover_config() == config
    assert cli_main(["preprocess", *args]) == 0
    metadata = json.loads((tmp_path / "output/preprocessing.json").read_text())
    assert metadata["arguments"]["config"] == str(config)
    assert metadata["run_command"][1] == "infer"
    override = tmp_path / "override.json"
    monkeypatch.setenv("BARCODECNV_PREPROCESS_CONFIG", str(override))
    assert discover_config() == override
    assert discover_config(config) == config


@pytest.mark.parametrize("exit_status", [0, 7])
def test_native_commands_and_output_stay_in_logs(tmp_path, capfd, caplog, exit_status):
    import logging
    import shlex
    import sys

    from barcodecnv.preprocessing import Commands

    run = Commands({"helper": sys.executable}, tmp_path)
    script = "import sys; print('native stdout'); print('native stderr', file=sys.stderr); sys.exit(int(sys.argv[1]))"
    with caplog.at_level(logging.INFO):
        if exit_status:
            with pytest.raises(
                RuntimeError, match="helper failed .*exit status 7.*see"
            ) as error:
                run("helper", "-c", script, exit_status)
            assert str(tmp_path / "00-helper.log") in str(error.value)
            assert "native stderr" not in str(error.value)
        else:
            run("helper", "-c", script, exit_status)
    console = capfd.readouterr()
    assert not console.out and not console.err
    assert script not in caplog.text
    log = (tmp_path / "00-helper.log").read_text()
    assert log.startswith("$ " + shlex.join(run.history[0]["argv"]) + "\n")
    assert "native stdout\n" in log and "native stderr\n" in log
    if exit_status:
        assert "Exit status: 7" in log
    # Even a mistakenly reused command runner cannot truncate an earlier log.
    with pytest.raises(FileExistsError):
        Commands({"helper": sys.executable}, tmp_path)("helper", "-c", script, 0)
    assert (tmp_path / "00-helper.log").read_text() == log
