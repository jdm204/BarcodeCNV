import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from barcodecnv import downloads as d
from barcodecnv import resources as r
from barcodecnv.cli import main, parser


class Response(io.BytesIO):
    def __init__(self, body, status=200, **headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


def test_verified_download_cache_and_corruption(tmp_path, monkeypatch):
    body = b"complete genomic data"
    sha = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(
        d.urllib.request,
        "urlopen",
        lambda *a, **k: Response(body, **{"Content-Length": str(len(body))}),
    )
    dest = tmp_path / "data"
    d.download("https://example.test/data", dest, sha256=sha)
    monkeypatch.setattr(
        d.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("cache hit attempted network"),
    )
    assert (
        d.download("https://example.test/data", dest, sha256=sha).read_bytes() == body
    )
    dest.write_bytes(b"corrupt genomic data")
    with pytest.raises(ValueError, match="modified cached"):
        d.download("https://example.test/data", dest, sha256=sha)


def test_invalid_checksum_cannot_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(d.urllib.request, "urlopen", lambda *a, **k: Response(b"bad"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        d.download("https://example.test/data", tmp_path / "data", sha256="0" * 64)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("honours_range", [False, True])
def test_partial_resume_and_ignored_range(tmp_path, monkeypatch, honours_range):
    (tmp_path / "data.partial").write_bytes(b"abc")
    d.write_json(
        tmp_path / "data.partial.json",
        dict(url="https://example.test/data", validator='"v1"'),
    )

    def response(request, **kwargs):
        assert request.get_header("Range") == "bytes=3-"
        assert request.get_header("If-range") == '"v1"'
        return (
            Response(b"def", 206, **{"Content-Range": "bytes 3-5/6"})
            if honours_range
            else Response(b"abcdef")
        )

    monkeypatch.setattr(d.urllib.request, "urlopen", response)
    assert (
        d.download(
            "https://example.test/data",
            tmp_path / "data",
            sha256=hashlib.sha256(b"abcdef").hexdigest(),
        ).read_bytes()
        == b"abcdef"
    )


def test_truncated_download_not_published_then_resumed(tmp_path, monkeypatch):
    responses = iter(
        [
            Response(b"abc", **{"Content-Length": "6", "ETag": '"v1"'}),
            Response(b"def", 206, **{"Content-Range": "bytes 3-5/6"}),
        ]
    )
    monkeypatch.setattr(d.urllib.request, "urlopen", lambda *a, **k: next(responses))
    with pytest.raises(OSError, match="incomplete"):
        d.download("https://example.test/data", tmp_path / "data", attempts=1)
    assert not (tmp_path / "data").exists()
    assert (
        d.download("https://example.test/data", tmp_path / "data").read_bytes()
        == b"abcdef"
    )


def test_map_preparation_ignores_traversal_and_repairs(tmp_path):
    archive = tmp_path / "maps.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped", b"unsafe")
        for chrom in r.CHROMOSOMES:
            z.writestr(
                f"chr_in_chrom_field/plink.chrchr{chrom}.GRCh38.map",
                f"chr{chrom} p0 0 1\nchr{chrom} p1 1 1001\n",
            )
    maps = r.ensure_maps(archive, tmp_path)
    assert not (tmp_path.parent / "escaped").exists()
    selected = maps / "plink.chrchr22.GRCh38.map"
    selected.write_text("bad")
    r.ensure_maps(archive, tmp_path)
    assert selected.read_text() == "chr22 p0 0 1\nchr22 p1 1 1001\n"


def test_setup_dry_run_defaults_no_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        r, "download", lambda *a, **k: pytest.fail("dry run downloaded")
    )
    assert main(["setup", "--resource-dir", str(tmp_path / "absent"), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["expression_panel"] == list(r.EXPRESSION_PANELS["b-cells-v1"])
    assert set(report["panels"]) == {*map(str, range(1, 23)), "X"}
    assert not (tmp_path / "absent").exists()
    with pytest.raises(SystemExit):
        parser().parse_args(["setup", "--chromosomes", "1,chr1"])
    with pytest.raises(SystemExit):
        parser().parse_args(["setup", "--genome", "mm10"])


def test_config_precedence_and_named_panel(tmp_path, monkeypatch):
    from barcodecnv.preprocessing import discover_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BARCODECNV_PREPROCESS_CONFIG", raising=False)
    monkeypatch.setenv("BARCODECNV_RESOURCE_DIR", str(tmp_path / "cache"))
    # Isolate from any source-checkout legacy setup.
    import barcodecnv.preprocessing as p

    monkeypatch.setattr(
        p, "__file__", str(tmp_path / "source/src/barcodecnv/preprocessing.py")
    )
    config = r.default_config()
    config.parent.mkdir(parents=True)
    panel = config.parent / "expression_panels/b-cells-v1"
    panel.mkdir(parents=True)
    d.write_json(config, dict(expression_panels={"b-cells-v1": str(panel)}))
    assert discover_config() == config
    assert r.resolve_expression_panel("b-cells-v1") == panel
    local = tmp_path / ".preprocessing/config.json"
    local.parent.mkdir()
    local.write_text("{}")
    assert discover_config() == local
    monkeypatch.setenv("BARCODECNV_PREPROCESS_CONFIG", str(tmp_path / "missing"))
    assert discover_config() == tmp_path / "missing"
    assert discover_config(config) == config


def test_expression_archive_rejects_links(tmp_path, monkeypatch):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        member = tarfile.TarInfo("expression.h5")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        t.addfile(member)
    monkeypatch.setattr(r, "download", lambda *a, **k: archive)
    with pytest.raises(ValueError, match="unsafe"):
        r.ensure_expression_panel(tmp_path, "b-cells-v1")
    assert not (tmp_path / "expression_panels/b-cells-v1").exists()


def test_failed_setup_preserves_previous_config(tmp_path, monkeypatch):
    root = tmp_path / "hg38"
    root.mkdir()
    previous = b'{"genome":"hg38","resources":{"genes":"existing"}}'
    (root / "config.json").write_bytes(previous)
    monkeypatch.setattr(
        r,
        "ensure_tools",
        lambda *a: (_ for _ in ()).throw(RuntimeError("tool install interrupted")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        r.setup(resource_dir=tmp_path)
    assert (root / "config.json").read_bytes() == previous
    assert json.loads((root / "setup.json").read_text())["status"] == "failed"


def test_nonpar_conversion_and_filter_failures(tmp_path):
    # Tiny real bcftools/Java tools are exercised in the integration smoke test.
    # Here executable fakes inspect exact filters and emulate both process errors.
    bcftools = tmp_path / "bcftools"
    java = tmp_path / "java"
    bcftools.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "'
        + str(tmp_path / "args")
        + '"\nprintf "sample genotypes with sufficient length"\n'
    )
    java.write_text("#!/bin/sh\ncat\n")
    bcftools.chmod(0o755)
    java.chmod(0o755)
    tools = {"bcftools": str(bcftools), "java": str(java)}
    dest = tmp_path / "x.bref3"
    r.convert_panel(tmp_path / "x.vcf", dest, "X", tools, tmp_path / "bref3.jar")
    args = (tmp_path / "args").read_text().splitlines()
    assert args[args.index("-t") + 1] == "chrX:2781480-155701382"
    assert args[:8] == ["view", "-v", "snps", "-m2", "-M2", "-c", "1", "-t"]
    bcftools.write_text(
        '#!/bin/sh\nprintf "truncated but nonempty genotypes"\nexit 1\n'
    )
    with pytest.raises(RuntimeError, match="filtering failed"):
        r.convert_panel(
            tmp_path / "x.vcf",
            tmp_path / "failed.bref3",
            "22",
            tools,
            tmp_path / "bref3.jar",
        )
    assert not (tmp_path / "failed.bref3").exists()
