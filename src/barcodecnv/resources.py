"""Explicit hg38 provisioning, adapted from the Julia resource/preprocessing code.

    barcodecnv setup --genome hg38 --expression-panel b-cells-v1

Only this command downloads/installs. Analysis consumes the published config.
Native dependencies are provisioned through Bioconda/conda-forge, not copied
from the developer's machine. Completed files carry hashes and source records.
"""

import argparse
import fcntl
import json
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

from .downloads import digest, download, write_json

LOG = logging.getLogger(__name__)
CHROMOSOMES = tuple(map(str, range(1, 23))) + ("X",)
PANEL_BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
# Published 20220804_manifest.txt checksums (MD5 upstream; SHA-256 also recorded locally).
PANEL_MD5 = {
    "1": "589762a43c412a34ff2dd76b009f06c6",
    "2": "0f00e90423da451798a87c037a440669",
    "3": "e13293cc80e060d731a0f3ce453168a1",
    "4": "637cb88918fb8816cefb893f2422fe6e",
    "5": "662dd03be57e9239eb693917edcce3fe",
    "6": "cc120eac122f422e7022817ba944acea",
    "7": "58a1da5c2cc6b3662f952887a54d187e",
    "8": "9c7359e9bd06a923fa7a2daa55dd2a82",
    "9": "c6832a28e977dafb9613b3d4f9a941df",
    "10": "1f0666ceb4818bc60936013a9e64473c",
    "11": "76e53c142f60f7305cc0c0380f02326a",
    "12": "7609c8f8b1116d51357bbcb3f3c8b194",
    "13": "d6b1dbeaa98f375824172ad28b9d0c34",
    "14": "daac8b8431cd8eb360e86f709361dfa7",
    "15": "b24a888029664f0f7a3b2bf8dffe00d1",
    "16": "4566018235a48c73b449c7d888427fd8",
    "17": "bdebafdef3fbbcf3a60ef114aa9ff362",
    "18": "64440c87109e9d12222300912e71cff2",
    "19": "cf5c060fbcc0fcd2fa089f27fd6ac78e",
    "20": "34a9a5dbfb285e7eb9d1f2881953014b",
    "21": "1eb1f48bf5608d3abacf5047c8889456",
    "22": "a2653cc7a1c8d03a96ca4f14d0fabdd2",
    "X": "d2c80aa7b3bcb8f895f98fd5779fb448",
}
BEAGLE_BASE = "https://faculty.washington.edu/browning/beagle"
# URL, local filename, expected SHA-256; versions match the validated Julia inputs.
ASSETS = {
    "genes": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz",
        "gencode.v44.annotation.gtf.gz",
        "01f817afed65feee863361b4baf30a95e722ee5c5d508ff77b04106ef7ba20d3",
    ),
    "snp_vcf": (
        "https://downloads.sourceforge.net/project/cellsnp/SNPlist/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz",
        "genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz",
        "88e22ea3c4d416aed925e5b3efeb371ef023c954599a325dcb334dffcf6ad707",
    ),
    "maps": (
        "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip",
        "plink.GRCh38.map.zip",
        "521549889b9ce0236142a4fb7db45d3f00035ec645e01465490d55f5b8ef26d6",
    ),
    "beagle_jar": (
        f"{BEAGLE_BASE}/beagle.27Feb25.75f.jar",
        "beagle.27Feb25.75f.jar",
        "7319f4af9638be05c18dcc1bfb8fb41a58a09293507ebf0d54617d0e40df5a70",
    ),
    "bref3_jar": (
        f"{BEAGLE_BASE}/bref3.27Feb25.75f.jar",
        "bref3.27Feb25.75f.jar",
        "6166426f63b2c1cfed9e2cda2f9ba59ef3b3c19fdfc988a9ec6036358457e3f2",
    ),
}
EXPRESSION_PANELS = {
    "b-cells-v1": (
        "https://gitlab.com/api/v4/projects/84721962/packages/generic/expression-panels/v1/b-cells-v1.tar.gz",
        "1fcbdd2528aac0bb95d129d3946caf33b66bfecf12e3ce4a61fb0046314a15d9",
    ),
}
MICROMAMBA = (
    "https://github.com/mamba-org/micromamba-releases/releases/download/2.3.2-0/micromamba-linux-64",
    "ffc3cb8d52d4d6b354bdbb979c407719c485392b74e462cbd50811aa88e58f85",
)
TOOL_SPECS = ("cellsnp-lite=1.2.3", "bcftools=1.22", "openjdk=17")


def cache_root():
    if value := os.environ.get("BARCODECNV_RESOURCE_DIR"):
        return Path(value).expanduser().resolve()
    return (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        / "barcodecnv"
    ).resolve()


def default_config():
    return cache_root() / "hg38/config.json"


def chromosomes(value):
    items = [x.strip().removeprefix("chr") for x in value.split(",")]
    if len(set(items)) != len(items) or not set(items) <= set(CHROMOSOMES):
        raise argparse.ArgumentTypeError(
            "chromosomes must be unique and drawn from 1..22,X"
        )
    return tuple(c for c in CHROMOSOMES if c in items)


def add_arguments(parser):
    parser.add_argument("--genome", choices=["hg38"], default="hg38")
    parser.add_argument(
        "--resource-dir",
        type=Path,
        default=None,
        help="cache root; default BARCODECNV_RESOURCE_DIR or XDG data/barcodecnv",
    )
    parser.add_argument(
        "--chromosomes",
        type=chromosomes,
        default=CHROMOSOMES,
        help="allele resources to provision; default 1..22,X",
    )
    parser.add_argument(
        "--tools",
        choices=["managed", "path"],
        default="managed",
        help="private Bioconda environment (Linux x86_64), or existing tools on PATH",
    )
    parser.add_argument(
        "--expression-panel",
        choices=sorted(EXPRESSION_PANELS),
        default="b-cells-v1",
        help="normal-expression panel (default: b-cells-v1; B-cell samples)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print source URLs and plan without network access or filesystem writes",
    )
    return parser


@contextmanager
def cache_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another setup is using {root}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checked_tools(paths):
    versions = {}
    for name in ("cellsnp-lite", "bcftools", "java"):
        path = paths.get(name)
        if not path or not os.access(path, os.X_OK):
            raise ValueError(
                f"{name} unavailable; use --tools managed or install it on PATH"
            )
        result = subprocess.run(
            [str(path), "-version" if name == "java" else "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        versions[name] = (result.stdout + result.stderr).strip()
    return versions


def ensure_tools(root, mode):
    if mode == "path":
        paths = {
            name: shutil.which(name) for name in ("cellsnp-lite", "bcftools", "java")
        }
        return paths, checked_tools(paths)
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError(
            "automatic native tool installation currently supports Linux x86_64; use --tools path on other Unix platforms"
        )
    manager = download(MICROMAMBA[0], root / "tools/micromamba", sha256=MICROMAMBA[1])
    manager.chmod(0o755)
    prefix = root / "tools/env"
    paths = {
        name: str(prefix / "bin" / name)
        for name in ("cellsnp-lite", "bcftools", "java")
    }
    stamp = root / "tools/installed.json"
    if not stamp.exists() or json.loads(stamp.read_text()).get("specs") != list(
        TOOL_SPECS
    ):
        LOG.info(
            "[setup] Installing private cellSNP-lite, bcftools and Java environment"
        )
        command = "install" if (prefix / "conda-meta/history").exists() else "create"
        subprocess.run(
            [
                str(manager),
                "--no-rc",
                command,
                "--yes",
                "--root-prefix",
                str(root / "tools/mamba"),
                "--prefix",
                str(prefix),
                "--override-channels",
                "-c",
                "conda-forge",
                "-c",
                "bioconda",
                "--strict-channel-priority",
                *TOOL_SPECS,
            ],
            check=True,
        )
        versions = checked_tools(paths)
        packages = [
            json.loads(p.read_text())
            for p in sorted((prefix / "conda-meta").glob("*.json"))
        ]
        write_json(stamp, dict(specs=TOOL_SPECS, versions=versions, packages=packages))
    return paths, checked_tools(paths)


def ensure_maps(archive, root):
    output = root / "chr_in_chrom_field"
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for chrom in CHROMOSOMES:
            name = f"plink.chrchr{chrom}.GRCh38.map"
            # Read only explicit regular members; never extract arbitrary archive paths.
            content = source.read("chr_in_chrom_field/" + name)
            rows = content.decode().splitlines()
            if not rows or any(
                len(row.split()) != 4 or row.split()[0] != "chr" + chrom for row in rows
            ):
                raise ValueError(f"invalid genetic map for chr{chrom}")
            temporary = output / (name + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(output / name)
    return output


def panel_url(chrom):
    suffix = ".v2" if chrom == "X" else ""
    return f"{PANEL_BASE}/1kGP_high_coverage_Illumina.chr{chrom}.filtered.SNV_INDEL_SV_phased_panel{suffix}.vcf.gz"


def convert_panel(source, destination, chrom, tools, converter):
    """Match Julia: biallelic polymorphic SNPs; fixed-ploidy chrX non-PAR."""
    temporary = destination.with_name(destination.name + ".partial")
    log_path = destination.with_name(destination.name + ".log")
    command = [tools["bcftools"], "view", "-v", "snps", "-m2", "-M2", "-c", "1"]
    if chrom == "X":
        command += ["-t", "chrX:2781480-155701382"]
    command += ["-Ov", str(source)]
    java = [tools["java"], "-Xmx4g", "-jar", str(converter)]
    with log_path.open("wb") as log, temporary.open("wb") as output:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log) as filtering:
            try:
                subprocess.run(
                    java, stdin=filtering.stdout, stdout=output, stderr=log, check=True
                )
            except BaseException:
                filtering.terminate()
                raise
            finally:
                filtering.stdout.close()
            if filtering.wait() != 0:
                raise RuntimeError(f"panel SNP filtering failed; see {log_path}")
    if temporary.stat().st_size < 16:
        raise ValueError(f"empty phasing panel; see {log_path}")
    temporary.replace(destination)
    return [command, java]


def ensure_panel(root, chrom, tools, converter):
    directory = root / "1kGP_HC"
    directory.mkdir(exist_ok=True)
    target = directory / f"1kGP_HC_chr{chrom}{'_nonPAR' if chrom == 'X' else ''}.bref3"
    stamp = target.with_name(target.name + ".json")
    if target.exists() and stamp.exists():
        saved = json.loads(stamp.read_text())
        if saved.get("source_url") == panel_url(chrom) and digest(target) == saved.get(
            "sha256"
        ):
            LOG.info("Cached panel: chr%s", chrom)
            return target
        raise ValueError(f"modified panel: {target}; remove it to rebuild")
    if target.exists():
        raise ValueError(f"unverified panel: {target}; remove it to rebuild")
    source = download(
        panel_url(chrom),
        root / "downloads" / Path(panel_url(chrom)).name,
        md5=PANEL_MD5[chrom],
    )
    LOG.info("[setup] Converting chr%s high-coverage panel to Beagle format", chrom)
    commands = convert_panel(source, target, chrom, tools, converter)
    write_json(
        stamp,
        dict(
            source_url=panel_url(chrom),
            source_sha256=digest(source),
            sha256=digest(target),
            converter_sha256=digest(converter),
            commands=commands,
        ),
    )
    source.unlink()  # Only our temporary downloaded source; retain source hash record.
    return target


def ensure_expression_panel(root, name):
    from .reference import read_panel

    url, sha = EXPRESSION_PANELS[name]
    target = root / "expression_panels" / name
    if target.exists():
        panel = read_panel(target)
    else:
        archive = download(url, root / "downloads" / f"{name}.tar.gz", sha256=sha)
        target.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            stage = Path(tmp)
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    path = Path(member.name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or not (member.isfile() or member.isdir())
                    ):
                        raise ValueError("unsafe expression-panel archive member")
                source.extractall(stage, filter="data")
            panel = read_panel(stage)
            if panel.id != name or panel.genome != "hg38":
                raise ValueError("expression panel identity/build mismatch")
            stage.rename(target)
    if panel.id != name or panel.genome != "hg38":
        raise ValueError("expression panel identity/build mismatch")
    return target


def resolve_expression_panel(value, config=None):
    value = Path(value)
    if value.is_dir() or str(value) not in EXPRESSION_PANELS:
        return value
    from .preprocessing import discover_config

    configured = discover_config(config)
    settings = json.loads(configured.read_text()) if configured else {}
    path = Path(
        settings.get("expression_panels", {}).get(
            str(value), cache_root() / "hg38/expression_panels" / value
        )
    )
    if not path.is_dir():
        raise ValueError(
            f"panel {value} is not installed; run barcodecnv setup --expression-panel {value}"
        )
    return path


def setup(args):
    root = (args.resource_dir or cache_root()).expanduser().resolve() / "hg38"
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    genome="hg38",
                    destination=str(root),
                    tools=args.tools,
                    tool_packages=TOOL_SPECS,
                    assets=ASSETS,
                    panels={c: panel_url(c) for c in args.chromosomes},
                    expression_panel=None
                    if not args.expression_panel
                    else EXPRESSION_PANELS[args.expression_panel],
                ),
                indent=2,
            )
        )
        return root / "config.json"
    with cache_lock(root):
        status = root / "setup.json"
        state = dict(
            genome="hg38",
            status="running",
            chromosomes=args.chromosomes,
            tools=args.tools,
        )
        write_json(status, state)
        try:
            tools, versions = ensure_tools(root, args.tools)
            LOG.info("[setup] Downloading hg38 annotation, SNP sites, maps and Beagle")
            paths = {
                key: download(url, root / filename, sha256=sha)
                for key, (url, filename, sha) in ASSETS.items()
            }
            paths["genetic_map"] = ensure_maps(paths.pop("maps"), root)
            converter = paths.pop("bref3_jar")
            for chrom in args.chromosomes:
                ensure_panel(root, chrom, tools, converter)
            paths["phasing_panel"] = root / "1kGP_HC"
            expression = {}
            if args.expression_panel:
                expression[args.expression_panel] = str(
                    ensure_expression_panel(root, args.expression_panel)
                )
            config = root / "config.json"
            previous = json.loads(config.read_text()) if config.exists() else {}
            expression = {**previous.get("expression_panels", {}), **expression}
            available = set(previous.get("chromosomes", [])) | set(args.chromosomes)
            # Revalidate all advertised chromosome resources, including earlier subsets.
            from .preprocessing import resolve_resource

            for chrom in available:
                resolve_resource(paths["genetic_map"], "chr" + chrom, "map")
                resolve_resource(paths["phasing_panel"], "chr" + chrom, "panel")
            write_json(
                config,
                dict(
                    genome="hg38",
                    tools=tools,
                    tool_versions=versions,
                    resources={k: str(v) for k, v in paths.items()},
                    chromosomes=[c for c in CHROMOSOMES if c in available],
                    expression_panels=expression,
                    default_reference_panel="b-cells-v1",
                ),
            )
            state.update(status="complete", config=str(config))
            write_json(status, state)
            LOG.info("[setup] Ready: %s", config)
            if root != default_config().parent:
                LOG.info(
                    "Use --config %s with run/preprocess, or set BARCODECNV_PREPROCESS_CONFIG",
                    config,
                )
            return config
        except BaseException as error:
            state.update(status="failed", error=str(error))
            write_json(status, state)
            raise
