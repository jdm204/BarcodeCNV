#!/usr/bin/env python3
"""Provision a local tool/resource bundle from existing installations (no downloads)."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resources",
        required=True,
        type=Path,
        help="existing Julia-style hg38 resource directory",
    )
    parser.add_argument("--beagle-jar", required=True, type=Path)
    parser.add_argument(
        "--tool-dir",
        action="append",
        default=[],
        type=Path,
        help="additional executable search directory; repeatable",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / ".preprocessing",
        help="new local bundle directory",
    )
    args = parser.parse_args()
    # Resolve everything before creating anything. No parent Julia checkout is needed.
    resources = {
        name: args.resources / filename
        for name, filename in {
            "genes": "gencode.v44.annotation.gtf.gz",
            "snp_vcf": "genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz",
            "genetic_map": "chr_in_chrom_field",
            "phasing_panel": "1kGP_HC",
        }.items()
    }
    for path in [*resources.values(), args.beagle_jar]:
        if not path.exists():
            parser.error(f"missing resource: {path}")
    tools = {}
    for name in ("cellsnp-lite", "bcftools", "java"):
        matches = [p / name for p in args.tool_dir if (p / name).is_file()]
        found = str(matches[0]) if matches else shutil.which(name)
        if not found:
            parser.error(f"{name} missing; use --tool-dir or install it on PATH")
        tools[name] = Path(found).resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "bin").mkdir()
    (out / "resources").mkdir()
    configured_tools = {}
    for name, path in tools.items():
        link = out / "bin" / name
        link.symlink_to(path)
        configured_tools[name] = str(link)
    configured_resources = {}
    for name, path in resources.items():
        link = out / "resources" / name
        link.symlink_to(path.resolve(), target_is_directory=path.is_dir())
        configured_resources[name] = str(link)
    jar = out / "beagle.jar"
    shutil.copy2(args.beagle_jar, jar)
    configured_resources["beagle_jar"] = str(jar)
    with jar.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    (out / "config.json").write_text(
        json.dumps(
            dict(
                genome="hg38",
                tools=configured_tools,
                resources=configured_resources,
                beagle_sha256=digest,
                installation="Local symlinks to native tools/resources; Beagle jar copied. Re-run setup on a different machine.",
                sources={
                    "cellsnp-lite": "https://github.com/single-cell-genetics/cellsnp-lite",
                    "beagle": "https://faculty.washington.edu/browning/beagle/beagle.html",
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Configured {out}/config.json")


if __name__ == "__main__":
    main()
