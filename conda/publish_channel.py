"""Merge built packages into the persistent Pages tree without replacing releases."""

import hashlib
import shutil
import sys
from pathlib import Path


def publish(output: Path, site: Path):
    packages = sorted((output / "linux-64").glob("barcodecnv-*.conda"))
    packages += sorted((output / "linux-64").glob("barcodecnv-*.tar.bz2"))
    if not packages:
        raise ValueError("No Linux x86_64 BarcodeCNV package was built")
    channel = site / "channel"
    (channel / "noarch").mkdir(parents=True, exist_ok=True)
    (channel / "linux-64").mkdir(parents=True, exist_ok=True)
    (site / ".gitignore").write_text(".cache/\n")
    for package in packages:
        target = channel / "linux-64" / package.name
        if target.exists():
            if (
                hashlib.sha256(target.read_bytes()).digest()
                != hashlib.sha256(package.read_bytes()).digest()
            ):
                raise ValueError(
                    f"Refusing to replace {target.name}; increment the recipe build number"
                )
        else:
            shutil.copy2(package, target)
    (site / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>BarcodeCNV conda channel</title>"
        "<h1>BarcodeCNV conda channel</h1>"
        "<p>Linux x86_64 packages. "
        '<a href="https://github.com/jdm204/BarcodeCNV/blob/main/docs/CONDA.md">'
        "Installation and release instructions</a>.</p>"
        '<p><a href="channel/linux-64/">Browse packages</a></p></html>\n'
    )


if __name__ == "__main__":
    publish(Path(sys.argv[1]), Path(sys.argv[2]))
