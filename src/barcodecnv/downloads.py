"""Atomic, integrity-checked downloads for the explicit resource setup command."""

import hashlib
import http.client
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

LOG = logging.getLogger(__name__)


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download(url, destination, *, sha256=None, md5=None, attempts=3, timeout=60):
    """Resume only an HTTP-validated partial; publish only a verified whole file.

    SHA-256 pins take precedence over recorded first-download hashes. For sources
    without a published checksum, record SHA-256 for subsequent cache checks;
    this detects later corruption, not substitution at the source on first use.
    Caller serializes access to the cache. No credentials are sent.
    """
    target = Path(destination)
    record = target.with_name(target.name + ".download.json")
    if target.is_file():
        saved = json.loads(record.read_text()) if record.exists() else {}
        expected = sha256 or (saved.get("sha256") if saved.get("url") == url else None)
        if (
            expected
            and digest(target) == expected
            and (not md5 or digest(target, "md5") == md5)
        ):
            LOG.info("Cached: %s", target.name)
            return target
        raise ValueError(
            f"unverified or modified cached file: {target}; remove this file to download it again"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial_record = partial.with_name(partial.name + ".json")
    for attempt in range(attempts):
        try:
            saved = (
                json.loads(partial_record.read_text())
                if partial_record.exists()
                else {}
            )
            validator = saved.get("validator") if saved.get("url") == url else None
            offset = partial.stat().st_size if partial.exists() and validator else 0
            headers = {
                "User-Agent": "barcodecnv/0.1 resource-setup",
                "Accept-Encoding": "identity",
            }
            if offset:
                headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
            LOG.info(
                "Downloading %s%s",
                target.name,
                f" (resuming at {offset:,} bytes)" if offset else "",
            )
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if not offset or not match or int(match[1]) != offset:
                        raise ValueError("invalid HTTP range response")
                    total = int(match[3])
                else:
                    offset = 0  # Server ignored Range or validator changed: replace, never append.
                    total = (
                        int(response.headers["Content-Length"])
                        if response.headers.get("Content-Length")
                        else None
                    )
                validator = response.headers.get("ETag") or response.headers.get(
                    "Last-Modified"
                )
                write_json(partial_record, dict(url=url, validator=validator))
                count = offset
                report_at = count + 256 * 1024**2
                with partial.open("ab" if offset else "wb") as out:
                    while chunk := response.read(1024**2):
                        out.write(chunk)
                        count += len(chunk)
                        if count >= report_at:
                            LOG.info(
                                "  %s: %.0f MB%s",
                                target.name,
                                count / 1e6,
                                f" / {total / 1e6:.0f} MB" if total else "",
                            )
                            report_at = count + 256 * 1024**2
                if count == 0 or (total is not None and count != total):
                    raise OSError(f"incomplete download: {count} / {total} bytes")
            actual = digest(partial)
            if (sha256 and actual != sha256) or (md5 and digest(partial, "md5") != md5):
                partial.unlink()
                partial_record.unlink(missing_ok=True)
                raise ValueError(f"checksum mismatch: {url}")
            write_json(
                record,
                dict(
                    url=url,
                    bytes=count,
                    sha256=actual,
                    expected_sha256=sha256,
                    expected_md5=md5,
                ),
            )
            partial.replace(target)
            partial_record.unlink(missing_ok=True)
            return target
        except urllib.error.HTTPError as error:
            if error.code == 416:  # A complete/stale partial is safely restarted.
                partial_record.unlink(missing_ok=True)
            elif error.code < 500 and error.code not in (408, 429):
                raise
            if attempt + 1 == attempts:
                raise
            LOG.warning("Download interrupted (%s); retrying", error)
            time.sleep(2**attempt)
        except (OSError, http.client.HTTPException) as error:
            if attempt + 1 == attempts:
                raise
            LOG.warning("Download interrupted (%s); retrying", error)
            time.sleep(2**attempt)
    raise RuntimeError("download did not complete")
