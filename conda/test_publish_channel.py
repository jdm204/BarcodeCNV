"""Protect existing releases when merging new channel packages."""

import tempfile
import unittest
from pathlib import Path

from publish_channel import publish


class ChannelTests(unittest.TestCase):
    def test_preserves_old_releases_and_allows_identical_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, site = root / "output", root / "site"
            (output / "linux-64").mkdir(parents=True)
            package = output / "linux-64/barcodecnv-0.1.0-py313_0.conda"
            package.write_bytes(b"first release")
            publish(output, site)
            publish(output, site)
            package.unlink()
            (output / "linux-64/barcodecnv-0.2.0-py313_0.conda").write_bytes(b"next")
            publish(output, site)
            self.assertEqual(
                (site / "channel/linux-64" / package.name).read_bytes(),
                b"first release",
            )
            self.assertEqual(len(list((site / "channel/linux-64").iterdir())), 2)
            self.assertTrue((site / "channel/noarch").is_dir())

    def test_rejects_replacing_a_published_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, site = root / "output", root / "site"
            (output / "linux-64").mkdir(parents=True)
            package = output / "linux-64/barcodecnv-0.1.0-py313_0.conda"
            package.write_bytes(b"original")
            publish(output, site)
            package.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                publish(output, site)
            self.assertEqual(
                (site / "channel/linux-64" / package.name).read_bytes(), b"original"
            )

    def test_requires_a_linux_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "No Linux"):
                publish(root / "output", root / "site")


if __name__ == "__main__":
    unittest.main()
