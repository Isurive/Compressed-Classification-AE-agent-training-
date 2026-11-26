"""Checks for src/build_jpeg_dataset.py on synthetic images (Pillow + numpy only).

    py -B -m unittest discover -s ai_harness/artifacts/ENG-001 -p test_build_jpeg_dataset.py -v
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, JpegImagePlugin

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))

import build_jpeg_dataset as b  # noqa: E402
from compress_jpeg import compress_jpeg  # noqa: E402
from make_synthetic import make_synthetic  # noqa: E402

QUALITIES = [90, 50, 10]


class BuildJpegDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="eng001_"))
        self.src = self.tmp / "source"
        self.sources = make_synthetic(self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, **kwargs):
        out = self.tmp / "out"
        return out, b.build(self.src, out, QUALITIES, **kwargs)

    def _manifest(self, out):
        with open(out / "manifest.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_manifest_matches_files_and_formulas(self):
        out, summary = self._build()
        rows = self._manifest(out)
        self.assertEqual(len(rows), len(self.sources) * len(QUALITIES))
        for r in rows:
            data = (out / r["jpeg_path"]).read_bytes()
            w, h = int(r["width"]), int(r["height"])
            self.assertEqual(len(data), int(r["jpeg_bytes"]))
            self.assertEqual(int(r["header_bytes"]) + int(r["payload_bytes"]), len(data))
            self.assertEqual(data[int(r["header_bytes"]):int(r["header_bytes"]) + 2], b"\xff\xda")
            self.assertAlmostEqual(float(r["bpp"]), 8 * len(data) / (w * h), places=5)
            self.assertEqual(int(r["raw_bytes"]), int(r["channels"]) * w * h)
            self.assertEqual(int(r["source_bytes"]), (self.src / r["source_path"]).stat().st_size)
            with Image.open(out / r["jpeg_path"]) as im:
                self.assertEqual(im.size, (w, h))
                for key in ("icc_profile", "exif", "comment"):
                    self.assertNotIn(key, im.info)
        for q in QUALITIES:
            q_rows = [r for r in rows if int(r["quality"]) == q]
            agg = summary["by_quality"][str(q)]["all"]
            jpeg = sum(int(r["jpeg_bytes"]) for r in q_rows)
            source = sum(int(r["source_bytes"]) for r in q_rows)
            self.assertEqual(agg["total_jpeg_bytes"], jpeg)
            self.assertAlmostEqual(agg["R_existing"], source / jpeg)
            self.assertAlmostEqual(agg["savings_percent_vs_existing"], 100 * (1 - jpeg / source))
            split_total = sum(summary["by_quality"][str(q)][s]["total_jpeg_bytes"] for s in ("train", "test"))
            self.assertEqual(split_total, jpeg)
        totals = [summary["by_quality"][str(q)]["all"]["total_jpeg_bytes"] for q in QUALITIES]
        self.assertGreater(totals[0], totals[1])
        self.assertGreater(totals[1], totals[2])
        self.assertEqual(summary["byte_identical_sources_shared_between_splits"], {})

    def test_modes_orientation_and_subsampling(self):
        out, _ = self._build()
        rows = {(r["source_path"], r["quality"]): r for r in self._manifest(out)}
        oriented = rows[("train/cat/oriented_jpeg_source.jpg", "50")]
        self.assertEqual((oriented["width"], oriented["height"]), ("32", "40"))  # EXIF rotation applied
        self.assertEqual(oriented["source_format"], "JPEG")
        self.assertEqual(rows[("train/dog/gray64.png", "50")]["subsampling"], "none (grayscale)")
        self.assertEqual(rows[("train/dog/gray16.png", "50")]["channels"], "1")
        self.assertEqual(rows[("train/cat/rgba32.png", "50")]["channels"], "3")
        self.assertEqual(rows[("train/dog/palette64.png", "50")]["subsampling"], "4:2:0")
        out2 = self.tmp / "out444"
        b.build(self.src, out2, [50], subsampling=0)
        with Image.open(out2 / "q50/train/cat/rgb32_0.jpg") as im:
            self.assertEqual(JpegImagePlugin.get_sampling(im), 0)

    def test_level_mapping_matches_compress_jpeg(self):
        source = self.src / "train/cat/rgb32_0.png"
        for level in (0, 20, 50, 80, 100):
            quality = b.level_to_quality(level)
            ours = self.tmp / f"ours_{level}.jpg"
            b.encode(b.prepare_pixels(source)[0], ours, quality, 2)
            theirs = compress_jpeg(source, self.tmp / f"theirs_{level}.jpg", level)
            with Image.open(ours) as a, Image.open(theirs) as c:
                self.assertEqual(a.quantization, c.quantization, f"level {level}")

    def test_refuses_nonempty_output_and_output_inside_source(self):
        out, _ = self._build()
        with self.assertRaises(FileExistsError):
            b.build(self.src, out, QUALITIES)
        b.build(self.src, out, QUALITIES, overwrite=True)
        with self.assertRaises(ValueError):
            b.build(self.src, self.src / "encoded", QUALITIES)

    def test_stem_collision_rejected(self):
        Image.new("RGB", (8, 8)).save(self.src / "train/cat/rgb32_0.bmp")
        with self.assertRaises(ValueError):
            self._build()

    def test_cross_split_duplicate_reported(self):
        shutil.copy(self.src / "train/cat/rgb32_0.png", self.src / "test/cat/leaked.png")
        _, summary = self._build()
        self.assertEqual(summary["byte_identical_sources_shared_between_splits"], {"test&train": 1})

    def test_split_selection(self):
        _, summary = self._build(splits=["test"])
        self.assertEqual(summary["n_source_images"], 4)
        self.assertEqual(set(summary["by_quality"]["50"]), {"all", "test"})


if __name__ == "__main__":
    unittest.main()
