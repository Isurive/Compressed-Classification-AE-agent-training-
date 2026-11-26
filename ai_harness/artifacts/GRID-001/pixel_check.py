"""Check on real ImageNet images that stored JPEG variants decode to the same pixels main.py trains on.

    python -B ai_harness/artifacts/GRID-001/pixel_check.py   (from Final Project, ai360 env)

For 500 random pool images (seed 0) and every grid level, compares
  (a) compress_jpeg.model_image(original, level)        -- main.py's in-memory training input
  (b) the stored file written by build_jpeg_dataset.py    -- what the storage claim counts
as decoded RGB arrays, before resize/crop/normalization (which are identical for both).
"""
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from build_jpeg_dataset import level_to_quality  # noqa: E402
from compress_jpeg import model_image  # noqa: E402

LEVELS = [20, 60, 80, 85, 90, 95, 100]
rows = list(csv.DictReader(open(ROOT / "Datasets/_manifests/imagenet-1k-val.csv", encoding="utf-8")))
random.Random(0).shuffle(rows)
same = differ = 0
worst = []
for r in rows[:500]:
    src = ROOT / "Datasets/imagenet-1k/val" / r["relative_path"]
    stem = Path(r["relative_path"]).with_suffix(".jpg")
    with Image.open(src) as im:
        for level in LEVELS:
            a = np.asarray(model_image(im, level))
            with Image.open(ROOT / "runs/GRID-001/jpeg_variants" / f"q{level_to_quality(level)}" / "val" / stem) as v:
                b = np.asarray(v.convert("RGB"))
            if a.shape == b.shape and np.array_equal(a, b):
                same += 1
            else:
                differ += 1
                diff = np.abs(a.astype(int) - b.astype(int)).max() if a.shape == b.shape else -1
                worst.append((r["relative_path"], level, a.shape, b.shape, int(diff)))
print(f"identical: {same}  different: {differ}  (500 images x {len(LEVELS)} levels)")
for w in worst[:20]:
    print("DIFF", w)
