"""Generate a small deterministic synthetic ImageFolder tree for testing build_jpeg_dataset.py.

    py ai_harness/artifacts/ENG-001/make_synthetic.py OUTPUT_DIR

These images only exercise the code paths (modes, metadata, orientation, sizes).
Their byte rates say nothing about any real dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# A tiny but structurally valid sRGB-like ICC payload is not needed: Pillow stores
# any bytes given as icc_profile, which is all the metadata-stripping test requires.
FAKE_ICC = b"\x00" * 128 + b"synthetic-icc-profile-for-testing"


def _rgb(rng: np.random.Generator, width: int, height: int) -> Image.Image:
    """Smooth color gradient with a random blob and mild noise, so JPEG rates vary by quality."""
    y, x = np.mgrid[0:height, 0:width] / max(width, height)
    cx, cy, r = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.3)
    blob = ((x - cx) ** 2 + (y - cy) ** 2 < r ** 2).astype(float)
    base = np.stack([x, y, 1 - x * y], axis=-1) * 200 + blob[..., None] * rng.uniform(-80, 80, 3)
    noisy = base + rng.normal(0, 8, base.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGB")


def make_synthetic(root: Path, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    written = []

    def save(img: Image.Image, rel: str, **kwargs) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, **kwargs)
        written.append(path)

    for i in range(4):
        save(_rgb(rng, 32, 32), f"train/cat/rgb32_{i}.png")
    rgba = _rgb(rng, 32, 32).convert("RGBA")
    rgba.putalpha(Image.linear_gradient("L").resize((32, 32)))
    save(rgba, "train/cat/rgba32.png")
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise when displayed
    save(_rgb(rng, 40, 32), "train/cat/oriented_jpeg_source.jpg",
         quality=95, exif=exif, icc_profile=FAKE_ICC, comment=b"synthetic comment")

    for i in range(3):
        save(_rgb(rng, 64, 64), f"train/dog/rgb64_{i}.png")
    save(_rgb(rng, 64, 64).convert("L"), "train/dog/gray64.png")
    gray16 = (np.asarray(_rgb(rng, 64, 64).convert("L"), dtype=np.uint16) * 257)
    save(Image.fromarray(gray16), "train/dog/gray16.png")
    save(_rgb(rng, 64, 64).convert("P", palette=Image.Palette.ADAPTIVE, colors=64), "train/dog/palette64.png")

    for label, size in (("cat", 32), ("dog", 64)):
        for i in range(2):
            save(_rgb(rng, size, size), f"test/{label}/test_{size}_{i}.png")
    return written


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_synthetic.py OUTPUT_DIR")
    for p in make_synthetic(Path(sys.argv[1])):
        print(p)
