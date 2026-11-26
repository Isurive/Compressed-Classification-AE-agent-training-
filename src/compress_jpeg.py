"""Encode an image as JPEG. Higher compression levels give lower JPEG quality.

    python -m pip install Pillow
    python compress_jpeg.py input.png output.jpg --compression-level 70

Omit the output path to create input_compressed.jpg beside the source image.
For animated or multipage images, the tool uses the first frame.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from PIL import Image, ImageOps


def _jpeg_pixels(image: Image.Image) -> Image.Image:
    if image.mode.startswith("I;16"):
        # Scale each 16-bit gray level into JPEG's 8-bit range. A PNG color key
        # must be matched before quantization so nearby gray levels stay opaque.
        levels = [round(sample / 257) for sample in range(65536)]
        if "transparency" in image.info:
            levels[image.info["transparency"]] = 255
        return image.convert("I").point(levels, "L")

    if image.mode == "La":
        image = image.convert("LA")
    palette_alpha = image.palette is not None and image.palette.mode.endswith("A")
    if image.mode in {"LA", "RGBA", "RGBa", "PA"} or "transparency" in image.info or palette_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background.convert("L") if image.mode in {"1", "L", "LA"} else background
    if image.mode in {"RGB", "L", "CMYK"}:
        return image
    return image.convert("L" if image.mode == "1" else "RGB")


def write_jpeg(original: Image.Image, output: BinaryIO, compression_level: int = 20) -> None:
    """Use the same encoder for files and for in-memory model inputs."""
    if not isinstance(compression_level, int) or not 0 <= compression_level <= 100:
        raise ValueError("compression_level must be an integer between 0 and 100.")
    image = _jpeg_pixels(ImageOps.exif_transpose(original))
    profile = original.info.get("icc_profile")
    if image.mode == "RGB" and original.mode not in {"RGB", "RGBA", "RGBa", "P", "PA"}:
        profile = None
    quality = round(95 - 94 * compression_level / 100)
    image.save(output, format="JPEG", quality=quality, optimize=True, icc_profile=profile)


def model_image(original: Image.Image, compression_level: int | None) -> Image.Image:
    """Return RGB pixels after one JPEG round trip. None uses the source pixels."""
    if compression_level is None:
        return _jpeg_pixels(ImageOps.exif_transpose(original)).convert("RGB")
    with BytesIO() as buffer:
        write_jpeg(original, buffer, compression_level)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def compress_jpeg(
    input_path: str | Path,
    output_path: str | Path | None = None,
    compression_level: int = 20,
) -> Path:
    """Write a JPEG and return its path. Level 0 uses quality 95 and 100 uses quality 1."""
    source = Path(input_path)
    target = (Path(output_path) if output_path is not None
              else source.with_name(f"{source.stem}_compressed.jpg"))
    if target.exists() and target.samefile(source):
        raise ValueError("Choose an output path different from the input image.")

    temp_path = None
    try:
        with Image.open(source) as original:
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=target.parent, suffix=".jpg", delete=False) as temp:
                temp_path = Path(temp.name)
                write_jpeg(original, temp, compression_level)
        temp_path.replace(target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress one image to JPEG.")
    parser.add_argument("input", type=Path, help="Source image, such as a PNG, JPEG or WebP.")
    parser.add_argument("output", type=Path, nargs="?", help="Output file (default: <input_stem>_compressed.jpg).")
    parser.add_argument("-c", "--compression-level", type=int, default=20, metavar="0-100",
                        help="0 = highest quality, 100 = strongest compression (default: 20).")
    args = parser.parse_args()
    try:
        target = compress_jpeg(args.input, args.output, args.compression_level)
        print(f"{target}  ({args.input.stat().st_size:,} -> {target.stat().st_size:,} bytes)")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
