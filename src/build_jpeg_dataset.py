"""Encode an ImageFolder-style dataset at several JPEG qualities and account for stored bytes.

    python src/build_jpeg_dataset.py SOURCE_ROOT OUTPUT_ROOT --qualities 90 50 20 10
    python src/build_jpeg_dataset.py SOURCE_ROOT OUTPUT_ROOT --levels 0 50 80 --splits train val

SOURCE_ROOT has the layout <split>/<class>/<image>, for example train/cat/0001.png.
OUTPUT_ROOT receives one ImageFolder tree per quality, q<Q>/<split>/<class>/<stem>.jpg,
plus manifest.csv (one row per source image and quality) and summary.json (totals
per quality and split).

Storage protocol (AGENTS.md, ai_harness/artifacts/REASON-001/report.md section 4):
- Every variant is encoded directly from the same source file. No variant is made
  from another variant, so no compression chain arises inside this tool. A source
  that is itself a JPEG is flagged in the manifest (source_format) because it was
  already lossy.
- Pixels are prepared exactly as src/compress_jpeg.py does (EXIF orientation applied,
  alpha flattened onto white, 16-bit gray scaled to 8 bits). Pixels are never resized.
  The recorded width and height are taken after the EXIF orientation is applied, so a
  rotated source reports swapped dimensions.
- Metadata policy "strip": no EXIF, ICC profile, or comment is written. Only Pillow's
  JFIF header remains, and its bytes are counted.
- Chroma subsampling is fixed by --subsampling and is the same for every condition.
  The subsampling actually written is read back from each file.
- jpeg_bytes is the complete file size on disk. header_bytes is everything before
  the first start-of-scan (SOS) marker: JFIF/APP segments, quantization and Huffman
  tables, and the frame header. payload_bytes = jpeg_bytes - header_bytes. Use
  jpeg_bytes for storage claims.
- bpp = 8 * jpeg_bytes / (width * height). raw_bytes = channels * width * height.
  Aggregate ratios are ratios of totals, never means of per-image ratios.

--levels uses the same mapping as src/compress_jpeg.py: quality = round(95 - 0.94 * level).
Requires Pillow. Does not need torch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image, ImageOps, JpegImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_jpeg import _jpeg_pixels  # noqa: E402  (reuse the user's pixel preparation)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
SUBSAMPLING_NAMES = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0", -1: "none (grayscale)"}
CHANNELS = {"L": 1, "RGB": 3, "CMYK": 4}
MANIFEST_FIELDS = [
    "image_id", "split", "class", "source_path", "source_format", "source_mode", "source_sha256",
    "source_bytes", "width", "height", "channels", "raw_bytes", "quality", "jpeg_path",
    "jpeg_bytes", "header_bytes", "payload_bytes", "bpp", "subsampling", "metadata_policy",
]


def level_to_quality(level: int) -> int:
    """Same mapping as src/compress_jpeg.py: level 0 -> quality 95, level 100 -> quality 1."""
    if not 0 <= level <= 100:
        raise ValueError("compression levels must be between 0 and 100")
    return round(95 - 94 * level / 100)


def jpeg_header_bytes(data: bytes) -> int:
    """Return the byte offset of the first SOS marker, i.e. the bytes before entropy-coded data."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file (missing SOI marker)")
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise ValueError(f"malformed JPEG marker at byte {pos}")
        marker = data[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker == 0xDA:  # start of scan
            return pos
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # standalone markers carry no length
            pos += 2
            continue
        pos += 2 + int.from_bytes(data[pos + 2:pos + 4], "big")
    raise ValueError("JPEG file has no SOS marker")


def find_images(source_root: Path, splits: list[str] | None) -> list[tuple[str, str, Path]]:
    """Return sorted (split, class, path) triples. Fails on output-name collisions."""
    split_dirs = ([source_root / s for s in splits] if splits
                  else sorted(p for p in source_root.iterdir() if p.is_dir()))
    found = []
    for split_dir in split_dirs:
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split folder not found: {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            stems: dict[str, Path] = {}
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if path.stem in stems:
                    raise ValueError(f"two source files would both become {path.stem}.jpg: "
                                     f"{stems[path.stem]} and {path}")
                stems[path.stem] = path
                found.append((split_dir.name, class_dir.name, path))
    if not found:
        raise ValueError(f"no images found under {source_root}")
    return found


def prepare_pixels(path: Path) -> tuple[Image.Image, str, str]:
    """Decode once and return metadata-free pixels, the source format, and the source mode."""
    with Image.open(path) as original:
        source_format, source_mode = original.format or "", original.mode
        pixels = _jpeg_pixels(ImageOps.exif_transpose(original))
        pixels.load()
        clean = Image.frombytes(pixels.mode, pixels.size, pixels.tobytes())  # drops all .info
    if clean.mode not in CHANNELS:
        raise ValueError(f"{path}: unexpected mode {clean.mode} after pixel preparation")
    return clean, source_format, source_mode


def encode(pixels: Image.Image, target: Path, quality: int, subsampling: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    options = {"quality": quality, "optimize": True}
    if pixels.mode == "RGB":
        options["subsampling"] = subsampling
    pixels.save(target, format="JPEG", **options)


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]  # noqa: E731
    return {"min": ordered[0], "p05": pick(0.05), "median": statistics.median(ordered),
            "mean": statistics.fmean(ordered), "p95": pick(0.95), "max": ordered[-1]}


def aggregate(rows: list[dict]) -> dict:
    source = sum(r["source_bytes"] for r in rows)
    jpeg = sum(r["jpeg_bytes"] for r in rows)
    header = sum(r["header_bytes"] for r in rows)
    raw = sum(r["raw_bytes"] for r in rows)
    pixels = sum(r["width"] * r["height"] for r in rows)
    return {
        "n_images": len(rows),
        "total_source_bytes": source,
        "total_jpeg_bytes": jpeg,
        "total_header_bytes": header,
        "total_payload_bytes": jpeg - header,
        "header_fraction": header / jpeg,
        "total_raw_bytes": raw,
        "R_existing": source / jpeg,
        "savings_percent_vs_existing": 100 * (1 - jpeg / source),
        "R_raw": raw / jpeg,
        "aggregate_bpp": 8 * jpeg / pixels,
        "aggregate_payload_bpp": 8 * (jpeg - header) / pixels,
        "per_image_bpp": describe([r["bpp"] for r in rows]),
        "per_image_jpeg_bytes": describe([r["jpeg_bytes"] for r in rows]),
    }


def build(source_root: Path, output_root: Path, qualities: list[int], splits: list[str] | None = None,
          subsampling: int = 2, overwrite: bool = False) -> dict:
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("OUTPUT_ROOT must be outside SOURCE_ROOT")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_root} is not empty; choose a new folder or pass --overwrite")
    if not qualities or any(not 1 <= q <= 95 for q in qualities) or len(set(qualities)) != len(qualities):
        raise ValueError("qualities must be distinct integers between 1 and 95")
    images = find_images(source_root, splits)

    rows = []
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    for split, label, path in images:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hashes_by_split[split].add(digest)
        pixels, source_format, source_mode = prepare_pixels(path)
        width, height = pixels.size
        channels = CHANNELS[pixels.mode]
        rel = path.relative_to(source_root).as_posix()
        for quality in qualities:
            target = output_root / f"q{quality}" / split / label / f"{path.stem}.jpg"
            encode(pixels, target, quality, subsampling)
            encoded = target.read_bytes()
            with Image.open(target) as check:
                if check.size != (width, height):
                    raise RuntimeError(f"{target}: size changed to {check.size}")
                written_subsampling = JpegImagePlugin.get_sampling(check)
                if "icc_profile" in check.info or "exif" in check.info or "comment" in check.info:
                    raise RuntimeError(f"{target}: metadata was written despite the strip policy")
            header = jpeg_header_bytes(encoded)
            rows.append({
                "image_id": rel, "split": split, "class": label, "source_path": rel,
                "source_format": source_format, "source_mode": source_mode, "source_sha256": digest,
                "source_bytes": len(data), "width": width, "height": height, "channels": channels,
                "raw_bytes": channels * width * height, "quality": quality,
                "jpeg_path": target.relative_to(output_root).as_posix(), "jpeg_bytes": len(encoded),
                "header_bytes": header, "payload_bytes": len(encoded) - header,
                "bpp": 8 * len(encoded) / (width * height),
                "subsampling": SUBSAMPLING_NAMES.get(written_subsampling, str(written_subsampling)),
                "metadata_policy": "strip",
            })

    split_names = sorted(hashes_by_split)
    duplicates = {f"{a}&{b}": len(hashes_by_split[a] & hashes_by_split[b])
                  for i, a in enumerate(split_names) for b in split_names[i + 1:]
                  if hashes_by_split[a] & hashes_by_split[b]}

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "pillow_version": PIL.__version__,
        "python_version": sys.version.split()[0],
        "qualities": qualities,
        "requested_subsampling": SUBSAMPLING_NAMES[subsampling],
        "metadata_policy": "strip (no EXIF, ICC, or comment; JFIF header kept and counted)",
        "optimize_huffman": True,
        "n_source_images": len(images),
        "byte_identical_sources_shared_between_splits": duplicates,
        "by_quality": {},
    }
    for quality in qualities:
        q_rows = [r for r in rows if r["quality"] == quality]
        summary["by_quality"][str(quality)] = {
            "all": aggregate(q_rows),
            **{split: aggregate([r for r in q_rows if r["split"] == split]) for split in split_names},
        }

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows({**r, "bpp": f"{r['bpp']:.6f}"} for r in rows)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_root", type=Path, help="Folder with <split>/<class>/<image> files.")
    parser.add_argument("output_root", type=Path, help="Empty or new folder for the encoded datasets.")
    rate = parser.add_mutually_exclusive_group(required=True)
    rate.add_argument("--qualities", type=int, nargs="+", metavar="Q", help="Pillow JPEG quality values, 1-95.")
    rate.add_argument("--levels", type=int, nargs="+", metavar="L",
                      help="compress_jpeg.py compression levels, 0-100 (converted to quality).")
    parser.add_argument("--splits", nargs="+", help="Split folders to encode (default: every subfolder).")
    parser.add_argument("--subsampling", choices=["4:4:4", "4:2:2", "4:2:0"], default="4:2:0",
                        help="Chroma subsampling for color images (default: 4:2:0).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Write into a non-empty OUTPUT_ROOT. Same-named files are replaced; other "
                             "files stay and are not in the new manifest, so prefer a fresh folder.")
    args = parser.parse_args()
    try:
        qualities = args.qualities or [level_to_quality(level) for level in args.levels]
        code = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}[args.subsampling]
        summary = build(args.source_root, args.output_root, qualities, args.splits, code, args.overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(f"Encoded {summary['n_source_images']} images at qualities {summary['qualities']}")
    for quality, stats in summary["by_quality"].items():
        a = stats["all"]
        print(f"  q{quality}: {a['total_jpeg_bytes']:,} bytes  R_existing {a['R_existing']:.2f}  "
              f"savings {a['savings_percent_vs_existing']:.1f}%  R_raw {a['R_raw']:.1f}  "
              f"bpp {a['aggregate_bpp']:.3f}  header {100 * a['header_fraction']:.1f}%")
    if summary["byte_identical_sources_shared_between_splits"]:
        print("WARNING: byte-identical source images appear in more than one split:",
              summary["byte_identical_sources_shared_between_splits"])


if __name__ == "__main__":
    main()
