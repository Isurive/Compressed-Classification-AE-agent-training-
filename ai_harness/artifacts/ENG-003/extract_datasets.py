"""Unpack the downloaded dataset files into Final Project/Datasets with original image bytes.

    python -I extract_datasets.py imagenetv2 ARCHIVE.tar(.gz) DATASETS_DIR
    python -I extract_datasets.py imagenet-val PARQUET_DIR DATASETS_DIR WNIDS_TXT

imagenetv2   -> DATASETS_DIR/imagenetv2-matched-frequency/<0-999>/<sha1>.jpeg
imagenet-val -> DATASETS_DIR/imagenet-1k/val/<wnid>/<original filename>
Each also writes DATASETS_DIR/_manifests/<name>.csv with class, relative path, bytes and SHA-256.
Manifests live outside the image folders on purpose: src/image_data.py would treat a
manifest.csv inside an image folder as the dataset itself.

Image bytes are copied unchanged (no decode/re-encode), so file sizes are the dataset's
real stored bytes. Refuses to write into an existing non-empty target folder.
"""

from __future__ import annotations

import csv
import hashlib
import sys
import tarfile
from pathlib import Path, PurePosixPath


def _target_dir(root: Path, *parts: str) -> Path:
    target = root.joinpath(*parts)
    if target.exists() and any(target.iterdir()):
        sys.exit(f"{target} already contains files; remove it first to re-extract.")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _safe_name(name: str) -> str:
    if not name or name in {".", ".."} or any(c in name for c in '/\\:'):
        raise ValueError(f"unsafe file or class name: {name!r}")
    return name


def _write_manifest(root: Path, name: str, rows: list[tuple[str, str, int, str]]) -> None:
    out = root / "_manifests" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "relative_path", "bytes", "sha256"])
        w.writerows(rows)
    print(f"manifest: {out} ({len(rows)} rows)")


def imagenetv2(archive: Path, root: Path) -> None:
    target = _target_dir(root, "imagenetv2-matched-frequency")
    rows = []
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) != 3 or not parts[1].isdigit() or not 0 <= int(parts[1]) <= 999:
                raise ValueError(f"unexpected archive member: {member.name}")
            label, filename = _safe_name(parts[1]), _safe_name(parts[2])
            data = tar.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError(f"short read: {member.name}")
            out = target / label / filename
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(data)
            rows.append((label, f"{label}/{filename}", len(data), hashlib.sha256(data).hexdigest()))
    _write_manifest(root, "imagenetv2-matched-frequency", rows)


def imagenet_val(parquet_dir: Path, root: Path, wnids_txt: Path) -> None:
    import pyarrow.parquet as pq

    wnids = wnids_txt.read_text().split()
    if len(wnids) != 1000:
        raise ValueError("expected 1,000 WNIDs")
    target = _target_dir(root, "imagenet-1k", "val")
    rows = []
    files = sorted(parquet_dir.rglob("validation-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no validation parquet files under {parquet_dir}")
    for path in files:
        table = pq.read_table(path, columns=["image", "label"])
        for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
            data, filename = image["bytes"], _safe_name(PurePosixPath(image["path"]).name)
            if not 0 <= label < 1000:
                raise ValueError(f"{filename}: label {label} outside 0-999")
            # HF names files ILSVRC2012_val_<8 digits>_<wnid>.JPEG. Check the embedded WNID
            # against the label, then restore the official name ILSVRC2012_val_<8 digits>.JPEG.
            stem, _, suffix = filename.rpartition(".")
            base, _, embedded = stem.rpartition("_")
            if embedded.startswith("n") and len(embedded) == 9:
                if embedded != wnids[label]:
                    raise ValueError(f"{filename}: label {label} is {wnids[label]}, name says {embedded}")
                filename = f"{base}.{suffix}"
            if not data.startswith(b"\xff\xd8"):
                raise ValueError(f"{filename}: not JPEG data")
            out = target / wnids[label] / filename
            if out.exists():
                raise ValueError(f"duplicate file name {filename}")
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(data)
            rows.append((wnids[label], f"{wnids[label]}/{filename}", len(data), hashlib.sha256(data).hexdigest()))
        print(f"{path.name}: total {len(rows):,} images", flush=True)
    _write_manifest(root, "imagenet-1k-val", rows)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "imagenetv2":
        imagenetv2(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 5 and sys.argv[1] == "imagenet-val":
        imagenet_val(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        sys.exit(__doc__)
