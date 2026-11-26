# ENG-001: Dataset-level JPEG encoding with a byte manifest

**Agent:** `claude-orient-20261122-a`. **Role:** engineer. **Date:** 2026-11-22.
**Status:** Submitted for independent review. Tested on synthetic images only. No dataset was downloaded, nothing was installed, and nothing was trained.

## Deliverables

| File | Purpose |
| --- | --- |
| [`src/build_jpeg_dataset.py`](../../../src/build_jpeg_dataset.py) | New CLI and library. Encodes `<split>/<class>/<image>` trees at several JPEG qualities into `q<Q>/<split>/<class>/<stem>.jpg`, and writes `manifest.csv` and `summary.json`. |
| [`make_synthetic.py`](make_synthetic.py) | Deterministic 16-image synthetic tree covering RGB, RGBA, L, I;16, P, and a JPEG source with EXIF orientation, an ICC profile, and a comment. |
| [`test_build_jpeg_dataset.py`](test_build_jpeg_dataset.py) | 7 unittest cases. |
| [`pilot/`](pilot/) | Synthetic source, outputs at q90/50/20/10/5 (`out/`), the `--levels` run (`out_levels/`), and `cli_output.txt`. |

User files untouched. SHA-256 hashes before and after:
`src/compress_jpeg.py` 674F290B…3C8F61, `src/resnet18_sparse_ae.py` B05E8448…621906. Re-verified at submission (see §5).

## How it meets the storage protocol (AGENTS.md; REASON-001 §4)

- **Reuse:** it imports `_jpeg_pixels` from the user's `compress_jpeg.py`, so pixel preparation is identical. `--levels` uses the same level→quality mapping, and a test confirms the quantization tables match `compress_jpeg()` at levels 0, 20, 50, 80, and 100.
- **No compression chains:** every quality variant is encoded from the source file. A source that was already a JPEG is flagged in the manifest's `source_format` column.
- **Fixed, recorded settings:**
  - Subsampling is fixed by `--subsampling` (default 4:2:0), and the value actually written is read back from each file.
  - Metadata is stripped: no EXIF, ICC profile, or comment. Pixels are rebuilt from raw bytes, so no `.info` carries over. A test confirms the source had all three and the outputs have none.
  - Huffman tables are optimized per file. The Pillow and Python versions are recorded.
- **Identities:** the same image IDs, labels, and splits are kept, and pixels are never resized (a test checks decoded sizes). SHA-256 hashes of the sources detect byte-identical images shared across splits, which is a leakage check. It does not catch near-duplicates.
- **Byte accounting per image:** `jpeg_bytes` is the full file. `header_bytes` is everything before the first SOS marker, and `payload_bytes` is the rest. The manifest also records `bpp = 8·jpeg_bytes/(W·H)` and `raw_bytes = channels·W·H`.
- **Byte accounting in aggregate:** per quality, both overall and per split, the summary reports totals, `R_existing = Σsource/Σjpeg`, savings %, `R_raw`, aggregate bpp, payload-only bpp, header fraction, and bpp and size distributions. All ratios are ratios of totals.
- **Safety:** the tool refuses a non-empty output folder unless `--overwrite` is passed, an output folder inside the source tree, and two source files that would produce the same `.jpg` name.

## Measured result (synthetic images; code-path evidence, not a dataset result)

`py -B -m unittest discover -s ai_harness/artifacts/ENG-001 -p test_build_jpeg_dataset.py -v` → **7 tests OK** (3.6 s).

Average bytes per RGB image from `pilot/out/manifest.csv`:

| Size | Quality | Header B | Payload B | File B | R_raw |
| --- | --- | --- | --- | --- | --- |
| 32×32 | 90 | 310 | 352 | 662 | 4.6 |
| 32×32 | 5 | 274 | 37 | 310 | 9.9 |
| 64×64 | 50 | 298 | 340 | 638 | 19.3 |
| 64×64 | 5 | 275 | 62 | 337 | 36.4 |

**Finding that affects the 50:1 target (measured on synthetic images; header size depends little on content):** a standalone baseline JPEG from this encoder carries about **275–325 bytes of headers and tables**, whatever the quality. A 50:1 ratio against raw RGB allows 61 bytes for a 32×32 image and 246 bytes for 64×64. Both budgets are **smaller than the header alone**. So with complete, independent JPEG files:

- At CIFAR-sized (32×32) or Tiny-ImageNet-sized (64×64) images, a full-file 50:1 ratio against raw RGB cannot be reached. At q5 the synthetic pilot reached about 10:1 at 32×32 and about 36:1 at 64×64.
- **[inference]** Reaching 50:1 with standalone files needs larger native images. The header alone needs at least about 68×68 pixels before any payload. Realistic payloads point to roughly 96–128 px or more, depending on content. This needs measuring on the chosen dataset.
- **[inference]** Another route is to store one shared set of quantization and Huffman tables for the whole dataset (abbreviated JPEG streams) and count those bytes once. That is a legitimate storage format, but it is a different claim from "a folder of ordinary JPEG files" and must be described as such. The Berkeley report's header-excluded rates (LIT-001) are closer to this definition.

## Unexecuted / limits

- Not run on any real dataset, so real per-image rates and attainable ratios are **unmeasured**.
- Not checked: `optimize=False` (standard tables), progressive JPEG, or abbreviated or shared-table streams. Header size under those options is unmeasured.
- No parallelism. **[inference]** At about 1 ms per small encode, CIFAR-10 (60k images × 4 qualities) should take a few minutes, but this is unmeasured.
- `--overwrite` does not delete stale files from earlier runs; the help text says so. A fresh output folder is recommended.
- Duplicate detection is exact-bytes only.
- The tool encodes by quality factor. Targeting per-image byte or bpp budgets (searching for a quality per image) is not implemented.

## Suggested next steps (proposals)

1. **The user picks a dataset with the header finding in mind.** If 50:1 full-file storage stays a goal, prefer native resolution ≥ ~96–128 px (for example a small ImageNet subset at native size), or adopt and disclose a shared-table storage format. Otherwise state the target against `R_existing` or at a lower ratio.
2. Once a dataset is chosen, run this tool on its train/val split to measure the actual quality→bytes curve before fixing the experiment grid.
3. An optional follow-up task could add a shared-table mode and measure how much it saves.

## 5. Completion evidence

Commands run: the unittest command above; `make_synthetic.py pilot/source`; `build_jpeg_dataset.py` with `--qualities 90 50 20 10 5`, then with `--levels 0 50 80 100`, then a deliberate rerun into a non-empty folder, which was refused with exit code 1 as expected. Files written: `src/build_jpeg_dataset.py` and this folder only. All runs used `py -B`, so no `__pycache__` was written to `src/`.
