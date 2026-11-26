# ENG-003: ImageNet-1K val and ImageNetV2 into `Datasets/`

**Agent:** `claude-orient-20261122-a`. **Role:** engineer. **Date:** 2026-11-24.
**Status:** Submitted for review. Downloads were approved by the user, and the folder location `Final Project/Datasets` (inside OneDrive) is the user's explicit choice.

## Result

| Folder | Images | Classes | Bytes | Verified |
| --- | --- | --- | --- | --- |
| `Datasets/imagenet-1k/val/<wnid>/ILSVRC2012_val_XXXXXXXX.JPEG` | 50,000 | 1,000 × 50 | 6,706,296,736 | All 50,000 SHA-256s match the manifest. All 50,000 labels match `src/imagenet1k_val_labels.txt.gz`. File names are unique. |
| `Datasets/imagenetv2-matched-frequency/<0-999>/<sha1>.jpeg` | 10,000 | 1,000 × 10 | 1,244,630,473 | All 10,000 SHA-256s match the manifest. Classes map to ImageNet rows 0…999. |

Full provenance (revisions, archive hash, commands, terms note) is in [`Datasets/DATASETS.md`](../../../Datasets/DATASETS.md). Per-image manifests are in `Datasets/_manifests/`. They are deliberately kept outside the image folders, because `src/image_data.py` would treat a `manifest.csv` inside an image folder as the dataset itself.

## Compatibility with the user's code (executed)

- `image_data.sources()` reads both folders. `reference_ids(classes, "auto")` returns 0…999 for both, so the original ImageNet fc and folded initialization line up.
- `python -B main.py train --data ../Datasets/imagenet-1k/val --holdout 0.10 --train-limit 0 --val-limit 0 --batch-size 128 --epochs 10 --plan-only` printed `{"split": "single-pool holdout", "training_images": 45000, "validation_images": 5000, "epochs": 10, "batch_size": 128, "batches_per_epoch": 352, "planned_optimizer_steps": 3520}`. This exactly matches `src/README.md`. No training was run.

## Method

1. I checked gated access with `HfApi().auth_check`. It failed twice because no access request had been submitted. The user then submitted it and the check passed.
2. `hf download` used pinned revisions and wrote to a temporary folder outside OneDrive (session scratchpad). ImageNet val took 349 s; ImageNetV2 took about 60 s.
3. [`extract_datasets.py`](extract_datasets.py), run with `python -I`, writes the original image bytes with no decode or re-encode. It also:
   - checks that every name is safe;
   - for ImageNet, checks that the WNID embedded in each Hugging Face file name matches the label, and restores the official name `ILSVRC2012_val_XXXXXXXX.JPEG`;
   - refuses to write into a non-empty target;
   - writes the manifests.
4. ImageNetV2's `.tar.gz` is actually an **uncompressed tar** (it starts with a PaxHeader, not the gzip magic bytes), so it is opened with `r:*`.
5. I deleted the temporary downloads after verification. `src/` was not modified. All runs used `-B`, and no `__pycache__` was left behind.

Tools: conda env `ai360` with huggingface_hub 2.4.0 and pyarrow 26.0.1. **pyarrow is a new dependency, installed for this task.**

## Notes for whoever uses this next

- **Recompression chain:** both datasets are already JPEGs. Report every compression level's bytes against these stored bytes (`R_existing`: 6.71 GB for the val pool) and against raw RGB. Use `src/build_jpeg_dataset.py` on this folder to get the per-level bytes (ENG-001; its pixels are verified identical to `main.py`'s in-memory path).
- **OneDrive:** about 7.95 GB will sync. A disk-usage scan of the new folder stalled while OneDrive was processing it. If training reads are slow, check that the files are set to "Always keep on this device".
- **Test-set discipline:** ImageNetV2 must not be used for model selection or tuning. It is the independent final test set. Byte-identical overlap with the val pool was not checked; the two were collected separately by construction (Recht et al. 2019).
- **Terms:** ImageNet terms limit use to non-commercial research and education. Do not redistribute these files or commit them anywhere public.

## Unexecuted

- No training or evaluation was run on these images.
- Byte-identical overlap between the two datasets was not checked.
- No image-decoding pass was run over every file. Only hashes and JPEG magic bytes (ImageNet) were checked.
