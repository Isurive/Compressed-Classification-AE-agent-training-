# Datasets

Local copies of the image data for the AI 360 final project. Created by harness task **ENG-003** (`ai_harness/artifacts/ENG-003/`). The location inside the OneDrive-synced Final Project folder is the user's choice (2026-11-24). Expect OneDrive to upload these files.

Do not edit, re-encode, or add files here. Compressed variants for experiments belong in a separate output folder; `src/build_jpeg_dataset.py` refuses to write inside its source tree. To rebuild a folder, delete it and rerun the extractor.

## Contents

| Folder | Status | Images | Classes | Bytes | Role |
| --- | --- | --- | --- | --- | --- |
| `imagenetv2-matched-frequency/<0-999>/<sha1>.jpeg` | ✅ extracted 2026-11-24 | 10,000 | 1,000 (10 each) | 1,244,630,473 | Independent final **test** set. Never used for training or model selection. |
| `imagenet-1k/val/<wnid>/ILSVRC2012_val_XXXXXXXX.JPEG` | ✅ extracted 2026-11-24 | 50,000 | 1,000 (50 each) | 6,706,296,736 | Training/validation pool (README: 45k/5k holdout or 10-fold CV) |
| `_manifests/*.csv` | one per extracted folder | | | | `class, relative_path, bytes, sha256` for every image |

## Provenance

**ImageNetV2 matched-frequency**
- Source: Hugging Face dataset `vaishaal/ImageNetV2`, file `imagenetv2-matched-frequency.tar.gz`, revision `d626240be2538720e83103a0e1178d24aca8b12c`.
- The archive is an uncompressed tar despite its `.gz` name. Its SHA-256 is `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c`, and its top folder is `imagenetv2-matched-frequency-format-val/`.
- Class folder `N` is ImageNet class index `N` (canonical order). `src/image_data.py` maps these directly. Verified: `reference_ids` = 0…999.
- Paper: Recht et al., "Do ImageNet Classifiers Generalize to ImageNet?" (ICML 2019). Project page: https://github.com/modestyachts/ImageNetV2

**ImageNet-1K validation**
- Source: Hugging Face dataset `ILSVRC/imagenet-1k`, which is gated. The user's account accepted the ImageNet terms of access, which restrict use to non-commercial research and education. Do not redistribute these files.
  - Files: `data/validation-00000…00013-of-00014.parquet` (6.69 GB), revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`.
  - Download: 349 s on 2026-11-24.
- Layout: `imagenet-1k/val/<wnid>/ILSVRC2012_val_XXXXXXXX.JPEG`.
  - The WNID comes from `src/imagenet1k_wnids.txt`, using the label as the row number.
  - Hugging Face stores names as `ILSVRC2012_val_XXXXXXXX_<wnid>.JPEG`. The extractor checks that the embedded WNID equals the label's WNID, then restores the official file name.
  - JPEG bytes are written unchanged.
- Verified:
  - Every SHA-256 matches the manifest.
  - All 50,000 labels agree with the official validation label table (`src/imagenet1k_val_labels.txt.gz`).
  - There are 50 images in each of 1,000 classes.
  - `image_data.reference_ids` = 0…999.
  - `python main.py train --data ../Datasets/imagenet-1k/val --holdout 0.10 --train-limit 0 --val-limit 0 --batch-size 128 --epochs 10 --plan-only` reports 45,000 / 5,000 images, 352 batches/epoch, and 3,520 steps.
- ImageNet-1K val and ImageNetV2 were collected separately (Recht et al. 2019), so the test set does not share images with the training pool by construction. Byte-identical overlap was not checked.

## How it was made (reproducible)

Environment: conda env `ai360` (Python 3.12.16, huggingface_hub 2.4.0, pyarrow 26.0.1). Downloads go to a temporary folder outside OneDrive and are deleted after verification.

```powershell
$hf = "C:\Users\ericw\miniconda3\envs\ai360\Scripts\hf.exe"
$py = "C:\Users\ericw\miniconda3\envs\ai360\python.exe"
$tmp = "$env:TEMP\ai360_hf_dl"
& $hf download vaishaal/ImageNetV2 imagenetv2-matched-frequency.tar.gz --repo-type dataset --revision d626240be2538720e83103a0e1178d24aca8b12c --local-dir "$tmp\imagenetv2"
& $py -I ai_harness\artifacts\ENG-003\extract_datasets.py imagenetv2 "$tmp\imagenetv2\imagenetv2-matched-frequency.tar.gz" Datasets
# Requires gated access to ILSVRC/imagenet-1k for the logged-in account (huggingface.co/settings/gated-repos shows "Accepted"):
& $hf download ILSVRC/imagenet-1k --repo-type dataset --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5 --include "data/validation-*" --local-dir "$tmp\imagenet-1k"
& $py -I ai_harness\artifacts\ENG-003\extract_datasets.py imagenet-val "$tmp\imagenet-1k\data" Datasets src\imagenet1k_wnids.txt
```

## Using it from `src/`

```powershell
conda activate ai360
cd src
python main.py train --data ../Datasets/imagenet-1k/val --holdout 0.10 --train-limit 0 --val-limit 0 --epochs 10 --batch-size 128 --cache memory --workers 4 --out runs/<run>
python main.py evaluate --checkpoint runs/<run>/best.pt --data ../Datasets/imagenetv2-matched-frequency --entire-input --limit 0 --baseline
```

These images are already JPEGs. Any further compression in experiments re-encodes from these files, which is a recompression chain. Report savings against these stored bytes (`R_existing`) as well as against raw RGB.
