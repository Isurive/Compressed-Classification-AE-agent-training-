# ORIENT-002: Audit of the user's expanded `src/` workflow

**Agent:** `claude-orient-20261122-a`. **Role:** engineer. **Date:** 2026-11-23.
**Status:** Submitted for independent review. Read-only. Nothing was installed or trained, and `src/` was not edited.

The user added these files to `src/` on 2026-11-23: `main.py`, `sparse_workflow.py`, `image_data.py`, `README.md`, `imagenet1k_wnids.txt`, and `imagenet1k_val_labels.txt.gz`. They also revised `compress_jpeg.py` and `resnet18_sparse_ae.py`. The user owns `src/` (see DECISIONS.md, "Adopted implementation"). `src/README.md` is the user's own guide; read it for commands.

## 1. What the pipeline does (code facts)

```
source image ─► model_image(level): in-memory JPEG encode+decode (compress_jpeg.py)
            ─► TorchVision ResNet-18 IMAGENET1K_V1 transforms (resize 256, crop 224, normalize)
            ─► frozen ResNet-18 → 512-d pooled feature (optionally cached once in RAM)
            ─► sparse_codes: encoder(x − center) → top-k (k=32) of M=2048, ReLU
            ├─► new fc: Linear(M, classes) → cross-entropy        (kept at deployment)
            └─► decoder: unit-norm dictionary D (+ optional MLP residual) → feature MSE (training only)
```

- **Loss** (`sparse_workflow.py:241-258`): `ce_weight·CE + reconstruction_weight·MSE + 0.1·dependency + 0.05·coherence`. CE and MSE weights default to 1. The reconstruction target is the frozen feature of the *same compressed input* (line 251), so no clean-image target is hidden in it.
- **Deployment** (`LatentClassifier`, lines 72-99): backbone → sparse code → new fc. The decoder is used only in training. `best.pt` is chosen by validation top-1.
- **fc initialization** (lines 175-191): either "folded" from the original ImageNet fc (`W·Dᵀ`, `b + W·c`), Xavier, or the PyTorch default.
- **Data** (`image_data.py`): reads class folders, the flat ILSVRC2012 val folder (using the bundled label table), a CSV, or a Hugging Face dataset. It makes a stratified holdout (lines 187-197) or k-fold split (200-216), class-balanced limits (219-237), and a split fingerprint hash (256-261).
- **The README's main experiment** trains on the ImageNet-1K **val** pool, with 45k train / 5k holdout or 10-fold CV, and suggests ImageNetV2 MatchedFrequency as the independent test set.
- **Same level for training and validation** (`sparse_workflow.py:392-393`): `ImageView` applies `cfg.compression_level` to both splits. `main.py evaluate --levels source 0 20 …` re-scores the same held-out images at other levels.
- **Commands** (`main.py`): `check`, `train`, `crossval`, `resume`, `evaluate`, `predict`, `encode`, `compress`, `convert`, `info`. `check` runs offline on synthetic images with a random backbone and verifies gradients, frozen weights, and the checkpoint round trip.

## 2. Verified by me (read-only)

| Check | Result |
| --- | --- |
| In-memory `model_image(level)` vs. ENG-001 `build_jpeg_dataset.py` stored file at the same level, decoded to RGB | **Identical pixels in all 80 cases** (16 synthetic images × levels 0/20/50/80/100). Stored files from ENG-001 are therefore exactly what this model trains on. |
| Bytes: `write_jpeg` vs. ENG-001 file | `write_jpeg` is 0–200 B larger per image. It keeps the ICC profile and comment; ENG-001 strips them. Pixels are unaffected. |
| `py ai_harness/harness.py check` | ok. ENG-001's submitted `build_jpeg_dataset.py` hash is still valid, and `_jpeg_pixels` still exists in the revised `compress_jpeg.py`. |
| PyTorch available? | **No.** Missing from `py` (3.14), miniconda base, and conda env `dsci430`. |

**Unexecuted:** `python main.py check` and any training or evaluation, because torch is unavailable. No ImageNet data was found in the project folder, and I did not look for it elsewhere. I have not checked whether the README's runtime and memory figures are accurate.

## 3. How it fits the research question (inference, for review)

Strengths:
1. It is a genuine neural dictionary classifier: an explicit D, top-k sparse codes with M and k separate, a reconstruction role, and a classifier reading the sparse code. This meets the AGENTS.md bar.
2. ImageNet images are mostly around 400×350 px, so the ~300 B JPEG header limit from ENG-001 does **not** block 50:1 against raw RGB at native resolution. That target is reachable in size terms, but this has not been measured on real images.
3. Split control is careful: stratified holdout and CV, a fingerprint hash, a fixed seed, and the README already warns that a val-pool holdout is not a test score.

Gaps for the training-storage claim (REASON-001 §2 and §6):
1. **No conventional baseline trained under the same conditions.** `--baseline` uses the original ImageNet fc, which was trained on original ImageNet-train images, so it is not trained on the compressed data. The four-condition comparison needs a non-dictionary head (for example a linear probe on the 512-d feature, or a dense bottleneck of matching width without top-k) trained on the same compressed split with the same budget. Without it, "the dictionary helps" cannot be tested.
2. **The evaluation must also use originals.** For training-storage, the primary score is a model trained at level L and evaluated on **source** (uncompressed) held-out images. That is `evaluate --levels source`. The `best.pt` selection currently uses compressed validation images, which is acceptable if disclosed.
3. **Claim scope.** The backbone is frozen and was pretrained on original ImageNet-train. Supported claim: *the sparse head and classifier can be trained on compressed images*, not that a classifier can learn from compressed data alone. The backbone's own training set also overlaps the label space. A `random_backbone` setting exists (`sparse_workflow.py:49`) but is wired only into `check`.
4. **Storage accounting.** Compression happens in memory, so no stored dataset or byte manifest is produced. Pair each training level with an ENG-001 run on the same split. Pixels are proven identical (§2), so the bytes it reports describe what the model trained on. ImageNet sources are already JPEGs, which makes this a recompression chain: report `R_existing` against the original files as well as `R_raw`.
5. **Level → bytes.** "Compression level" is a quality setting, so the actual bytes and bpp per level must be measured on the chosen split before claiming any ratio.
6. **Dataset size.** ImageNet-1K val is about 6.4 GB and 50k images. That is far more practical than the full train set, but it is still a download that needs the user's go-ahead. ImageNetV2 adds roughly 1.2 GB. **[unverified sizes; check before downloading]**

## 4. Earlier statements this supersedes

The earlier artifacts are under review and locked by their hashes, so their text was not edited. Read them with these corrections:
- ORIENT-001 §4.2 and DECISIONS.md "Adopted implementation": "cross-entropy is off by default" and "frozen original `fc` classifies the reconstruction" describe only the **standalone** `resnet18_sparse_ae.py`. The adopted entry point is now `main.py`, where CE weight = 1, a **new fc reads the sparse code**, and `best.pt` is chosen by validation top-1.
- ORIENT-001 §4.2 item 5 ("ImageNet-1K out of scope"): the workflow now targets the **50k val pool**, not the 1.28M train set.
- ORIENT-001 §1 ("author unknown") was already superseded by COORD-001.
- DECISIONS.md open choices: the dataset now has a user-proposed default (ImageNet val pool, ImageNetV2 test). It is not yet recorded as a decision, so the coordinator should confirm with the user.

## 5. Suggested next tasks (proposals)

1. **ENG-002 (needs the user's OK for torch):** install torch/torchvision in a fresh env with Python 3.10–3.12 and CUDA, then run `python main.py check`.
2. **ENG-003:** add a conventional-head baseline (linear probe and/or dense bottleneck) behind the same `main.py` data and caching path, preferably as a new module and flag, agreed with the user since they own `src/`.
3. **ENG-004 (after a dataset download is approved):** run `build_jpeg_dataset.py` on the 45k/5k split at the planned levels to get real bytes, bpp, and `R_existing` / `R_raw`.
4. LIT-002 still applies: verify the Zheng et al. arXiv 2604.17568 citation now repeated in `src/README.md`.
