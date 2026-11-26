# REASON-002: Main experiment plan (GRID-001)

**Agent:** `claude-orient-20261122-a`. **Date:** 2026-11-24. **Status:** Proposed. Nothing in this plan has been run, except the 1,000-image byte sample in §2.
**Builds on:** REASON-001 (design and controls), ENG-001 (byte tool), ENG-003 (data), ENG-004/005 (pilots and timings).

## 1. Question and claim scope

> With a frozen ImageNet-pretrained ResNet-18, how much can the **stored training images** be JPEG-compressed before the accuracy of a newly trained head on **original, independent test images** drops? Does a sparse-dictionary head tolerate stronger compression than a dense-dictionary head or a plain linear head?

**Claim scope:** this concerns the data used to train the **head**. The backbone was pretrained on original ImageNet-train images, so the results do not show learning from compressed data alone (AGENTS.md; REASON-001 §6). The proposal must say this.

## 2. Compression levels and what they mean (measured)

These are measured on 1,000 random ImageNet-val images (seed 0), averaging 206k pixels, with metadata stripped and 4:2:0 subsampling: [byte_sample.txt](byte_sample.txt). Each level is a compression setting; the bytes are the measurement.

| Level (`--compression-level`) | JPEG quality | bpp | R_raw (vs raw RGB) | R_existing (vs stored ImageNet files) | Saving vs stored |
| --- | --- | --- | --- | --- | --- |
| source (original files) | — | 5.03 | 4.8:1 | 1:1 | 0% |
| 20 | 76 | 1.53 | 15.6:1 | 3.3:1 | 69.5% |
| 60 | 39 | 0.82 | 29.3:1 | 6.2:1 | 83.7% |
| 80 | 20 | 0.50 | 47.9:1 | 10.0:1 | 90.0% |
| 85 | 15 | 0.40 | **59.7:1** | 12.5:1 | 92.0% |
| 90 | 10 | 0.29 | 82.1:1 | 17.2:1 | 94.2% |
| 95 | 6 | 0.19 | 124.6:1 | 26.1:1 | 96.2% |
| 100 | 1 | 0.10 | 228.6:1 | 47.9:1 | 97.9% |

**Proposed grid: source, 20, 60, 80, 85, 90, 95, 100 (8 levels).** The user's 50:1-against-raw target falls between levels 80 and 85. Final numbers come from encoding the full 50k pool with `build_jpeg_dataset.py`, not from this sample.

## 3. Models (3 heads, everything else identical)

| Name | How | Role |
| --- | --- | --- |
| `sparse` | `main.py` defaults: M = 2048, k = 32, CE + MSE + 0.1·dep + 0.05·coh | Proposed method |
| `dense` | the same with `--k 2048` | Isolates top-k sparsity |
| `linear` | `baseline_head.py`, Linear(512, 1000), CE | Conventional baseline |

Shared settings: frozen ResNet-18 (IMAGENET1K_V1); 10 epochs; batch 128; Adam lr 1e-3; `--encoder-init he --fc-init xavier`; the same seeds.

## 4. Data, selection and test protocol (fixes the pilot's selection bias)

- **Training and selection:** `Datasets/imagenet-1k/val` with `--holdout 0.10` gives 45,000 train / 5,000 validation, using `training_splits(…, seed)`. Train and validation images both use the condition's compression level, because only compressed data would exist. `best.pt` is chosen on this 5k validation set (top-1, then CE), as in `main.py`.
- **Test, primary:** **ImageNetV2 matched-frequency at `source`** (10,000 original images, collected separately). This is never used for any choice. It is evaluated once, after all training is frozen.
- **Test, secondary:** ImageNetV2 at the same level as training, reported as deployment robustness and labeled separately.
- **Diagnostic only:** the 5k validation scores. These are biased by selection.
- **Seeds:** 42, 43, 44. In `main.py` the seed sets **both** the split and the initialization. All models and levels within a seed share the identical split (confirmed by digest), so comparisons are paired. The test set is the same for every run.

## 5. Byte accounting (storage claim)

1. Run `build_jpeg_dataset.py Datasets/imagenet-1k/val <variants> --levels 20 60 80 85 90 95 100` once on the full 50k pool. Output: stored files, `manifest.csv` (per-image bytes) and `summary.json`.
2. For each seed, sum the bytes of **its 45k training images** at each level using that manifest. Report total bytes, bpp, R_raw, R_existing, and the saving against the 6.71 GB original pool. State whether the 5k validation images are counted.
3. **Check:** for 500 real images × all levels, the decoded stored-variant pixels must equal the pixels from `main.py`'s in-memory path. This was already shown 80/80 on synthetic images in ORIENT-002. This justifies training in memory while claiming the stored bytes.
4. Disk: about 4.7 GB of variant images, estimated from the sample. Keep the manifests and summary. The images can be deleted after measurement, or kept for a "train from stored files" spot-check.

## 6. Execution (a new orchestrator, no edits to the user's files)

A new `src/run_grid.py`:
- For each level, extract the frozen 512-d features of the **whole 50k pool once** (about 3–4 min/level, from ENG-004/005 timings) into a `FoldFeatureCache`.
- Then, for each seed, train `sparse` and `dense` through the user's `sparse_workflow.train(…, partitions=…, fold_cache=…)` (the existing cross-validation path) and train `linear` on the same cached tensors.
- Extract ImageNetV2 features once at `source` and once per level (about 45 s each). Score every `best.pt` on them.
- Write one `results.csv` row per (model, train level, seed, eval set, eval level): top-1, top-5, CE, best epoch, and split digest. Add the bytes table and plots.

| Cost | Estimate |
| --- | --- |
| Pool feature extraction | 8 levels × about 4 min ≈ 32 min |
| ImageNetV2 features | 9 × about 45 s ≈ 7 min |
| Head training | 8 levels × 3 models × 3 seeds = **72 runs** × about 40 s ≈ 50 min |
| Byte measurement (CPU; can overlap) | about 30 min |
| **Total** | **about 1.5–2 h on the laptop** |

Run outputs: `best.pt` for sparse/dense is about 57 MB because it includes the backbone, and `last.pt` is about 97 MB. Across 48 dictionary runs that is about 7 GB.

## 7. Analysis (pre-declared before seeing test results)

- **Primary plot:** ImageNetV2-source top-1 (mean ± sd over 3 seeds) against training-set bytes (log scale), one line per model.
- **Drop:** `drop(m, L) = A(m, source) − A(m, L)`. **Interaction:** `drop(linear, L) − drop(sparse, L)`, and the same against dense. Show absolute accuracies alongside, as REASON-001 §5 requires.
- **Tolerance (proposed default):** a level "retains accuracy" for a model if its mean is within **1.0 percentage point** of the same model trained on source. Report the strongest tested level that passes, and its measured storage.
- The 98%-at-50:1 target is not reachable in absolute terms: the original fc reaches only 70% here. Report it as unattainable in this setting rather than redefining it.

## 8. Decisions for the user (defaults in bold)

1. **Tune first, or run with the pilot settings?** **Recommended: a short validation-only tuning step first** (about 15 min). Try sparse k ∈ {32, 128, 512} and `--fc-init` {xavier, auto}, and linear weight decay {0, 1e-4, 1e-3}, at levels source and 90. Freeze the best per model, then run the grid. Each model gets a comparable search budget. The alternative is to run now with the pilot settings, accepting that k = 32 looked weak (ENG-005).
2. **Where large outputs go:** **outside OneDrive, `C:\Users\ericw\ai360_work\` (variants and run folders, about 12 GB)**, with results CSVs, plots and manifests copied into the project. The alternative is `Final Project\runs\`, which syncs to OneDrive.
3. **Levels:** **the 8 above**, or fewer to save time.
4. **Tolerance:** **1.0 pp**, or 2.0 pp.
5. **Optional phase 3, about 35 min:** the original dictionary-size question. Sweep M ∈ {512, 1024, 2048, 4096} at the chosen k, levels {source, 80, 90, 95}, 3 seeds, reusing the cached features.

## 9. Risks

- Sparse/dense training was unstable late in the pilot (dense fell 6 points at epoch 9). `best.pt` selection guards against this. Report the best epochs.
- Three seeds give only rough error bars; describe differences under about 1 pp as inconclusive.
- CPU contention if byte measurement overlaps extraction; run them one after the other if timings matter.
- Some ImageNet-val images are CMYK or grayscale. `_jpeg_pixels` handles these, and both paths prepare pixels identically.
