# GRID-001: compressed-training grid, results

**Agent:** `claude-orient-20261122-a`. **Date:** 2026-11-24. **Plan:** [REASON-002/plan.md](../REASON-002/plan.md), approved by the user with: tune first, outputs in `Final Project/runs`, include the M sweep.
**Status of evidence:** everything below is a **measured result** from runs on this laptop unless labeled otherwise. Single machine, 3 seeds.

## 0. Headline (read this first)

1. **Accuracy holds well under heavy training-set compression, for every head.** On original ImageNetV2 test images, training the head on level-80 JPEGs costs **0.8–1.2 pp**. Level 80 is quality 20. The 45k training set at that level measures 0.47 bpp, **50.3:1 against raw RGB** and **9.8:1 against the stored ImageNet files**, an 89.8% saving in full-file bytes. Level 90 (quality 10, 86:1 against raw, 94.0% saving) costs about **2 pp**. Level 100 (quality 1) costs 8–11 pp.
2. **The sparse-dictionary head does not beat the plain linear head.** The linear head is **1.5–2.3 pp more accurate at every compression level** (paired, all three seeds). Differences in *drop* between heads are under 0.5 pp up to level 95 and inside seed noise. Sparse vs dense (same dictionary, k = 512 vs 2048) is indistinguishable at every level.
3. Under the pre-declared 1.0 pp tolerance, the strongest passing level is **80 for sparse** (drop 0.81 ± 0.67) and **60 for dense and linear** (linear at 80: 1.21 ± 0.36). That difference is not meaningful: it comes from sparse's lower and noisier starting point, not better retention.
4. **No trained head beats the original ImageNet classifier** (57.28% on ImageNetV2-source), even trained on uncompressed images (best 53.95%). All three tuned heads start from the original fc weights (tuning chose `auto`/`original` over Xavier for every model), and linear heads peak at epoch 1–3. So this grid mainly measures *how much fine-tuning a pretrained head on compressed data damages it*, which is a narrower claim than "learning from compressed data". See section 6.
5. **The M sweep (k = 512 fixed) is inconclusive about M** (revised after REVIEW-002, G1-1). The smaller dictionaries selected **epoch-1 checkpoints**: M = 1024 in 9 of 9 compressed-level runs, and M = 512 at epochs 1–3. Those checkpoints are still close to the inherited ImageNet classifier. M ≥ 2048 selected epochs 4–10. Accuracy differences across M therefore mix dictionary size with how far training moved from the original fc, and the effect of M is **unresolved** under this initialization.

## 1. Claim scope (unchanged from plan)

- Frozen ImageNet-pretrained ResNet-18 (IMAGENET1K_V1). Only the head is trained on compressed data, so this is **not** evidence about learning from compressed data alone.
- **Additional disclosure found during tuning:** all three frozen settings initialize the new classifier from the **original ImageNet fc** (`sparse`/`dense`: `--fc-init auto`, which folds the mapped fc; `linear`: `--fc-init original`). That classifier was trained on original ImageNet-train images. This is extra clean-data information inside every head, and it applies equally to all three models.
- Training-storage claim: train **and** validation images of each condition use that compression level. The primary test is the original (source) ImageNetV2 matched-frequency images, used only after `tuned.json` was frozen.

## 2. What was run

| Phase | Runs | Wall time | Log |
| --- | --- | --- | --- |
| Feature extraction: 8 levels × (50k pool + 10k ImageNetV2) | 16 feature files | 31 min (17:08–17:39) | [features_log.txt](features_log.txt) |
| Tune: seed 42, levels source & 90, validation only | 14 candidates × 2 = 28 | 15 min | [tune_log.txt](tune_log.txt) |
| Grid: 3 models × 8 levels × seeds 42/43/44 | 72 | 39 min | [grid_log.txt](grid_log.txt) |
| M sweep: M ∈ {512, 1024, 4096}, k = 512, levels source/80/90/95, 3 seeds (M = 2048 reused from grid) | 36 | 26 min | [msweep_log.txt](msweep_log.txt) |
| Reference: original ImageNet fc on ImageNetV2 features per level | 8 evals | <1 min | [reference_log.txt](reference_log.txt) |
| Byte measurement: full 50k pool × 7 levels (CPU, overlapped) | 350k files | see [bytes_log.txt](bytes_log.txt) | [bytes_log.txt](bytes_log.txt) |

- Code: [src/run_grid.py](../../../src/run_grid.py) (new orchestrator) using the user's `sparse_workflow.train` for sparse/dense and `src/baseline_head.py` for linear. Analysis: [analyze.py](analyze.py) → [analysis.md](analysis.md), [analysis.json](analysis.json).
- Raw results: `runs/GRID-001/results.csv` (198 rows), `summary.csv`, `tune_results.json`, `tuned.json`, `reference_original_fc.json`. Per-run `best.pt` and `done.json` are under `runs/GRID-001/{tune,grid,msweep}/`. `history.json` exists only for linear runs. For sparse/dense runs the per-epoch history is in the phase logs, and the settings are in `last.pt` (correction, REVIEW-002 G1-4).
- Shared settings: 45,000 train / 5,000 validation from the 50k ImageNet-val pool (`--holdout 0.10`); 10 epochs; batch 128; Adam lr 1e-3; `--encoder-init he`; `best.pt` by validation top-1, then CE.
- **Pairing verified:** every run of a given seed records the same split digest. There is one digest per seed across all 198 result rows.

### Tuning (seed 42, mean validation top-1 over source and level 90; ImageNetV2 not read)

| Model | Candidates | Chosen | Val top-1 (mean / source / L90) |
| --- | --- | --- | --- |
| sparse | k ∈ {32, 128, 512} × fc-init {xavier, auto} | **k = 512, auto** | 62.93 / 66.76 / 59.10 |
| dense | fc-init {xavier, auto} | **auto** | 62.83 / 66.82 / 58.84 |
| linear | wd ∈ {0, 1e-4, 1e-3} × init {xavier, original} | **wd = 0, original** | 63.28 / 67.48 / 59.08 |

- Budgets were unequal (6 / 2 / 6 candidates). The dense control has no k to tune because k = M by definition.
- `auto` beat `xavier` by 2–3.4 pp for every model. Larger k was monotonically better (k = 32: 55.67; 128: 60.67; 512: 62.93).
- The tuned sparse head keeps **512 of 2048 atoms active (25%)**. That is weak sparsity, and it is why it behaves like the dense control. The pilot's k = 32 was the worst setting.

## 3. Primary results: ImageNetV2 at **source**, mean ± sd over seeds 42/43/44 (top-1 %)

| Train level (quality) | sparse M2048 k512 | drop | dense M2048 k2048 | drop | linear | drop | original fc (no training)* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source | 52.07 ± 0.06 | — | 52.20 ± 0.11 | — | **53.95 ± 0.46** | — | 57.28 |
| 20 (q76) | 51.83 ± 0.13 | 0.24 ± 0.07 | 51.93 ± 0.12 | 0.27 ± 0.23 | 53.76 ± 0.65 | 0.19 ± 0.19 | |
| 60 (q39) | 51.52 ± 0.10 | 0.55 ± 0.05 | 51.52 ± 0.12 | 0.68 ± 0.11 | 53.17 ± 0.51 | 0.78 ± 0.49 | |
| 80 (q20) | 51.27 ± 0.62 | **0.81 ± 0.67** | 51.08 ± 0.23 | 1.11 ± 0.32 | 52.74 ± 0.24 | 1.21 ± 0.36 | |
| 85 (q15) | 50.76 ± 0.23 | 1.31 ± 0.28 | 50.59 ± 0.09 | 1.60 ± 0.13 | 52.56 ± 0.10 | 1.39 ± 0.42 | |
| 90 (q10) | 49.94 ± 0.49 | 2.13 ± 0.54 | 50.09 ± 0.46 | 2.10 ± 0.43 | 52.14 ± 0.47 | 1.81 ± 0.77 | |
| 95 (q6) | 48.98 ± 0.25 | 3.10 ± 0.24 | 48.68 ± 0.29 | 3.52 ± 0.19 | 51.25 ± 0.27 | 2.70 ± 0.69 | |
| 100 (q1) | 41.25 ± 0.35 | 10.82 ± 0.29 | 43.20 ± 4.39 | 9.00 ± 4.41 | 46.30 ± 1.10 | 7.65 ± 1.20 | |

\* The original fc row is a single deterministic evaluation of the unmodified ImageNet classifier on source ImageNetV2 images. It is a reference, not a trained condition.

Drops are paired by seed (`A(source, seed) − A(level, seed)`), in pp.

**Tolerance (pre-declared 1.0 pp on the mean drop):**

| Model | Strongest passing level | Drop there | First failing level |
| --- | --- | --- | --- |
| sparse | 80 | 0.81 ± 0.67 | 85 (1.31) |
| dense | 60 | 0.68 ± 0.11 | 80 (1.11) |
| linear | 60 | 0.78 ± 0.49 | 80 (1.21) |

The 80 vs 60 split is within seed noise (sparse's level-80 sd is 0.62 pp). The plan stated that differences under about 1 pp are inconclusive with 3 seeds.

**Interaction, `drop(other) − drop(sparse)`, paired (pp; positive means sparse retained more):**

| Level | vs dense | vs linear | Absolute gap, linear − sparse |
| --- | --- | --- | --- |
| 20 | 0.03 ± 0.26 | −0.05 ± 0.25 | +1.93 |
| 60 | 0.13 ± 0.15 | 0.23 ± 0.53 | +1.65 |
| 80 | 0.31 ± 0.39 | 0.40 ± 0.80 | +1.48 |
| 85 | 0.29 ± 0.16 | 0.08 ± 0.62 | +1.80 |
| 90 | −0.03 ± 0.12 | −0.32 ± 1.20 | +2.20 |
| 95 | 0.42 ± 0.07 | −0.40 ± 0.91 | +2.28 |
| 100 | −1.82 ± 4.16 | −3.18 ± 0.99 | +5.05 |

- Sparse vs dense: the only consistent signal is a small advantage for sparse at level 95 (0.42 ± 0.07 pp), in absolute terms +0.30 pp. This is far below the 1 pp threshold.
- Sparse vs linear: no level where sparse retains reliably more. Linear is ahead in absolute accuracy everywhere.

## 4. Storage (bytes)

**Measured on all 50,000 pool images.** `build_jpeg_dataset.py` wrote 7 stored variants into `runs/GRID-001/jpeg_variants/` with metadata stripped, 4:2:0 subsampling and Pillow 13.0.0. Outputs: `manifest.csv` (per-image full, header and payload bytes) and `summary.json`. The run took 5,443 s on the CPU, overlapped with GPU training. Log: [bytes_log.txt](bytes_log.txt).

Totals for the **45,000 training images of seed 42**, as used for each condition. Seeds 43 and 44 differ by under 0.1% in every column; all three are in [analysis.md](analysis.md).

| Level (quality) | Full-file bytes | bpp | R_raw | R_existing | Saving vs stored | Header share |
| --- | --- | --- | --- | --- | --- | --- |
| source (stored ImageNet JPEGs) | 6,040,232,194 | 4.63 | 5.1 | 1.00 | 0% | — |
| 20 (q76) | 1,889,729,601 | 1.45 | 16.4 | 3.20 | 68.7% | 0.8% |
| 60 (q39) | 1,005,281,955 | 0.771 | 30.8 | 6.01 | 83.4% | 1.5% |
| **80 (q20)** | **615,578,103** | **0.472** | **50.3** | **9.81** | **89.8%** | 2.3% |
| 85 (q15) | 494,880,253 | 0.380 | 62.6 | 12.21 | 91.8% | 2.8% |
| 90 (q10) | 360,039,910 | 0.276 | 86.0 | 16.78 | 94.0% | 3.8% |
| 95 (q6) | 238,198,791 | 0.183 | 130.0 | 25.36 | 96.1% | 5.6% |
| 100 (q1) | 131,331,537 | 0.101 | 235.8 | 45.99 | 97.8% | 9.8% |

- **R_raw** is the uncompressed 8-bit size divided by full-file bytes. Correction (REVIEW-002 G1-2): the uncompressed size counts each image's **actual channel count**, so 899 grayscale pool images count as 1 channel, not 3. That gives 30,963,814,297 B for these 45k images. Counting every image as 3-channel RGB would give R_raw ≈ 50.8 at level 80 instead of 50.3, so the reported figure is the conservative one.
- **R_existing** is the stored source bytes divided by full-file bytes. The variants have metadata stripped (EXIF, ICC and comments) while the source files keep theirs. R_existing and the saving therefore include the bytes removed by stripping, which were not measured separately (REVIEW-002 G1-3).
- **Headers are counted.** Every figure is complete-file bytes, standalone JPEG headers included. The header share is reported separately.
- **Validation images.** The 5,000 validation images of each condition are stored at the same level but are **not** included above. Their totals are in [analysis.md](analysis.md), and their compression ratios are within 0.1 of the training set's.
- **Whole pool.** The 50k pool stores 6,706,296,736 B. At level 80 it drops to 683,922,845 B (R_raw 50.2).
- **The user's 50:1-against-raw target is first met at level 80.** The 1,000-image sample in the plan put it between 80 and 85; the full pool shows level 80 just crosses it.
- Disk used by the variant images: 5,261,061,450 B ≈ 5.3 GB (sum of the 7 variants). They are kept so the "train from stored files" spot-check stays possible.

## 5. M sweep (sparse, k = 512, ImageNetV2-source top-1 %, mean ± sd, paired drop)

| M (atoms) | source | L80 (drop) | L90 (drop) | L95 (drop) |
| --- | --- | --- | --- | --- |
| 512 (= k, so dense) | 50.62 ± 0.57 | 49.43 (1.19 ± 0.68) | 48.47 (2.14 ± 0.78) | 47.80 (2.82 ± 0.76) |
| 1024 | 50.63 ± 0.88 | 50.49 (0.14 ± 1.08) | 50.22 (0.41 ± 1.24) | 49.36 (1.27 ± 1.09) |
| 2048 (grid) | 52.07 ± 0.06 | 51.27 (0.81 ± 0.67) | 49.94 (2.13 ± 0.54) | 48.98 (3.10 ± 0.24) |
| 4096 | **52.21 ± 0.03** | **51.61** (0.59 ± 0.18) | **50.92** (1.28 ± 0.37) | **49.69** (2.52 ± 0.10) |

**Selected (best) epoch per run**, seeds 42/43/44, from `results.csv`:

| M | source | L80 | L90 | L95 |
| --- | --- | --- | --- | --- |
| 512 | 1, 1, 1 | 1, 2, 1 | 3, 1, 2 | 1, 2, 2 |
| 1024 | 5, 1, 6 | 1, 1, 1 | 1, 1, 1 | 1, 1, 1 |
| 2048 | 5, 6, 5 | 6, 6, 7 | 7, 7, 8 | 7, 7, 6 |
| 4096 | 5, 4, 5 | 7, 6, 6 | 5, 6, 7 | 10, 5, 6 |

- **Checkpoint-selection confound (revised; REVIEW-002 G1-1).** With `--fc-init auto`, epoch 1 is still close to the original ImageNet classifier. For M ≤ 1024, validation accuracy *falls* after epoch 1. For example, M1024 L80 seed 42 goes from 62.42% at epoch 1 to 60.40%, and M512 source seed 42 from 64.22% to 61.36%. For M ≥ 2048 it rises, e.g. M4096 L80 seed 42 reaches 65.14%. The small-M rows therefore largely report a lightly fine-tuned original classifier.
- **The effect of M is unresolved.** The earlier wording ("absolute accuracy improves with M") is withdrawn.
- **M = 1024's small drops are not evidence of robustness.** They come from epoch-1 checkpoints, not from a noisy baseline as previously stated.
- **The sparsity level varies with M.** k = 512 is 100% of the atoms at M = 512 (the dense case), 50% at M = 1024, 25% at M = 2048 and 12.5% at M = 4096, so M and sparsity vary together. k was tuned only at M = 2048.
- **A clean M sweep would need** Xavier init (no inherited classifier) and a k that is sparse at every M, e.g. a fixed k/M ratio. Not run.

## 6. Interpretation, caveats and alternative explanations

- **Heads inherit the ImageNet classifier.** With `auto`/`original` init every head starts at the original fc. Linear runs peak at epoch 1–3 (12 of 24 grid runs at epoch 1), and sparse/dense at epochs 5–7. Training on 45 images per class moves the head *away* from a better clean-data classifier. Compression then changes *how* it moves. This explains why all trained heads sit below 57.28%. It may also explain the small compression drops: the inherited weights carry clean-data knowledge. **Unresolved:** how the curves look with Xavier init (no inherited classifier). That is the cleaner "head learned from compressed data only" test; tuning shows it is 2–3 pp lower on validation.
- **Over-fitting:** sparse/dense reach 99.9% train top-1 by epoch 10. The bigger heads (2048-atom dictionary + decoder) have far more parameters than linear (513k) while seeing the same 45k images. That is a plausible reason linear wins.
- **Domain shift:** ImageNetV2 is harder than ImageNet-val (original fc: 57.28% vs about 70% on val). Absolute numbers are lower than validation numbers for that reason, not because of leakage.
- **Matched-level test (secondary, deployment robustness):** if test images are compressed to the same level, accuracy falls much faster: level 90 costs 8.5–9 pp, and level 95 17–18 pp, for every head (see [analysis.md](analysis.md), `v2_matched`). This is a different claim from training-storage savings and is reported separately.
- **The 98%-at-50:1 target** is unattainable here in absolute terms: no head exceeds 54% on ImageNetV2, and the original classifier gets 57%. In relative terms (retention vs the same head on source): level 80, the first level at or above 50:1 against raw (50.3:1), keeps **98.5% (sparse)**, 97.9% (dense) and 97.8% (linear) of source accuracy. Level 85 keeps 97.5%, 96.9% and 97.4%. Sparse passing a "98% relative" line at 50:1 is within seed noise: its level-80 drop sd is 0.67 pp, which is about 1.3% of its accuracy. Linear is still 1.5 pp higher in absolute terms. Do not report this as the sparse head meeting the target.
- **Seeds:** 3 seeds give rough error bars. One dense L100 seed collapsed (sd 4.4 pp).
- **Duplicate pool images (found by REVIEW-002, G1-5):** 29 groups of byte-identical ImageNet-val images carry conflicting labels. 3, 3 and 7 of these groups span train and validation for seeds 42, 43 and 44. This affects only validation-based checkpoint selection, not the ImageNetV2 test (no ImageNetV2 file's sha256 appears in the pool).
- **Tuning budget** was larger for sparse and linear (6 candidates each) than dense (2). Selection used only seed 42 validation; the test set was never read during tuning.

## 7. Checks

- **Pixel equivalence:** [pixel_check.py](pixel_check.py) → [pixel_check_output.txt](pixel_check_output.txt): **3,500 of 3,500 identical** (500 random pool images × 7 levels). The decoded stored variant equals `compress_jpeg.model_image` exactly. Training in memory therefore used the same pixels whose bytes are counted in section 4.
- **Test set isolation:** `cmd_tune` never calls `test_run` or `test_features`. `results.csv` holds only grid and msweep rows: 135 + 63 = 198, matching the expected run × evaluation count.
- User-owned `src/` files: hashes compared with [src_hashes_before.txt](src_hashes_before.txt). Result after all phases: **8 of 8 unchanged** (including `baseline_head.py` and `build_jpeg_dataset.py`).
- No `__pycache__` in `src/` (all runs used `python -B`). Verified: none present.

## 8. Disk use

`runs/GRID-001/`: grid 7.0 GB, msweep 5.0 GB, tune 0.9 GB, features 0.9 GB, plus JPEG variants 5.3 GB (section 4), all inside OneDrive as the user chose. Most of the size is `last.pt` (dictionary runs, about 97 MB each) and `best.pt` (about 57 MB). `last.pt` files in grid/msweep can be deleted to save about 7 GB; the analysis needs only `results.csv` and `done.json`. The dictionary runs' settings live in `last.pt` and the phase logs, so keep the logs if `last.pt` is deleted. **Nothing has been deleted.**

## 9. Recommended next steps

1. **Xavier-init grid** (no inherited classifier), the same 3 models × 8 levels × 3 seeds, about 40 min GPU with the cached features. This turns the result into a "head learned from compressed data" claim and tests whether the sparse head helps when it cannot lean on the original fc.
2. If the sparse method must be defended: a smaller-capacity or regularized sparse head (e.g. stronger weight decay, fewer epochs, or a lower-rank decoder), since overfitting rather than compression is what separates it from linear.
3. Independent review: done in REVIEW-002 (see §10).
4. If M matters for the write-up: run a Xavier-init M sweep with k scaled to M (proposal, not run).

## 10. Revision log

- **2026-11-25, after REVIEW-002** ([report](../REVIEW-002/report.md); verdict *revise*, no blocking findings):
  - G1-1: rewrote headline 5 and §5. The M effect is now marked unresolved, with best epochs per run listed.
  - G1-2: corrected the R_raw definition (actual channel count).
  - G1-3: disclosed that metadata stripping is part of the saving.
  - G1-4: corrected where run histories and settings live.
  - G1-5: disclosed duplicate pool images.
  - No numbers changed and no runs were redone. The reviewer independently reproduced every table.
