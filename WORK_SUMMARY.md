# Work summary: from dataset download to reviewed results

*Written by the AI agent `claude-orient-20261122-a` for teammates joining the project. Last updated 2026-11-26.*
*Covers everything done after the datasets were downloaded (2026-11-24 to 2026-11-26), up to the learning-rate checks (LR-001, LR-002) and the confirmatory re-test at tuned learning rates (RETEST-001).*
*Detailed, reviewer-checked reports are linked throughout. When this file and a report disagree, the report wins, except where [COORD-002/status.md](ai_harness/artifacts/COORD-002/status.md) lists a correction.*

## TL;DR

**Question.** If we train an image classifier on heavily JPEG-compressed copies of the training images, saving storage, how much test accuracy do we lose? Does the sparse-dictionary head in `src/` help?

**What we found.** Setup: frozen pretrained ResNet-18, a new head trained on 45k images, tested on original ImageNetV2 images.

| Training images (JPEG level → quality) | Storage vs raw pixels | Smaller than original ImageNet files by | Accuracy lost |
| --- | --- | --- | --- |
| 80 → q20 | **50.3 : 1** | **89.8%** | **mean 0.5–0.85 points** at tuned lrs (RETEST-001; 95% CIs up to 1.17) |
| 90 → q10 | 86 : 1 | 94.0% | 1.5–2.2 points at tuned lrs (RETEST-001) |
| 95 → q6 | 130 : 1 | 96.1% | about 2.7–5 points |
| 100 → q1 | 236 : 1 | 97.8% | about 8–19 points |

- **Storage:** compressed training data is cheap up to about level 90.
- **The sparse-dictionary head does not beat a properly tuned linear head** (RETEST-001, confirmatory, ImageNetV2, seeds 45–51, every head at a tuned lr and 20 epochs, accepted in review):
  - sparse (2048 atoms) − linear: **+0.02 points**, 95% CI [−0.25, +0.29]: no detected difference;
  - sparse − dense (same width, all atoms active): **+0.09 points**: too small to matter, and not significant after correcting for 12 tests;
  - the **4096-atom head** (k = 1024) is the most accurate head: **+0.63** over the 2048-atom head and **+0.65** over linear, in all 7 seeds, but below the 1-point tolerance. M, k and parameter count double together, so the cause can't be pinned down.
- **Why this differs from earlier results:** the earlier "sparse beats linear by about 1 point" and "sparsity adds about 0.4 points" (GRID-002, SEED-001) were measured at Adam lr 1e-3, which was wrong for every head; the linear head was under-tuned by about 2 points (LR-001, LR-002). Those measurements are correct as measured, but the head claims are **superseded**.
- **Compression tolerance:** no head makes accuracy drop less under compression. At level 90 the dictionary heads lose 0.5–0.7 points *more* than linear (exploratory), so their edge over linear shrinks toward level 90 (sparse M4096 +0.29 points, sparse M2048 −0.33).
- All test results are conditional on the one fixed ImageNetV2 test set (10,000 images) and on frozen pretrained ResNet-18 features.
- **Status:**
  - Every experiment was checked by independent reviewer agents and accepted. The reviewers reproduced the numbers.
  - SEED-001, MSWEEP-001 (the M sweep), LR-001 and LR-002 (the lr checks), RETEST-001 (the re-test) and their reviews are accepted (REVIEW-010 was cross-reviewed and accepted by `reviewer-20261126-l`).

## 1. Setup you need to reproduce anything

| Item | Detail |
| --- | --- |
| Hardware used | RTX 3080 Ti Laptop GPU (16 GB), i9-12900H, 32 GB RAM. The laptop is enough. |
| Python env | conda env `ai360`: Python 3.12, torch 2.11.0+cu128, torchvision 0.26.0, Pillow 13.0.0. Path on this machine: `C:\Users\ericw\miniconda3\envs\ai360\python.exe`. |
| Run style | Always `python -B ...`, so no `__pycache__` lands in `src/`. Most scripts run from `src/`. |
| Training/validation data | `Datasets/imagenet-1k/val/<wnid>/*.JPEG`: the 50,000 ImageNet-1K validation images, used as our training pool. |
| Test data | `Datasets/imagenetv2-matched-frequency/<0-999>/*.jpeg`: 10,000 ImageNetV2 images, **test only, never used for tuning or checkpoint choice**. |
| Data notes | `Datasets/DATASETS.md` (sources, Hugging Face revisions, hashes). ImageNet is gated on Hugging Face; each person must accept its non-commercial terms. **Do not upload the images anywhere.** |
| Code repository | <https://github.com/Isurive/Compressed-Classification-AE-agent-training-> (public). Datasets, checkpoints and cached features are git-ignored. Small result tables, reports and code are included. |

Your own code is in `src/`: `main.py`, `sparse_workflow.py`, `image_data.py`, `compress_jpeg.py` and `resnet18_sparse_ae.py`. It was **not modified**; its hashes were checked after every experiment. Everything the agent added is in **new** files:

| New file | What it does |
| --- | --- |
| `src/build_jpeg_dataset.py` | Writes compressed copies of a dataset to disk and records the exact byte size of every file, with headers counted separately, in `manifest.csv` and `summary.json`. |
| `src/baseline_head.py` | A plain linear classifier on the same frozen features: the conventional baseline. |
| `src/run_grid.py` | Main experiment runner. Caches features once per compression level, then tunes, trains and tests every head. |
| `src/run_grid_xavier.py` | The same grid, with every head starting from random (Xavier) weights. |
| `src/run_ablation.py` | Turns individual loss terms off to find what makes the sparse head work. |
| `src/run_seeds.py` | Repeats the key comparison on 7 new seeds. |
| `src/run_msweep.py` | Dictionary-size sweep at a fixed active fraction (k = M/4). |

**How "dictionary learning" appears in the code.**
- `SparseDictionaryAE` in `resnet18_sparse_ae.py` learns M = 2048 unit-length **atoms**: the dictionary.
- Each image's 512-number ResNet feature is rebuilt as a sparse combination of those atoms. Only the top-k atoms are kept (k = 512 in our tuned runs).
- The reconstruction (MSE) loss is what trains the atoms.
- The classifier reads the sparse code.

## 2. Timeline of what was done

### Step 1: first real training run and timing (ENG-004, 2026-11-24)
- **Run:** `main.py train` with the README's 45k/5k settings: k = 32 of M = 2048 atoms, JPEG level 20, 10 epochs.
- **Timing:** feature extraction runs at about 330 images/s and is limited by the CPU. One run takes about 3.5 minutes.
- **Caveat (added in review):** its "held-out" scores used the same 5k images that picked the best checkpoint, so they are slightly optimistic.
- **Report:** `ai_harness/artifacts/ENG-004/report.md`.

### Step 2: a baseline and a dense control (ENG-005)
- **Added:** a linear baseline (`baseline_head.py`) and a dense control (`main.py --k 2048`, all atoms active).
- **Single-seed pilot, same 5k images:**

  | Head | Top-1 |
  | --- | --- |
  | sparse k = 32 | 57.6% |
  | dense | 63.2% |
  | linear | 62.0% |
  | original ImageNet classifier | 70.0% |

- **Caveat:** selection-biased, as above. That is why the real experiments test on ImageNetV2.
- **Report:** `ai_harness/artifacts/ENG-005/report.md`.

### Step 3: experiment plan (REASON-002)
- **Plan:** `ai_harness/artifacts/REASON-002/plan.md`. Approved with three choices: tune first, store outputs in `Final Project/runs`, include a dictionary-size (M) sweep.
- **Fixed in advance:**
  - 8 compression levels: source (original), 20, 60, 80, 85, 90, 95, 100;
  - 3 heads: sparse, dense, linear;
  - seeds 42/43/44;
  - test on original ImageNetV2;
  - a 1.0-point tolerance: only differences ≥ 1 point count as meaningful.

### Step 4: main grid (GRID-001)
- **Ran:**
  - Validation-only tuning: 28 runs. It chose k = 512 for sparse, and chose to start every head **from the original ImageNet classifier's weights**.
  - 72 grid runs.
  - A 36-run M sweep.
  - Byte measurement of all 50k images at 7 levels.
- **Byte result:** level 80 = 50.3:1 against raw pixels (counting each image's real channel count; 50.8 if all were RGB) and 89.8% smaller than the original files. Stored files decode to exactly the pixels we trained on (3,500 of 3,500).
- **Accuracy result:** the linear head was best. No trained head beat the untouched ImageNet classifier (57.3% on ImageNetV2). Because every head started from those weights, this grid mostly measured *how much fine-tuning on compressed data damages a pretrained classifier*.
- **M sweep:** inconclusive. The small dictionaries kept epoch-1 checkpoints.
- **Report:** `ai_harness/artifacts/GRID-001/report.md`.

### Step 5: random-start grid (GRID-002)
- **Ran:** the same 72-run grid, but every head starts from random (Xavier) weights. This is the cleaner test of "learning a head from compressed data". The settings were frozen from the existing validation tuning.
- **Result:** sparse is ahead of linear on every seed, by ≥ 1 point only at source and level 20. Starting from the original classifier's weights was worth 3.3–6.8 points.
- **Report:** `ai_harness/artifacts/GRID-002/report.md`.

### Step 6: ablation (ABL-001)
- **Ran:** 60 runs switching off loss terms, with random-start heads.
- **Result:**

  | Change | Effect |
  | --- | --- |
  | A wide (2048) head with plain cross-entropy | 1.0 point *worse* than linear (overfits 45 images per class) |
  | Adding the feature-reconstruction (MSE) loss | +1.4 (dense) / +2.0 (sparse) points |
  | Dependency/coherence regularizers | no measurable effect |
  | Top-k sparsity | about 0 without MSE, +0.46 with MSE (resolved in Step 9) |

- **Not ruled out:** any regularizer might do what MSE does, since the wide head had no weight decay.
- **Report:** `ai_harness/artifacts/ABL-001/report.md`.

### Step 7: first independent review (REVIEW-002)
- **Process:** a reviewer agent re-derived every table and checked checkpoints, bytes, test-set isolation and split pairing.
- **Withdrawn or softened:**
  - the M-sweep conclusion;
  - "sparse beats linear" as established;
  - "MSE alone explains the gain".
- **Outcome:** all three experiments were accepted after revisions.

### Step 8: second review round, covering every remaining task (REVIEW-003 to REVIEW-005, 2026-11-25)
- **Process:** three more reviewer agents ran in parallel and reviewed the 11 tasks still waiting:
  - the code audits;
  - the byte tool;
  - the environment;
  - the dataset download (sampled 400 files: all hashes match; no ImageNetV2 image is in the training pool);
  - the pilots;
  - the plan;
  - the first review;
  - my results summary.
- **Outcome:** 8 were accepted outright and 3 sent back for small fixes. All three were fixed and accepted. The reviewers then cross-reviewed each other's reports.
- **Fixes that came out of it:**
  - A results summary with a **corrections table**. It covers wrong numbers in already-accepted reports (e.g. 3.5 → 3.3 points; per-seed gaps 0.95/0.90/0.89) and two harness summaries that still state withdrawn claims.
  - `ai_harness/DECISIONS.md` and `RESEARCH_STATUS.md` were updated with a dated "Current state" section. Before this, they still described the project as of 2026-11-22.
  - A small **harness bug fix** (HARNESS-001). `harness.py check` used to fail whenever a later task updated a shared file. It now accepts the latest registered version and still catches tampering.

### Step 9: more seeds (SEED-001, 2026-11-26)
- **Ran:** 7 new seeds (45–51) × 3 heads (sparse and dense with MSE, linear) × levels source/80/90 = 63 runs in 28 minutes. The decision rule was written down before the first run.
- **Pre-registered result (new seeds only; pooled over the 3 levels):**

  | Comparison | Gap | 95% CI | Seeds positive | Verdict |
  | --- | --- | --- | --- | --- |
  | sparse − dense | +0.41 | [0.29, 0.53] | 7/7 (per level 6, 6, 7) | **detected, below 1 point** |
  | sparse − linear | +0.97 | [0.78, 1.15] | 7/7 | detected; ≥ 1 point only on uncompressed images (1.23, borderline) |

- **With all 10 seeds:** the numbers are nearly the same. All 8 pre-registered tests survive a multiple-comparison correction.
- **Exploratory, not pre-registered:**
  - dense − linear is +0.55 points.
  - No change in the sparse − dense gap with compression was detected, consistent with a roughly constant ~0.4-point benefit.
  - The wide heads' advantage over linear narrows toward level 90.
- **First review (REVIEW-006):** *revise*, for wording only:
  - per-level sign counts;
  - marking dense − linear and the compression trend as exploratory;
  - scoping the sparsity claim to k = 512.

  Every number reproduced. After the revision, SEED-001 was **accepted**. A second reviewer then accepted REVIEW-006 itself. It noted one more detail: if all 12 printed tests are corrected together, sparse − dense at L80 only just passes (0.0235 against 0.025). The 8 pre-registered tests pass comfortably.
- **Report:** `ai_harness/artifacts/SEED-001/report.md`.

### Step 10: clean dictionary-size sweep (MSWEEP-001, 2026-11-26)
- **Ran:**
  - Dictionary sizes M = 512, 1024, 2048 and 4096, with 1 in 4 atoms active at every size (k = M/4).
  - Random-start heads with the reconstruction loss.
  - Seeds 45–51; levels source/80/90.
  - 63 new runs; M = 2048 reused from SEED-001.
  - Pre-registered.
- **Interruption:** the first attempt was killed overnight after 57 runs, with no error (the laptop or app shut down). The remaining 6 runs were resumed the next day. The reviewer confirmed there are no duplicate or corrupted runs.
- **Result (pre-registered; pooled over the 3 levels):**

  | M (k) | Accuracy | vs M = 2048 | Verdict |
  | --- | --- | --- | --- |
  | 512 (128) | 44.33% | −2.95 points | real, ≥ 1 point |
  | 1024 (256) | 46.12% | −1.16 points | real, ≥ 1 point |
  | 2048 (512) | 47.28% | — | — |
  | 4096 (1024) | 47.73% | +0.46 points | real, below 1 point |

  The trend is **+1.14 points per doubling of M** (95% CI 1.07–1.20), at every compression level.
- **Important caveats:**
  - M, the number of active atoms and the parameter count (about 1 M to 8 M) all grow together. So this shows "a bigger dictionary at the same sparsity helps", not "more atoms alone helps".
  - **Training becomes unstable as M grows** at the fixed learning rate (1e-3, tuned at M = 2048). The training loss rose partway through 0/21, 0/21, 15/21 and 21/21 runs for M = 512 → 4096. The best-checkpoint selection keeps the numbers valid, but the small gain above 2048 may be held back by the learning rate. This also affects the M = 2048 heads in the earlier experiments. The reviewer found it.
- **Exploratory:**
  - Gains shrink with each doubling (+1.8, +1.2, +0.5), possibly limited by the learning rate.
  - M = 512 is about 2 points below the linear head.
  - No detected change in the M effect with compression.
- **My mistake:** the pre-registration text counted the tests as 10 when its own rule defines 13. The stricter 13 were used, and the same tests come out significant either way.
- **Review (REVIEW-007):** first *revise*: the reviewer added the instability finding and asked for wording fixes. After the revision, MSWEEP-001 was **accepted**.
- **Report:** `ai_harness/artifacts/MSWEEP-001/report.md`.

### Step 11: learning-rate check (LR-001, 2026-11-26)
- **Why:** MSWEEP-001's reviewer found that the large dictionary heads diverge late in training at lr 1e-3.
- **Ran:** a **validation-only** check with no ImageNetV2 data, pre-registered before the first run.
  - Heads: sparse M2048 (k 512), sparse M4096 (k 1024) and dense M2048.
  - lr ∈ {1e-3, 3e-4, 1e-4}; levels source/L090; seeds 42–44.
  - 54 runs in 23 minutes.
  - The 12 repeated lr-1e-3 cells reproduced ABL-001 exactly.
- **Result (pre-registered rule):** lr 1e-3 is too high for all three heads.

  | Head | Best lr tried | Validation gain vs 1e-3 | Unstable runs at 1e-3 → lower lr |
  | --- | --- | --- | --- |
  | sparse M2048 | 3e-4 (1e-4 tied) | +0.33 points (marginal) | 4/6 → 0/6 |
  | sparse M4096 | 1e-4 | +1.11 points | 6/6 → 0/6 |
  | dense M2048 | 1e-4 | +1.12 points | 6/6 → 0/6 |

- **Not finished:** for sparse M4096 and dense, accuracy is still rising at 1e-4, the lowest lr tried (+0.45 and +0.60 points over 3e-4). Their best lr is not located yet.
- **Exploratory (validation only):**
  - At a stable lr the sparse advantage over dense **disappears**: −0.19 points at 1e-4, with a CI that touches 0. So SEED-001's "sparsity adds +0.4 points" is consistent with dense being hurt more by the too-high lr.
  - The M4096 − M2048 gap grows from +0.53 to +1.33 points, so MSWEEP-001 probably understates the large-dictionary gain.
  - The linear head was not tested; it also used lr 1e-3.
- **Review (REVIEW-008):** first *revise*. Every number reproduced exactly, but the reviewer found a wrong median (9.5 → 8.5) and an understated grid-edge caveat. After the revision, LR-001 was **accepted**.
- **Report:** `ai_harness/artifacts/LR-001/report.md`.

### Step 12: learning-rate check, part 2, including the linear head (LR-002, 2026-11-26)
- **Why:** LR-001 left the best lr for two heads at the edge of its grid and never checked the linear head.
- **Ran:** a **validation-only** check, pre-registered.
  - Dictionary heads at lr 1e-4, 3e-5 and 1e-5; linear head at 3e-3, 1e-3, 3e-4 and 1e-4.
  - Every run lasted 20 epochs. The lr is constant and each epoch is reseeded, so epochs 1–10 equal a 10-epoch run; both budgets come from one run.
  - 78 runs in 47 minutes.
  - 24 repeated cells reproduced LR-001 and GRID-002 exactly.
- **Result (pre-registered rule):**

  | Head | Setting to use | vs old setting |
  | --- | --- | --- |
  | sparse M2048 | 3e-4, 10 epochs (1e-4 at 20 epochs is +0.28, just under the bar) | +0.33 (LR-001) |
  | sparse M4096 | 1e-4, 10 epochs | +1.11 (LR-001) |
  | dense M2048 | 1e-4, 10 epochs | +1.12 (LR-001) |
  | **linear** | **3e-4, 20 epochs** | **+2.14 points** |

  - Lower lrs (3e-5, 1e-5) are worse even with 20 epochs, so the dictionary heads' best lr is now located, within 20 epochs.
- **Exploratory (validation; selected on the same seeds):**

  | Gap (points) | Old lr 1e-3 | Equal 10 epochs | Equal 20 epochs |
  | --- | --- | --- | --- |
  | sparse M2048 − linear | +1.89 | +0.66 | +0.36 |
  | sparse M2048 − dense | +0.62 | — | +0.00 |
  | sparse M4096 − sparse M2048 | +0.53 | +1.33 | +1.04 |

  The earlier sparse-over-linear advantage was mostly the linear head's bad lr. The residual of about 0.4–0.7 points is below the 1-point tolerance.
- **Review (REVIEW-009):** first *revise*. The first report called the heads "tied", but that compared 10-epoch dictionary heads with a 20-epoch linear head. The revision added equal-budget numbers and a separate post-hoc script. After the revision, LR-002 was **accepted**.
- **Report:** `ai_harness/artifacts/LR-002/report.md`; post-hoc numbers are in `posthoc.md`.

### Step 13: confirmatory ImageNetV2 re-test at tuned learning rates (RETEST-001, 2026-11-26)
- **Why:** every earlier head comparison used lr 1e-3. LR-001/LR-002 showed on validation that this was wrong for every head, so the head claims needed a fair test.
- **Ran:** pre-registered (design, 12-test family and write-up rule in `task.json`, before any run).
  - 4 heads: sparse M2048 k512, sparse M4096 k1024, dense M2048 (all lr 1e-4) and linear (lr 3e-4); **all 20 epochs**.
  - Levels source/80/90; seeds 45–51 (never used for lr choice; same splits as SEED-001); 84 runs in 48 minutes; ImageNetV2 scored once per run.
- **Result (ImageNetV2-source top-1, mean of 7 seeds):**

  | Head | source | level 80 | level 90 |
  | --- | --- | --- | --- |
  | sparse M2048 | 49.29 | 48.45 | 47.10 |
  | **sparse M4096** | **49.90** | **49.11** | **47.72** |
  | dense M2048 | 49.10 | 48.42 | 47.06 |
  | linear | 48.94 | 48.42 | 47.43 |

  | Pre-registered test (points) | Mean | 95% CI | Verdict |
  | --- | --- | --- | --- |
  | sparse M2048 − linear | +0.02 | [−0.25, +0.29] | no detected difference |
  | sparse M2048 − dense | +0.09 | [+0.00, +0.17] | below 1 point; not significant after Holm correction |
  | sparse M4096 − sparse M2048 | +0.63 | [+0.47, +0.80] | robust (7/7, Holm), below 1 point |
  | sparse M4096 − linear | +0.65 | [+0.45, +0.85] | robust (7/7, Holm), below 1 point |
  | cost of level 80, per head | 0.53–0.85 | upper ends 0.90–1.17 | below 1 point on average |
  | cost of level 90, per head | 1.51–2.19 | all above 1 | a meaningful cost for every head |

  - Checks: split digests match SEED-001; no training instability; no head cut off by the 20-epoch budget (median selected epoch 11–15).
  - Exploratory: tuning raised every head by 1.0–2.0 points over the lr 1e-3 runs (linear the most). At level 90 the dictionary heads lose 0.5–0.7 points more than linear, a pattern already seen at lr 1e-3.
- **Caveats:** ImageNetV2 has been scored many times in the project; the CIs cover seed variation only, on the same 10,000 images (image-sampling error not quantified); M, k and parameters are confounded in the M4096 result; k, init and weight decay were not re-tuned at the new lrs.
- **Review (REVIEW-010, `reviewer-20261126-k`):** first *revise*, wording only; every number, the timing and the code reproduced. The fixes: give the CI with the level-80 headline, don't call the M4096 gain "capacity", don't call the level-90 pattern a reversal, and state that the CIs are conditional on the fixed test images. **Process slip:** I edited the report before the reviewer's verdict was recorded, so the ledger refused it. The task was recovered, reclaimed and resubmitted, with the revise verdict written in the recover note. Revision 1 was then **accepted**.
- **Report:** `ai_harness/artifacts/RETEST-001/report.md`.

## 3. What we can and cannot claim

**Can claim** (measured, reviewed):
- **Storage cost (RETEST-001, tuned lrs):** storing the 45k training images at JPEG quality 20 (50.3:1 against raw pixels; 89.8% smaller than the stored ImageNet files, a saving that includes stripped metadata) costs a mean of 0.5–0.85 points of ImageNetV2 accuracy for a head on frozen ResNet-18 features. The 95% CIs lie below 1 point for the linear and dense heads but reach 1.08–1.17 for the sparse heads. At quality 10 (86:1) it costs 1.5–2.2 points.
- **The sparse-dictionary head is no better than a tuned linear head** (RETEST-001): +0.02 points, no detected difference. Top-k sparsity at equal width adds +0.09 points, below the tolerance and not robust to multiplicity.
- **The 4096-atom head (k = 1024) is the most accurate**, by +0.63 over the 2048-atom head and +0.65 over linear (robust, all 7 seeds), but below the 1-point tolerance. M, k and the parameter count double together, so this does not show that more atoms, or sparsity, are what helps.
- **lr 1e-3 was a poor setting for every head** (LR-001, LR-002, confirmed on ImageNetV2 by RETEST-001): tuning raised every head by 1.0–2.0 points.
- **Superseded (correct as measured at lr 1e-3, but lr-conditional):** "sparse beats linear by about 1 point" (GRID-002, SEED-001), "sparsity adds about 0.4 points" (SEED-001) and "the reconstruction loss is what lets a wide head beat linear" (ABL-001), which also rested on comparisons with the under-tuned linear head.

**Cannot claim:**
- **Training from compressed data alone.** The backbone was pretrained on original ImageNet; only the head was trained on compressed data.
- **"Sparse dictionaries make training robust to compression."** No component reduced the accuracy drop.
- **"98% accuracy at 50:1"** in absolute terms. Heads reach about 46–54% against 57% for the original classifier. Relative to each head's own uncompressed accuracy, level 80 retains about 98%, within noise.
- **"More available atoms, with everything else equal, helps."** MSWEEP-001 shows that a larger dictionary at a fixed active fraction helps (+1.1 points per doubling), but M, k and the parameter count rise together. Results at M ≥ 2048 are conditional on the learning rate.

**Caveats:**
- ImageNetV2 results guided which experiment to run next, though no settings were tuned on it.
- Test CIs cover seed-to-seed variation on one fixed set of 10,000 test images; the image-sampling component was not quantified (REVIEW-010 S-4).
- 29 duplicate images with conflicting labels exist in the ImageNet-val pool. They affect validation only.
- Effects under 1 point are below the project's own tolerance, even when statistically detected.

## 4. How to reproduce (from `Final Project`, `ai360` env)

```powershell
cd src
python -B -u run_grid.py features      # ~31 min: cache ResNet-18 features, 8 levels x (50k pool + 10k test)
python -B -u run_grid.py tune          # ~15 min: validation-only tuning -> runs/GRID-001/tuned.json
python -B -u run_grid.py grid          # ~39 min: 72 runs + ImageNetV2 test
python -B -u run_grid.py msweep        # ~26 min
python -B -u run_grid.py reference     # original ImageNet classifier on the test features
python -B -u run_grid.py summary
python -B -u run_grid_xavier.py freeze; python -B -u run_grid_xavier.py grid; python -B -u run_grid_xavier.py summary   # ~40 min
python -B -u run_ablation.py grid; python -B -u run_ablation.py summary                                               # ~30 min
python -B -u run_seeds.py grid; python -B -u run_seeds.py summary                                                     # ~28 min
python -B -u run_msweep.py grid; python -B -u run_msweep.py summary                                                   # ~25 min
python -B -u run_lrcheck.py grid                                                                                       # ~23 min, validation only
python -B -u run_lrcheck2.py grid                                                                                      # ~47 min, validation only
python -B -u run_retest.py grid                                                                                        # ~48 min, ImageNetV2 re-test
cd ..
python -B -u src/build_jpeg_dataset.py Datasets/imagenet-1k runs/GRID-001/jpeg_variants --levels 20 60 80 85 90 95 100 --splits val   # ~90 min CPU
python -B ai_harness/artifacts/GRID-001/pixel_check.py
python -B ai_harness/artifacts/GRID-001/analyze.py      # likewise GRID-002, ABL-001, SEED-001, MSWEEP-001, LR-001, LR-002, RETEST-001
```

- **Resuming:** every phase skips runs that already finished (it checks for `done.json`).
- **Warning:** the `analyze.py` scripts overwrite `analysis.md`/`analysis.json` next to themselves. Those files are recorded in the harness ledger, so copy the script elsewhere before experimenting with it.

## 5. Where things are

| Path | Contents | Size |
| --- | --- | --- |
| `runs/GRID-001/` | `results.csv` and `summary.csv` (all numbers), tuning files, cached features, checkpoints, JPEG variants with `manifest.csv` | about 14 GB + 5.3 GB of JPEG variants |
| `runs/GRID-002/`, `runs/ABL-001/`, `runs/SEED-001/`, `runs/MSWEEP-001/`, `runs/LR-001/`, `runs/LR-002/`, `runs/RETEST-001/` | Results and `best.pt` checkpoints (`last.pt` deleted). The LR-001/LR-002/RETEST-001 `done.json` files also hold the per-epoch history; RETEST-001's `results.csv` has no lr column, so join on `done.json`. | 2.7, 3.2, 2.3, 3.3, 3.1, 3.2 and 3.7 GB |
| `runs/ENG-004/`, `runs/ENG-005/` | The pilot runs | small |
| `ai_harness/artifacts/<TASK>/` | Report, logs and analysis for every task | small |
| `ai_harness/artifacts/COORD-002/status.md` | **One-page list of accepted results and corrections** | small |

- All of this lives inside OneDrive. Only the small files are on GitHub.
- About 7 GB of `last.pt` files in `runs/GRID-001` could be deleted if space is needed. Nothing has been deleted without approval.

## 6. Team coordination (the harness)

- **What it is:** a local task ledger.
  - Start with `AGENTS.md`, `ai_harness/START_HERE.md` and the "Current state" section of `ai_harness/DECISIONS.md`.
  - `py ai_harness/harness.py status` shows the queue.
  - `py ai_harness/harness.py check` verifies every submitted file's checksum.
- **State on 2026-11-26:**
  - 31 tasks done; nothing in review.
  - SEED-001, MSWEEP-001, LR-001, LR-002, RETEST-001 and REVIEW-006 to REVIEW-010 accepted.
  - **Small known slips in REVIEW-008, found by `reviewer-20261126-h`; none changes a decision:**
    - Of 4 tied-maximum runs, 2 kept the later epoch, not the earlier one; all 4 follow the "top-1 then CE" rule.
    - Dense source − L090 at lr 1e-3 is 8.21 points, not 8.22. This figure was copied into LR-001's report.
    - The sparse M2048 "plateau" holds only within the 10-epoch budget.
  - **Small known slips in LR-002 / REVIEW-009, none changes a decision:**
    - LR-002 says switching the dictionary heads to 20 epochs has a validation "cost of −0.02 to +0.30 pp". It is a *gain* of +0.02 to +0.30 (found by `reviewer-20261126-i`).
    - REVIEW-009's "removes about 1.2–1.5 pp" credits the linear lr fix alone, but the sparse head's lr also changed (found by `reviewer-20261126-j`).
    - REVIEW-009 counted 4 exact-tie windows; there are 2 more in the LR-001 histories.
    - `reviewer-20261126-i` accepted LR-002's revision 46 s after resubmission. It had verified hashes and every post-hoc number against its earlier recompute; that acceptance was not independently re-checked.
  - **RETEST-001 process slip:** the coordinator edited the report before REVIEW-010's *revise* verdict was recorded, so the ledger refused the verdict. The task was recovered (the reason note records the revise), reclaimed, resubmitted and then accepted. Lesson: wait for the reviewer's verdict to be recorded before editing submitted files.
    - More precisely (found by `reviewer-20261126-l`): the edits (analyze.py at 21:55:12 local) came before REVIEW-010 was even submitted (21:56:39 local), not just before the verdict.
    - REVIEW-010 presents the lr 1e-3 cost differences +0.53 and +0.37 as new; they were already in SEED-001's report (from REVIEW-006). Only the sparse M4096 − linear figure (+0.23) is new.
  - REPRO-001, REVIEW-001, SYNTH-001 and LIT-002 have never been claimed. LIT-002 is verifying the "Diverse Dictionary Learning" citation in `src/`.
- **Known stale text:** the one-line ledger summaries of ABL-001 and GRID-002 still state claims that were later withdrawn. Use the reports, or `COORD-002/status.md`, instead.
- **Reviewer agents:** twelve so far, `reviewer-20261125-a` to `-d` and `reviewer-20261126-e` to `-l`. They also cross-reviewed each other's reports. A task is never accepted by the agent that wrote it.

## 7. Next steps

**Next step (recommended): the write-up** (task SYNTH-001). The head question is now settled at the project's level of evidence: the storage result is the main finding, and the sparse-dictionary head is no better than a tuned linear head. Fold `COORD-002/status.md`, RETEST-001 and the superseded claims into the proposal and final report.

**Also open:**
1. **A paired bootstrap over test images** (no training, a few minutes): re-score RETEST-001's 84 `best.pt` files, keep per-image correctness, and add image-sampling error to the CIs (REVIEW-010 S-4). The most fragile result is sparse − dense.
2. **A fixed-k arm and a dense M4096 control** to separate available atoms M, active atoms k and parameter count in the M4096 result: for example k = 256 at M = 1024, 2048 and 4096, at tuned lrs.
3. **A regularized wide head:** a 2048-wide head with tuned weight decay instead of the reconstruction loss. This tests whether any regularizer matches what MSE does.
4. **Plot:** a storage-vs-accuracy plot needs matplotlib, which is not installed yet; ask before adding it.
5. **Optional:**
   - precision/recall/F1 from the saved checkpoints (no retraining);
   - end-to-end fine-tuning of the whole ResNet on compressed images, which is a different and larger claim, and the team's decision.
