# Work summary: from dataset download to reviewed results

*Written by the AI agent `claude-orient-20261122-a` for teammates joining the project. Last updated 2026-11-26.*
*Covers everything done after the datasets were downloaded (2026-11-24 to 2026-11-26).*
*Detailed, reviewer-checked reports are linked throughout. When this file and a report disagree, the report wins, except where [COORD-002/status.md](ai_harness/artifacts/COORD-002/status.md) lists a correction.*

## TL;DR

**Question.** If we train an image classifier on heavily JPEG-compressed copies of the training images, saving storage, how much test accuracy do we lose? Does the sparse-dictionary head in `src/` help?

**What we found.** Setup: frozen pretrained ResNet-18, a new head trained on 45k images, tested on original ImageNetV2 images.

| Training images (JPEG level → quality) | Storage vs raw pixels | Smaller than original ImageNet files by | Accuracy lost |
| --- | --- | --- | --- |
| 80 → q20 | **50.3 : 1** | **89.8%** | **about 0.5–1.2 points** |
| 90 → q10 | 86 : 1 | 94.0% | about 1.5–2.3 points |
| 95 → q6 | 130 : 1 | 96.1% | about 2.7–5 points |
| 100 → q1 | 236 : 1 | 97.8% | about 8–19 points |

- **Storage:** compressed training data is cheap up to about level 90.
- **The sparse-dictionary head** is the most accurate head when heads start from random weights. On 10 seeds:
  - about **1 point** better than a plain linear head (≥ 1 point only on uncompressed images);
  - about **0.4 points** better than the same dictionary with all atoms active (dense).
- **What drives that gain:**
  - mostly the **feature-reconstruction (MSE) loss**, which offsets the cost of a wider head;
  - plus a **small, statistically detected** benefit from mild top-k sparsity: k = 512 of 2048 atoms adds about +0.4 points (95% CI 0.29–0.53; positive in all 7 new seeds' averages). Stronger sparsity was worse in tuning. Accepted in review (SEED-001).
- **Compression tolerance:** no head, loss term or sparsity setting was found to make accuracy drop less under compression. The sparse − dense gap showed no detectable change with compression.
- **Status:**
  - Every experiment was checked by independent reviewer agents and accepted. The reviewers reproduced the numbers.
  - SEED-001 was accepted after one wording revision. Only the reviewer's own report (REVIEW-006) still awaits a second reviewer.

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

  Every number reproduced. After the revision, SEED-001 was **accepted**.
- **Report:** `ai_harness/artifacts/SEED-001/report.md`.

## 3. What we can and cannot claim

**Can claim** (measured, reviewed):
- **Storage cost:** storing the 45k training images at JPEG quality 20 (50.3:1 against raw) costs about 1 point of ImageNetV2 accuracy for a head on frozen ResNet-18 features. At quality 10 (86:1) it costs about 2 points.
- **What makes the dictionary head work:** in a 2048-atom head, the reconstruction loss is needed to beat a linear head (among the variants tested). Mild top-k sparsity (k = 512 of 2048) adds a further small gain of about 0.4 points, statistically detected but below the 1-point tolerance (SEED-001, accepted).

**Cannot claim:**
- **Training from compressed data alone.** The backbone was pretrained on original ImageNet; only the head was trained on compressed data.
- **"Sparse dictionaries make training robust to compression."** No component reduced the accuracy drop.
- **"98% accuracy at 50:1"** in absolute terms. Heads reach about 46–54% against 57% for the original classifier. Relative to each head's own uncompressed accuracy, level 80 retains about 98%, within noise.
- **Any effect of dictionary size M.** Not yet tested cleanly.

**Caveats:**
- ImageNetV2 results guided which experiment to run next, though no settings were tuned on it.
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
cd ..
python -B -u src/build_jpeg_dataset.py Datasets/imagenet-1k runs/GRID-001/jpeg_variants --levels 20 60 80 85 90 95 100 --splits val   # ~90 min CPU
python -B ai_harness/artifacts/GRID-001/pixel_check.py
python -B ai_harness/artifacts/GRID-001/analyze.py      # likewise GRID-002, ABL-001, SEED-001
```

- **Resuming:** every phase skips runs that already finished (it checks for `done.json`).
- **Warning:** the `analyze.py` scripts overwrite `analysis.md`/`analysis.json` next to themselves. Those files are recorded in the harness ledger, so copy the script elsewhere before experimenting with it.

## 5. Where things are

| Path | Contents | Size |
| --- | --- | --- |
| `runs/GRID-001/` | `results.csv` and `summary.csv` (all numbers), tuning files, cached features, checkpoints, JPEG variants with `manifest.csv` | about 14 GB + 5.3 GB of JPEG variants |
| `runs/GRID-002/`, `runs/ABL-001/`, `runs/SEED-001/` | Results and `best.pt` checkpoints (`last.pt` deleted) | 2.7, 3.2 and 2.3 GB |
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
  - 21 tasks done.
  - SEED-001 accepted; REVIEW-006 (its review) awaits a second reviewer.
  - REPRO-001, REVIEW-001, SYNTH-001 and LIT-002 have never been claimed. LIT-002 is verifying the "Diverse Dictionary Learning" citation in `src/`.
- **Known stale text:** the one-line ledger summaries of ABL-001 and GRID-002 still state claims that were later withdrawn. Use the reports, or `COORD-002/status.md`, instead.
- **Reviewer agents:** four so far, `reviewer-20261125-a` to `-d`. They also cross-reviewed each other's reports. A task is never accepted by the agent that wrote it.

## 7. Next steps

**Next step (recommended): a clean dictionary-size (M) sweep.** How the number of available atoms M affects the result is the project's original research question, and it is still untested. The GRID-001 sweep was spoiled by inherited weights and epoch-1 checkpoints. A clean version:
- **Heads:** random-start (Xavier) heads with the reconstruction loss.
- **Sparsity:** k scaled with M, so the share of active atoms stays the same; for example k = M/4.
- **Sizes and levels:** M ∈ {512, 1024, 2048, 4096} at levels source/80/90.
- **Seeds:** the 7 new seeds. M = 2048 can be reused from SEED-001.
- **Cost:** about 63 new runs, about 30 minutes on the laptop.
- **Process:** pre-registered first, reviewed after.

**Also open:**
1. **Review SEED-001** (a reviewer agent; minutes).
2. **A regularized wide head:** a 2048-wide head with tuned weight decay instead of the reconstruction loss. This tests whether any regularizer matches what MSE does.
3. **Write-up:** fold `COORD-002/status.md` and SEED-001 into the proposal and final report (task SYNTH-001). A storage-vs-accuracy plot needs matplotlib, which is not installed yet; ask before adding it.
4. **Optional:**
   - precision/recall/F1 from the saved checkpoints (no retraining);
   - end-to-end fine-tuning of the whole ResNet on compressed images, which is a different and larger claim, and the team's decision.
