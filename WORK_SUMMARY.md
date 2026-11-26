# Work summary: from dataset download to reviewed results

*Written 2026-11-25 by the AI agent `claude-orient-20261122-a` for teammates joining the project.*
*Covers everything done after the datasets were downloaded (2026-11-24 to 2026-11-25).*
*Detailed, reviewer-checked reports are linked throughout; when this file and a report disagree, the report wins.*

## TL;DR

**Question.** If we train an image classifier on heavily JPEG-compressed copies of the training images, saving storage, how much test accuracy do we lose? Does the sparse-dictionary head in `src/` help?

**What we found.** Setup: frozen pretrained ResNet-18, 45k training images, original ImageNetV2 test images, 3 seeds.

| Training images (JPEG level → quality) | Storage vs raw pixels | Smaller than original ImageNet files by | Accuracy lost |
| --- | --- | --- | --- |
| 80 → q20 | **50.3 : 1** | **89.8%** | **about 0.5–1.2 points** |
| 90 → q10 | 86 : 1 | 94.0% | about 1.5–2.3 points |
| 95 → q6 | 130 : 1 | 96.1% | 3–5 points |
| 100 → q1 | 236 : 1 | 97.8% | 8–19 points |

- **Storage:** compressed training data is cheap up to about level 90.
- **The sparse-dictionary head:** it beats a plain linear head only when neither starts from the original ImageNet classifier's weights, and then by under 1 point at most compression levels.
- **What drives that gain:** the ablation says it comes mostly from the **feature-reconstruction (MSE) loss**, which offsets the cost of a wider head. Top-k sparsity itself has an unresolved effect, about 0.5 points at most.
- **Compression tolerance:** no head or loss term made accuracy drop less under compression.
- **Status:** all three experiments were checked by an independent reviewer agent and accepted. The reviewer reproduced every number.

## 1. Setup you need to reproduce anything

| Item | Detail |
| --- | --- |
| Hardware used | RTX 3080 Ti Laptop GPU (16 GB), i9-12900H, 32 GB RAM. The laptop is enough. |
| Python env | conda env `ai360`: Python 3.12, torch 2.11.0+cu128, torchvision 0.26.0, Pillow 13.0.0. Path on this machine: `C:\Users\ericw\miniconda3\envs\ai360\python.exe`. |
| Run style | Always `python -B ...`, so no `__pycache__` lands in `src/`. Most scripts run from `src/`. |
| Training/validation data | `Datasets/imagenet-1k/val/<wnid>/*.JPEG`: the 50,000 ImageNet-1K validation images, used as our training pool. |
| Test data | `Datasets/imagenetv2-matched-frequency/<0-999>/*.jpeg`: 10,000 ImageNetV2 images, **test only, never used for any choice**. |
| Data notes | `Datasets/DATASETS.md` (sources, Hugging Face revisions, hashes). ImageNet is gated on Hugging Face; each person must accept its non-commercial terms. **Do not upload the images anywhere.** |

Your own code is in `src/`: `main.py`, `sparse_workflow.py`, `image_data.py`, `compress_jpeg.py` and `resnet18_sparse_ae.py`. It was **not modified**; its hashes were checked after every experiment. Everything the agent added is in **new** files:

| New file | What it does |
| --- | --- |
| `src/build_jpeg_dataset.py` | Writes compressed copies of a dataset to disk and records the exact byte size of every file, with headers counted separately, in `manifest.csv` and `summary.json`. |
| `src/baseline_head.py` | A plain linear classifier on the same frozen features: the conventional baseline. |
| `src/run_grid.py` | Main experiment runner. Caches features once per compression level, then tunes, trains and tests every head. |
| `src/run_grid_xavier.py` | The same grid, with every head starting from random (Xavier) weights. |
| `src/run_ablation.py` | Turns individual loss terms off to find what makes the sparse head work. |

## 2. Timeline of what was done

### Step 1: first real training run and timing (task ENG-004, 2026-11-24)
- **Run:** `main.py train` with the README's 45k/5k settings: k = 32 of M = 2048 atoms, JPEG level 20, 10 epochs.
- **Timing:** feature extraction runs at about 330 images/s and is limited by the CPU. One run takes about 3.5 minutes, so a grid of dozens of runs is feasible.
- **Report:** `ai_harness/artifacts/ENG-004/report.md`.

### Step 2: a baseline and a dense control (ENG-005)
- **Added:** a linear baseline (`baseline_head.py`) and a dense control (`main.py --k 2048`, all atoms active).
- **Single-seed pilot, same 5k held-out images:**

  | Head | Top-1 |
  | --- | --- |
  | sparse k = 32 | 57.6% |
  | dense | 63.2% |
  | linear | 62.0% |
  | original ImageNet classifier | 70.0% |

- **Caveat:** these numbers were selected and reported on the same images, so they are biased. That is why the real experiment uses ImageNetV2 as the test set.
- **Report:** `ai_harness/artifacts/ENG-005/report.md`.

### Step 3: experiment plan (REASON-002)
- **Plan:** `ai_harness/artifacts/REASON-002/plan.md`. Approved with three choices: tune first, store outputs in `Final Project/runs`, include a dictionary-size (M) sweep.
- **Fixed in advance, before seeing any test result:**
  - 8 compression levels: source (original), 20, 60, 80, 85, 90, 95, 100;
  - 3 heads: sparse, dense, linear;
  - seeds 42/43/44;
  - test on original ImageNetV2;
  - a 1.0-point tolerance rule;
  - only differences ≥ 1 point count as real with 3 seeds.

### Step 4: main grid (GRID-001)
- **Ran:**
  - Validation-only tuning: 28 runs. It chose k = 512 for sparse, and chose to start every head **from the original ImageNet classifier's weights**.
  - 72 grid runs.
  - A 36-run M sweep.
  - Byte measurement of all 50k images at 7 levels.
- **Byte result:** level 80 = 50.3:1 against raw pixels and 89.8% smaller than the original files. A pixel check confirmed the stored files decode to exactly the pixels we trained on (3,500 of 3,500).
- **Accuracy result:** the linear head was best, 1.5–2.3 points ahead of sparse. But no trained head beat the untouched ImageNet classifier (57.3% on ImageNetV2). Because every head started from those weights, this grid mostly measured *how much fine-tuning on compressed data damages a pretrained classifier*.
- **M sweep:** inconclusive. The small dictionaries kept their epoch-1 checkpoints.
- **Report:** `ai_harness/artifacts/GRID-001/report.md`.

### Step 5: random-start grid (GRID-002)
- **Ran:** the same 72-run grid, but every head starts from random (Xavier) weights. This is the cleaner test of "learning a head from compressed data". The settings were frozen from the existing validation tuning, with no test data used.
- **Result:** the ranking reversed to sparse > dense > linear.
  - **Sparse vs linear:** sparse is ahead on every seed, but by ≥ 1 point only at source and level 20; elsewhere by 0.4–1.0.
  - **Sparse vs dense:** 0.2–0.8 points, not established.
  - **What the original classifier's weights were worth:** 3.3–6.8 points. The GRID-002 report says 3.5; the true minimum is 3.3.
- **Report:** `ai_harness/artifacts/GRID-002/report.md`.

### Step 6: ablation (ABL-001)
- **Ran:** 60 runs switching off loss terms, with random-start heads at levels source, 60, 80, 90 and 95.
- **Result:**

  | Change | Effect |
  | --- | --- |
  | A wide (2048) head trained with plain cross-entropy | 1.0 point *worse* than linear (overfits 45 images per class) |
  | Adding the feature-reconstruction (MSE) loss | +1.4 (dense) / +2.0 (sparse) points |
  | The dependency/coherence regularizers | about 0 |
  | Top-k sparsity | about 0 without MSE, +0.46 with MSE (unresolved) |

- **Alternative explanations not ruled out:**
  - any regularizer might do what MSE does, since the wide head had no weight decay;
  - code-scale effects from the unit-norm atoms;
  - different training speed.
- **Report:** `ai_harness/artifacts/ABL-001/report.md`.

### Step 7: independent review (REVIEW-002)
- **Process:** a separate reviewer agent re-derived every table, checked checkpoints, byte totals, test-set isolation and split pairing, and requested wording fixes. First verdict: *revise*, with no blocking problems.
- **Main corrections:**
  - The M-sweep conclusion was withdrawn because of the epoch-1 checkpoints.
  - "Sparse beats linear" was downgraded from "established" to "consistent in sign, under 1 point at most levels".
  - "MSE explains the gain" was restated as above.
- **Outcome:** after the revisions, all three were **accepted**.
- **Reports:** `ai_harness/artifacts/REVIEW-002/report.md`. A one-page summary of the accepted results is in `ai_harness/artifacts/COORD-002/status.md`.

## 3. What we can and cannot claim

**Can claim** (measured, reviewed):
- Storing the 45k training images at JPEG quality 20 (50.3:1 against raw) costs about 1 point of ImageNetV2 accuracy for a head on frozen ResNet-18 features. At quality 10 (86:1) it costs about 2 points.
- In a 2048-wide head, the feature-reconstruction loss is what lets it beat a linear head. The advantage is under 1 point.

**Cannot claim:**
- **Training from compressed data alone.** The backbone was pretrained on original ImageNet, and only the head was trained on compressed data.
- **"Sparse dictionaries help."** Sparsity's own effect is unresolved.
- **"98% accuracy at 50:1"** in absolute terms. Heads reach about 48–54% against 57% for the original classifier. Relative to each head's own uncompressed accuracy, level 80 retains about 98%, within noise.
- **Any effect of dictionary size M.**

**Caveats:**
- Only 3 seeds.
- ImageNetV2 results guided which experiment to run next, though no settings were tuned on it.
- 29 duplicate images with conflicting labels exist in the ImageNet-val pool. They affect validation only.

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
cd ..
python -B -u src/build_jpeg_dataset.py Datasets/imagenet-1k runs/GRID-001/jpeg_variants --levels 20 60 80 85 90 95 100 --splits val   # ~90 min CPU
python -B ai_harness/artifacts/GRID-001/pixel_check.py
python -B ai_harness/artifacts/GRID-001/analyze.py      # likewise GRID-002/analyze.py, ABL-001/analyze.py
```

- Every phase skips runs that already finished (it checks for `done.json`).
- **Warning:** the `analyze.py` scripts overwrite `analysis.md`/`analysis.json` next to themselves. Those files are recorded artifacts in the harness ledger, so copy the script elsewhere before experimenting with it.

## 5. Where things are

| Path | Contents | Size |
| --- | --- | --- |
| `runs/GRID-001/` | `results.csv` and `summary.csv` (all numbers), tuning files, cached features, checkpoints, JPEG variants with `manifest.csv` | about 14 GB + 5.3 GB of JPEG variants |
| `runs/GRID-002/`, `runs/ABL-001/` | Results and `best.pt` checkpoints (`last.pt` deleted) | 2.7 GB and 3.2 GB |
| `runs/ENG-004/`, `runs/ENG-005/` | The pilot runs | small |
| `ai_harness/artifacts/<TASK>/` | Report, logs and analysis for every task | small |

- All of this lives inside OneDrive.
- About 7 GB of `last.pt` files in `runs/GRID-001` could be deleted if space is needed. Nothing has been deleted without approval.

## 6. Team coordination (the harness)

- **What it is:** a local task ledger. Start with `AGENTS.md` and `ai_harness/START_HERE.md`. `py ai_harness/harness.py status` shows the queue, and `ai_harness/BOARD.md` is a readable snapshot.
- **Known error in the ledger:** the ABL-001 one-line summary in the ledger and `BOARD.md` still states a claim that was withdrawn in review. Use the report or `COORD-002/status.md` instead.
- **Waiting for a reviewer other than the original author:**
  - REVIEW-002 and COORD-002;
  - the earlier setup tasks ORIENT-001/002, COORD-001, ENG-001…005 and REASON-002.
- **Never claimed:** REPRO-001, REVIEW-001, SYNTH-001, and LIT-002 (verify a citation used in `src/`).

## 7. Suggested next steps (not done)

1. **More seeds (about 7)** for sparse vs dense with the MSE loss, at levels source/80/90. This decides whether sparsity matters at all.
2. **A weight-decay-tuned wide head**, to test whether any regularizer matches the MSE loss.
3. **A clean M sweep:** random-start heads, with k scaled to M.
4. **A storage-vs-accuracy plot for the write-up.** matplotlib is not installed; ask before adding it.
5. **Fold `COORD-002/status.md` into the proposal and write-up.** Keep the claim-scope limits from section 3.
