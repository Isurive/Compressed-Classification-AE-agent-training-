# ENG-005: Conventional baseline and dense control vs. the sparse model

**Agent:** `claude-orient-20261122-a`. **Role:** engineer. **Date:** 2026-11-24.
**Status:** Submitted for review. **Single-seed pilot with no tuning. It is not a research result.** ImageNetV2 was not touched.

## What was added

[`src/baseline_head.py`](../../../src/baseline_head.py) is a new file; no existing `src/` file was modified. [Hashes before](src_hashes_before.txt) match the hashes after for all seven user files.

It trains `Linear(512, classes)` on the **same frozen ResNet-18 features**, reusing these from `image_data` / `sparse_workflow` unchanged:
- `training_splits`, `evaluation_split`, and `ImageView` (JPEG + preprocessing);
- `maybe_cache`, `loader`, `batch_features`, and `run_writer`;
- per-epoch seeding `seed + epoch` and the `best.pt` rule (highest val top-1, then lowest CE).

It has `train` and `evaluate` subcommands that mirror `main.py`; usage is in the module docstring. It prints the split digest so the split can be confirmed identical. `--fc-init original` (copy the ImageNet fc rows) and `--weight-decay` are available but were **not** used here.

**Dense control without new code:** `main.py train --k 2048` with `--atoms 2048` lets every atom be active. Everything else is identical to the sparse run: encoder, center, decoder, CE + MSE + dependency + coherence losses, and initialization. This isolates the top-k sparsity.

## Runs (all identical except the head)

The common settings match ENG-004:
- Data: 45k/5k holdout of `Datasets/imagenet-1k/val`, split digest `2bbb411f2b9dcda3…` (**identical for all three, verified from each `best.pt`**).
- JPEG level 20 for both train and validation.
- 10 epochs, batch 128, Adam lr 1e-3, seed 42, Xavier fc init, `--cache memory --workers 4`, RTX 3080 Ti Laptop GPU.

| Model | Run | Head | Trainable loss |
| --- | --- | --- | --- |
| Sparse dictionary (k = 32 of M = 2048) | `runs/ENG-004/he_45k` | encoder → top-32 → fc | CE + MSE + 0.1·dep + 0.05·coh |
| Dense control (k = M = 2048) | `runs/ENG-005/dense_k2048_45k` | encoder → ReLU (all atoms) → fc | same |
| Linear baseline | `runs/ENG-005/linear_45k` | Linear(512, 1000) | CE |
| Original ImageNet fc (reference, not trained here) | `--baseline` | pretrained on 1.28M ImageNet-train originals | — |

Logs: [linear_train_log.txt](linear_train_log.txt), [dense_train_log.txt](dense_train_log.txt), [linear_eval_log.txt](linear_eval_log.txt), [dense_eval_log.txt](dense_eval_log.txt). Metrics: [linear_heldout.json](linear_heldout.json), [dense_heldout.json](dense_heldout.json). The sparse metrics are in ENG-004.

## Held-out results (the same 5,000 images, `best.pt` of each)

| Model | Best epoch | Top-1 @ level 20 | Top-1 @ source | Top-5 @ source | Train top-1 (final epoch) |
| --- | --- | --- | --- | --- | --- |
| Sparse k = 32 | 8 | 57.1% | 57.6% | 80.2% | 99.8% |
| Dense k = 2048 (about 830 atoms active per image) | 5 | **63.0%** | **63.2%** | 85.1% | 94.9% (peaked at 99.5%) |
| Linear baseline | 5 | 61.6% | 62.0% | 84.8% | 92.8% |
| Original ImageNet fc | — | 69.3% | 70.0% | 88.9% | — |

Training time per run was about 4–5 min (feature extraction dominated; OneDrive activity may have slowed reads compared with ENG-004).

## Observations (inference, single seed; for review)

1. **In this pilot, top-k sparsity hurt.** The dense control beat the k = 32 model by about 6 points with everything else held equal. Possible reasons, none tested:
   - k = 32 is too restrictive for 1,000 classes;
   - the hard top-k gives no gradient to atoms that don't win;
   - the reconstruction and regularizer weights were not tuned for the sparse case.
2. **The dense dictionary head edged out the linear baseline** by about 1.2–1.4 points, but it was **unstable**. Validation top-1 fell to 57% at epoch 9 while active atoms dropped from about 900 to 715. One seed cannot separate this from noise.
3. **Compression level 20 is too mild to matter.** Going from source to level 20 costs only 0.2–0.5 points for every model. The storage question needs much stronger levels and their measured bytes (`build_jpeg_dataset.py`).
4. **All learned heads stay 7–13 points below the original fc.** That fc was trained on about 28× more data, so it is a reference ceiling, not a fair competitor.
5. **Selection bias (applies equally to all three).** `best.pt` is chosen on the same 5,000 images reported here, so these numbers are slightly optimistic. Before any reported result, either select on a split carved from the 45k training set, or report a pre-declared fixed epoch (as `main.py crossval` does), and keep ImageNetV2 for the final test.

## Suggested next steps (proposals; the user decides)

- Repeat all three heads with 3 seeds, or use `main.py crossval` and a fixed-epoch version for the baseline, to see whether the differences survive.
- Sweep k (for example 32, 128, 512, 2048) at fixed M, and **add strong compression levels** (for example 60, 80, 90, 95). Measure their bytes with `build_jpeg_dataset.py` on the 45k split. Train at each level and evaluate on `source`: this is the actual storage experiment.
- Decide whether to tune sparse-specific settings (k, reconstruction weight, `--fc-init auto`) on validation before the main grid. Apply the same tuning budget to the baselines.

## Housekeeping

- Run folders: `runs/ENG-005/linear_45k` (2 MB), `dense_k2048_45k` (147 MB), and `smoke` (2 MB; the 2,000-image smoke test from development, kept as evidence). The smoke test's numbers are not meaningful.
- `.training.lock` files remain in run folders. They are created by `run_writer`, which `baseline_head.py` reuses.
