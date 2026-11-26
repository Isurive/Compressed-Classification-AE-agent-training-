# ABL-001: where does the sparse head's gain over the linear head come from?

**Agent:** `claude-orient-20261122-a`. **Date:** 2026-11-24. **Approved by the user:** "Yes, run ablation", proposed in [GRID-002 report](../GRID-002/report.md) §8.
**Evidence status:** measured results, 3 seeds, this laptop. Tables: [analysis.md](analysis.md) / [analysis.json](analysis.json), produced by [analyze.py](analyze.py).

## 0. Headline

1. **Inside the 2048-wide heads, the feature-reconstruction (MSE) auxiliary loss clearly helps.** Adding it raises ImageNetV2-source top-1 by **+2.0 pp (sparse)** and **+1.4 pp (dense)**, averaged over the 5 levels.
   - Every seed at every level is positive (15 of 15 for each head).
   - The 15 pairs come from 3 splits. The per-seed mean effects are 1.98, 2.09 and 1.90 for sparse, and 1.42, 1.36 and 1.38 for dense (REVIEW-002 A-4).
2. **Top-k sparsity alone does nothing.** Without the auxiliary loss, sparse (k = 512) vs dense (k = 2048) is −0.14 pp on average (sparse ahead in 5 of 15 seed × level pairs). With MSE, sparse is ahead by +0.46 pp (13 of 15 pairs). That is a possible small sparsity × MSE interaction, but it is **below the pre-declared 1 pp threshold**.
3. **Width alone hurts.** A CE-only 512→2048→1000 ReLU MLP (dense head, no auxiliary loss) is **1.0 pp worse than the linear head**, with linear ahead in all 15 pairs. This is consistent with overfitting 45 images per class.
4. **The dependency and coherence regularizers have no measurable effect** (−0.07 pp for sparse, +0.01 pp for dense).
5. **None of these components improves robustness to compression.** Drops at levels 80/90/95 are similar in every cell, and the MSE benefit **shrinks** with compression: for sparse, +2.2 pp at source and +1.5 pp at level 90.

**Answer to the GRID-002 question** (revised after REVIEW-002 A-1). On this ablation's levels (source, 60, 80, 90, 95), the sparse head's gain over linear is **0.39–1.19 pp, mean 0.78**. It decomposes approximately as follows (mean over levels, pp):

| Step | Effect |
| --- | --- |
| Going 2048-wide with CE only (MLP − linear) | **−1.00** |
| Adding the MSE auxiliary loss (dense-mse − MLP) | **+1.39** |
| So dense-mse − linear | +0.39 (12 of 15 pairs positive; per-seed 0.12, 0.44, 0.60) |
| Adding top-k sparsity with MSE (sparse-mse − dense-mse) | +0.46 (13 of 15) |
| Adding the dependency/coherence regularizers (sparse-full − sparse-mse) | ≈ 0 |

- **MSE is necessary.** It offsets the penalty of the wider head, and without it the dictionary heads lose to linear.
- **The remaining advantage over linear is below the 1 pp threshold.** It comes partly from MSE and partly from sparsity *in combination with* MSE. Neither part is individually established.
- **The defensible statement:** "in a 2048-wide head trained on compressed images, a feature-autoencoding auxiliary loss is needed to beat a linear head, and the resulting advantage is under 1 pp."
- **Not supported:** "sparse dictionaries help," or "the MSE loss alone explains the gain over linear." The earlier wording made that second claim.

## 1. Design

Everything matches GRID-002 except the loss weights:
- Xavier classifier init and He encoder init.
- M = 2048; k = 512 (sparse) or 2048 (dense).
- 10 epochs, batch 128, Adam lr 1e-3.
- Same cached features, same splits (one split digest per seed across GRID-002 and ABL-001, verified), same ImageNetV2 test protocol.

| Cell | CE | Feature MSE | Dependency | Coherence | Source |
| --- | --- | --- | --- | --- | --- |
| full | 1 | 1 | 0.1 | 0.05 | GRID-002 |
| mse | 1 | 1 | 0 | 0 | ABL-001 `abl-mse` |
| none | 1 | 1e-6 * | 0 | 0 | ABL-001 `abl-none` |
| linear | 1 | — | — | — | GRID-002 (wd 1e-4) |

\* `Settings.validate` in the user's `sparse_workflow.py` requires a positive reconstruction weight. Training feature MSE in these runs is 1.04–3.12, so the weighted term is about 1e-6 to 3e-6, against a training CE of 0.019–3.5. The cell is CE-only in effect. (Corrected after REVIEW-002 A-3; the earlier "about 2e-7" was wrong.) The decoder still exists and Adam gives it updates of normal size, but it does not affect predictions.

- **Run count:** levels source, 60, 80, 90, 95 × seeds 42/43/44 × 2 heads × 2 new loss variants = 60 runs (about 30 min). The full and linear cells are reused from GRID-002.
- **What the MSE target is:** the frozen backbone's 512-d features of the **same compressed image**. It is not an original-image target, so the compressed-only training claim (with a pretrained backbone) still holds.

## 2. Results: ImageNetV2-source top-1 % (mean ± sd)

| Cell | source | L60 | L80 | L90 | L95 |
| --- | --- | --- | --- | --- | --- |
| sparse-full | 48.34 ± 0.55 | 48.00 ± 0.19 | 47.62 ± 0.32 | 46.08 ± 0.15 | 43.36 ± 0.08 |
| sparse-mse | **48.37 ± 0.27** | **48.22 ± 0.11** | 47.52 ± 0.20 | 46.08 ± 0.50 | **43.54 ± 0.33** |
| dense-full | 47.95 ± 0.26 | 47.72 ± 0.19 | 46.85 ± 0.75 | 45.85 ± 0.27 | 43.09 ± 0.19 |
| dense-mse | 47.90 ± 0.34 | 47.50 ± 0.44 | 47.23 ± 0.50 | 45.80 ± 0.21 | 43.00 ± 0.20 |
| linear | 47.15 ± 0.21 | 47.12 ± 0.13 | 46.64 ± 0.19 | 45.61 ± 0.27 | 42.97 ± 0.25 |
| dense-none (MLP) | 46.16 ± 0.40 | 45.96 ± 0.25 | 45.67 ± 0.43 | 44.62 ± 0.19 | 42.08 ± 0.27 |
| sparse-none | 46.20 ± 0.13 | 45.82 ± 0.13 | 45.40 ± 0.42 | 44.59 ± 0.27 | 41.78 ± 0.12 |

## 3. Effects (pp, paired by seed; [n] = seeds with a positive difference, of 3)

| Effect | source | L60 | L80 | L90 | L95 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| **MSE aux, sparse** (mse − none) | 2.17 [3] | 2.41 [3] | 2.11 [3] | 1.49 [3] | 1.76 [3] | **1.99** |
| **MSE aux, dense** (mse − none) | 1.74 [3] | 1.54 [3] | 1.56 [3] | 1.18 [3] | 0.93 [3] | **1.39** |
| Sparsity, no aux (sparse − dense) | 0.04 [2] | −0.14 [1] | −0.27 [1] | −0.03 [1] | −0.29 [0] | −0.14 |
| Sparsity, MSE only | 0.47 [3] | 0.72 [3] | 0.29 [2] | 0.28 [2] | 0.54 [3] | 0.46 |
| Sparsity, full loss | 0.39 [2] | 0.29 [3] | 0.77 [3] | 0.23 [2] | 0.27 [3] | 0.39 |
| Dep + coh, sparse (full − mse) | −0.03 | −0.22 | 0.11 | 0.00 | −0.18 | −0.07 |
| Dep + coh, dense (full − mse) | 0.05 | 0.22 | −0.38 | 0.05 | 0.09 | 0.01 |
| **Width** (MLP − linear) | −0.99 [0] | −1.17 [0] | −0.97 [0] | −0.99 [0] | −0.90 [0] | **−1.00** |
| Whole method vs linear (sparse-full − linear) | 1.19 [3] | 0.88 [3] | 0.99 [3] | 0.47 [3] | 0.39 [3] | 0.78 |

Standard deviations are in [analysis.md](analysis.md).

**Paired drops vs source** (all cells in [analysis.md](analysis.md)):
- **Level 80:** 0.5–1.1 pp in every cell.
- **Level 90:** 1.5–1.6 pp for the cells without the auxiliary loss (linear, MLP, sparse-none), and 2.1–2.3 pp for the cells with it.
- **Level 95:** 4.1–5.0 pp.

The auxiliary loss adds accuracy but adds no compression tolerance. Its benefit is largest on lightly compressed data.

## 4. Interpretation and caveats

- **Why MSE helps (inference, not tested):** reconstructing the 512-d feature from the 2048-d code forces the code to keep the feature's information. That acts as a regularizer against fitting 45 images per class. The CE-only MLP overfits more than linear, which supports this reading. Compressed images carry less feature detail, so there is less for reconstruction to preserve, which fits the shrinking benefit.
- **Alternative explanations for the MSE effect (added after REVIEW-002 A-2):**
  - **Generic regularization:** the CE-only MLP has no weight decay or dropout, while linear had a tuned weight decay of 1e-4. Any reasonable regularizer might give part of what MSE gives. MSE is shown to beat *no* regularization, not to beat weight decay or dropout.
  - **Code-scale anchoring:** the dictionary atoms are unit-norm (`F.normalize` in `resnet18_sparse_ae.py`), and reconstruction ties code magnitudes to feature magnitudes. MSE may help by fixing the scale of the codes rather than by preserving information.
  - **Training dynamics:** CE-only runs peak at epochs 4–6, and MSE runs at 6–9, with an untuned learning rate and a fixed 10 epochs. Part of the gap may come from optimization speed rather than final quality.
- **Scope of the conclusion:** the MSE benefit is measured against an **unregularized, CE-only head of the same width** under GRID-002's training budget. The width-hurts finding is likewise about this budget, not MLPs in general. A weight-decay-tuned MLP would test the generic-regularization explanation.
- **Sparsity × MSE:** +0.46 pp with MSE vs −0.14 pp without. It is consistent in sign (13 of 15 vs 5 of 15) but under 1 pp, so it is **unresolved**. Settling it needs more seeds.
- **Same scope as GRID-002:** frozen ImageNet-pretrained backbone, head trained from Xavier init on compressed images, tested on original ImageNetV2.

## 5. Checks

- 60 of 60 `done.json`; `results.csv` has 108 rows (15 source runs × 1 evaluation + 45 compressed runs × 2); no tracebacks. Logs: [grid_log.txt](grid_log.txt), [summary_log.txt](summary_log.txt).
- Settings recorded in `runs/ABL-001/settings.json`; best epochs 4–9.
- No ImageNetV2-based choices: nothing was tuned, and the variants were fixed in the task spec ([task.json](task.json)) before running.
- Code: [src/run_ablation.py](../../../src/run_ablation.py) imports `run_grid.py` unchanged. The 8 user `src/` files are unchanged; the 22 GRID-001 and 12 GRID-002 submitted artifacts match their ledger hashes; there is no `__pycache__`. `runs/ABL-001` is 3.2 GB.
- `last.pt` deleted for the new runs only. `runs/ABL-001` holds `best.pt` and `done.json`, but **no `history.json`** (corrected, REVIEW-002 A-5). The per-epoch history is in [grid_log.txt](grid_log.txt). The loss weights are in each `done.json` and in `runs/ABL-001/settings.json`, not in `results.csv`, where the variant is encoded in the `phase` column.

## 6. Implications and next steps (proposals)

1. **Reframe the method** (revised after REVIEW-002 A-1): describe it as "a wide dictionary head that needs a feature-autoencoding auxiliary loss to beat a linear head". The advantage is under 1 pp. Sparsity, and the alternative explanations in §4, are unresolved secondary factors.
   - Do not claim that the sparse dictionary, or the MSE loss alone, explains the gain over linear.
2. **More seeds (about 7) for sparse-mse vs dense-mse** at source, 80 and 90, to decide the 0.5 pp sparsity × MSE effect.
3. **Fair MLP/linear baseline with an auxiliary loss:** a linear head can't reconstruct, but an MLP + MSE is exactly dense-mse. Tuning weight decay for the MLP would make the width comparison fair.
4. **Independent review**: done in REVIEW-002.
5. **Test reuse:** ImageNetV2 has informed the design of the follow-up experiments (GRID-001 → GRID-002 → ABL-001). Confirm the effects under 1 pp with more seeds before final claims.

## 7. Revision log

- **2026-11-25, after REVIEW-002** ([report](../REVIEW-002/report.md); verdict *revise*, no blocking findings):
  - A-1: restated the headline answer and §6.1. MSE is necessary to offset the width penalty; the residual advantage over linear is under 1 pp.
  - A-2: added three alternative explanations and scoped the claim.
  - A-3: corrected the size of the 1e-6 term.
  - A-4: added per-seed means.
  - A-5: corrected where run histories and loss weights are stored.
  - No numbers changed. The reviewer reproduced every table.
