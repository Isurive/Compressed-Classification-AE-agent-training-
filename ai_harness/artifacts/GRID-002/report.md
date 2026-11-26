# GRID-002: Xavier-init grid (heads without the inherited ImageNet classifier)

**Agent:** `claude-orient-20261122-a`. **Date:** 2026-11-24. **Approved by the user:** "Yes, run it", the follow-up recommended in [GRID-001 report](../GRID-001/report.md) §9.
**Evidence status:** every number below is a **measured result** (3 seeds, this laptop) unless labeled otherwise. Tables: [analysis.md](analysis.md) / [analysis.json](analysis.json), produced by [analyze.py](analyze.py).

## 0. Headline

1. **Without the inherited classifier, the sparse-dictionary head is the most accurate head.** On original ImageNetV2 test images, sparse beats linear at every level from source to 95, in all 21 seed × level pairs. The gap is **consistent in sign but small**:

| Level | Sparse − linear (pp) |
| --- | --- |
| source | 1.19 |
| 20 | 1.57 |
| 60 | 0.88 |
| 80 | 0.99 |
| 85 | 0.90 |
| 90 | 0.47 |
| 95 | 0.39 |

The pre-declared 1 pp rule is reached only at source and level 20. The 21 positive pairs come from only 3 independent splits, with per-seed mean gaps of 0.76, 0.80 and 0.79 pp. Revised after REVIEW-002 G2-1: the earlier wording called this gap "established", which overstated it.

Sparse vs dense (same dictionary, k = 512 vs 2048) is a smaller **0.2–0.8 pp** edge (21 of 24 pairs positive). The benefit of *top-k sparsity itself* is **not established**.
2. **Sparse does not tolerate compression better.** Its accuracy drops are equal to or larger than linear's at every level, though the drop differences are mostly within noise. The sparse advantage is a roughly constant offset that shrinks at heavy compression: +0.4 pp at level 95, and linear wins by 2.1 pp at level 100.
3. **Heavy training compression still costs little.** Level 80 (quality 20; the 45k training set is 50.3:1 against raw and 89.8% smaller than the stored files, per [GRID-001](../GRID-001/report.md) §4) costs **0.5–1.1 pp**. Level 85 costs 0.7–1.0 pp. Level 90 costs 1.5–2.3 pp. These are similar to GRID-001 up to level 90. Without the inherited classifier the drops grow much faster beyond that: level 95 costs 4–5 pp and level 100 costs 15–19 pp.
4. **Starting from the original fc was worth a lot.** Same-model Xavier heads are 3.7 (sparse), 4.3 (dense) and 6.8 (linear) pp lower than the GRID-001 heads at source. So the GRID-001 ranking (linear first) came from the linear head inheriting the most. All heads remain far below the untouched original fc (57.28%).

## 1. Claim scope

- **Backbone:** a frozen ImageNet-pretrained ResNet-18 (trained on original ImageNet-train). The head is learned from 45k compressed images **with no inherited classifier weights**. The sparse/dense heads use a feature-MSE auxiliary loss against the frozen backbone's features of the *same compressed* image. That is not an original-image target.
- **Allowed claim:** a head trained from scratch on compressed training images, over fixed pretrained features. **Not allowed:** a claim that a model learned from compressed data alone.
- **Test protocol:** the same as GRID-001. Splits match it per seed (verified by split digest). The primary test is the original ImageNetV2 matched-frequency images, used only after the settings were frozen.

## 2. Settings (frozen before any GRID-002 test evaluation)

The settings come from `runs/GRID-001/tune_results.json` (seed 42, 5k validation only). The rule is the one pre-declared in GRID-001, restricted to Xavier candidates. Log: [freeze_log.txt](freeze_log.txt).

| Model | Xavier candidates | Chosen | Val top-1 (mean of source & L90) |
| --- | --- | --- | --- |
| sparse | k ∈ {32, 128, 512} | M 2048, k 512 | 60.03 |
| dense | 1 (k = M) | M 2048, k 2048 | 59.43 |
| linear | wd ∈ {0, 1e-4, 1e-3} | wd 1e-4 | 58.41 |

- **Tuning budget:** 3 / 1 / 3 candidates (sparse / dense / linear).
- **Shared settings:** as in GRID-001: 10 epochs, batch 128, Adam lr 1e-3, `--encoder-init he`, and `best.pt` chosen by validation top-1, then CE.
- **Best epochs:** 4–10 for every model (in GRID-001, linear peaked at epochs 1–3).

## 3. Primary results: ImageNetV2 at source, top-1 % (mean ± sd over seeds 42/43/44)

| Level (quality) | sparse | drop | dense | drop | linear | drop |
| --- | --- | --- | --- | --- | --- | --- |
| source | **48.34 ± 0.55** | — | 47.95 ± 0.26 | — | 47.15 ± 0.21 | — |
| 20 (q76) | **48.57 ± 0.08** | −0.23 ± 0.48 | 47.85 ± 0.41 | 0.10 ± 0.15 | 47.00 ± 0.18 | 0.15 ± 0.10 |
| 60 (q39) | **48.00 ± 0.19** | 0.33 ± 0.37 | 47.72 ± 0.19 | 0.23 ± 0.17 | 47.12 ± 0.13 | 0.03 ± 0.13 |
| 80 (q20) | **47.62 ± 0.32** | 0.71 ± 0.58 | 46.85 ± 0.75 | 1.10 ± 0.50 | 46.64 ± 0.19 | 0.51 ± 0.16 |
| 85 (q15) | **47.32 ± 0.17** | 1.02 ± 0.39 | 46.99 ± 0.28 | 0.96 ± 0.12 | 46.41 ± 0.26 | 0.74 ± 0.23 |
| 90 (q10) | **46.08 ± 0.15** | 2.26 ± 0.66 | 45.85 ± 0.27 | 2.10 ± 0.25 | 45.61 ± 0.27 | 1.54 ± 0.10 |
| 95 (q6) | **43.36 ± 0.08** | 4.98 ± 0.61 | 43.09 ± 0.19 | 4.86 ± 0.36 | 42.97 ± 0.25 | 4.18 ± 0.30 |
| 100 (q1) | 29.69 ± 0.49 | 18.64 ± 0.49 | 29.21 ± 0.11 | 18.73 ± 0.30 | **31.75 ± 0.41** | 15.40 ± 0.48 |

Drops are paired by seed, in pp.

**1.0 pp tolerance** (all weaker levels must also pass):

| Model | Passes up to | First failure |
| --- | --- | --- |
| sparse | **80** (0.71) | 85 (1.02, borderline) |
| dense | 60 | 80 (1.10). Level 85 (0.96) passes again, so the dense curve is noisy near 1 pp. |
| linear | **85** (0.74) | 90 (1.54) |

**Sparse vs the other heads, paired (pp).** Gap = A(sparse) − A(other). Interaction = drop(other) − drop(sparse); a positive interaction means sparse retained more.

| Level | Gap vs dense | Interaction vs dense | Gap vs linear | Interaction vs linear |
| --- | --- | --- | --- | --- |
| source | 0.39 ± 0.73 | — | 1.19 ± 0.45 | — |
| 20 | 0.72 ± 0.48 | 0.33 ± 0.44 | 1.57 ± 0.13 | 0.38 ± 0.52 |
| 60 | 0.29 ± 0.25 | −0.10 ± 0.54 | 0.88 ± 0.07 | −0.31 ± 0.38 |
| 80 | 0.77 ± 0.64 | 0.38 ± 0.10 | 0.99 ± 0.15 | −0.20 ± 0.43 |
| 85 | 0.33 ± 0.42 | −0.06 ± 0.32 | 0.90 ± 0.32 | −0.28 ± 0.18 |
| 90 | 0.23 ± 0.23 | −0.16 ± 0.90 | 0.47 ± 0.33 | −0.72 ± 0.75 |
| 95 | 0.27 ± 0.22 | −0.12 ± 0.76 | 0.39 ± 0.33 | −0.80 ± 0.76 |
| 100 | 0.48 ± 0.60 | 0.09 ± 0.78 | −2.06 ± 0.71 | −3.25 ± 0.96 |

Per-seed sparse − linear gaps, source through level 95, are all positive (0.06 to 1.67 pp). At level 100 all three are negative.

## 4. Comparison with GRID-001 (inherited classifier), same model, paired by seed

| Level | sparse | dense | linear |
| --- | --- | --- | --- |
| source | −3.74 | −4.25 | −6.80 |
| 80 | −3.64 | −4.23 | −6.11 |
| 90 | −3.86 | −4.24 | −6.53 |
| 95 | −5.62 | −5.59 | −8.28 |
| 100 | −11.56 | −13.98 | −14.55 |

Values are Xavier minus inherited, in pp; all levels are in [analysis.md](analysis.md).

- **Inherited classifier:** it adds a near-constant 3.5–6.8 pp up to level 90, and much more at levels 95–100. In other words, the original fc supplies clean-data class knowledge that compressed training cannot recover at extreme compression.
- **Ranking reversal:** the order of heads flips between the two grids.

| Grid | Ranking |
| --- | --- |
| GRID-001 (inherited classifier) | linear > sparse ≈ dense |
| GRID-002 (Xavier) | sparse > dense > linear |

  **Inference, not tested (REVIEW-002 G2-2):** the reversal is plausibly explained by how much each head inherits. The linear head inherits the whole classifier. Sparse and dense inherit a folded version through an untrained dictionary, which training then alters.

## 5. Secondary: matched-level test (ImageNetV2 compressed like training)

| Level | sparse | dense | linear |
| --- | --- | --- | --- |
| 80 | 45.20 | 44.36 | 44.11 |
| 90 | 40.23 | 39.97 | 39.17 |
| 95 | 32.13 | 31.79 | 30.73 |

Sparse is also first when the test images are compressed the same way. This is deployment robustness, a different claim from storage savings.

## 6. Interpretation and alternative explanations

- **What helps is probably capacity or regularization, not sparsity.** Sparse beats linear consistently, but sparse vs dense is within noise.
  - Possible contributors: the extra 2048-wide encoder/decoder layer, the feature-MSE auxiliary loss, and the dependency/coherence regularizers. Each could produce the gain over linear without top-k mattering.
  - **Unresolved:** which of these parts matters. An ablation (dense head without the MSE/regularizers, or an MLP of matched width) would separate them.
- **k = 512 is weak sparsity (25% of atoms active).** Smaller k was worse in validation tuning (k = 32: 53.3%).
- **Tuning budget was unequal (3 / 1 / 3).** Dense had no free choice. This can favor sparse over dense by a small amount, which is the same order as the observed gap.
- **Linear's learning rate and epoch count were not tuned (REVIEW-002 G2-4).** Only weight decay and initialization were searched. 9 of 24 linear runs peaked at epoch 9 or 10, but their validation curves are flat (±0.5 pp), so longer training is unlikely to change the ranking much. It is untested.
- **Seeds:** 3. Gaps under about 1 pp are inconclusive by the pre-declared rule. The sparse > linear gap is consistent in sign at every level from source to 95. Its mean is at least 1 pp only at source (1.19) and level 20 (1.57); it is 0.88–0.99 at levels 60–85 and 0.39–0.47 at levels 90–95. So it reaches the threshold only for lightly compressed training data.
- **The 98%-at-50:1 target.** At level 80 (50.3:1 against raw), relative retention is 98.5% for sparse, 97.7% for dense and 98.9% for linear, all against each head's own source accuracy. In absolute terms every head is about 48% vs the original fc's 57%. Report the relative numbers only with that caveat.

## 7. Checks

- **Splits:** identical to GRID-001 for every seed (one split digest per seed, equal to GRID-001's): `splits_match_grid001: true` in [analysis.json](analysis.json).
- **Results file:** 72 runs complete (72 `done.json`). `results.csv` has 135 rows (9 source runs × 1 evaluation + 63 compressed runs × 2), and the logs contain no tracebacks. Logs: [grid_log.txt](grid_log.txt), [summary_log.txt](summary_log.txt).
- **Test set never read early:** `freeze` reads only `tune_results.json`. The first ImageNetV2 evaluation happened after `runs/GRID-002/tuned.json` was written.
- **Code:** [src/run_grid_xavier.py](../../../src/run_grid_xavier.py) imports `run_grid.py` unchanged and redirects only the output folder. The 8 user `src/` files are unchanged against `GRID-001/src_hashes_before.txt`. All 22 GRID-001 submitted artifacts match their ledger hashes. There is no `__pycache__` in `src/`.
- **Disk:** `runs/GRID-002` is 2.7 GB. `last.pt` was deleted for these new runs only, as stated in the task. `best.pt` and `done.json` are kept. Correction (REVIEW-002 G2-3): `history.json` exists only for the 24 linear runs.
  - For the 48 dictionary runs, the per-epoch history is only in [grid_log.txt](grid_log.txt).
  - Their settings are in `done.json` (M, k, fc-init) and `runs/GRID-002/tuned.json`. The loss weights are the `Settings` defaults (1 / 0.1 / 0.05), which the reviewer confirmed from the per-epoch losses.
  - `best.pt` holds no settings.

## 8. Next steps (proposals, not run)

1. **Ablation of the dictionary head** (about 40 min GPU): dense with no MSE/regularizers, and a 512→2048→1000 MLP. This tests whether the sparse > linear gain is from sparsity, the auxiliary loss, or width.
2. **More seeds (5–10) for sparse vs dense** at levels source, 80 and 90, to decide whether the 0.2–0.8 pp sparsity edge is real.
3. **Independent review**: done in REVIEW-002.
4. **Test reuse:** ImageNetV2 informed the design of the next experiment: GRID-001's results led to this grid, and this grid's led to ABL-001. No settings were tuned on it. Before final claims, confirm the effects under 1 pp with more seeds.

## 9. Revision log

- **2026-11-25, after REVIEW-002** ([report](../REVIEW-002/report.md); verdict *revise*, no blocking findings):
  - G2-1: restated the sparse > linear claim. It is consistent in sign but reaches 1 pp only at source and level 20.
  - G2-2: labeled the inheritance explanation as an inference.
  - G2-3: corrected where run histories and settings are stored.
  - G2-4: added that linear's learning rate and epochs were not tuned.
  - Added the adaptive-reuse note to §8.
  - No numbers changed. The reviewer reproduced every table.
