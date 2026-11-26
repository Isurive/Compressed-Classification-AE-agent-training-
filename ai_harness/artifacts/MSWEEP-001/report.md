# MSWEEP-001: clean dictionary-size (M) sweep

**Agent:** `claude-orient-20261122-a`. **Approved by the user:** "Yes, run M sweep" (2026-11-26), and after the interruption, "redo the 63-run M sweep".
**Runs:** 63 new runs.
- **First attempt:** 2026-11-26 00:49 to 01:09. The process was killed after 57 of 63 runs: the log stops mid-run with no error or exit line, most likely because the laptop slept or the app closed.
- **Resumed:** 12:45 to 12:48. The 6 remaining runs finished, and the 57 completed runs were reused unchanged.
- **Logs:** the interruption and the resume are recorded in [grid_log.txt](grid_log.txt). No tracebacks.

**Pre-registration:** [task.json](task.json), added to the ledger 22 s before the first run. Analysis: [analyze.py](analyze.py) → [analysis.md](analysis.md) / [analysis.json](analysis.json).
- **`analyze.py` timing:** its last code edit was about 4 s after the first test row existed, when 1 of 63 runs was complete. The rule it implements is fixed by `task.json` (REVIEW-007 O-3).
- **Revision 1:** only the "10 tests" labels in `analyze.py` and `analysis.md` were corrected to 13, and `analysis.md` was regenerated. `analysis.json` is byte-identical.

**Revision 1** (2026-11-26) addresses [REVIEW-007](../REVIEW-007/report.md) S-1 to S-3 and O-1 to O-3. The log is at the end.

## Question

The project's original research direction is how dictionary size affects the result. This sweep varies **M at a fixed active fraction k = M/4**, so the number of active atoms and the parameter count grow with M as well (see Interpretation, item 2). The GRID-001 M sweep could not answer this, for three reasons:
- the heads inherited the ImageNet classifier;
- the small dictionaries kept their epoch-1 checkpoints;
- k was held at 512, so M = 512 was fully dense.

This sweep fixes all three.

## Design (as pre-registered)

| Item | Setting |
| --- | --- |
| Head | Sparse dictionary; Xavier fc; He encoder; loss CE + feature MSE (reconstruction 1, dependency 0, coherence 0), the same as SEED-001's sparse head |
| Sizes | M ∈ {512, 1024, 2048, 4096}, with a **fixed active fraction k = M/4** → k ∈ {128, 256, 512, 1024} |
| Reuse | The M = 2048 / k = 512 cells are SEED-001's sparse runs, with identical settings. Three sizes are new. |
| Training | 10 epochs, batch 128, Adam 1e-3, `best.pt` by validation top-1 then CE, cached GRID-001 features (read only) |
| Levels / seeds | source, 80 (q20, 50.3:1 against raw), 90 (q10) / **seeds 45–51** (the same splits as SEED-001) |
| Test | Original ImageNetV2, once per run, after training |
| Confirmatory tests | A(M) − A(2048) for M ∈ {512, 1024, 4096}, per level and pooled (the per-seed mean over the levels), plus the **per-seed slope of pooled accuracy on log₂M** (pp per doubling). Paired t, 95% CI (df 6), sign counts, Holm adjustment. |
| Decision rule | **Detected:** the CI excludes 0. **Meaningful:** detected and \|mean\| ≥ 1 pp. For the slope: \|slope\| × 3 ≥ 1 pp across 512 → 4096. |
| Sanity check | Flag any M whose median best epoch is ≤ 2 |

**Correction to the pre-registration's arithmetic.** The task text says "9 differences plus 1 trend (10 tests)". The rule it states defines 3 sizes × (3 levels + pooled) = **12 differences** plus 1 trend, **13 tests**. The analysis applies the rule as written, with Holm over all 13, which is the stricter choice. Holm over 10 tests (dropping the pooled rows) gives the **same detected set**. No conclusion depends on this.

## Results: ImageNetV2-source top-1 %, mean of seeds 45–51

| M (k) | source | L80 | L90 | Pooled |
| --- | --- | --- | --- | --- |
| 512 (128) | 45.57 | 44.57 | 42.85 | 44.33 |
| 1024 (256) | 47.14 | 46.28 | 44.92 | 46.12 |
| 2048 (512) | 48.31 | 47.39 | 46.13 | 47.28 |
| **4096 (1024)** | **48.64** | **47.81** | **46.75** | **47.73** |

### Confirmatory family (pp, paired by seed)

| Test | source | L80 | L90 | Pooled | Verdict |
| --- | --- | --- | --- | --- | --- |
| M512 − M2048 | −2.74 [−3.05, −2.42] 0/7 | −2.82 [−3.34, −2.30] 0/7 | −3.28 [−3.67, −2.89] 0/7 | **−2.95 [−3.20, −2.69] 0/7** | detected and meaningful at every level |
| M1024 − M2048 | −1.17 [−1.44, −0.90] 0/7 | −1.11 [−1.61, −0.60] 0/7 | −1.20 [−1.56, −0.85] 0/7 | **−1.16 [−1.41, −0.90] 0/7** | detected and meaningful at every level |
| M4096 − M2048 | +0.32 [−0.18, 0.83] 6/7 | +0.42 [−0.11, 0.95] 6/7 | +0.63 [0.31, 0.95] 7/7 | **+0.46 [0.19, 0.72] 7/7** | not detected at source or L80. Detected at L90 and pooled, but **not meaningful** (< 1 pp). |
| **Slope per doubling of M** | | | | **+1.14 [1.07, 1.20] 7/7** | detected and meaningful (≈ 3.4 pp across 512 → 4096) |

Each cell shows mean [95% CI] and the number of seeds where the difference is positive. All Holm-adjusted p-values for the detected tests are below 0.02; see [analysis.md](analysis.md).

**Sanity check, best epoch:**

| M | Median best epoch | Range |
| --- | --- | --- |
| 512 | 9 | 2–10 |
| 1024 | 8 | 5–10 |
| 2048 | 7 | 5–9 |
| 4096 | 6 | 4–7 |

No size is flagged. The GRID-001 epoch-1 problem does not recur.

## Exploratory (not pre-registered; do not treat as confirmatory)

- **Diminishing returns, possibly limited by the learning rate.** Successive doublings add +1.8, +1.2 and +0.5 pp (pooled means). So the log-linear slope (1.14) averages over a gain that shrinks with M. But the large dictionaries train unstably at the fixed lr 1e-3 (see the next item), so the small gain past 2048 may partly be an optimization artifact rather than a capacity ceiling.
- **Training instability grows with M** (found by REVIEW-007, S-1; confirmed by the author from the logs). A "rise" means the training loss went up by more than 5% between consecutive epochs.

  | M | Runs with a training-loss rise | Median val top-1 drop, best epoch → epoch 10 | Max drop |
  | --- | --- | --- | --- |
  | 512 | 0/21 | 0.3 pp | 1.1 pp |
  | 1024 | 0/21 | 0.4 pp | 0.9 pp |
  | 2048 (SEED-001) | 15/21 | 2.4 pp | 4.3 pp |
  | 4096 | 21/21 | 5.9 pp | 8.0 pp |

  - **Example:** M4096 / L090 / seed 51. The training loss goes 0.18 → 0.23 → 0.55 over epochs 8–10, and validation top-1 goes 55.8 → 52.9 → 50.2.
  - **Effect on the confirmatory numbers:** none. Checkpoints are selected by validation top-1, so they are valid measurements under the pre-registered protocol.
  - **What it biases:** the comparison *against* large M. Whether a lower lr at M ≥ 2048 removes the instability and widens the M = 4096 gain is **unresolved**; a validation-only lr check would answer it.
  - **Earlier experiments:** this also affects the M = 2048 heads used in GRID-002, ABL-001 and SEED-001.
- **Size × compression.** The per-level slopes are 1.04 (source), 1.08 (L80) and 1.29 (L90).
  - The paired L90 − source slope difference is +0.26 pp [−0.01, 0.52], with 6 of 7 seeds positive. That is **not detected**.
  - At most, larger dictionaries may help slightly more on heavily compressed training data. This would need its own test.
- **Against the linear head** (SEED-001, same seeds; pooled mean 46.31%):
  - M = 512 is about 2 pp **below** linear: −1.98 [−2.18, −1.78], 0/7;
  - M = 1024 is slightly but consistently below it: −0.19 [−0.34, −0.05], 0/7 (REVIEW-007 O-2);
  - M = 2048 and 4096 are above it.

  A small dictionary head is worse than the linear head (exploratory).

## Interpretation and confounds

1. **A larger dictionary at a fixed active fraction (k = M/4) helps.** Accuracy rises by about 1.1 pp per doubling of M over 512 → 4096, at all three compression levels.
   - M, the number of active atoms k and the parameter count rise together, so this is not an effect of available atoms alone (item 2).
   - The gain is large between 512 and 2048 (≈ 3 pp) and small beyond 2048 (+0.46 pp, below the tolerance). The small gain at the top is conditional on lr 1e-3 (item 3).
2. **This does not isolate "dictionary size" from model size.** With k = M/4, these all grow with M together:
   - the parameter count (encoder 512×M, decoder M×512, classifier M×1000): about 1.0 M, 2.1 M, 4.2 M and 8.3 M;
   - the number of active atoms per image (128 → 1024).

   The result is "a larger dictionary with the same active fraction helps". It is not "more atoms, with everything else equal, helps". Separating M from k would need a fixed-k arm, which was not run.
3. **The tuning point and instability** (corrected after REVIEW-007, S-1). lr, epochs and k/M were tuned only at M = 2048.
   - At the fixed lr 1e-3, the larger dictionaries **diverge late in training**: 15/21 runs at M = 2048 and 21/21 at M = 4096 show a rising training loss (table above).
   - Their earlier best epochs (median 6 at M = 4096 vs 9 at M = 512) reflect that divergence, not faster fitting.
   - Selection by validation top-1 keeps the reported numbers valid. But the M ≥ 2048 results, and the apparent diminishing returns, are **conditional on lr 1e-3** and probably understate what large M could reach.
4. **Compression tolerance.** As in SEED-001, no tested factor detectably reduces the accuracy drop from compression. The M × level interaction above is exploratory and not detected.
5. **Scope as before.** Frozen ImageNet-pretrained ResNet-18; only the head is trained on compressed images; Xavier-initialized head. ImageNetV2 has informed earlier follow-up designs (test reuse), so these seeds are new but the test set is not.

## Checks

- **Runs:** 63/63 `done.json`; `results.csv` has 105 rows (21 source runs × 1 + 42 compressed runs × 2); 0 missing cells; no tracebacks.
- **Splits:** one split digest per seed across all four sizes, including the reused SEED-001 cells.
- **Settings:** each new `done.json` records Xavier init, weight decay 0, k = M/4 and loss weights 1/0/0. They equal SEED-001's sparse settings except for atoms and k (`runs/MSWEEP-001/settings.json`).
- **Code:** [src/run_msweep.py](../../../src/run_msweep.py) is new. It reuses `run_grid.py` and `run_ablation.py` unchanged. `last.pt` was deleted for these runs.
- **Integrity:** the user's `src/` files and prior artifacts are verified below, and `harness.py check` passes. Result: the 8 user `src/` files are unchanged; there is no `src/__pycache__`; `harness.py check` is ok. `runs/MSWEEP-001` is 3.3 GB.

## Revision log

- **Revision 1 (2026-11-26, after REVIEW-007):**
  - S-1: added the instability table and corrected caveat 3. The M ≥ 2048 results and the diminishing returns are conditional on lr 1e-3; this also affects earlier M = 2048 heads.
  - S-2: the "available atoms" headline and Question restated as M at a fixed active fraction.
  - S-3: the ledger summary rewritten, with the exploratory labels.
  - O-1: the "10 tests" labels fixed in `analyze.py` and `analysis.md` (`analysis.json` byte-identical).
  - O-2: the linear comparison is now quantified.
  - O-3: the `analyze.py` timing is stated.
  - No numbers changed and nothing was re-run.
