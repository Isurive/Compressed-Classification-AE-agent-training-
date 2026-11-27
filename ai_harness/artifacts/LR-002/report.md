# LR-002: validation-only lr/epoch extension for the dictionary heads and the linear head

**Agent:** `claude-orient-20261122-a`. **Approved by the user:** "Yes, run it" (2026-11-26), answering "Run the lr extension check next (validation only)".
**Pre-registration:** [task.json](task.json). The ledger `add` (2026-11-27 00:27:05 UTC = 17:27:05 local) predates the first run (smoke cell, 17:27:46 local; see [grid_log.txt](grid_log.txt)).
**Runs:** 78, 20 epochs each, 2026-11-26 17:27–18:14 local.
- Two smoke cells ran first (linear lr 1e-3 and sparse M2048 lr 1e-4, source, seed 42).
- The full grid then reused them through `done.json`.
- Exit 0, no tracebacks.

**Analysis:** [analyze.py](analyze.py) → [analysis.md](analysis.md) / [analysis.json](analysis.json).
- `analyze.py` was last written at 17:29:39 local, after the two smoke runs but before any other run completed (REVIEW-009 O-4). Its logic matches task.json.
- Every post-hoc number in this report comes from [posthoc.py](posthoc.py) → [posthoc.md](posthoc.md), added in revision 1 (REVIEW-009 O-1). It is a separate script, not pre-registered; `analyze.py` was not changed.

**Code:** [src/run_lrcheck2.py](../../../src/run_lrcheck2.py) (new).

**Revision 1** (2026-11-26) addresses [REVIEW-009](../REVIEW-009/report.md) S-1, S-2 and O-1 to O-5. The log is at the end.

**Commands:** from `src/`, `C:\Users\ericw\miniconda3\envs\ai360\python.exe -B -u`.
- Smoke: `run_lrcheck2.py grid --heads linear --lrs 1e-3 --levels source --seeds 42`, then the same command with `--heads sparse-M2048 --lrs 1e-4`.
- Full: `run_lrcheck2.py grid`.
- Analysis, from `Final Project`: `py -B ai_harness/artifacts/LR-002/analyze.py`.

**Validation only.**
- No ImageNetV2 or other test data was evaluated or read. The code never calls the test path, and the log has no test lines.
- Every number is the validation top-1 of the selected epoch, on the 5k ImageNet-val pool split, seeds 42–44.
- The metric is matched-compression: L090 runs are validated on L090 images.

## Question

LR-001 showed that lr 1e-3 is too high for the dictionary heads. It left two gaps:
- **Grid edge:** for sparse M4096 and dense M2048, accuracy was still rising at lr 1e-4, the lowest value tried (REVIEW-008 S-2).
- **Linear head:** it was never lr-checked.

This task locates the best (lr, epochs) setting for all four heads, so that a later ImageNetV2 re-test compares tuned heads.

## Design (as pre-registered)

| Item | Setting |
| --- | --- |
| Heads | sparse M2048 k512, sparse M4096 k1024, dense M2048 (CE + feature MSE 1/0/0, Xavier fc, He encoder); linear (Xavier, weight decay 1e-4, CE) |
| New runs | dictionary heads at lr 1e-4, 3e-5 and 1e-5; linear at 3e-3, 1e-3, 3e-4 and 1e-4. All 20 epochs. |
| Two budgets per run | lr is constant and each epoch is reseeded, so epochs 1–10 equal a 10-epoch run. The 10-epoch budget is read from the history: best epoch ≤ 10 by top-1, then lower CE. |
| Candidates | dictionary heads: LR-001's 3e-4 at 10 epochs, plus the 3 new lrs × {10, 20}. Linear: 4 lrs × {10, 20}. |
| Cells | source and L090 × seeds 42–44, paired by cell (the same cells as LR-001) |
| Baseline b | LR-001's recommendation at 10 epochs: sparse M2048 3e-4, sparse M4096 1e-4, dense 1e-4. Linear: the current 1e-3. |
| Switch rule | the best candidate c* replaces b only if its mean gain is ≥ 0.30 pp **and** it is positive in ≥ 5/6 cells |
| Flags | lower edge (dictionary 1e-5, linear 1e-4); upper edge (linear 3e-3); budget binds (adopted budget 20 with median selected epoch ≥ 19) |

## Results (pre-registered)

**Sanity: passed.** 24 of 24 reproductions are identical in top-1, selected epoch and split digest:
- the 10-epoch selection at lr 1e-4 matches all 18 LR-001 cells for the three dictionary heads;
- linear at lr 1e-3 over 10 epochs matches all 6 GRID-002 cells.

There is one split digest per seed.

**Mean validation top-1 (%), 6 cells.** Bold = adopted under the rule.

| Head | 3e-3 | 1e-3 | 3e-4 | 1e-4 | 3e-5 | 1e-5 |
| --- | --- | --- | --- | --- | --- | --- |
| sparse M2048 | | (LR-001: 60.04) | **60.38 @10** | 60.35 @10 · 60.65 @20 | 57.33 @10 · 60.05 @20 | 42.77 @10 · 54.22 @20 |
| sparse M4096 | | (LR-001: 60.57) | 61.22 @10 | **61.68 @10** · 61.70 @20 | 59.72 @10 · 61.62 @20 | 51.31 @10 · 58.12 @20 |
| dense M2048 | | (LR-001: 59.42) | 59.93 @10 | **60.54 @10** · 60.65 @20 | 58.13 @10 · 60.40 @20 | 46.00 @10 · 55.62 @20 |
| linear | 53.42 @10 · 53.54 @20 | 58.15 @10 · 58.15 @20 | 59.71 @10 · **60.30 @20** | 55.73 @10 · 59.73 @20 | | |

**Decisions:**

| Head | Best candidate | Gain vs baseline (pp, 95% CI, cells positive) | Adopted | Flags |
| --- | --- | --- | --- | --- |
| sparse M2048 | 1e-4 @ 20 | +0.28 [0.18, 0.38], 6/6. Just under the 0.30 threshold. | **3e-4 @ 10** (unchanged) | none, located |
| sparse M4096 | 1e-4 @ 20 | +0.02 [−0.03, 0.07], 1/6 | **1e-4 @ 10** (unchanged) | none, located |
| dense M2048 | 1e-4 @ 20 | +0.11 [−0.16, 0.38], 2/6 | **1e-4 @ 10** (unchanged) | none, located |
| linear | 3e-4 @ 20 | **+2.14 [1.84, 2.44], 6/6** | **3e-4 @ 20** (switch from 1e-3 @ 10) | none, located (median selected epoch 16.5) |

**Conclusions:**
1. **Within budgets of up to 20 epochs, the dictionary heads' best lr is about 1e-4 to 3e-4, and LR-001's settings stand.**
   - Lower rates are worse, even with 20 epochs: vs baseline, 3e-5 @ 20 is −0.06 to −0.32 pp and 1e-5 @ 20 is −3.6 to −6.2 pp.
   - Doubling to 20 epochs at 1e-4 adds at most +0.28 pp vs baseline (sparse M2048), below the switch threshold. Against 1e-4 @ 10 it adds +0.30 pp (REVIEW-009 O-5).
   - The edge concern from LR-001 / REVIEW-008 S-2 is resolved within this budget: below 1e-4 the curve turns down.
   - **"Located" holds only up to 20 epochs** (REVIEW-009 O-2). The dictionary heads at 3e-5 are still rising at epoch 20 (median selected epoch 19.5–20), so a much longer schedule at a lower lr was not explored.
2. **The linear head was badly under-tuned at lr 1e-3.**
   - lr 3e-4 for 20 epochs is **+2.14 pp** better (6/6). That is a larger gain than any dictionary head got from its lr fix.
   - lr 3e-3 is much worse (−4.7 pp), so the lr is bracketed on both sides.
   - At 3e-4 the linear curve is flat from epoch 11 to 20 (means 59.8–60.0); at 1e-4 it is still rising at epoch 20 (59.36 → 59.54 → 59.68). See posthoc.md.
3. **No instability** at lr ≤ 1e-4 for any dictionary head over 20 epochs (0/54 runs).

## Exploratory (not pre-registered decisions; validation only; selected on the same cells)

Gaps in pp, paired by cell: mean [95% CI, df 5], cells positive. Rows marked **(pre)** are pre-declared in task.json (at the adopted settings); all others are **post hoc**, generated by [posthoc.py](posthoc.py) (corrected after REVIEW-009 S-2).

**Budgets matter** (REVIEW-009 S-1). The adopted settings give the dictionary heads 10 epochs and the linear head 20: sparse M2048's 20-epoch candidate missed the switch threshold by 0.02 pp. So the head gaps are shown at the adopted settings **and** at equal budgets.

| Gap | lr 1e-3 @ 10 (old setting) | Adopted settings (dictionary 10 ep, linear 20 ep) | Equal 10 epochs | Equal 20 epochs |
| --- | --- | --- | --- | --- |
| sparse M2048 − linear | +1.89 [1.57, 2.21] 6/6 | **(pre)** +0.08 [−0.20, 0.36] 3/6 | +0.66 [0.49, 0.84] 6/6 | +0.36 [0.08, 0.64] 5/6 |
| dense M2048 − linear | — | +0.24 [0.01, 0.47] 5/6 | +0.82 [0.56, 1.09] 6/6 | +0.35 [0.04, 0.67] 5/6 |
| sparse M2048 − dense M2048 | +0.62 [0.42, 0.83] 6/6 (LR-001) | **(pre)** −0.16 [−0.35, 0.03] 1/6 | — (different lrs) | +0.00 [−0.33, 0.33] 3/6 (both 1e-4) |
| sparse M4096 − sparse M2048 | +0.53 [0.28, 0.78] 6/6 (LR-001) | **(pre)** +1.30 [0.98, 1.62] 6/6 (mixed lrs) | +1.33 [0.96, 1.70] 6/6 (both 1e-4) | +1.04 [0.82, 1.27] 6/6 (both 1e-4) |
| sparse M4096 − linear | — | +1.38 [0.97, 1.79] 6/6 | — | — |

Settings used in the equal-budget columns:
- **10 epochs:** sparse M2048 3e-4, dense 1e-4, linear 3e-4; for the M gap, both sparse heads at 1e-4.
- **20 epochs:** each head's best candidate at 20 epochs, which is 1e-4 for every dictionary head and 3e-4 for linear.

Without checkpoint selection, sparse M2048 (3e-4, epoch 10) − linear (3e-4, epoch 20) is +0.25 [−0.37, +0.87], 5/6. Best-of-20 selection on the linear head's flat plateau flatters it somewhat.

Per level at the adopted settings (mean %, source / L090 / drop):
- sparse M2048: 64.67 / 56.08 / 8.59
- sparse M4096: 66.02 / 57.33 / 8.69
- dense M2048: 64.71 / 56.37 / 8.34
- linear: 64.42 / 56.17 / 8.25

So the compression drop (source − L090) is about 8.2–8.7 pp for every head.

**What this suggests (to be tested on ImageNetV2):**
- **Fixing the linear head's lr removes most of its validation deficit,** about 1.2–1.8 pp of the 1.89 pp gap at lr 1e-3. Whether a residual 0.3–0.7 pp advantage for the 2048-wide dictionary heads remains **depends on the epoch budget**:
  - about 0 at the adopted (unequal) settings;
  - +0.36 at equal 20 epochs;
  - +0.66 at equal 10 epochs.

  So the earlier "sparse beats linear by about 1 pp" (GRID-002, SEED-001) is **largely, possibly fully, explained by the linear head's lr**. It is not shown to be fully an artifact. Any residual is below the project's 1 pp tolerance.
- **Sparse vs dense:** at a common lr and budget (1e-4 @ 20) the gap is +0.00 [−0.33, +0.33]. SEED-001's "sparsity adds about 0.4 pp" is consistent with dense being hurt more by lr 1e-3 (LR-001), and no sparsity benefit is visible at a stable lr.
- **The larger head (sparse M4096) is clearly ahead:** +1.0 to +1.4 pp over sparse M2048 depending on the setting, and +1.38 over linear at the adopted settings.
  - It has about 8.3 M trainable parameters vs 4.2 M for M2048, including the decoder, which serves only the auxiliary loss. The deployed encoder + classifier is about 6.2 M vs 3.1 M (REVIEW-009 O-3). It also has twice as many active atoms.
  - This is a capacity effect at fixed active fraction, not evidence that sparsity itself helps. There is no dense M4096 control, so width and active-atom count cannot be separated.
- **No head reduces the compression drop,** consistent with every earlier task.

## Caveats

1. **Validation only, with repeated selection on the same cells.** Seeds 42–44 (source and L090) have now been used to choose settings twice (LR-001, LR-002). The adopted settings and the gaps above are therefore optimistic. Only a test on new seeds and ImageNetV2 can confirm them.
2. **Grid holes:**
   - 3e-4 for 20 epochs was not run for the dictionary heads.
   - lr between 3e-4 and 1e-3 was not run for any head.
   - Since 1e-4 @ 20 gains at most +0.28 pp vs baseline, a large missed gain is unlikely, but it is not excluded.
   - Budgets beyond 20 epochs were not tried (conclusion 1).
3. **Other hyperparameters were not re-tuned** at the new lrs: k (512 / 1024), the dictionary heads' zero weight decay, the linear head's weight decay 1e-4, and initialisation were all tuned at lr 1e-3. A sparse-specific setting (k) could be mistuned at the new lr.
4. **Unequal budgets at the adopted settings:** the linear head trains for 20 epochs and the dictionary heads for 10. This is the pre-registered outcome, because 20 epochs did not help the dictionary heads by ≥ 0.30 pp. It changes the head gaps by up to about 0.6 pp, so the exploratory table shows equal-budget columns, and the proposed re-test uses one common budget (REVIEW-009 S-1).
5. **Selection optimism differs across curve shapes** (REVIEW-008 O-4, REVIEW-009 S-1):
   - Best-of-n selection flatters flat or noisy curves.
   - Best-of-20 on the linear head's plateau selects 60.30 against epoch-by-epoch means of 59.8–60.0.
   - So the linear head's adopted-setting number is slightly optimistic relative to the 10-epoch dictionary heads.

## What this means for earlier results

Earlier ImageNetV2 results stay valid as measured (lr 1e-3). The head comparisons drawn from them are **largely lr-dependent**; this rests on validation evidence from seeds 42–44, already used twice for selection, so it is not a test result (REVIEW-009 O-3):
- **GRID-002 / SEED-001 "sparse > linear by about 1 pp":** the linear head was under-tuned by about 2 pp on validation. At tuned lrs the validation residual is about 0 to 0.7 pp, depending on the budget.
  - GRID-002's dictionary heads used the full loss (reconstruction 1, dependency 0.1, coherence 0.05); LR-002 uses the MSE-only loss of SEED-001. ABL-001 found the extra terms had no measurable effect, so this difference is minor.
- **SEED-001 "sparsity adds about 0.4 pp":** no sparsity benefit is visible at a stable lr on validation (LR-001; +0.00 at a common 1e-4 @ 20 here).
- **MSWEEP-001 "+0.46 pp for M4096 over M2048":** on validation, the gap is about +1.0 to +1.3 pp at a stable common lr (1e-4 @ 20 or @ 10).

The compression-cost results (level 80 costs 0.5–1.2 pp, level 90 1.5–2.3 pp, no head reduces the drop) are not contradicted. The validation drops here are similar for every head.

## Proposed next step (requires separate user approval; not run)

A **confirmatory ImageNetV2 re-test with one common budget** (revised after REVIEW-009 S-1, so the outcome cannot hinge on a 0.02 pp threshold miss):
- **Settings:** every head trains for **20 epochs** at its best lr at that budget: sparse M2048, sparse M4096 and dense M2048 at 1e-4; linear at 3e-4.
  - For the dictionary heads this is a post-hoc departure from the rule's adopted 10-epoch settings. The validation cost is −0.02 to +0.30 pp, i.e. 20 epochs is never worse.
  - Equal budgets and one lr for all dictionary heads make the comparisons clean.
- **Data:** levels source / 80 / 90; **new seeds 45–51**, not used for any lr choice. That is 84 runs, about 50 minutes.
- **Pre-registration:**
  - primary: sparse M2048 − linear, sparse M2048 − dense, and sparse M4096 − sparse M2048 (plus M4096 − linear);
  - per-head compression cost (source − L80, source − L90);
  - the same detected / meaningful (≥ 1 pp) rule and Holm correction as before.
- **Outcome:** this decides which head claims go into the write-up.

## Checks

- 78/78 `done.json` files, each with a 20-epoch history and fields matching its path. `analyze.py` asserts this, and asserts that each run's best epoch equals its history's selection.
- `results.csv` has 78 rows. No `last.pt` remains; linear runs also keep `history.json`.
- 24/24 sanity reproductions are identical.
- The user's `src/` files are unchanged: only `src/run_lrcheck2.py` is new. `harness.py check` passes at submission.
- `runs/LR-002` (3.2 GB) holds `best.pt` checkpoints and is not committed. `results.csv` is the full table; the smoke runs' `results_partial.csv` was deleted in revision 1 (REVIEW-009 O-5).

## Revision log

- **Revision 1 (2026-11-26, after REVIEW-009):**
  - **S-1:** the exploratory table now shows equal 10- and 20-epoch budgets next to the adopted settings.
    - "Tied within ±0.25 pp" became "the linear lr fix removes most of the gap; the 0–0.7 pp residual depends on the budget".
    - "Probably lr artifacts" became "largely lr-dependent".
    - The proposed ImageNetV2 re-test now uses one common 20-epoch budget, and caveats 4 and 5 are restated.
  - **S-2:** row provenance is marked in the table: (pre) rows vs post hoc.
  - **O-1:** a new `posthoc.py` / `posthoc.md` generates every post-hoc number. It is separate from the pre-registered `analyze.py`, which is unchanged, as are its outputs.
  - **O-2:** "located" is now conditional on budgets of up to 20 epochs.
  - **O-3:** the implications now say "on validation"; they note the GRID-002 loss difference, the mixed lrs in the M gap (now also given at a common lr), deployed vs total parameters, and the missing dense M4096 control.
  - **O-4:** the `analyze.py` write time is recorded.
  - **O-5:** "vs baseline" was added; `results_partial.csv` was deleted.
  - No runs were added, and no pre-registered number changed.
