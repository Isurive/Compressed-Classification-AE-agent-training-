# LR-001: validation-only learning-rate check for large dictionary heads

**Agent:** `claude-orient-20261122-a`. **Approved by the user:** "Yes, run lr check" (2026-11-26).
**Pre-registration:** [task.json](task.json); the ledger `add` (22:53:53 UTC = 15:53:53 local) predates the first run (15:54:40 local, smoke cell).
**Runs:** 54, 2026-11-26 15:54–16:18 local, in [grid_log.txt](grid_log.txt).
- The first cell (sparse M2048, lr 1e-3, source, seed 42) ran alone as a smoke test.
- The full grid then reused it from its `done.json`.
- Exit 0, no tracebacks.

A game was using the GPU at the same time. That changes timing only; the twelve repeated cells reproduce ABL-001 exactly (Sanity).

**Analysis:** [analyze.py](analyze.py) → [analysis.md](analysis.md) / [analysis.json](analysis.json). **Code:** [src/run_lrcheck.py](../../../src/run_lrcheck.py) (new).

**Validation only.** No ImageNetV2 or other test data was evaluated or read; the log contains no test line. Every number below is best-checkpoint top-1 on the 5,000-image validation split of the ImageNet-val pool.
- The validation metric is **matched-compression**: L090 runs are validated on L090-compressed validation images, source runs on source images. It is not a common held-out set across levels (REVIEW-008 O-5).

**Revision 1** (2026-11-26) addresses [REVIEW-008](../REVIEW-008/report.md) S-1, S-2 and O-1 to O-6. The log is at the end.

**Commands** (from `src/`, `C:\Users\ericw\miniconda3\envs\ai360\python.exe -B -u`, output prefixed with local time and appended to `grid_log.txt`):
- Smoke cell: `run_lrcheck.py grid --heads sparse-M2048 --lrs 1e-3 --levels source --seeds 42`
- Full grid: `run_lrcheck.py grid`
- Analysis (from `Final Project`): `py -B ai_harness/artifacts/LR-001/analyze.py`

## Question

MSWEEP-001 / REVIEW-007 (S-1) found that at Adam lr 1e-3 the training loss rises late in training in 15/21 M = 2048 and 21/21 M = 4096 sparse runs. Two questions follow:
- Does a lower learning rate remove this instability?
- Does it change accuracy enough that the project's M ≥ 2048 results should be re-run?

## Design (as pre-registered)

| Item | Setting |
| --- | --- |
| Heads | sparse M2048 k512; sparse M4096 k1024; dense M2048 k2048. Xavier fc, He encoder, loss CE + feature MSE (1 / 0 / 0), as in SEED-001 and MSWEEP-001. The linear head is **not** tested. |
| lr | 1e-3 (current), 3e-4, 1e-4 |
| Cells | levels source and L090 × seeds 42, 43, 44 = 6 cells per head and lr, paired by cell. Seeds 45–51 were deliberately not used. |
| Fixed | 10 epochs, batch 128, Adam, `best.pt` by validation top-1 then CE, cached GRID-001 features |
| Switch rule | per head, lr* = highest mean validation top-1. Replace 1e-3 by lr* only if the paired gain is ≥ 0.30 pp **and** positive in ≥ 5/6 cells |
| Instability | a run is unstable if its training loss rises > 5% between consecutive epochs. "Resolved" at an lr if ≤ 1/6 runs are unstable |
| Truncation flag | median best epoch at lr* equal to 10 |

## Results (pre-registered)

| Head | lr | Mean val top-1 | Unstable runs | Median / max drop, best epoch → epoch 10 | Median best epoch |
| --- | --- | --- | --- | --- | --- |
| sparse M2048 | 1e-3 | 60.04 | 4/6 | 2.54 / 4.14 pp | 6 |
| | **3e-4** | **60.38** | 0/6 | 0.11 / 0.58 | 8.5 |
| | 1e-4 | 60.35 | 0/6 | 0.16 / 0.52 | 9.5 |
| sparse M4096 | 1e-3 | 60.57 | 6/6 | 4.87 / 8.14 | 5 |
| | 3e-4 | 61.22 | 0/6 | 0.12 / 0.46 | 6.5 |
| | **1e-4** | **61.68** | 0/6 | 0.04 / 0.48 | 9.5 |
| dense M2048 | 1e-3 | 59.42 | 6/6 | 5.09 / 6.24 | 6 |
| | 3e-4 | 59.93 | 0/6 | 0.15 / 0.76 | 8 |
| | **1e-4** | **60.54** | 0/6 | 0.20 / 0.64 | 8.5 |

**Lower lr minus 1e-3** (pp, paired by cell; mean [95% CI, df 5], cells positive):

| Head | 3e-4 | 1e-4 |
| --- | --- | --- |
| sparse M2048 | +0.33 [0.13, 0.54] 6/6 | +0.30 [0.06, 0.54] 5/6 |
| sparse M4096 | +0.65 [0.38, 0.93] 6/6 | +1.11 [0.71, 1.50] 6/6 |
| dense M2048 | +0.51 [0.30, 0.72] 6/6 | +1.12 [0.82, 1.42] 6/6 |

**Decisions under the rule:**

| Head | lr* | Switch? | Instability at 1e-3 → at lower lr | Truncation flag |
| --- | --- | --- | --- | --- |
| sparse M2048 | 3e-4 | **yes**, but marginal: +0.33 pp, just over the 0.30 threshold. 1e-4 is a near-tie (+0.30, 5/6) and also meets the rule. | 4/6 → 0/6 | not flagged |
| sparse M4096 | 1e-4 | **yes** (+1.11 pp, 6/6) | 6/6 → 0/6 | not flagged (median 9.5) |
| dense M2048 | 1e-4 | **yes** (+1.12 pp, 6/6) | 6/6 → 0/6 | not flagged (median 8.5) |

**Conclusion:** lr 1e-3 is too high for all three heads.
- Lowering the learning rate to 3e-4 or 1e-4 removes the late-training divergence completely (0/18 unstable runs at each lower lr).
- It raises validation top-1 by 0.3 pp (sparse M2048) to about 1.1 pp (sparse M4096, dense M2048).

**Sanity check: passed.** The 12 lr 1e-3 cells repeat ABL-001 `abl-mse` runs with identical settings. All 12 reproduce exactly: the same validation top-1 (difference 0.00 pp), best epoch and split digest. There is one split digest per seed across all heads and lrs.

## Caveats on the decision

1. **Edge of the grid: the optimum for sparse M4096 and dense M2048 is not located** (corrected after REVIEW-008 S-1, S-2). For these two heads lr* = 1e-4 is the lowest value tried, and accuracy is still rising steeply there. Paired 1e-4 − 3e-4 (from REVIEW-008 §5, validation):
   - sparse M4096: **+0.45 pp [0.29, 0.62], 6/6**
   - dense M2048: **+0.60 pp [0.32, 0.89], 6/6**
   - sparse M2048: −0.03 pp [−0.14, +0.08], 3/6, so this head is on a plateau between 3e-4 and 1e-4.

   A further step down could therefore gain as much as the effects under study, not just "a little". The epoch budget may also bind: median best epoch at 1e-4 is 9.5 for sparse M4096 (3 of 6 runs peaked at epoch 10) and for sparse M2048, and 8.5 for dense M2048. The pre-registered truncation flag (median = 10) does not fire for any head.
2. **Selection optimism.** lr* is chosen on the same validation cells used to score it, so the gains for lr* are slightly optimistic. The full table for every lr does not depend on that choice.
   - task.json says the best-checkpoint bias applies to all lr arms "alike". That is probably wrong in direction (REVIEW-008 O-4): best-of-10-epoch selection flatters the peaked, noisy curves at 1e-3 more than the near-monotone curves at lower lr.
   - Without selection (epoch-10 validation top-1) the lower-lr advantage is larger: means at 1e-3 / 3e-4 / 1e-4 are 57.60 / 60.17 / 60.15 (sparse M2048), 55.10 / 61.03 / 61.55 (M4096) and 54.29 / 59.68 / 60.26 (dense), from REVIEW-008 §5. The selected-checkpoint gains above are therefore conservative.
3. **Validation, not test.** These are pool-validation numbers on seeds 42–44, not ImageNetV2 results. Validation and ImageNetV2 accuracies differ in level (about 60% vs 47%); the direction is likely to transfer, but that is unverified.
4. **Linear head untested.** It used lr 1e-3 with weight decay 1e-4 throughout. If the dictionary heads are re-run at a lower lr, comparing them with an untuned linear head would favour the dictionary heads. A fair re-test should give the linear head the same lr check.

## Exploratory (not pre-registered; validation only; do not treat as confirmatory)

These change how earlier results should be read. **None of the gaps at 1e-4 is a comparison at each head's optimum** (caveat 1): sparse M2048 is on its plateau there while sparse M4096 and dense M2048 are still improving. The same holds for the "each head at its lr*" column, because two of the three lr* values sit at the grid edge.

| Gap (pp, paired by cell) | at 1e-3 | at 3e-4 | at 1e-4 | each head at its lr* |
| --- | --- | --- | --- | --- |
| sparse M4096 − sparse M2048 | +0.53 [0.28, 0.78] 6/6 | +0.85 [0.60, 1.09] 6/6 | +1.33 [0.96, 1.70] 6/6 | +1.30 [0.98, 1.62] 6/6 |
| sparse M2048 − dense M2048 | +0.62 [0.42, 0.83] 6/6 | +0.44 [0.27, 0.62] 6/6 | **−0.19 [−0.36, −0.02] 1/6** | **−0.16 [−0.35, 0.03] 1/6** |

- **The sparse advantage over dense disappears at a stable lr (on validation).**
  - At lr 1e-3, sparse beats dense by about 0.6 pp on validation, in line with SEED-001's test result (+0.41 pp, detected but below the 1 pp tolerance).
  - Dense diverged more often at 1e-3 (6/6 vs 4/6 runs) and gains more from a lower lr. At 1e-4 the gap is −0.19 pp, but it is not robust: −0.16 [−0.35, +0.03] at each head's lr*, and −0.11 [−0.48, +0.27], 2/6, without checkpoint selection (REVIEW-008 O-1). The supported statement is "no sparse advantage", not "dense is ahead".
  - The SEED-001 sparse-vs-dense result is **consistent with** dense being hurt more by the too-high lr, rather than with a benefit of sparsity.
  - Untested alternative: k = 512 (and wd 0, Xavier) were tuned at lr 1e-3, so the sparse head's extra hyperparameter k may be mistuned at 1e-4.
  - Dense is still improving at 1e-4 (caveat 1), so the gap at each head's real optimum is unknown. Not yet tested on ImageNetV2.
- **Large dictionaries gain more at a lower lr.** The M4096 − M2048 gap grows from +0.53 pp (1e-3) to +1.33 pp (1e-4).
  - This supports MSWEEP-001's caveat: the "diminishing returns past 2048" were at least partly an optimisation artifact.
  - The MSWEEP-001 M ≥ 2048 numbers **probably understate** the large-M gain (validation evidence only; REVIEW-008 O-2).
- **Sparse vs linear** cannot be assessed here (linear not tested; caveat 4).

## What this means for earlier results

Earlier results stay valid as measured under their pre-registered protocol (lr 1e-3, best-checkpoint selection). But they are conditional on a learning rate that is now shown to be too high for these heads.
- **Statements that may change:** sparse > dense (SEED-001), the size of the M4096 − M2048 gain (MSWEEP-001), and the sparse − linear margin (GRID-002, SEED-001).
- **Not tested here:** the compression costs (L80 0.5–1.2 pp, L90 1.5–2.3 pp) and "no head reduces the compression drop". For the dictionary heads, those ImageNetV2 numbers are also conditional on lr 1e-3 (REVIEW-008 O-5).
  - Descriptively, the validation source − L090 differences barely move with lr: sparse M2048 8.46 / 8.59 / 8.55 pp and dense 8.22 / 8.51 / 8.34 pp at 1e-3 / 3e-4 / 1e-4 (REVIEW-008 §5).

## Proposed next steps (each requires separate user approval; none run)

Revised after REVIEW-008 S-2: the optimum must be located before a confirmatory test.

1. **Locate the optimum (validation only).**
   - Extend the lr grid downward (3e-5, possibly 1e-5) and/or train for 20 epochs, for **all three dictionary heads**. Dense M2048 and sparse M4096 are still rising at 1e-4; sparse M2048 is included so every head is checked under the same rule.
   - Add the same validation-only lr check for the **linear head** (caveat 4).
   - Same seeds 42–44, levels source / L090, and a pre-registered switch rule. Roughly 30–40 runs, about 20 minutes.
2. **Then a confirmatory ImageNetV2 re-test** at each head's located lr.
   - Heads: sparse M2048, dense M2048, sparse M4096 and linear.
   - Data: seeds 45–51, levels source / L80 / L90.
   - Pre-registration: sparse − dense, sparse − linear and M4096 − M2048 as before.
   - If one common lr is used for simplicity, that is a post-hoc choice: for sparse M2048 this check's rule recommends 3e-4, not 1e-4, although the two are tied within noise (−0.03 pp, 3/6; REVIEW-008 O-3).

## Checks

- 54/54 `done.json` files, each with its 10-epoch history. `results.csv` has 54 rows. No `last.pt` remains.
- No test evaluation: `run_lrcheck.py` never calls `rg.test_run` or loads ImageNetV2, and the log has no test lines.
- The user's `src/` files are unchanged: only `src/run_lrcheck.py` is new. `runs/LR-001` is 3.1 GB (best.pt only; not committed). `harness.py check` passes at submission.
- Each run directory also has a 1-byte `.training.lock` left by `sparse_workflow`. It is harmless, and ABL-001 run directories have it too.

## Revision log

- **Revision 1 (2026-11-26, after REVIEW-008):**
  - S-1: dense M2048's median best epoch at lr* corrected to 8.5 in the decision table; caveat 1 made head-specific.
  - S-2: caveat 1 restated with the 1e-4 − 3e-4 slopes. The exploratory section now says the 1e-4 and lr* gaps are not at-optimum comparisons. The next step now locates the optimum for all three dictionary heads and the linear head before any confirmatory test.
  - O-1: "dense is ahead" changed to "the sparse advantage disappears"; "plausibly explained" changed to "consistent with". The k-tuned-at-1e-3 alternative is added.
  - O-2: the MSWEEP-001 statement is hedged.
  - O-3: the common-lr choice is labeled post hoc.
  - O-4: added the selection-optimism direction and epoch-10 means. task.json is the pre-registration, so it is not edited; the correction is here.
  - O-5: the metric is stated as matched-compression; compression costs are noted as lr-conditional.
  - O-6: the command lines are recorded. `analyze.py` now asserts 10-epoch histories and that the head, lr, level and seed fields match each path. `analysis.md` and `analysis.json` were regenerated and are byte-identical.
  - No runs were added and no numbers changed.
