# REVIEW-008: independent review of LR-001

**Reviewer:** `reviewer-20261126-g` (did not author LR-001). **Date:** 2026-11-26.
**Reviewed:** `ai_harness/artifacts/LR-001/` (task.json, report.md, analyze.py, analysis.md/.json, grid_log.txt), `src/run_lrcheck.py`, `runs/LR-001/` (54 `done.json`, `results.csv`, `settings.json`), the ledger record (`harness.py show LR-001`), and `runs/ABL-001/abl-mse/{sparse,dense}/{source,L090}/seed4x/done.json`.
All artifact SHA-256 hashes match the ledger's submitted hashes, so this review covers the submitted versions.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of LR-001) saved it here verbatim, with no edits to content, on 2026-11-26.

Verdict: revise

Two small must-fix items (one factual error in the decision table, one understated grid-edge caveat that matters for the proposed next step). The pre-registered analysis itself is implemented correctly. Every pre-registered number and decision reproduces exactly from the raw `done.json` files. The validation-only claim holds.

## What I checked and how

Independent recomputation used my own standard-library script, kept outside the project in the session scratchpad and run with `py -I`. It reads only `done.json`, `results.csv` and the ABL-001 `done.json` files. I did not run `analyze.py`, did not train, did not use the GPU and did not load any checkpoint. I wrote nothing outside `ai_harness/artifacts/REVIEW-008`.

### 1. Pre-registration timing: verified

| Event | UTC |
| --- | --- |
| `ai_harness/artifacts/LR-001/task.json` created (file system) | 22:53:52.93 |
| Ledger `add` | 22:53:53 |
| Ledger `claim` | 22:53:54 |
| `src/run_lrcheck.py` created | 22:54:14 |
| `runs/LR-001` created | 22:54:36 |
| Log `start` (15:54:29 local, UTC−7) | 22:54:29 |
| First run line `[15:54:40]` and first `.training.lock` | 22:54:40 |
| Earliest `done.json` | 22:55:06 |

- The ledger spec text is identical to task.json, and its hash matches the ledger.
- No other `runs/` directory was created on 2026-11-26 after SEED-001 / MSWEEP-001, so there is no sign of an earlier unlogged LR run.
- The gap between `add` and the first run is only 47 s. That is consistent with the design being written first and the code second; the timestamps cannot show more.

### 2. No test data evaluated: verified

- **Code.** `run_lrcheck.py` only calls `rg.pool_split`, `rg.cache_for` (which reads `runs/GRID-001/features/pool_<level>.pt`) and `sparse_workflow.train`, which validates on `fold_cache.select(vi)`.
  - It never calls `rg.test_run`, `rg.test_features` or `rg.TEST`.
  - Importing `run_grid` and `run_ablation` runs only constant and function definitions, with no data loading. (`run_ablation` sets `rg.OUT` on import; `run_lrcheck` resets it before use.)
- **Log.** `grid_log.txt` (1,302 lines) has no case-insensitive match for `test`, `v2`, `ImageNetV2` or `matched`. It has no `Traceback`, `Error`, `Warning` or `nan`.
  - It has 54 `[lr-check]` start lines: the smoke cell, then 53 in the full grid, which reused the smoke cell via `done.json`.
  - Both segments end `exit 0`.
- **Results.** No `done.json` or `results.csv` field refers to a test set. `settings.json` records `"test_evaluation": "none (validation only)"`.
- **Files.** Each run directory holds only `done.json`, `best.pt` and a 1-byte `.training.lock`, so no `last.pt` remains. The lock file is a `sparse_workflow` convention that ABL-001 run directories also have.
- **Tracked files.** `git status` / `git diff --stat` show no change to any tracked file. Only the LR-001 paths and `src/run_lrcheck.py` are new. `harness.py check` reports `ok: true`.

### 3. analyze.py implements task.json exactly: verified

| Pre-registered rule | analyze.py | Matches |
| --- | --- | --- |
| lr* = lr with the highest mean val top-1 over 6 cells | `max(LRS, key=mean)` (no ties occurred) | yes |
| Switch if lr* ≠ 1e-3, mean paired diff ≥ 0.30 pp, and ≥ 5/6 cells positive | `best_lr == 1e-3 → no`; `mean >= 0.0030 and positive >= 5`, with positive meaning strictly `> 0` | yes. The marginal sparse M2048 case is 0.003333 ≥ 0.0030, so it is not a floating-point edge. |
| Unstable if total training loss rises > 5% between any two consecutive epochs | `b > a * 1.05` on `history[].train_loss`, the total loss from `r["train"]["loss"]` | yes |
| Resolved if ≤ 1/6 unstable | `unstable <= 1` | yes |
| Drop from best epoch to epoch 10 (median, max) | `val_top1 − history[-1].val_top1` | yes |
| Truncation flag if median best epoch at lr* == 10 | `== 10` | yes |
| Sanity: 12 lr 1e-3 cells vs ABL-001 abl-mse; abs diff and digest | yes, and it also checks the best epoch | yes |
| Paired-t 95% CI with df 5 | t = 2.571 (exact value 2.5706) | yes (negligible rounding) |

`analyze.py` does not itself check that each history has 10 epochs, or that the `lr` field matches the directory. My script checked both: all 54 histories have epochs 1–10, and every `lr`, `head`, `train_level` and `seed` field matches its path. Every `val_top1` equals `history[best_epoch−1].val_top1` and equals the history maximum. In four runs the maximum is tied, and the earlier epoch was kept, consistent with "top-1 then CE". `results.csv` (54 rows) matches the `done.json` files exactly.

### 4. Independent recomputation: all numbers reproduce

| Head | lr | Mean val top-1 | Unstable | Median / max drop (pp) | Median best epoch | Largest epoch-to-epoch loss change (max over 6 runs) |
| --- | --- | --- | --- | --- | --- | --- |
| sparse M2048 | 1e-3 | 60.043 | 4/6 | 2.54 / 4.14 | 6 | +25.8% |
| | 3e-4 | 60.377 | 0/6 | 0.11 / 0.58 | 8.5 | −10.1% |
| | 1e-4 | 60.347 | 0/6 | 0.16 / 0.52 | 9.5 | −10.5% |
| sparse M4096 | 1e-3 | 60.570 | 6/6 | 4.87 / 8.14 | 5 | +89.4% |
| | 3e-4 | 61.223 | 0/6 | 0.12 / 0.46 | 6.5 | −12.4% |
| | 1e-4 | 61.677 | 0/6 | 0.04 / 0.48 | 9.5 | −12.6% |
| dense M2048 | 1e-3 | 59.420 | 6/6 | 5.09 / 6.24 | 6 | +104.8% |
| | 3e-4 | 59.933 | 0/6 | 0.15 / 0.76 | **8** | −11.2% |
| | 1e-4 | 60.537 | 0/6 | 0.20 / 0.64 | **8.5** | −10.3% |

At both lower lrs the training loss falls by at least 10% every epoch in every run, so "0/18 unstable" is not near the 5% threshold.

Paired differences, lower lr minus 1e-3 (mean [95% CI], cells positive). These are identical to the report:

| Head | 3e-4 | 1e-4 | lr* | Switch |
| --- | --- | --- | --- | --- |
| sparse M2048 | +0.333 [0.127, 0.540] 6/6 | +0.303 [0.064, 0.543] 5/6 | 3e-4 | yes |
| sparse M4096 | +0.653 [0.381, 0.926] 6/6 | +1.107 [0.715, 1.499] 6/6 | 1e-4 | yes |
| dense M2048 | +0.513 [0.303, 0.724] 6/6 | +1.117 [0.818, 1.415] 6/6 | 1e-4 | yes |

Truncation flag: not fired for any head. The medians at lr* are 8.5, 9.5 and 8.5.

**Sanity.** All 12 cells match ABL-001 exactly: val top-1 difference 0, val CE difference 0, same best epoch, same split digest, and the same atoms, k, fc_init, weight_decay and reconstruction weight. There is one split digest per seed (42, 43, 44) across all heads and lrs.

**Exploratory gaps.** All reproduce:

| Gap | 1e-3 | 3e-4 | 1e-4 | At each head's lr* |
| --- | --- | --- | --- | --- |
| M4096 − M2048 | +0.53 | +0.85 | +1.33 | +1.30 |
| sparse M2048 − dense | +0.62 | +0.44 | −0.19 [−0.36, −0.02], 1/6 | −0.16 [−0.35, +0.03], 1/6 |

### 5. Additional diagnostics (mine; descriptive, validation only)

- **Slope at the grid edge**, paired 1e-4 − 3e-4:
  - sparse M2048: −0.03 [−0.14, +0.08], 3/6
  - sparse M4096: **+0.45 [0.29, 0.62], 6/6**
  - dense M2048: **+0.60 [0.32, 0.89], 6/6**
- **Without checkpoint selection** (epoch-10 val top-1), sparse M2048 − dense:
  - +3.31 at 1e-3
  - +0.49 [0.27, 0.72] at 3e-4
  - **−0.11 [−0.48, +0.27], 2/6** at 1e-4
- Epoch-10 means at 1e-3 / 3e-4 / 1e-4: sparse M2048 57.60 / 60.17 / 60.15; M4096 55.10 / 61.03 / 61.55; dense 54.29 / 59.68 / 60.26. The direction of "lower lr is better" holds without selection too.
- The validation metric is matched-compression: L090 runs are validated on L090-compressed validation images (`fold_cache.select(vi)` over `pool_L090.pt`).

## Findings

### Must fix before acceptance

**S-1. Factual error in the decision table.** `report.md` "Decisions under the rule", dense M2048 row, says "not flagged (median 9.5)".
- The median best epoch for dense M2048 at lr* = 1e-4 is **8.5**: best epochs [8, 9, 8, 9, 9, 8]. The report's own results table and `analysis.md` give 8.5.
- Caveat 1 has the same problem: "the median best epoch is 9.5" is presented as true "at 1e-4" generally. It is true for sparse M4096 (and sparse M2048), not dense.
- **Repair:** change the dense entry to 8.5, and make caveat 1 head-specific.

**S-2. The grid-edge caveat understates the evidence, and the proposed next step does not follow from it.** Caveat 1 says a lower lr "might gain a little more", and the optional pre-check covers only "M4096".
- The data show the accuracy curve still rising steeply at the edge for exactly the two heads whose lr* = 1e-4:
  - 1e-4 − 3e-4 = +0.45 pp (6/6) for M4096
  - +0.60 pp (6/6) for dense
  - For sparse M2048 it is flat: −0.03, 3/6.
- "A little" is therefore not supported. The optimum for M4096 and dense is unlocated, and the gain from a further step could be comparable to the effects under study.
- This matters for the report's own exploratory claims and its proposal:
  - At 1e-4, the sparse-vs-dense comparison pits a head on its plateau (sparse M2048) against a head still improving (dense).
  - The M4096 − M2048 comparison has the same asymmetry in the other direction.
  - So neither the dense-ahead reversal nor the size of the M gap is a comparison at each head's optimum.
- The proposed single-lr 1e-4 confirmatory test would inherit this.
- **Repair:**
  - Restate caveat 1 with the slopes.
  - Say explicitly that the exploratory gaps at 1e-4 are not at-optimum comparisons.
  - Extend the optional pre-confirmation validation check (lr 3e-5 and/or 20 epochs) to dense M2048 as well as M4096, ideally all three dictionary heads, or say why dense is excluded.
  - No new runs are required for this revision.

### Optional

**O-1. The sparse-vs-dense reversal is not robust.** The reversal is labeled exploratory and hedged ("may be", "plausibly", "not yet confirmed"), which is broadly the right strength. However:
- At each head's lr* the CI includes 0: −0.16 [−0.35, +0.03].
- Without checkpoint selection it is −0.11 [−0.48, +0.27], 2/6.

The robust statement is "the sparse advantage disappears at a stable lr (on validation)", not "dense is ahead". Suggested rewording of "So the SEED-001 sparse-vs-dense result is plausibly explained by dense being hurt more by the too-high lr" is "is consistent with". Also note an untested alternative: k = 512, wd = 0 and Xavier were tuned (GRID-001/002) at lr 1e-3, so the sparse head's extra tuned hyperparameter (k) may be suboptimal at 1e-4.

**O-2. One MSWEEP statement is unhedged.** "The MSWEEP-001 M ≥ 2048 numbers understate the large-M gain" is stated as fact. The evidence is exploratory, validation-only, from seeds 42–44 at two levels, while MSWEEP-001's numbers are ImageNetV2 test results. Write "probably understate (validation evidence only)", matching the wording MSWEEP-001 itself uses after REVIEW-007.

**O-3. The proposed lr for sparse M2048 is a deviation from the rule.** The proposal picks a single lr of 1e-4 for all dictionary heads because "it meets the switch rule for all three". The pre-registered rule recommends lr* (= 3e-4) for sparse M2048; it does not test non-lr* values. Using 1e-4 is reasonable (−0.03 pp vs 3e-4, 3/6, and one common lr simplifies comparisons). Label it as a post-hoc choice that departs from the rule's recommended lr.

**O-4. Selection optimism is probably not equal across lr arms.** task.json says the bias "applies to all lr arms alike". Best-of-10-epoch optimism depends on curve shape: peaked, noisy curves at 1e-3 vs near-monotone curves at lower lr. The epoch-10 comparison favours lower lr even more strongly. So the pre-registered conclusion is robust, but the "alike" wording should be softened, for example to "likely favours 1e-3, making the lower-lr gains conservative".

**O-5. The metric description and one "does not touch" statement need small qualifications.**
- State that L090 cells are validated on L090-compressed validation images, so the validation metric is matched-compression, not a common held-out set.
- "Statements this check does not touch: compression costs ... and no head reduces the compression drop" is fair as "not tested here". But for the dictionary heads those ImageNetV2 numbers are also lr-1e-3-conditional; say so.
- Descriptively, the validation source − L090 differences are similar across lrs: sparse M2048 8.46 / 8.59 / 8.55 pp; dense 8.22 / 8.51 / 8.34 pp.

**O-6. Minor reproducibility gaps.**
- The exact command lines and environment are not recorded in the report or log header: the docstring gives `python -B -u run_lrcheck.py grid`, and the smoke run presumably used `--heads/--lrs/--levels/--seeds` filters.
- `analyze.py` could assert history length and the per-file lr/head/level/seed fields, as my script did.
- The leftover 1-byte `.training.lock` files are harmless, and ABL-001 has them too.

## Overclaiming checklist (requested items)

| Item | Status |
| --- | --- |
| Everything labeled validation-only | Yes: header line, conclusion wording, caveat 3, exploratory header, ledger summary. Minor gap in O-5. |
| Grid-edge lr* | Stated, but understated (S-2) |
| Selection optimism | Stated (caveat 2); "alike" wording in task.json is questionable (O-4) |
| Linear head untested | Clearly stated (design table, caveat 4, exploratory bullet). The fairness implication for a re-test is correctly drawn. |
| Sparse-vs-dense reversal labeled exploratory | Yes. Strength mostly appropriate; "dense ahead" is not robust (O-1). |
| M-gap widening labeled exploratory | Yes; one unhedged sentence (O-2) |
| Implications for SEED-001 / MSWEEP-001 | "What this means for earlier results" is appropriately hedged ("may change", "valid as measured"). The exploratory bullets need O-1 / O-2 wording. |
| Proposed test needs separate approval | Yes, stated and not run |

## Re-review scope

On resubmission, a reviewer needs to check only:
- the S-1 number;
- the S-2 caveat, exploratory-gap wording and next-step text;
- any O-items the author chooses to adopt.

No recomputation is needed unless the data or `analyze.py` change.
