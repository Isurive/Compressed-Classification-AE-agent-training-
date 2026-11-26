# REVIEW-007: independent review of MSWEEP-001

Reviewer `reviewer-20261126-e`. Date 2026-11-26. I did not author MSWEEP-001.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of MSWEEP-001) saved it here verbatim, with no edits to content, on 2026-11-26.

Everything outside this folder was read-only. There was no training, no GPU work and no checkpoint loading. I did not run `analyze.py`.

My own scripts were kept in the session scratchpad and run with `py -I`. They recomputed everything from:
- `runs/MSWEEP-001/results.csv` (phase `msweep-clean`);
- `runs/SEED-001/results.csv` (phase `seed-mse`, model `sparse`, for M = 2048; phase `seed-linear` for the exploratory linear comparison);
- the 63 MSWEEP-001 and 21 SEED-001 sparse `done.json` files;
- `grid_log.txt` of both tasks.

Severity scale:
- **Blocking:** invalidates results.
- **Should-fix:** a claim not supported as stated; correct it before acceptance.
- **Optional:** an improvement.

## Verdict: revise (no blocking findings)

Every confirmatory number reproduces, and the resume left no trace in the data. The pre-registration predates the first run.

One substantive issue was not disclosed (S-1): training becomes unstable as M grows. Every M = 4096 run shows a rising training loss. The fixed learning rate was tuned at M = 2048, so the M = 4096 result and the "diminishing returns" reading depend on that learning rate. Two wording issues (S-2, S-3) overstate what the design isolates. No re-run is needed for acceptance.

## 1. Pre-registration timing and rule adherence

**Timing** (UTC; local is UTC−7):

| Event | Time |
| --- | --- |
| Ledger `add` for MSWEEP-001 (seq 113) | 07:49:42Z |
| `task.json` last modified (00:49:42.62 local) | 07:49:42Z |
| `src/run_msweep.py` last modified | 07:49:57Z |
| `grid_log.txt` "start" line | 07:49:57Z |
| First run line (`M=512 k=128 source seed 45`) | 07:50:04Z |
| First `done.json` (M512/source/seed45) and its ImageNetV2 rows | 07:50:22Z |
| `analyze.py` last modified | 07:50:26Z |
| Submission (seq 116) | 19:49:55Z |

- The design and decision rule in `task.json` predate the first run.
- The final edit to `analyze.py` came about 4 s after the first ImageNetV2 result existed. At that point only 1 of 63 runs was complete. It was never modified after that, including at the resume. The rule it implements is fixed by `task.json`, so this has no practical effect (O-3).

**Rule adherence in `analyze.py`:**

| Rule element | Matches |
| --- | --- |
| Data | MSWEEP-001 `msweep-clean`, plus SEED-001 `seed-mse` sparse for M = 2048; seeds 45–51; levels source, L080, L090; ImageNetV2-source top-1 |
| Differences | A(M) − A(2048) for M ∈ {512, 1024, 4096}, per level and pooled (the per-seed mean over levels) |
| Trend | Per-seed least-squares slope of pooled accuracy on log₂M; mean, paired-t CI, df 6 |
| Detected / meaningful | CI excludes 0 / \|mean\| ≥ 1 pp; for the slope, \|slope\| × 3 ≥ 1 pp |
| Holm | Over all 13 tests |
| Sanity check | Median best epoch per M, flagged at ≤ 2 |

**The 10 vs 13 tests discrepancy.** The pre-registration says "These 9 differences plus 1 trend (10 tests)". However, its own item (1) defines 3 sizes × (3 levels + pooled) = 12 differences. The report's reading, that the count was miscounted and the rule is what was written, is the natural one. Holm over 13 tests is stricter than over 10. I checked the claim that no conclusion depends on this:
- Holm over all 13 detects the same 11 tests as the unadjusted CIs. The largest Holm p among detected tests is 0.0173 (M4096 pooled).
- Holm over the 10 non-pooled tests detects the same tests on that shared set: all M512 and M1024 levels, M4096 at L90, and the slope.

The disclosure is honest and adequate in `report.md`. But the generated `analysis.md` still has the heading "Confirmatory family (pre-registered; 10 tests)" above a 13-row table. The `analyze.py` docstring and comment also still say 10 (O-1).

## 2. Independent recomputation (seeds 45–51, paired by seed)

All 84 cells (4 sizes × 3 levels × 7 seeds) are present, with no duplicates.

**Mean accuracy (%):**

| M | source | L80 | L90 | Pooled |
| --- | --- | --- | --- | --- |
| 512 | 45.57 | 44.57 | 42.85 | 44.33 |
| 1024 | 47.14 | 46.28 | 44.92 | 46.12 |
| 2048 | 48.31 | 47.39 | 46.13 | 47.28 |
| 4096 | 48.64 | 47.81 | 46.75 | 47.73 |

These match the report.

**Confirmatory family.** Mean [95% CI], seeds positive, unadjusted p:

| Test | source | L80 | L90 | Pooled |
| --- | --- | --- | --- | --- |
| M512 − M2048 | −2.74 [−3.06, −2.42] 0/7 | −2.82 [−3.34, −2.30] 0/7 | −3.28 [−3.67, −2.89] 0/7 | −2.95 [−3.20, −2.69] 0/7 |
| M1024 − M2048 | −1.17 [−1.44, −0.90] 0/7 | −1.11 [−1.61, −0.60] 0/7, p 0.0017 | −1.20 [−1.56, −0.85] 0/7 | −1.16 [−1.41, −0.90] 0/7 |
| M4096 − M2048 | +0.32 [−0.18, 0.83] 6/7, p 0.167 | +0.42 [−0.11, 0.95] 6/7, p 0.101 | +0.63 [0.31, 0.95] 7/7, p 0.0030 | +0.46 [0.19, 0.72] 7/7, p 0.0058 |
| Slope per doubling | | | | **+1.14 [1.07, 1.20] 7/7**, t = 40.8 |

All values match `report.md` and `analysis.md` to 0.01 pp. There is one rounding difference: I get a lower bound of −3.055 for M512 at source, which the report rounds to −3.05.

The verdicts follow from the rule:
- M512 and M1024 are detected and meaningful at every level.
- M4096 is detected at L90 and pooled but not meaningful, and is not detected at source or L80.
- The slope is detected and meaningful.

The Holm p-values I computed equal `analysis.md`. For example, M4096 L090 is 0.0120, M4096 pooled is 0.0173, and M4096 source and L80 are both 0.2018.

**Exploratory numbers:**
- **Doubling increments:** +1.79, +1.16 and +0.46 pp.
- **Per-level slopes:** 1.04, 1.08 and 1.29.
- **L90 − source slope difference:** +0.26 [−0.01, 0.52], 6/7 seeds positive, p = 0.057.
- **Pooled linear accuracy:** 46.31%.

All four match the report.

## 3. Resume integrity

| Check | Result |
| --- | --- |
| `runs/MSWEEP-001/results.csv` | 105 rows, all phase `msweep-clean`. No duplicate (phase, run_dir, eval_set, eval_level) keys. |
| Result sets per run directory | 63 directories. Each has exactly 1 row (source runs) or 2 rows (compressed runs: eval source and its own level). |
| `done.json` | 63. `best_epoch`, `split_digest`, `atoms` and `k` agree with `results.csv` for every run. |
| `best.pt` / `last.pt` | 63 / 0 |
| Interrupted run `M512/L080/seed51` | Killed at 01:09:54 local, after the "single-pool holdout" line and before epoch 1. It left only a `.training.lock` file. On resume it restarted from scratch: an initialization line followed by 10 fresh epochs. Its `done.json` (best epoch 9, val top-1 0.5734) matches that resumed run's epoch-9 log line. |
| Skip logic | `run_ablation.train_variant` returns early on an existing `done.json` and raises on `best.pt` without `done.json`. `run_grid.test_run` skips rows already in `results.csv`. So the 57 finished runs were neither retrained nor re-scored. The log shows only the 6 remaining runs after "resume start". |
| Log | The line "interrupted: process killed after 01:09:54 ..." is an annotation added after the fact, not process output. It is clearly worded and is consistent with the timestamps. |

There was no corruption or duplication.

## 4. Settings equivalence and splits

- **Settings:** all 84 `done.json` files (63 MSWEEP-001 and 21 SEED-001 sparse) have one identical signature apart from atoms and k:
  - model `sparse`;
  - fc_init `xavier`;
  - weight decay 0;
  - loss weights 1 / 0 / 0.

  k = M/4 holds in every file.
- **settings.json:** `runs/MSWEEP-001/settings.json` `dictionary_loss` equals SEED-001's. Its per-size configs equal SEED-001 `base.sparse` except for atoms and k.
- **Code path:** `src/run_msweep.py` builds the configs from `GRID-002/tuned.json["sparse"]`, which has the same base as SEED-001, and calls the same `train_variant` and `test_run`.
- **Shared constants:** epochs, batch size and learning rate come from the shared `run_grid` constants. The resumed log shows "Adam lr=0.001" and He/Xavier initialization.
- **Splits:** there is one split digest per seed across all four sizes, including the SEED-001 sparse and linear cells. There are 7 distinct digests.

## 5. Findings

**S-1, should-fix. Training instability grows with M and is not disclosed. Caveat 3 misreads it.**
- **Location:** `report.md` "Interpretation and confounds" item 3 ("Larger M peaked earlier ... a fixed 10-epoch budget may favour some sizes"); "Exploratory: Diminishing returns".
- **Evidence:** from the epoch lines in `grid_log.txt` of both tasks. "Rise" means the training loss went up by more than 5% from one epoch to the next.

  | M | Runs with a training-loss rise | Median val top-1 drop from best epoch to epoch 10 | Max drop |
  | --- | --- | --- | --- |
  | 512 | 0/21 | 0.3 pp | 1.1 pp |
  | 1024 | 0/21 | 0.4 pp | 0.9 pp |
  | 2048 (SEED-001) | 15/21 | 2.4 pp | 4.3 pp |
  | 4096 | 21/21 | 5.9 pp | 8.0 pp |

  An example is M4096/L090/seed51: the training loss goes 0.18 → 0.23 → 0.55 over epochs 8–10, and val top-1 goes 55.8 → 52.9 → 50.2.
- **Implication:**
  - Selection by validation top-1 means the reported accuracies are valid measurements under the pre-registered protocol. The confirmatory results stand as measured.
  - However, at lr 1e-3 (tuned at M = 2048), large dictionaries train unstably. The earlier "peak" at M = 4096 is likely due to divergence rather than faster fitting.
  - The small M4096 − M2048 gain (+0.46) and the "diminishing returns past 2048" reading may therefore partly reflect the optimizer setting, not a capacity ceiling. That would bias against large M.
- **Repair:**
  - Add the instability table, or a sentence with these counts.
  - Reword caveat 3: larger M diverges late at the fixed lr, so the M ≥ 2048 results are conditional on lr 1e-3.
  - State that diminishing returns may be partly an optimization artifact.
  - Optional follow-up: a validation-only learning-rate check at M = 4096.

**S-2, should-fix. The headline names "available atoms", but the design varies M, k and parameter count together.**
- **Location:** `report.md` "Question" ("how the number of available dictionary atoms (M) affects the result"); "Interpretation" item 1 heading, "**Available atoms matter.**"
- **Evidence:** item 2 of the same section says the result "is not 'more atoms, with everything else equal, helps'". With k = M/4, the number of active atoms also rises (128 → 1024) and the head grows from about 1.0 M to 8.3 M parameters. `AGENTS.md` asks that available atoms M be kept distinct from active atoms k.
- **Implication:** the confound is disclosed, but the bold headline contradicts it. Readers who quote only headlines will carry over the stronger claim.
- **Repair:**
  - Change the heading to "A larger dictionary at a fixed active fraction (k = M/4) helps". Note that M, k and parameter count rise together.
  - Optionally note that separating M from k would need a fixed-k arm.

**S-3, should-fix. The ledger summary states "Diminishing returns past 2048" without the exploratory label.**
- **Location:** the MSWEEP-001 ledger summary (seq 116).
- **Evidence:** the pre-registration lists only the 13 tests as confirmatory. The doubling increments (+1.79, +1.16, +0.46) were not tested, and S-1 bears on them. The report body labels them correctly.
- **Repair:** on resubmit, write "(exploratory) diminishing returns past 2048, possibly lr-limited (S-1)".

**O-1, optional. The generated analysis still says "10 tests".**
- `analysis.md` has the heading "Confirmatory family (pre-registered; 10 tests)" over 13 rows.
- The `analyze.py` docstring ("(9) ... (1)") and the "Holm adjustment over the 10-test family" comment say the same.
- The code correctly uses 13. Either fix the label and regenerate, disclosing the label-only change, or add a pointer in `report.md` saying the heading is stale.

**O-2, optional. The exploratory linear comparison is slightly soft for M = 1024.**
- `report.md` says M = 1024 is "about level with" linear.
- Pooled, it is −0.19 pp [−0.34, −0.05], 0/7 seeds positive, so it is consistently but slightly below linear.
- M512 − linear is −1.98 [−2.18, −1.78], 0/7, which supports "about 2 pp below".
- "A small dictionary is worse than no dictionary" would read better as "... worse than the linear head (exploratory)".
- The section is already correctly labeled exploratory, so this is wording only.

**O-3, optional. State the `analyze.py` timing precisely.**
- `analyze.py` was last modified 4 s after the first test row was written, when 1 of 63 runs was complete.
- This is harmless because the rule is fixed in `task.json`, but say so rather than implying the script predates all results.

## 6. What is supported as stated

- **Pre-registered timing** of the design and decision rule.
- **All confirmatory numbers and verdicts:**
  - M512 −2.95 and M1024 −1.16: detected and meaningful;
  - M4096 +0.46: detected, not meaningful;
  - slope +1.14 [1.07, 1.20].
- **The 10-vs-13 correction:** honestly disclosed, and it has no effect on the detected set.
- **The best-epoch sanity check:** no M is flagged. Medians 9 / 8 / 7 / 6; the M512 minimum is 2.
- **Resume integrity, settings equivalence and splits.**
- **Disclosure** of the capacity and k confound, the tuning point, and the claim scope.
- **The exploratory size × compression interaction:** correctly reported as not detected.

## 7. Checks run

1. `harness.py claim REVIEW-007`; `show MSWEEP-001` (spec, events, artifacts); artifact sha256 compared with the ledger, all match.
2. Timestamps: ledger events; `ls --time-style=full-iso` on `task.json`, `analyze.py`, `run_msweep.py`, `runs/MSWEEP-001/*` and the first and interrupted run folders; `grid_log.txt` start, first run, interruption and resume lines.
3. Read in full: `task.json`, `report.md`, `analyze.py`, `analysis.md`, `src/run_msweep.py`, both `settings.json` files. Also read `train_variant` (`run_ablation.py`) and `recorded` / `test_run` (`run_grid.py`).
4. Own recompute script (`py -I`):
   - duplicates and per-run-directory result sets;
   - `done.json` vs CSV;
   - settings signatures;
   - split digests;
   - all 13 confirmatory statistics, Holm over 13 and over 10;
   - the exploratory slopes and interaction;
   - the comparisons with linear.
5. Instability scan of the per-epoch training loss and validation top-1 in both `grid_log.txt` files.
6. `sha256sum -c GRID-001/src_hashes_before.txt` (8/8 OK); no `src/__pycache__`; `harness.py check` ok (29 tasks; the COORD-001 files are listed as superseded).

## 8. Not checked

- Training was not re-run, and no predictions were re-scored. The test accuracies come from `results.csv`, which agrees with `done.json` in structure.
- Parameter counts were checked only arithmetically: 512M + 512M + 1000M, without biases, gives about 1.04 M to 8.29 M. No model was instantiated.
- Whether a different learning rate removes the M = 4096 instability; that would need training.
- The user's approval wording quoted in `report.md`.
