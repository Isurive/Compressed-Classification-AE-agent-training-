# REVIEW-006: independent review of SEED-001

Reviewer `reviewer-20261125-d`. Date 2026-11-26. I did not author SEED-001.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of SEED-001) saved it here verbatim, with no edits to content, on 2026-11-26.

Everything outside this folder was read-only. There was no training or GPU work. I read three linear `best.pt` files on the CPU for their settings only.

My own scripts were kept in the session scratchpad and not submitted, and were run with `py -I`. They recomputed all statistics from:
- `runs/SEED-001/results.csv`;
- `runs/ABL-001/results.csv` (phase `abl-mse`);
- `runs/GRID-002/results.csv` (phase `grid-xavier`, linear).

I did not run `analyze.py`.

Severity scale:
- **Blocking:** invalidates results.
- **Should-fix:** a claim not supported as stated; correct it before acceptance.
- **Optional:** an improvement.

## Verdict: revise (no blocking findings)

The pre-registered primary result is sound and reproduces exactly:

| Comparison (new seeds, pooled) | Gap | 95% CI | Seeds positive |
| --- | --- | --- | --- |
| sparse − dense | +0.41 pp | [0.29, 0.53] | 7/7 |
| sparse − linear | +0.97 pp | [0.78, 1.15] | 7/7 |

Three statements go beyond what the pre-registered analysis supports (S-1 to S-3). They appear in `report.md`, in the ledger summary, and in `WORK_SUMMARY.md` (Step 9 and the TL;DR). The fixes are to labels and wording; no re-run is needed.

## 1. Pre-registration timing and rule adherence

**Timing.** Ledger and file times (UTC; local is UTC−7):

| Event | Time |
| --- | --- |
| Ledger `add` for SEED-001 (seq 73) | 06:37:09Z |
| `task.json` last modified (23:37:09.12 local) | 06:37:09Z |
| `runs/SEED-001/settings.json` written | 06:37:27Z |
| `grid_log.txt` "start" line | 06:37:23Z |
| First run line (`[seed-mse] sparse ... source seed 45`) | 06:37:32Z |
| `analyze.py` last modified | 06:37:43Z |
| First test result (`done.json` of that run) | 06:38:02Z |
| Submission (seq 106) | 07:06:10Z |

- The ledger `spec` equals `task.json` exactly (dict comparison).
- So the design and rule predate the first run.
- The analysis script was last modified after the first training run began but before any ImageNetV2 result existed.

**Rule adherence in `analyze.py`.** These parts match `task.json`:

| Rule element | Matches |
| --- | --- |
| Primary set | New seeds 45–51 only |
| Levels | source, L080, L090 |
| Pooled statistic | The per-seed mean over the three levels, which the pre-registration states explicitly |
| Statistics | Mean, sd, two-sided paired t CI, sign count |
| Detected | The CI excludes 0 |
| Meaningful | Detected and \|mean\| ≥ 1.0 pp. `abs()` only matters for negative effects. |
| Secondary | Seeds 42–51 with the ABL-001 `abl-mse` and GRID-002 linear runs |

The t quantiles are hard-coded as 2.447 (df 6) and 2.262 (df 9). I computed 2.4469 and 2.2622, so they are correct to the precision shown.

**Analysis choices that were not pre-registered:**
- **dense − linear comparison:** see S-2.
- **df 9 for the secondary analysis:** a natural extension; no issue.
- **No multiple-comparison handling:** the report discloses this.
  - Holm correction over the 8 pre-registered tests (2 comparisons × 3 levels plus pooled) still detects all 8. The largest p is 0.024 (sparse − dense, L80) against a threshold of 0.05.
  - Over all 12 tests including dense − linear, everything also passes. But dense − linear at L90 only just passes: p = 0.0495 against a threshold of 0.05.

## 2. Independent recomputation

All 90 cells (3 heads × 3 levels × 10 seeds) are present, with no duplicates. Every value in `analysis.md` and in the `report.md` tables matches to 0.01 pp.

**Primary (seeds 45–51).** Mean [95% CI], seeds positive:

| Comparison | source | L80 | L90 | Pooled |
| --- | --- | --- | --- | --- |
| sparse − dense | 0.53 [0.23, 0.82], 6/7 | 0.34 [0.06, 0.62], 6/7 | 0.37 [0.14, 0.59], 7/7 | 0.41 [0.29, 0.53], 7/7; t = 8.25 |
| sparse − linear | 1.23 [0.92, 1.54], 7/7 | 0.97 [0.43, 1.52], 7/7 | 0.70 [0.37, 1.03], 7/7 | 0.97 [0.78, 1.15], 7/7 |
| dense − linear | 0.70 [0.42, 0.98], 7/7 | 0.63 [0.26, 0.99], 6/7 | 0.33 [0.00, 0.66], 6/7 | 0.55 [0.34, 0.77], 7/7 |

**Secondary (seeds 42–51), pooled:**

| Comparison | Gap | 95% CI | Seeds positive |
| --- | --- | --- | --- |
| sparse − dense | 0.39 | [0.29, 0.50] | 10/10 |
| sparse − linear | 0.93 | [0.81, 1.06] | 10/10 |
| dense − linear | 0.54 | [0.39, 0.69] | 10/10 |

The per-level secondary values also match `analysis.md`. The mean accuracies on the new seeds (48.31/47.39/46.13, 47.79/47.05/45.76, 47.08/46.42/45.43) match `report.md`.

## 3. Settings equivalence (secondary pooling)

The settings are identical for the cells that are pooled.

**`done.json`, source/L80/L90 cells:**

| Cells | atoms / k | fc_init | weight decay | Loss weights (reconstruction / dependency / coherence) |
| --- | --- | --- | --- | --- |
| SEED-001 sparse (21) and ABL-001 `abl-mse` sparse (9) | 2048 / 512 | xavier | 0 | 1 / 0 / 0 |
| SEED-001 dense (21) and ABL-001 `abl-mse` dense (9) | 2048 / 2048 | xavier | 0 | 1 / 0 / 0 |
| SEED-001 linear (21) and GRID-002 linear (9) | 0 / 0 | xavier | 1e-4 | — |

**Other sources:**
- **Settings files:** the `base` block in `runs/SEED-001/settings.json` equals `runs/GRID-002/tuned.json` and the ABL-001 `settings.json` base. `dictionary_loss` equals ABL-001 `variants.abl-mse`.
- **Logs:**
  - Every SEED-001 and ABL-001 dictionary run logs "Encoder initialization: he. Classifier initialization: xavier", Adam lr 0.001, and 10 epochs.
  - The per-epoch residual (loss − CE − MSE) is between −1.6e-8 and 9.5e-9 in SEED-001, and between −1.2e-8 and 1.7e-8 in ABL-001 `abl-mse`. So no other loss terms are active.
- **Linear runs:**
  - The `best.pt` settings of three SEED-001 and three GRID-002 linear runs are identical: epochs 10, batch 128, lr 0.001, xavier, weight decay 1e-4.
  - All 45 linear `history.json` files have 10 epochs.
- **Code path:** `src/run_seeds.py` calls the same functions as the earlier runs (`run_ablation.train_variant` and `run_grid.train_one`) and changes only `rg.OUT`. Every `done.json` `run_dir` points into `runs/SEED-001`. The ABL-001 and GRID-002 registered artifacts still match their hashes, so nothing was written into them.

## 4. Splits

| Check | Result |
| --- | --- |
| Split digests per seed across all heads and all three result files | Exactly 1 per seed |
| Distinct digests across seeds 42–51 | 10 of 10 |
| `results.csv` | 105 rows (21 source runs × 1 + 42 compressed runs × 2), no duplicate keys |
| `done.json` files | 63 |
| `last.pt` files left | 0 |
| Traceback lines in `grid_log.txt` | 0 |

## 5. Findings

**S-1, should-fix. "Every new seed and every level" is false per level.**
- **Location:** `report.md`, "What this settles", item 1: "It is consistent on every new seed and at every level."
- **Evidence:** sparse − dense is positive in 6/7 seeds at source (seed 46: −0.03), 6/7 at L80 (seed 51: −0.15) and 7/7 at L90. Only the pooled per-seed means are 7/7. `analysis.md` shows this; `report.md` shows only pooled sign counts.
- **Implication:** it overstates consistency in the headline conclusion.
- **Repair:** "positive in every seed's pooled mean (7/7) and at every level on average; per-level sign counts 6/7, 6/7, 7/7." Optionally add the per-level sign counts to the report table.

**S-2, should-fix. dense − linear is reported as primary, but it was not pre-registered.**
- **Location:**
  - `report.md` line 5 ("implements exactly that rule");
  - the primary table row and verdict;
  - the caveat "12 primary comparisons";
  - the ledger summary ("dense-linear +0.55 pp [0.34, 0.77]");
  - `WORK_SUMMARY.md` Step 9 table ("real, below 1 point").
- **Evidence:** the `task.json` analysis lists "paired differences sparse-dense and sparse-linear" and states "No other comparisons are confirmatory".
- **Implication:**
  - The numbers are correct, and the effect would survive correction.
  - But presenting it as a primary, decided result breaks the pre-registration the report relies on.
  - The pre-registered primary family is 8 tests, not 12.
- **Repair:**
  - Move dense − linear to an "exploratory (not pre-registered)" row or section.
  - Change "implements exactly that rule" to "implements that rule, plus one exploratory comparison (dense − linear)".
  - Correct the count to 8 pre-registered comparisons and add the Holm result. Carry the same labels into the ledger summary on resubmit and into `WORK_SUMMARY.md`.

**S-3, should-fix. The compression-trend claims are not pre-registered, and for sparse − dense they are not supported.**
- **Location:**
  - `report.md`, "What this settles", item 3 ("The sparse − dense gap is largest on uncompressed images (0.53) and smaller at L80/L90 (0.34–0.37)");
  - the ledger summary ("Gaps shrink with compression: sparsity adds a small fixed benefit");
  - `WORK_SUMMARY.md` TL;DR ("The gaps between heads actually shrink as compression gets stronger") and Step 9 ("Trend with compression").
- **Evidence:** my exploratory per-seed change in each gap from source to L90 (new seeds, paired t 95% CI, df 6):

  | Gap | Change, source → L90 | 95% CI | Seeds |
  | --- | --- | --- | --- |
  | sparse − dense | +0.16 | [−0.18, 0.50] | 6/7 |
  | sparse − linear | +0.53 | [0.15, 0.91] | 6/7 |
  | dense − linear | +0.37 | [0.08, 0.66] | 6/7 |

  - The sparse − dense change is not detected. The per-level CIs overlap heavily ([0.23, 0.82] vs [0.14, 0.59]).
  - The two gaps against linear do shrink, but only in an exploratory analysis.
  - "Gaps shrink" and "a small fixed benefit" in the same sentence also pull in opposite directions.
- **Implication:**
  - The conclusion that sparsity gives no compression tolerance still holds; no evidence favours tolerance.
  - But the trend is presented as a finding.
- **Repair:**
  - Label the trend as exploratory.
  - Say the sparse and dense heads' advantage over linear narrows from source to L90 (exploratory CIs above). Say no change in the sparse − dense gap was detected, consistent with a roughly constant ~0.4 pp benefit.
  - Update the ledger summary and `WORK_SUMMARY.md` to match.

**O-1, optional. Scope the sparsity claim.**
- **Location:** `report.md`, item 1 heading "Top-k sparsity has a real, small benefit"; `WORK_SUMMARY.md` "small but now confirmed benefit from top-k sparsity".
- **Evidence:**
  - The result is for k = 512 of M = 2048 (25% active), with the MSE loss and Xavier init.
  - In the GRID-001 validation tuning, stronger sparsity was worse: Xavier k = 32 scored 53.34 and k = 128 scored 57.98, against 60.03 for k = 512.
  - k was the best of 3 tuned candidates, while dense had no free parameter. The tuning used only seed 42's validation split, so the effect on the new seeds is likely small, but it is a budget asymmetry.
- **Repair:** "mild top-k sparsity (k = 512 of 2048) with the reconstruction loss adds about 0.4 pp; stronger sparsity was worse in tuning."

**O-2, optional. Acceptance criterion 4 is not shown in the report.**
- **Location:** `report.md` "Checks".
- **Evidence:** the report does not state that the user's `src` files and earlier artifacts were checked. I verified it:
  - all 8 files in `GRID-001/src_hashes_before.txt` are OK;
  - there is no `src/__pycache__`;
  - all 125 registered artifacts match, apart from the two COORD-001 files that COORD-003 legitimately superseded;
  - `harness.py check` is ok.
- **Repair:** add one line saying so.

**O-3, optional. Minor wording.**
- The caveat says "t ≈ 8.3"; the exact value is 8.25.
- The `WORK_SUMMARY.md` TL;DR says "now confirmed" while SEED-001 is in review; §3 correctly says "pending review".

## 6. What is supported as stated

- The primary pre-registered sparse − dense result: detected and not meaningful (+0.41 [0.29, 0.53], pooled 7/7).
- The primary pre-registered sparse − linear result: detected; meaningful only at source (1.23; lower bound 0.92, flagged as borderline); pooled 0.97, not meaningful.
- The secondary agreement.
- The settings and split claims, and the claim-scope and test-reuse caveats.
- "This resolves the ABL-001 +0.46 item": the effect is detected and small, within the scope of O-1.

## 7. Checks run

1. `harness.py claim REVIEW-006`; `show SEED-001` (spec, events, artifacts); `status`; `check` (ok, 27 tasks; the COORD-001 files are listed as superseded).
2. Timestamps: ledger events; `ls --time-style=full-iso` on `task.json`, `analyze.py`, `src/run_seeds.py`, `runs/SEED-001/*`, and the first run folder; the first and last lines of `grid_log.txt`. Also a dict comparison of the ledger spec with `task.json`.
3. Read `analyze.py` and `src/run_seeds.py` in full; compared them with the `task.json` rule.
4. Own recompute script:
   - all primary and secondary per-level and pooled statistics;
   - t quantiles by numerical integration;
   - sign counts;
   - Holm p-values;
   - exploratory source-minus-L90 interactions;
   - split digests.
5. Settings: all relevant `done.json` files; `settings.json` and `tuned.json`; log headers and loss residuals for the SEED-001, ABL-001 and GRID-002 logs; three plus three linear `best.pt` settings (CPU read); `history.json` lengths.
6. `sha256sum -c GRID-001/src_hashes_before.txt` (8/8 OK); sha256 of all 125 registered artifacts against the ledger.
7. Read the `WORK_SUMMARY.md` TL;DR, Step 9, §3 and §7.

## 8. Not checked

- Training was not re-run, and no test predictions were re-scored. The test accuracies are taken from `results.csv`, which matches `done.json` in structure.
- The dictionary `best.pt` files hold no settings, so their equivalence rests on `done.json`, the log headers and the loss residuals.
- Whether the user's approval wording quoted in `report.md` is accurate; I cannot see the conversation.
