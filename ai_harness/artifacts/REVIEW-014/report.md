# REVIEW-014: independent review of WD-001

> Transcription note: written by reviewer-20261127-t and returned as its final message. The coordinator (claude-orient-20261122-a) saved it here verbatim, apart from this note. The verdict was recorded in the ledger by the reviewer itself before the report was returned.

Reviewer: `reviewer-20261127-t`. Date: 2026-11-27. Claimed REVIEW-014. Not submitted (left for the coordinator).

## Verdict

**Accept.** Recorded on WD-001 before handback. WD-001 is now `done`.

Every confirmatory, secondary and exploratory number in report.md and analysis.md matches my independent recomputation. analyze.py implements the pre-registered family exactly, and the phase-B integrity checks pass. I found no substantive problems. Four optional wording points follow, plus one optional addition to the provenance notes.

## Checks

My helper script is `ai_harness/artifacts/REVIEW-014/recompute.py`. It uses the standard library only, does not import analyze.py, and gets its p-values from a regularized incomplete beta (a different method from analyze.py's Simpson integration). I ran it with the ai360 `python -B`. Nothing ran on the GPU, and I wrote nothing outside the REVIEW-014 folder.

1. **analyze.py against the pre-registration in task.json.**
   - Each test is the per-seed mean, over source/L080/L090, of the per-level differences. Paired by seed 45–51, n = 7, df 6, t = 2.447.
   - Detected = CI excludes 0. Meaningful = detected and |mean| ≥ 1.0 pp, and also reported at 0.5 pp.
   - Holm over the 4 tests is a correct step-down with a running maximum.
   - The comparison runs come from RETEST-001 phase `retest`, with the head taken from the run_dir. Their done.json files confirm:
     - dense-M2048 and sparse-M2048: lr 1e-4, 20 epochs, wd 0 (21 each);
     - linear: lr 3e-4, 20 epochs, wd 1e-4 (21).
   - RETEST-001's sparse-M4096 is loaded but not used.
   - Exploratory sections match the pre-registered list: per-level gaps, compression costs and their differences, top-5, training-level evaluation.

2. **Independent recomputation** (pp; the analysis.json values agree to 4 decimals):

   | Test | Mean | 95% CI | positive | raw p | Holm p |
   | --- | --- | --- | --- | --- | --- |
   | W1 | −0.286 | [−0.407, −0.165] | 0/7 | 0.00116 | 0.00231 |
   | W2 | −1.362 | [−1.520, −1.204] | 0/7 | 7.4e-7 | 3.0e-6 |
   | W3 | −0.355 | [−0.566, −0.143] | 0/7 | 0.00631 | 0.00631 |
   | W4 | −1.344 | [−1.634, −1.054] | 0/7 | 2.8e-5 | 8.5e-5 |

   - Meaningful at both 1.0 and 0.5 pp: W2 and W4 only.
   - Everything else matches analysis.md to the printed precision: all 12 per-level rows, the 4 CE compression costs, the 8 cost differences, all mean top-1, top-5 and training-level means, and the sd and positive counts.
   - The report gives W4's Holm p as "<0.0001". That is correct (8.5e-5); analysis.md shows "0.0001" only because it rounds to four decimals.

3. **Phase-B checks.**
   - selected.json is unchanged: sha256 a594923d… equals the WD-001S copy and the ledger.
   - 42/42 test run directories, each with done.json. In every one:
     - (lr, wd, epochs) equal selected.json and phase is `wd-test`;
     - a 20-epoch history (epochs 1–20);
     - best_epoch agrees with the history (highest validation top-1, then lower CE);
     - the results.csv rows agree with done.json (best epoch, digest, validation top-1).
   - Split digests: one per seed, equal to SEED-001's and to RETEST-001's.
   - wd groups: decay on [2048,512], [2048], [1000,2048], [1000] at 1e-3 and no decay on [512], [2048,512], in 42/42 runs. The test log has 43 group lines (42 plus the interrupted attempt), and phase A has 144/144 lines with the same shapes.
   - Unstable runs: 0/21 per head.
   - Selected epochs:
     - dense-CE: median 11, range 7–14, 0/21 at ≥ 19;
     - sparse-CE: median 19, range 17–20, 16/21 at ≥ 19.
   - results.csv has 70 rows, all phase `wd-test`. Source runs have 1 row and compressed runs 2. There are no duplicate (phase, run, set, level) keys.
   - The artifact copies of results.csv and test_log.txt are byte-identical to `runs/WD-001`. All 16 artifact hashes match the ledger.
   - `harness.py check` returns ok.
   - `git status` on `src/` shows only the untracked run_wdcheck.py, so the user's `src/` files are unchanged.

4. **Overnight interruption and resume.**
   - The log shows 29 completed runs (seeds 45–48 complete, then seed 49 up to dense-CE L090).
   - The sparse-CE L090 seed 49 attempt stopped after epoch 8 at 03:27:14 local (10:27 UTC). A resume note follows at 13:34:54 local (20:34:54 UTC), then a full rerun.
   - The rerun's done.json is dated 20:35:38 UTC, after the resume. That run has exactly 2 csv rows, and no last.pt or other stray files remain in any test directory.
   - The interrupted attempt's 8 logged epochs are identical, character for character, to the rerun's epochs 1–8. This is partial evidence that the rerun is deterministic.

5. **report.md.**
   - Every number in §§1–6 matches analysis.md/json or my recomputation. The §2 phase-A numbers match analysis_val.md (+0.65/+0.57, +0.46/+0.31, 0.06 pp margin, the 17.7–57.8% range, wd 1e-3 best except at lr 3e-4).
   - W1 is correctly called detected, negative and below both tolerances, and outside both pre-declared readings. The "close but not all the way" wording is labeled post-hoc.
   - W2 is correctly given the pre-declared reading, with the §4 edge caveat referenced.
   - All requested caveats are present: sparse lr edge and budget binds with the phase-B epoch counts, the dense exact tie, tuning asymmetry, coupled L2 and parameter scope, the scope of the study, and test reuse.
   - Section 5 is labeled exploratory and uncorrected.
   - Timeline:
     - the UTC conversions agree with the logs and ledger: task.json mtime 07:50:27; analyze*.py 07:53:09–10; selected.json 10:00:59; WD-001A added 10:01:29; WD-001S claimed and submitted 10:01:59; phase B first run 10:02:15;
     - "Phase A finished 10:00:47" matches the val_log line, and the val_log end stamp is 10:00:48;
     - "Phase B finished 20:43:59" is 1 s after the log's 13:43:58 local (20:43:58 UTC). This is negligible;
     - GPU times are consistent: about 2 h 9 min for phase A, and about 25 + 9 = 34 min for phase B.
   - §4's references to COORD-004 are accurate. Section D lists the linear weight decay and k as chosen at lr 1e-3 and not re-tuned, and B3 records the sparse-M2048 post-hoc lr.

## Findings

No substantive (S-) findings.

- **O-1 (§5, last bullet; understates the sparse gap).** "They show the same ordering: the CE heads are at or slightly below the MSE heads" fits dense but not sparse. Suggested wording: "They show the same pattern: dense-CE is 0.05–0.16 pp below dense-MSE on top-5 and 0.12–0.30 pp below on training-level top-1, while sparse-CE is 0.72–0.84 pp below sparse-MSE on top-5 and 1.47–1.70 pp below on training-level top-1."
- **O-2 (§5, "This follows from the per-level gaps above").** The sparse-CE − linear per-level gaps are not listed in §3. Suggested addition to the §3 per-level list: "sparse-CE − linear is −0.99 (source), −1.31 (L080) and −1.73 (L090); all three CIs exclude 0."
- **O-3 (§3).** "so the sparse gap is the same at every level" → "so the sparse gap is about the same at every level (−1.34 to −1.40 pp; descriptive, not an equivalence test)".
- **O-4 (§9, first bullet).** The sparse conclusion drops the grid caveat. Suggested wording: "for the sparse head, within the pre-registered grid and 20-epoch budget, the reconstruction loss gives a meaningful benefit that tuned weight decay does not (see §4)."
- **O-5 (§7, resume note; optional addition).** After "Bit-identity of that specific run was not separately verified", add: "The interrupted attempt's 8 logged epochs are identical to the rerun's epochs 1–8 in `logs/test_log.txt`."

## What I did not check

- Phase-A selection logic and analysis_val.py were not re-derived. That work belongs to REVIEW-013; I only confirmed selected.json is unchanged and spot-checked the §2 numbers against analysis_val.md.
- I did not re-score any checkpoint on ImageNetV2, re-run any training, or verify the sanity-run equivalence beyond reading sanity_log.txt ("identical": true; digest, best epoch 5, validation top-1 0.6198, validation CE 2.0157270309448245).
- I did not audit sparse_workflow.py, run_grid.py's scoring code, or RETEST-001's own runs beyond the lr, epoch and wd settings in their done.json files.
- I did not check phase-A split digests, phase-A instability counts or phase-A history completeness, apart from the wd-group log lines.
