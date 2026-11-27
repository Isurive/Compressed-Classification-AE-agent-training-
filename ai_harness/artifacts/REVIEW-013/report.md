# REVIEW-013: independent review of WD-001S

> Transcription note: written by reviewer-20261127-r and returned as its final message. The coordinator (claude-orient-20261122-a) saved it here verbatim, apart from this note. The verdict was recorded in the ledger by the reviewer itself before the report was returned.

## Verdict

I accepted WD-001S. The verdict is recorded at 2026-11-27T10:06:55Z by reviewer-20261127-r, and WD-001S is now `done`. I recomputed the selection independently and it matches `selected.json` exactly. The timing is correct, I found no test access during phase A, and the four copies are byte-identical. I found no substantive problems. The minor notes below do not affect the lock. I have not submitted REVIEW-013; it is active and owned by reviewer-20261127-r.

## Checks

**1. Timing.** I read file mtimes and creation times with Python `os.stat`, the ledger with `harness.py show`, and the log heads. The logs are in local time (-07:00); all times below are converted to UTC.
- WD-001 `task.json` mtime is 07:50:27Z. The WD-001 ledger add and claim are both at 07:50:27Z.
- After that:
  - The sanity run log starts at 07:51:19Z.
  - `settings_val.json` was written at 07:52:05Z.
  - `val_log.txt` starts at 07:52:01Z, and the first `[wd-val]` line is at 07:52:09Z.
  - The first val `done.json` was written at 07:52:59Z and the last at 10:00:47Z. The "phase A finished" line is at 10:00:47Z.
- `src/run_wdcheck.py` mtime is 07:51:19Z.
- `analyze_val.py` mtime is 07:53:09.9Z (see O-1).
- The outputs in WD-001 were written at 10:00:59Z, and the WD-001S copies at 10:01:42Z.
- The WD-001S claim and submit are at 10:01:59Z. Phase B started after that:
  - `test_log.txt` was created at 10:02:05Z.
  - `settings_test.json` was written at 10:02:10Z.
  - The first `[wd-test]` line is at 10:02:15Z.
  - `runs/WD-001/test` was created at 10:02:15Z.
  - `runs/WD-001/results.csv` was created at 10:03:10Z.
- The submit precedes every phase-B artifact.

**2. Byte identity.** I ran `cmp` and sha256 on each pair in WD-001 and WD-001S. All four are identical:

| File | sha256 |
| --- | --- |
| `selected.json` | `a594923d719e34c489d48290fd96f1eac776d960070105ed768487f79ff32fe8` |
| `analysis_val.md` | `36fa8f38…` |
| `analysis_val.json` | `bb78301b…` |
| `analyze_val.py` | `2e694d60…` |

These hashes match the ledger artifact hashes. Phase B reads `ai_harness/artifacts/WD-001/selected.json`. The `selected` block in `settings_test.json` and the `[wd-test]` log lines (dense-CE lr 1e-4 wd 1e-3 20 ep; sparse-CE lr 3e-5 wd 1e-3 20 ep) match it.

**3. `analyze_val.py` against the pre-registration.** I read the code against WD-001 `task.json`. It matches:
- **Candidates:** 3 lr × 4 wd × budgets {10, 20}. The 10-epoch budget is read from the 20-epoch history as the best epoch ≤ 10 by top-1, then lower CE.
- **Selection:** the highest mean val top-1 over 6 cells (source/L090 × seeds 42–44), with ties broken by lower mean val CE.
- **Flags:**
  - wd upper edge = 1e-1.
  - wd lower edge = selected wd 0 with the best wd-1e-3 candidate within 0.10 pp (0.0010 as a fraction).
  - lr edge = 3e-4 or 3e-5.
  - budget binds = 20 ep with median selected epoch ≥ 19.
- **Exploratory comparisons:** paired, t(df 5) = 2.5706, against the LR-002 MSE heads (lr 1e-4, 20 ep).

The script asserts 144 runs, 20-epoch histories, and lr and wd matching the logged wd groups. Two interpretation choices are noted under O-2 and O-3.

**4. Independent recomputation.** I wrote `ai_harness/artifacts/REVIEW-013/recompute.py`. It does not import `analyze_val.py`. It finds runs by walking the directories and uses integer correct counts (`val_top1 × 5000`) so that ties are exact.
- **Setup checks:**
  - It found 144 runs: 2 heads × 3 lr × 4 wd × 6 cells.
  - Every history has epochs 1–20.
  - For every run, the best epoch in the history equals the `best_epoch` and `val_top1` derived from `best.pt`.
  - There is one split digest per seed: 42 `2bbb411f2b9d`, 43 `36aa0b4f4e44`, 44 `6b92e94b1eb1`.
- **dense-CE:**
  - Selected: lr 1e-4, wd 1e-3, 20 ep, 18100/30000 = 60.333%, mean CE 1.67430, median epoch 11.0.
  - It ties exactly with lr 3e-5, wd 1e-3, 20 ep (also 18100/30000, mean CE 1.68740, median epoch 19.0). The lower CE breaks the tie.
  - The next distinct candidate is lr 1e-4, wd 1e-3, 10 ep, at 60.237% (−0.097 pp).
  - Flags: none.
- **sparse-CE:**
  - Selected: lr 3e-5, wd 1e-3, 20 ep, 17674/30000 = 58.913%, mean CE 1.75328, selected epochs [19, 18, 20, 16, 19, 20], median 19.0.
  - The next candidate is lr 1e-4, wd 1e-3, 20 ep, at 58.853% (−0.060 pp).
  - Flags: lr edge and budget binds. Both are correct, and per the pre-registration neither changes the selection.
- **Exploratory paired CIs (df 5)** match `analysis_val.md` to 2 decimals:

| Head | Selected − wd0 (same lr and budget) | Selected − MSE | wd0 − MSE |
| --- | --- | --- | --- |
| dense-CE | +0.65 [+0.42, +0.89], 6/6 | −0.32 [−0.62, −0.02], 0/6 | −0.97 [−1.19, −0.75] |
| sparse-CE | +0.57 [+0.31, +0.83], 6/6 | −1.74 [−2.15, −1.33], 0/6 | −2.31 [−2.80, −1.82] |

- **Instability counts** match: 6/6 for dense lr3e-4 wd0; 3/6 for dense lr3e-4 wd1e-1; 6/6 for sparse lr3e-4 wd0 and wd1e-3; 3/6 for sparse lr3e-4 wd1e-1; 0 elsewhere.
- **Tie in floating point:** in `analysis_val.json` the two tied dense means are the identical float 0.6033333333333334. The tie-break therefore went through CE, not through floating-point noise.

**5. No test data touched in phase A.**
- **Code path:** `train_ce` uses `rg.pool_split` and `rg.cache_for`, which load `pool_*.pt` only. It calls neither `test_features` nor `test_run`. Only `cmd_test` calls `rg.test_run`. `sparse_workflow.py` has no `test`, `evaluation_split` or `ImageNetV2` references.
- **Records:**
  - `settings_val.json` has `"test_evaluation": "none (validation only)"`.
  - `val_log.txt` has 144 `[wd-val]` lines and no test, ImageNetV2 or v2 mentions.
  - `runs/WD-001/results.csv` has only `wd-test` rows (4 at the time of reading) and was created at 10:03:10Z, in phase B.
  - No `results.csv` under `runs/*` has a `wd-val` row.
- **Limit:** this rests on the code path and records, not on file access times (see the last section).

**6. Weight-decay groups and the wrapper; user `src/` files.**
- **Logged groups:** all 144 `done.json` files log the same groups:
  - decay `[[2048,512],[2048],[1000,2048],[1000]]` (encoder W/b, fc W/b), with `wd` equal to the run's `weight_decay`;
  - no decay `[[512],[2048,512]]` (centre, decoder atoms).
  - 144 "wd groups" lines appear in `val_log.txt`.
- **Wrapper:** `run_wdcheck.py` temporarily wraps `sw.new_parts` and `torch.optim.Adam`. It asserts that the parameter list equals `[*sae.parameters(), *model.fc.parameters()]` by id, that there are exactly 4 decay tensors, and that no `weight_decay` was passed. It applies coupled L2 through Adam param groups and restores both names in a `finally` block.
- **Other settings:** CE-only weights 1e-6/0/0, the model/k/atoms/Xavier settings, and He encoder init were checked from `done.json` and the code.
- **Sanity run:** the log reports `"identical": true` against ABL-001 abl-none dense source seed 42.
- **User `src/` files:** `git diff --quiet HEAD -- src` reports no diff, and `git status` lists only `src/run_wdcheck.py` as new under `src/`. `main.py`, `sparse_workflow.py`, `image_data.py`, `compress_jpeg.py`, `resnet18_sparse_ae.py`, `run_grid.py` and `run_ablation.py` are all tracked and unchanged.

**Other checks.**
- `harness.py check` returns `"ok": true`. It lists WD-001 as `stale`, which is expected: WD-001 is still active in phase B with no recent ledger event.
- **WD-001A disclosure:** WD-001A was added at 10:01:29Z, is `pending`, has no owner and no artifacts, and has no `ai_harness/artifacts/WD-001A` folder. Its write paths include four files under `ai_harness/artifacts/WD-001/`, which is why it overlaps WD-001. The WD-001S description states the replacement and says WD-001A should be ignored. I consider that disclosure adequate for the record. See O-4 for housekeeping.

## Findings

No substantive (S-) findings.

- **O-1. "Written before any phase-A result was read" is not provable from mtimes.** `analyze_val.py` was last saved at 07:53:09.9Z, 10.5 s after the first `done.json` existed (07:52:59Z, dense-CE lr3e-4 wd0 source seed 42, not a selected config). `analyze.py` was saved at 07:53:10Z. The selection rule is fully fixed by `task.json` (07:50:27Z), the code implements it faithfully, and 143 of 144 results did not exist yet. So this cannot have steered the selection. However, the WD-001S claim that it was "written … before any phase-A result was read" rests on the author's statement, not on the timestamps. Suggest "last saved 07:53Z, after 1 of 144 runs had finished; rule fixed by task.json at 07:50Z".
- **O-2. The exploratory "selected − wd0" comparison is at the same lr and budget, not against the best wd 0 candidate.** Against the best tuned wd 0 candidate the gain is smaller:
  - dense-CE: best wd 0 is lr 3e-5, 20 ep, at 59.87%, so the gain is +0.46 pp instead of +0.65.
  - sparse-CE: best wd 0 is lr 3e-4, 20 ep, at 58.60%, so the gain is +0.31 pp instead of +0.57.

  The pre-registration ("validation gap at wd 0 vs selected wd") allows either reading. The report should state which one it uses. All of these validation comparisons are selection-biased, which `analysis_val.md` already notes.
- **O-3. The wd lower edge flag compares against the best wd-1e-3 candidate at any lr and budget.** The pre-registration is ambiguous on this. It is moot here because neither head selected wd 0.
- **O-4. The dense-CE selection rests on a pre-registered CE tie-break between two exactly tied configs.** The losing config (lr 3e-5, median epoch 19) would have carried the lr edge and budget binds flags. The WD-001 report should mention that the dense choice was decided by the tie-break. Sparse-CE's margin is only 0.06 pp (18 images out of 30000), and its flags point to a possibly unconverged edge. These belong as caveats on W2/W4. Housekeeping: WD-001A should be closed or annotated by the coordinator so it does not appear as ready work.

## What I did not check

- I did not inspect file access times or OS-level reads of the `test_*.pt` feature files. The "no test data" finding rests on the code path, logs, `results.csv` and settings files.
- I did not re-run any training, run the sanity comparison myself, or verify the ABL-001 or LR-002 runs beyond reading their `done.json` histories.
- I did not review phase-B results, `analyze.py`, or the W1–W4 analysis.
- I did not verify that the split digests match SEED-001 or RETEST-001; I checked only one digest per seed within phase A.
- I did not touch the GPU, phase-B outputs, or any file outside `ai_harness/artifacts/REVIEW-013/`. The only file I wrote is `ai_harness/artifacts/REVIEW-013/recompute.py`.
