# REVIEW-009: independent review of LR-002

**Reviewer:** `reviewer-20261126-i` (did not author LR-002). **Date:** 2026-11-26.
**Reviewed:** `ai_harness/artifacts/LR-002/` (task.json, report.md, analyze.py, analysis.md/.json, grid_log.txt), `src/run_lrcheck2.py`, the relevant parts of `src/run_grid.py` (`train_linear`, module globals), `src/run_ablation.py` (module-level side effects) and `src/sparse_workflow.py` (`_fit` epoch loop and best-checkpoint logic), the 78 `runs/LR-002/**/done.json`, the LR-001 `lr3e-4`, `lr1e-4` and `lr1e-3` `done.json`, and the GRID-002 `grid/linear` `done.json` + `history.json` for seeds 42–44, source/L090. Context: LR-001 report, REVIEW-008, SEED-001 / GRID-002 / MSWEEP-001 reports (loss settings only).

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of LR-002) saved it here verbatim, with no edits to content, on 2026-11-26.

**Verdict: revise.** The pre-registered part is correct and reproduces exactly. Two fixes are needed in the exploratory interpretation and its labelling (S-1, S-2) before acceptance.

## What I checked and how

All numbers below come from my own standard-library script, kept outside the project (`C:\Users\ericw\AppData\Local\Temp\claude\5a605a89fa322016\2139bfc6-af5b-461d-bfca-a7c21622894d\scratchpad\review009.py`, run with `py -I`, output in `review009_out.txt` next to it). It does not import or run `analyze.py`. Selection within a budget n = the epoch ≤ n with the highest validation top-1, then lower validation CE (first such epoch on an exact tie, as `best.pt` uses a strict `>`). Paired-t 95% CI uses t = 2.5706, df 5. No training, no GPU, no checkpoint loading.

| Check | Result |
| --- | --- |
| Pre-registration timing | `task.json` mtime 17:27:04.7 local; ledger `add` 00:27:05Z = 17:27:05 local; `run_lrcheck2.py` 17:27:29; log `start` 17:27:39; first run 17:27:46. **Passes** (by ~40 s). The ledger spec text equals task.json. |
| `analyze.py` timing | Last written 17:29:39. At that time only the two smoke runs (done 17:28:47) had finished; the first non-smoke run finished at 17:29:47. It implements task.json (below), so this is not a problem, but the report does not say it (O-4). |
| Artifact integrity | SHA-256 of all 9 ledger artifacts equals the ledger values. `harness.py check` → `ok: true`. Tracked `src/` files are unmodified (`git status`: only `run_lrcheck.py`, `run_lrcheck2.py` untracked). |
| No test data | `run_lrcheck2.py` never references `rg.TEST`, `test_features` or `test_run`; `run_ablation` import only resets `rg.OUT`; no `test`/`imagenetv2` string in any of the 78 `done.json`/`history.json` files or in `grid_log.txt`. `settings.json` records `"test_evaluation": "none (validation only)"`. **Passes.** |
| Run inventory | 78/78 `done.json`; head, lr, level, seed and `epochs = 20` match each path; every history has epochs 1–20; the recorded `best_epoch`/`val_top1` equals my 20-epoch selection in 78/78. 78 `[lr-check2]` start lines in the log (2 smoke + 76; the 2 smoke cells were reused). `results.csv` 78 rows. 0 `last.pt`. One split digest per seed, also across the LR-001 lr 3e-4 cells. Exact top-1 ties occurred in 4 budget windows; the CE tiebreak is applied identically by my code and by `analyze.py`. |
| `analyze.py` vs task.json | Candidate set (LR-001 3e-4@10 for dictionary heads + 3 lrs × {10, 20}; linear 4 lrs × {10, 20}), 10-epoch budget read from 20-epoch histories by top-1 then CE, baselines, switch rule (≥ 0.30 pp and ≥ 5/6), lower/upper-edge and budget-binds flags, instability (> 5% rise, 20 epochs, dictionary heads), 18 + 6 sanity cells and the three pre-declared exploratory gaps are all implemented as written. |
| Linear-head wrapper | `train_linear` reads the module globals `LR` (optimizer, line 151; saved settings, line 173) and `EPOCHS` (loop, line 157) at call time, so the `try/finally` override of `rg.LR, rg.EPOCHS` takes effect and is restored. Empirically: the four linear lrs give different trajectories (it is not silently 1e-3), and every linear history has 20 epochs. Dictionary runs pass `lr` and `EPOCHS` explicitly, so they do not depend on the override. |
| 10-of-20 premise | Code: constant Adam lr (no scheduler), `torch.manual_seed(seed + epoch)` and loader seed `seed + epoch` each epoch in both `_fit` and `train_linear`, and nothing in the loop depends on the total epoch count. Data: in all 24 sanity cells the **entire** first-10-epoch trajectory (validation top-1 and CE for every epoch, plus training loss for dictionary heads) is bit-identical to the earlier 10-epoch run, not just the selected epoch. **Premise verified** at lr 1e-4 (dictionary) and 1e-3 (linear); at other lrs it rests on the same code path. |
| Sanity (pre-registered) | 24/24 identical in top-1, selected epoch and split digest (18 LR-001 lr 1e-4 cells, 6 GRID-002 linear cells; GRID-002 linear config confirmed Xavier, wd 1e-4). |

### Decision table (recomputed; agrees with analysis.md to the reported precision)

| Head | Best candidate (mean %) | Gain vs baseline, pp [95% CI], positive | Adopt? | Adopted | Median sel. epoch | Flags |
| --- | --- | --- | --- | --- | --- | --- |
| sparse M2048 | 1e-4@20 (60.65) | +0.277 [+0.18, +0.38], 6/6 | no (< 0.30) | 3e-4@10 | 8.5 | none |
| sparse M4096 | 1e-4@20 (61.70) | +0.020 [−0.03, +0.07], 1/6 | no | 1e-4@10 | 9.5 | none |
| dense M2048 | 1e-4@20 (60.65) | +0.113 [−0.16, +0.38], 2/6 | no | 1e-4@10 | 8.5 | none |
| linear | 3e-4@20 (60.30) | +2.143 [+1.84, +2.44], 6/6 | **yes** | 3e-4@20 | 16.5 | none |

All 28 candidate means, median epochs and the 25 vs-baseline rows in analysis.md reproduce exactly. Instability: 0/6 for every dictionary head × lr (largest consecutive-epoch loss ratio 0.964, i.e. no rise at all). The report's text numbers (3e-5@20 −0.06 to −0.32 pp; 1e-5@20 −3.56 to −6.16 pp; linear 3e-3 −4.74 pp) are correct.

### Exploratory gaps (recomputed)

| Gap (pp, paired, df 5) | Mine | Report |
| --- | --- | --- |
| sparse M2048 − linear @ adopted | +0.08 [−0.20, +0.36] 3/6 | same |
| sparse M2048 − dense @ adopted | −0.16 [−0.35, +0.03] 1/6 | same |
| dense − linear @ adopted (post hoc) | +0.24 [+0.01, +0.47] 5/6 | same |
| M4096 − M2048 @ adopted | +1.30 [+0.98, +1.62] 6/6 | same |
| M4096 − linear @ adopted (post hoc) | +1.38 [+0.97, +1.79] 6/6 | same |
| sparse M2048 − linear @ lr 1e-3 (10 ep) | +1.89 [+1.57, +2.21] 6/6 | same |
| sparse − dense / M4096 − M2048 @ 1e-3 | +0.62 [+0.42, +0.83] / +0.53 [+0.28, +0.78] | same |
| Per level, source / L090 (drop) | sparse 64.67 / 56.08 (8.59); M4096 66.02 / 57.33 (8.69); dense 64.71 / 56.37 (8.34); linear 64.42 / 56.17 (8.25) | same ("8.2–8.7") |

Additional sensitivity rows I computed (not in LR-002; validation, same cells, post hoc):

| Comparison | pp [95% CI], positive |
| --- | --- |
| sparse M2048 3e-4@10 − linear 3e-4@10 (equal 10-epoch budget, same lr) | **+0.66 [+0.49, +0.84] 6/6** |
| sparse M2048 1e-4@20 − linear 3e-4@20 (equal 20-epoch budget; each head's best candidate) | **+0.36 [+0.08, +0.64] 5/6** |
| dense 1e-4@10 − linear 3e-4@10 / dense 1e-4@20 − linear 3e-4@20 | +0.82 [+0.56, +1.09] 6/6 / +0.35 [+0.04, +0.67] 5/6 |
| sparse 1e-4@20 − dense 1e-4@20 | +0.00 [−0.33, +0.33] 3/6 |
| M4096 1e-4@20 − M2048 1e-4@20 | +1.04 [+0.82, +1.27] 6/6 |
| No selection: sparse 3e-4 epoch 10 − linear 3e-4 epoch 20 | +0.25 [−0.37, +0.87] 5/6 |
| Linear 3e-4, mean top-1 by epoch 10…20 | 59.62, 59.92, 59.91, 59.93, 59.80, 59.92, 60.03, 59.85, 59.99, 59.98, 59.93 (selected: 60.30) |
| Linear 1e-4, mean top-1 at epochs 18/19/20 | 59.36 / 59.54 / 59.68 (still rising) |

## Findings

### Must fix

**S-1. The "sparse M2048, dense and linear are tied" conclusion, and the "probably lr artifacts" verdict on GRID-002 / SEED-001, rest on unequal epoch budgets produced by the 0.30 pp threshold; disclose the equal-budget rows and soften.**
- The +0.08 pp gap compares sparse M2048 at 10 epochs (its 20-epoch candidate missed the switch threshold by 0.02 pp: +0.277, CI [0.18, 0.38], 6/6) with linear at 20 epochs. At an equal budget the dictionary heads stay ahead of linear: sparse − linear **+0.66 [0.49, 0.84] 6/6** at 10 epochs (same lr 3e-4) and **+0.36 [0.08, 0.64] 5/6** at 20 epochs with each head at its best candidate; dense − linear +0.82 / +0.35, CIs excluding 0.
- Best-of-20 selection on the linear head's flat plateau (epochs 11–20 all ≈ 59.8–60.0 mean, selected 60.30) also flatters it relative to best-of-10 for sparse (without selection the sparse − linear gap is +0.25, CI wide). This is the REVIEW-008 O-4 direction; caveat 5's "≤ 0.3 pp" compares 1e-4@20 with @10, which is a budget effect, not a measure of selection optimism.
- So the defensible reading is: correcting the linear head's lr removes most (about 1.2–1.5 pp) of the 1.89 pp validation gap at lr 1e-3, and whether a residual ~0.3–0.7 pp remains depends on the epoch budget. "Tied within ±0.25 pp" (line 102) and "probably lr artifacts" (line 124) should become "largely, possibly fully, explained by lr" with the equal-budget rows shown next to the adopted-setting rows, labelled post hoc. Caveat 4 already says budgets should be disclosed "in any comparison"; the tie bullet does not do so.
- The proposed confirmatory ImageNetV2 test inherits the asymmetry (sparse/dense 10 epochs, linear 20). Its pre-registration should either use a common budget for all heads (e.g. 20 epochs, with each head's best lr at that budget) or pre-declare both the adopted-setting and the equal-budget comparisons, so the outcome cannot hinge on a 0.02 pp threshold miss.

**S-2. Row provenance in the exploratory table is mislabelled.** The report says "The first three rows are pre-declared in task.json", but the table order is sparse − linear, sparse − dense, **dense − linear**, M4096 − M2048, **M4096 − linear**. The pre-declared rows are 1, 2 and 4; rows 3 and 5 are post hoc (the next sentence says so, contradicting the first). Mark post-hoc rows in the table itself.

### Optional

**O-1. Reproducibility of report-only numbers.** The dense − linear and M4096 − linear rows, the "At lr 1e-3" column (+1.89 for sparse − linear uses LR-001 lr 1e-3 `done.json`), and the per-level means are not produced by `analyze.py`/`analysis.md`. I reproduced them all, but they should be generated by a script (a separate post-hoc script is fine; do not alter the pre-registered analyze.py logic) or their source stated.

**O-2. "Located" is conditional on the 20-epoch cap.** The rule's flags fire only on the adopted setting, so they are correctly empty. But linear 1e-4 is still rising at epoch 20 (59.36 → 59.54 → 59.68; median selected epoch 20), as are the dictionary heads at 3e-5 (median 19.5–20). The optimum is located over lr × {10, 20} epochs, not over budgets in general. A linear run takes ~17 s, so a longer-budget check is cheap if the head comparison is to rest on ~0.3 pp. State the conditionality.

**O-3. Implications for earlier tasks: wording and scope.**
- GRID-002's sparse/dense heads used the full loss (reconstruction 1, dependency 0.1, coherence 0.05); LR-002 uses the MSE-only 1/0/0 setting of SEED-001. Attributing GRID-002's sparse > linear to lr alone mixes two settings (ABL-001 reported the extra terms had no measurable effect, so this is minor; say so).
- GRID-002 / SEED-001 / MSWEEP-001 numbers are ImageNetV2 test results; LR-002's are validation gaps on seeds already used twice for selection. Line 127 ("the gap is about 1.3 pp at a stable lr") should say "on validation", and the 1.3 pp mixes lrs (M4096 at 1e-4, M2048 at 3e-4); at a common setting it is +1.33 (1e-4@10) or +1.04 (1e-4@20).
- "Capacity effect at fixed active fraction, not evidence that sparsity itself helps" is a fair inference, but there is no dense M4096 control, so it is untested which part of the M4096 gain is width vs active-atom count. The parameter counts (≈ 8.3 M vs 4.2 M) include the decoder, which serves only the auxiliary loss; the deployed classifier is ≈ 6.2 M vs 3.1 M.

**O-4. Pre-registration record.** Add that `analyze.py` was last written at 17:29:39 local, after the smoke runs but before any other run completed, and that its logic matches task.json. This closes the question of whether the analysis was adjusted after seeing results.

**O-5. Minor.** Conclusion 1 says "Doubling to 20 epochs at 1e-4 adds at most +0.28 pp"; that is relative to the 3e-4@10 baseline. Against 1e-4@10 the sparse M2048 gain is +0.30 pp. Say "vs baseline". `results_partial.csv` from the smoke runs could be removed or labelled, as the report already notes.

## What is solid

- The pre-registration was in place before any run, the analysis implements it, and all 24 sanity cells reproduce bit-for-bit over the whole 10-epoch trajectory. That validates the 10-of-20-epochs design at the tested lrs.
- No test data was used. All decisions follow the rule: the dictionary heads keep LR-001's settings, and the linear head switches to 3e-4@20 (+2.14 pp, 6/6).
- The main substantive finding is well supported on validation: the linear head was under-tuned at lr 1e-3 by about 1.6–2.1 pp. That is large enough that the earlier sparse − linear test gaps should not be cited as a sparse-dictionary benefit without a re-test.
- Instability is absent at lr ≤ 1e-4 (0/54 runs).
- The caveats on repeated selection, grid holes and un-retuned k / weight decay are disclosed adequately.

Verdict: revise
