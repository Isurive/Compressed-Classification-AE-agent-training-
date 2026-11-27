# REVIEW-010: independent review of RETEST-001

**Reviewer:** `reviewer-20261126-k` (did not author RETEST-001). **Date:** 2026-11-26 (local, UTC-7).
**Reviewed:** `ai_harness/artifacts/RETEST-001/` (task.json, report.md, analyze.py, analysis.md, analysis.json, grid_log.txt), `src/run_retest.py`, and `runs/RETEST-001/` (results.csv, settings.json, 84 `done.json`, file presence). For comparison: `runs/SEED-001/results.csv`, `runs/MSWEEP-001/results.csv`, LR-002 `analysis.md` / `posthoc.md`, SEED-001 and MSWEEP-001 reports and task.json, GRID-001 `analysis.md`, and the helpers `src/run_grid.py`, `src/run_ablation.py`, `src/sparse_workflow.py`.
**Recompute script:** [review010.py](review010.py), standard library only, run as `py -I review010.py "<Final Project>"`. It does not import analyze.py. It computes p-values with a regularized incomplete beta function (analyze.py uses Simpson integration) and also gives CIs with the exact t quantile.
**Constraints kept:** read-only outside this folder. No training, no GPU, no torch. One disclosure: to confirm the linear-head lr override, I read the pickle **opcode stream** (`pickletools.genops`) of `data.pkl` inside the 21 linear `best.pt` zip files. Nothing was unpickled or executed and no model was built. If the coordinator counts this as loading a checkpoint, set check C-3b aside; the code reading and the indirect evidence give the same answer.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of RETEST-001) saved it here verbatim, with no edits to content, on 2026-11-26, and copied the reviewer's scratchpad script to [review010.py](review010.py) so the link resolves. The coordinator accepts check C-3b: reading the opcode stream without unpickling is within "do not load checkpoints".

## Verdict: **revise** (wording only)

Every number I recomputed matches analysis.md and report.md to the printed precision. The code trains what task.json specifies. Pre-registration timing holds. No result needs to be recomputed or rerun.

Four statements in report.md go beyond the evidence (S-1 to S-4). Each needs a one- or two-sentence rewrite. These sentences decide what goes into the write-up, so they should be fixed before acceptance.

## What I checked and how

### C-1. Pre-registration timing (verified)

| Event | Time (local, UTC-7) | Source |
| --- | --- | --- |
| task.json written | 20:54:45.5 | file mtime |
| Ledger `add` and `claim` RETEST-001 | 20:54:48 (2026-11-27T03:54:48Z) | `events` seq 144–145 in state.sqlite3 (read-only) |
| src/run_retest.py written | 20:55:10.0 | file mtime |
| First run starts (smoke, linear source seed 45) | 20:55:31 | grid_log.txt |
| analyze.py last written | 20:57:25.6 | file mtime |
| Last run done.json | 21:43:41.7 | file mtime |
| Grid end, exit 0 | 21:43:42 | grid_log.txt |
| analysis.md / analysis.json | 21:43:51 | file mtime |
| report.md | 21:45:30 | file mtime |

- The ledger spec equals task.json (title, description, write_paths, acceptance).
- When analyze.py was written, exactly 3 `done.json` existed: the two smoke cells and sparse-M4096/source/seed45. This matches the report. I cannot verify the claim that the third row was not read. The exposure is one cell, so the risk is low (O-2).
- analyze.py was not modified after the grid.
- No file in src/, SEED-001, MSWEEP-001 or LR-002 (artifacts or results.csv) has an mtime after 20:54, apart from the new run_retest.py. Tracked src/ files have no git diff against HEAD.
- `harness.py check` returns ok.
- The copies of grid_log.txt in runs/ and in the artifact folder are identical.

### C-2. analyze.py against task.json (verified, with one weak assertion)

These match task.json:

- Metric: ImageNetV2-source top-1, paired by seed 45–51, df 6, t = 2.447. The exact quantile is 2.44691, and CIs are identical to 3 decimals in pp.
- A1–A4 are pooled as the per-seed mean of the per-level differences over source/L080/L090. Their operand order matches task.json.
- B1–B4 are A(source) − A(L080) and B5–B8 are A(source) − A(L090), in the head order sparse M2048, sparse M4096, dense M2048, linear.
- Detected = CI excludes 0. Meaningful = detected and |mean| ≥ 1.0 pp. Holm runs step-down over the 12 tests with a running max.
- The wording rule is implemented exactly. That includes the "meaningful but not Holm-detected → small difference" branch.
- The checks match: median selected epoch, counts ≥ 19 and ≤ 2, the budget-binds flag at median ≥ 19, and instability as a consecutive training-loss rise > 5% (LR-001 definition).
- Exploratory sources are correct: SEED-001 `seed-mse` sparse/dense M2048 and `seed-linear`, and MSWEEP-001 `msweep-clean` M4096. Each gives 21 cells, with no duplicate keys and the same split digests as RETEST.
- The LR-002 constants (A1 +0.36, A2 +0.00, A3 +1.04) equal LR-002 posthoc.md, "Equal 20-epoch budget".

The weak assertion is reported under O-1.

### C-3. src/run_retest.py against task.json (verified)

1. **Heads and loss.**
   - Configuration: sparse 2048/512, sparse 4096/1024, dense 2048/2048 (dense = k = M, as in ABL-001), all with Xavier fc, He encoder, weight decay 0 and `ra.VARIANTS["abl-mse"]` = reconstruction 1 / dependency 0 / coherence 0. Linear: Xavier, weight decay 1e-4.
   - Every `done.json` records the intended head, lr and weights.
   - The log has 63 lines reading `Adam lr=0.0001.`, none with another lr, and 63 lines each for Xavier fc and He encoder.
2. **Linear lr and epoch override (C-3a).** `rg.LR` and `rg.EPOCHS` are set before `rg.train_linear` and restored in a `finally` block. `train_linear` reads `LR` and `EPOCHS` as module globals at call time, for the optimizer, the epoch loop and the saved settings. The same pattern is used in run_lrcheck2.py, whose LR-002 sanity reproductions were identical. The side effect of importing run_ablation (`rg.OUT` = ABL-001) is overwritten before use. `FEATURES` was bound to GRID-001 at import, so the features are read-only GRID-001 caches as intended.
3. **Linear settings in best.pt (C-3b, opcode read only).** All 21 linear `best.pt` files record lr 0.0003, epochs 20, weight decay 1e-4 and batch 128. Each checkpoint epoch equals `done.json` best_epoch. Indirect evidence agrees: the linear median selected epoch is 15, against 16.5 for lr 3e-4 at 20 epochs and 8 for lr 1e-3 in LR-002.
4. **Training loop.** 20 epochs at a constant lr, `torch.manual_seed(seed + epoch)` each epoch, best.pt on strictly greater (val top-1, −val CE), batch 128, 45k/5k split. Every one of the 84 histories has epochs 1..20. I re-derived best_epoch from each history with the same rule: 84/84 match `done.json`, and `val_top1` and `split_digest` in results.csv match `done.json`.
5. **Testing.** `rg.test_run` scores once at source and once at the training level (one row for source-trained runs). `recorded()` is taken once at grid start, so the two smoke cells were skipped. results.csv has **140 rows** (28 + 2 × 56), all `phase=retest` and `eval_set=imagenetv2`, with **no duplicate (run_dir, eval_level)** and no missing expected row. Labels and weight decay in each row match its run_dir.
6. **Files.** last.pt is absent in 84/84 runs and best.pt is present in 84/84. Total run folder size is 3.64 GiB.

### C-4. Pre-declared checks (recomputed, all match)

- 84/84 runs.
- One digest per seed, equal to SEED-001's (7/7), with 7 distinct digests.
- Median selected epoch: sparse M2048 14, sparse M4096 14, dense 11, linear 15.
- Runs selecting epoch ≥ 19: 2, 3, 2, 3. Runs selecting epoch ≤ 2: 0 for every head. No budget-binds flag.
- Instability: 0/63. The largest consecutive training-loss ratio in any dictionary run is 0.940, so the loss never rose.

## Recomputed numbers

**Mean ImageNetV2-source top-1 (%)**

| Head | source | L080 | L090 |
| --- | --- | --- | --- |
| sparse M2048 | 49.293 | 48.446 | 47.104 |
| sparse M4096 | 49.903 | 49.110 | 47.724 |
| dense M2048 | 49.103 | 48.417 | 47.064 |
| linear | 48.941 | 48.416 | 47.433 |

**Confirmatory family (pp; t 2.447, df 6; p from the incomplete beta function)**

| Test | Mean | sd | 95% CI | positive | p | Holm p | Rule outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 sparse M2048 − linear | +0.018 | 0.291 | [−0.252, +0.287] | 4/7 | 0.878 | 0.878 | no detected difference |
| A2 sparse M2048 − dense M2048 | +0.086 | 0.091 | [+0.002, +0.171] | 5/7 | 0.0470 | 0.0940 | small / not robust |
| A3 sparse M4096 − sparse M2048 | +0.631 | 0.178 | [+0.467, +0.796] | 7/7 | 0.00008 | 0.00074 | small (Holm-detected) |
| A4 sparse M4096 − linear | +0.649 | 0.215 | [+0.450, +0.848] | 7/7 | 0.00021 | 0.00144 | small (Holm-detected) |
| B1 sparse M2048 cost L080 | +0.847 | 0.349 | [+0.524, +1.170] | 7/7 | 0.00067 | 0.00336 | small |
| B2 sparse M4096 cost L080 | +0.793 | 0.312 | [+0.504, +1.081] | 7/7 | 0.00053 | 0.00316 | small |
| B3 dense M2048 cost L080 | +0.686 | 0.296 | [+0.412, +0.959] | 7/7 | 0.00086 | 0.00343 | small |
| B4 linear cost L080 | +0.526 | 0.399 | [+0.157, +0.895] | 7/7 | 0.0130 | 0.0391 | small |
| B5 sparse M2048 cost L090 | +2.189 | 0.297 | [+1.914, +2.463] | 7/7 | <1e-5 | 0.00001 | finding |
| B6 sparse M4096 cost L090 | +2.179 | 0.396 | [+1.812, +2.545] | 7/7 | 0.00001 | 0.00007 | finding |
| B7 dense M2048 cost L090 | +2.039 | 0.475 | [+1.599, +2.478] | 7/7 | 0.00003 | 0.00028 | finding |
| B8 linear cost L090 | +1.509 | 0.482 | [+1.063, +1.954] | 7/7 | 0.00017 | 0.00134 | finding |

A2 per-seed values (pp) are +0.133, +0.183, +0.183, +0.100, −0.003, −0.057, +0.063. Its detection rests on a CI lower bound of +0.002.

**Exploratory (all match analysis.md)**

- **Gain over the lr 1e-3 / 10-epoch runs, pooled:**
  - sparse M2048 +1.005 [+0.825, +1.185]
  - sparse M4096 +1.180 [+1.021, +1.340]
  - dense +1.331 [+1.160, +1.503]
  - linear +1.954 [+1.787, +2.120]
  - All 7/7 seeds.
- **Old (lr 1e-3) pooled gaps, same seeds and levels:**
  - A1 +0.966 [+0.781, +1.152]
  - A2 +0.412 [+0.290, +0.535]
  - A3 +0.456 [+0.190, +0.723]
  - A4 +1.422 [+1.191, +1.653]
- **Cost differences between heads, new runs:**
  - sparse M2048 − linear: L080 +0.321 [−0.145, +0.788]; L090 +0.680 [+0.238, +1.122], 6/7.
  - sparse M4096 − linear: L080 +0.267 [−0.205, +0.740]; L090 +0.670 [+0.272, +1.068], 6/7.
  - sparse M2048 − dense: L080 +0.161; L090 +0.150 (both CIs include 0).
  - Not computed in analyze.py: dense − linear at L090 +0.530 [+0.068, +0.992], 7/7; sparse M4096 − sparse M2048 at L090 −0.010.
- **Cost differences on the old lr 1e-3 runs, same seeds** (not in RETEST-001):
  - sparse M2048 − linear at L090 +0.530 [+0.152, +0.908], 6/7.
  - dense − linear at L090 +0.370 [+0.079, +0.661], 6/7.
  - sparse M4096 − linear at L090 +0.226 [−0.421, +0.872].
  - Old per-head L090 costs: sparse M2048 2.19, sparse M4096 1.88, dense 2.03, linear 1.66.
- **Per-level secondary gaps, top-5 and training-level evaluation:** all match analysis.md. The extra drop from scoring on compressed ImageNetV2 is 2.41–6.71 pp.
- **Other figures in report.md:**
  - LR-002 disclosure figures (+0.28, +0.02, +0.11; linear +2.14) match LR-002 analysis.md.
  - GRID-001 byte figures (q20 50.3:1, 89.8%; q10 86.0:1, 94.0%) match GRID-001 analysis.md.

## Findings

### Substantive (must fix before acceptance; wording only)

**S-1. The headline storage claim drops the CI qualifier (report.md, final paragraph).**

The report says training on q20 copies "costs under 1 pp of ImageNetV2 accuracy for every head". That holds for the point estimates (0.53–0.85 pp). It does not hold for the 95% CIs of the two sparse heads, which reach +1.17 (B1) and +1.08 (B2). Only dense (+0.96) and linear (+0.90) have CIs wholly below 1 pp.

Point 4 of the report states this correctly ("not an equivalence result"). The Consequences table also gives "CI up to 1.17". The headline sentence, the one most likely to be copied into the write-up, does not.

The pre-registered B wording ("not meaningful means … retained accuracy within 1 pp") is based on the point estimate. The report was right to decline that reading in point 4, and the headline should match point 4. Suggested wording: "Mean cost 0.5–0.85 pp. The 95% CIs lie below 1 pp for the linear and dense heads but reach 1.08–1.17 pp for the sparse heads. These CIs are conditional on the fixed test set (S-4)."

**S-2. A3/A4 are attributed to "a capacity effect" and "a larger dictionary", which the design cannot isolate.**

- Point 3 says "this is a capacity effect, not evidence for sparsity". The Consequences table says "a capacity effect". The final paragraph says "Its only robust advantage comes from a larger dictionary".
- The comparison is M4096/k1024 against M2048/k512. It doubles the available atoms M, the active atoms k and the parameter count together, and there is no dense-M4096 or parameter-matched control.
- Disclosure 4 itself says A3/A4 "do not isolate the number of atoms from the parameter count". It does not mention that k also doubles.
- AGENTS.md requires M and k to be kept distinct.

Suggested wording: "The M4096/k1024 head beats M2048/k512 by +0.63 pp (robust, below 1 pp). The design does not separate M, k and parameter count. A2 gives no evidence that top-k sparsity itself helps." "Capacity" can stay as a labeled inference.

**S-3. "Reverses the earlier impression" misstates the project record.**

The report says the compression-cost result "reverses the earlier impression that 'no head changes the compression cost'". The Consequences row is "Compression costs are similar for every head → Revised". Both are inaccurate:

- **The earlier claim was different.** The earlier project claims (SEED-001 item 3, MSWEEP-001 item 4, LR-002 line 151, WORK_SUMMARY) were that *no head reduces the drop*, that is, no head is more compression-tolerant than linear. A linear head that loses *less* is consistent with that claim, not a reversal of it.
- **The same pattern was already in the same-seed lr 1e-3 data:**
  - cost(sparse M2048) − cost(linear) at L090 = +0.53 [+0.15, +0.91], 6/7.
  - cost(dense) − cost(linear) = +0.37 [+0.08, +0.66], 6/7.
  - SEED-001 report item 3 already says the wide heads' advantage over linear "does narrow toward L90" (exploratory).
- **The new part is limited.** Only sparse M4096 − linear is newly detected (+0.67 against +0.23 before). Everything is uncorrected and exploratory.
- **The cost measure is confounded.** A lower absolute-pp cost for linear goes with linear's lower source accuracy. "The linear head is the most compression-tolerant" is a fair description of the point estimates, but it should be framed as "the dictionary heads' source-level advantage over linear shrinks toward L090", not as a property of the linear head.

Suggested wording: "Exploratory, replicates a SEED-001 pattern: at L090 the dictionary heads lose 0.5–0.7 pp more than linear (uncorrected). Consistent with the earlier claim that no head adds compression tolerance."

**S-4. Disclosure 5 states without support that test-sample variation "cancels largely in paired head differences".**

- The CIs cover seed variation only, on one fixed set of 10,000 images.
- A paired difference on a fixed test set still has an image-sampling component. Its size depends on how often the two models disagree on correctness, and it was not measured, because per-image predictions were not saved.
- **Inference, not measured:** for a single pair of models, discordance of 5–10% on 10,000 images corresponds to an image-sampling SE of roughly 0.2–0.3 pp. Averaging over seeds reduces this, but systematic image-level head effects do not average out. This is the same order as the A3/A4 effects and the B half-widths.
- The fragile result is A2, whose detection rests on a lower bound of +0.002.

Fix: say that the CIs are conditional on these 10,000 test images and that the image-sampling component was not quantified. Optional follow-up (no training): re-score the 84 best.pt on cached test features, save per-image correctness, and run a paired bootstrap over images.

### Optional / minor

- **O-1. Weak duplicate check in analyze.py.** Its duplicate-row assertion skips `train_level == source` rows, which is exactly where the smoke-resume duplicates would appear. Its 21-per-head count would not catch a duplicate either, because dict assignment overwrites. My independent check found 140 rows and no duplicate (run_dir, eval_level), so the report's "no duplicate test rows" is true. But analyze.py does not establish it. Add the assertion for source rows, or cite this check.
- **O-2. Third row at analysis-code time.** analyze.py was written after a third, non-smoke result row existed (sparse-M4096 source seed 45). The report discloses this. The not-read claim cannot be verified, and I found no deviation of the code from task.json. No action beyond keeping the disclosure.
- **O-3. Disclosure 2 omits the lr departure for sparse M2048.** LR-002's rule adopted lr 3e-4 at 10 epochs for sparse M2048, so RETEST departs in lr as well as epochs for that head. task.json mentions only "the 10-epoch settings". The +0.28 validation figure is already the right comparison. Say the lr part explicitly.
- **O-4. "Withdrawn".** The lr 1e-3 measurements remain correct (recomputed: A1 +0.97, A2 +0.41 on the same seeds). What is withdrawn is the head-architecture generalization. Suggest "superseded: lr-conditional; not supported at tuned lrs". For A1, the CI [−0.25, +0.29] lies inside ±1 pp. That can be stated descriptively; it is not a pre-registered equivalence test.
- **O-5. Gain attribution.** The "tuning gain" mixes the lr change with the change from 10 to 20 epochs. "That gap came from the under-tuned linear head" would be more exact as "the gap shrank because the linear head gained 0.95 pp more than the sparse head from the lr and budget change".
- **O-6. The LR-002 "prediction" column is not like-for-like.** It pools source/L090 on validation seeds 42–44, while RETEST pools source/L080/L090 on the test set. Label this. Also say in one sentence that the validation gaps overstated the test gaps (A1 +0.36 against +0.02, A3 +1.04 against +0.63).
- **O-7. 89.8% saving.** When quoting "89.8% smaller than the original files", add the COORD-002 qualifier that the saving against stored ImageNet files includes stripped metadata.
- **O-8. Budget edge.** "Medians of 11–15 show no head was cut off" is fine under the pre-registered rule. Still, 2–3 runs per head selected epoch ≥ 19, and 2 per head selected epoch 20. Say so in a clause.
- **O-9. Reproducibility.** results.csv has no lr or epochs column (RESULT_FIELDS). Per-run lr lives only in done.json, settings.json and, for linear, best.pt. A future analysis should join on done.json.

## Recommendation

Revise RETEST-001 with the four wording fixes S-1 to S-4. O-1 to O-9 are at the author's discretion. No recomputation or rerun is needed: the data, code, checks and confirmatory family are correct and reproduce independently.

Once the wording is fixed, the accepted conclusions are:

- **A1 (sparse M2048 vs linear):** no detected difference at tuned lrs. The lr 1e-3 gap was lr-conditional.
- **A2 (sparsity at equal width):** at most a small, non-robust effect (+0.09 pp).
- **A3/A4:** M4096/k1024 is robustly but sub-tolerance better (+0.6 pp). The cause (M, k or parameter count) cannot be identified.
- **L080 cost:** 0.5–0.85 pp, with CIs reaching about 1.2 pp for the sparse heads.
- **L090 cost:** 1.5–2.2 pp, meaningful for every head.
- All of the above are conditional on the fixed ImageNetV2 test set and on frozen pretrained ResNet-18 features.
