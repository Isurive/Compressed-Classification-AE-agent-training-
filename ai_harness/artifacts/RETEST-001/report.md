# RETEST-001: confirmatory ImageNetV2 re-test of the heads at tuned learning rates and a common 20-epoch budget

**Agent:** `claude-orient-20261122-a`. **Approved by the user:** "Yes, run re-test" (2026-11-26), answering "Run the confirmatory ImageNetV2 re-test (84 runs, ~50 min)?".
**Pre-registration:** [task.json](task.json) (written 20:54:45 local). The ledger `add` (2026-11-27 03:54:48 UTC = 20:54:48 local) predates the first run (smoke cell, 20:55:31 local; see [grid_log.txt](grid_log.txt)).
**Runs:** 84, 20 epochs each, 2026-11-26 20:55–21:43 local.
- Two smoke cells ran first (linear and sparse M2048, source, seed 45). The full grid then reused them through `done.json`, and `run_grid.recorded()` kept their test rows from being written twice.
- Exit 0, no tracebacks. Per-batch progress lines are filtered out of the log after the smoke cells.

**Analysis:** [analyze.py](analyze.py) → [analysis.md](analysis.md) / [analysis.json](analysis.json).
- `analyze.py` was last written at 20:57:25 local, while the grid was running. By then three runs had finished (the two smoke cells and sparse M4096 source seed 45). Only the two smoke-cell rows had been read, as a pipeline check (digest, file layout). The script was not changed after the grid finished.
- Its logic follows task.json. The family has exactly 12 tests, checked by an assertion.
- Revision 1 added only an explicit duplicate-test-row assertion over every row (REVIEW-010 O-1). Its outputs are byte-identical (same SHA-256 for analysis.md and analysis.json).

**Revision 1** (2026-11-26) addresses [REVIEW-010](../REVIEW-010/report.md) S-1 to S-4 and O-1, O-3 to O-8. The log is at the end. No run or number changed.

**Code:** [src/run_retest.py](../../../src/run_retest.py) (new). It reuses `run_grid.py`, `run_ablation.py` and `sparse_workflow.py` unchanged; ImageNetV2 scoring is `run_grid.test_run`, as in SEED-001 and MSWEEP-001.

**Commands:** from `src/`, `C:\Users\ericw\miniconda3\envs\ai360\python.exe -B -u`.
- Smoke: `run_retest.py grid --seeds 45 --levels source --heads linear sparse-M2048`.
- Full: `run_retest.py grid`.
- Analysis, from `Final Project`: `py -B ai_harness/artifacts/RETEST-001/analyze.py`.

## Question

LR-001 and LR-002 (validation only, seeds 42–44) showed that Adam lr 1e-3, used by every earlier head comparison, is wrong for every head. The linear head was under-tuned by about 2 pp. So the earlier ImageNetV2 claims "sparse beats linear by about 1 pp" (GRID-002, SEED-001), "top-k sparsity adds about 0.4 pp" (SEED-001) and "M4096 beats M2048 by 0.46 pp" (MSWEEP-001) were lr-conditional.

This task re-measures them on ImageNetV2 with every head at a tuned lr and the same 20-epoch budget, on seeds not used for any lr choice. Its results decide which head claims go into the write-up.

## Design (as pre-registered)

| Item | Setting |
| --- | --- |
| Heads | sparse M2048 k512, sparse M4096 k1024, dense M2048 k2048: Xavier fc, He encoder, CE + feature MSE (1/0/0), Adam **lr 1e-4**. Linear: Xavier, weight decay 1e-4, CE, Adam **lr 3e-4**. |
| Budget | **20 epochs for every head**, constant lr, per-epoch reseeding. `best.pt` = the epoch with the best validation top-1 (then lower CE), on the seed's own 5k validation split. |
| Data | Training on the 45k pool split at levels source, 80 (JPEG q20) and 90 (q10); matched-compression validation; cached GRID-001 features (frozen ImageNet-pretrained ResNet-18). |
| Seeds | 45–51 (7). Same splits as SEED-001 and MSWEEP-001; not used for any lr choice. |
| Test | Original ImageNetV2 matched-frequency (10,000 images), once per run after training. The training-level ImageNetV2 score is also recorded (exploratory). |
| Statistic | Paired by seed, n = 7, df 6: mean, sd, two-sided paired-t 95% CI, sign count, p. Holm over the 12-test family. |
| Decision | **Detected** = CI excludes 0. **Meaningful** = detected and \|mean\| ≥ 1.0 pp. **Write-up rule:** meaningful and Holm-detected → stated as a finding; detected only → "small difference below the 1 pp tolerance or not robust to multiplicity"; otherwise "no detected difference". |

Family: A1 sparse M2048 − linear, A2 sparse M2048 − dense M2048, A3 sparse M4096 − sparse M2048, A4 sparse M4096 − linear (pooled over the 3 levels); B1–B8 compression cost A(source) − A(level) for each head at L080 and L090.

## Confirmatory results (ImageNetV2-source top-1)

Mean accuracy (%), 7 seeds:

| Head | source | L080 | L090 |
| --- | --- | --- | --- |
| sparse M2048 | 49.29 | 48.45 | 47.10 |
| sparse M4096 | **49.90** | **49.11** | **47.72** |
| dense M2048 | 49.10 | 48.42 | 47.06 |
| linear | 48.94 | 48.42 | 47.43 |

The 12 tests (pp):

| Test | Mean | 95% CI | positive | Holm p | Detected | Meaningful | Write-up wording |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 sparse M2048 − linear | +0.02 | [−0.25, +0.29] | 4/7 | 0.878 | no | no | **no detected difference** |
| A2 sparse M2048 − dense M2048 | +0.09 | [+0.00, +0.17] | 5/7 | 0.094 | yes (p 0.047) | no | small, below 1 pp; not Holm-detected |
| A3 sparse M4096 − sparse M2048 | +0.63 | [+0.47, +0.80] | 7/7 | 0.0007 | yes | no | small, below 1 pp (Holm-detected) |
| A4 sparse M4096 − linear | +0.65 | [+0.45, +0.85] | 7/7 | 0.0014 | yes | no | small, below 1 pp (Holm-detected) |
| B1 sparse M2048 cost at L080 | +0.85 | [+0.52, +1.17] | 7/7 | 0.003 | yes | no | below 1 pp |
| B2 sparse M4096 cost at L080 | +0.79 | [+0.50, +1.08] | 7/7 | 0.003 | yes | no | below 1 pp |
| B3 dense M2048 cost at L080 | +0.69 | [+0.41, +0.96] | 7/7 | 0.003 | yes | no | below 1 pp |
| B4 linear cost at L080 | +0.53 | [+0.16, +0.89] | 7/7 | 0.039 | yes | no | below 1 pp |
| B5 sparse M2048 cost at L090 | +2.19 | [+1.91, +2.46] | 7/7 | <0.001 | yes | **yes** | **finding** |
| B6 sparse M4096 cost at L090 | +2.18 | [+1.81, +2.54] | 7/7 | <0.001 | yes | **yes** | **finding** |
| B7 dense M2048 cost at L090 | +2.04 | [+1.60, +2.48] | 7/7 | <0.001 | yes | **yes** | **finding** |
| B8 linear cost at L090 | +1.51 | [+1.06, +1.95] | 7/7 | 0.001 | yes | **yes** | **finding** |

(A2's CI lower bound is +0.0016 pp before rounding; revision 1 corrects "+0.003", found by REVIEW-010.) Full table with sd, t-based p and Holm p: [analysis.md](analysis.md).

### What the pre-registered rule says

1. **Sparse vs linear (A1): no detected difference** (+0.02 pp, CI [−0.25, +0.29]; descriptively inside ±1 pp, but this is not a pre-registered equivalence test). On the same seeds, levels and test images, the old lr 1e-3 / 10-epoch runs gave +0.97 pp. The gap shrank because the linear head gained 0.95 pp more than sparse M2048 from the change in lr and budget (REVIEW-010 O-5).
2. **Sparsity at equal width (A2): small, below 1 pp, not robust to multiplicity** (+0.09 pp; detection rests on a CI lower bound of +0.002). No evidence that top-k sparsity helps in practice at a tuned lr. The old lr 1e-3 runs gave +0.41.
3. **M4096/k1024 vs M2048/k512 (A3) and vs linear (A4): small, below 1 pp, but robust** (+0.63 and +0.65 pp, 7/7 seeds, Holm-detected). The 4096-atom head is the most accurate head at every level, but by less than the 1 pp tolerance. The comparison doubles the available atoms M, the active atoms k and the head's parameter count together, with no dense-M4096 or parameter-matched control, so the design cannot say which of them causes the gain (REVIEW-010 S-2). *Inference, not tested:* extra capacity is the simplest explanation; A2 gives no support for sparsity itself.
4. **Compression at level 80 (q20; 50.3:1 vs raw pixels; 89.8% smaller than the stored ImageNet files, a saving that includes stripped metadata; byte figures from GRID-001 and COORD-002)** costs 0.5–0.85 pp on average. Every cost is detected and its mean is below 1 pp. But this is not an equivalence result: the 95% CIs lie below 1 pp for the linear (0.90) and dense (0.96) heads, while for the sparse heads they reach 1.17 and 1.08.
5. **Compression at level 90 (q10, 86:1, 94.0% smaller)** costs 1.5–2.2 pp: a meaningful cost for every head.

All CIs here cover seed-to-seed variation on one fixed set of 10,000 test images (caveat 5).

## Checks (pre-registered)

| Check | Result |
| --- | --- |
| Runs | 84/84, each with a 20-epoch history |
| Split digest | one per seed, equal to SEED-001's for that seed (7/7) |
| Median selected epoch | sparse M2048 14, sparse M4096 14, dense 11, linear 15 |
| Runs selecting epoch ≥ 19 / ≤ 2 | 2/0, 3/0, 2/0, 3/0 of 21. No "budget binds" flag, but 2–3 runs per head peaked at the last epochs (REVIEW-010 O-8). |
| Instability (dictionary heads) | 0 of 63 runs (training loss never rises > 5% between epochs) |

## Exploratory (labeled; not confirmatory)

**Gain from the lr and budget change on ImageNetV2.** Against the earlier lr 1e-3 / 10-epoch runs on the same seeds and levels (SEED-001, MSWEEP-001), pooled: sparse M2048 +1.01, sparse M4096 +1.18, dense +1.33, linear **+1.95** pp (all 7/7 seeds). The gain mixes the lr change with the change from 10 to 20 epochs. The linear head gained the most, as LR-002's validation suggested (+2.14).

**Pooled head gaps, old vs new vs LR-002 validation (pp).** The validation column is not like-for-like: it pools source/L090 on validation seeds 42–44, while the test columns pool source/L080/L090 on ImageNetV2 (REVIEW-010 O-6). Validation overstated the test gaps (A1 +0.36 vs +0.02; A3 +1.04 vs +0.63).

| Gap | old lr 1e-3 (test, same seeds) | RETEST-001 (test) | LR-002 validation (equal 20 ep; different seeds and levels) |
| --- | --- | --- | --- |
| sparse M2048 − linear | +0.97 | +0.02 | +0.36 |
| sparse M2048 − dense M2048 | +0.41 | +0.09 | +0.00 |
| sparse M4096 − sparse M2048 | +0.46 | +0.63 | +1.04 |

**The dictionary heads' edge over linear shrinks toward L090** (exploratory; 6 tests, no correction). At level 90 the sparse heads lose more accuracy than linear:
- cost(sparse M2048) − cost(linear) at L090: +0.68 pp [+0.24, +1.12], 6/7;
- cost(sparse M4096) − cost(linear) at L090: +0.67 pp [+0.27, +1.07], 6/7;
- at L080 the same differences are +0.32 and +0.27, with CIs including 0;
- sparse vs dense costs do not differ (+0.16, +0.15; CIs include 0).

This replicates a pattern already present in the same-seed lr 1e-3 runs (sparse M2048 − linear +0.53 [+0.15, +0.91]; SEED-001 noted that the wide heads' advantage over linear narrows toward L90). Only sparse M4096 − linear is newly detected (+0.67, against +0.23 before; recomputed in REVIEW-010). It is consistent with the earlier claim that **no head adds compression tolerance**. It should not be read as the linear head being more tolerant as a property: a smaller absolute cost goes with linear's lower source accuracy (REVIEW-010 S-3).

**Per-level head gaps (secondary, descriptive):**
- sparse M2048 − linear: +0.35 at source, +0.03 at L080, −0.33 at L090.
- sparse M4096 − linear: +0.96 at source, +0.69 at L080, +0.29 at L090.
- dense − linear at L090: −0.37 [−0.63, −0.11], 1/7.

**Top-5 and training-level evaluation:** same ordering (sparse M4096 first, about +0.5–0.9 pp top-5 over the others at source). Scoring on ImageNetV2 compressed to the training level costs another 2.4–6.7 pp (exploratory; a different claim from training-storage).

## Disclosures and caveats

1. **ImageNetV2 is not untouched.** It was scored on these seeds at lr 1e-3 (SEED-001, MSWEEP-001) and many times across the project. The lrs were chosen on validation only (seeds 42–44), so the test was not used for selection here. But the project's choices of what to test were made with knowledge of earlier test results.
2. **Post-hoc budget for the dictionary heads.** LR-002's rule adopted 10 epochs for them (and, for sparse M2048, lr 3e-4); the 20-epoch, lr 1e-4 setting is the equal-budget proposal from REVIEW-009 S-1. So sparse M2048 departs from LR-002's adopted setting in lr as well as epochs (REVIEW-010 O-3). On validation it changed mean accuracy by +0.28 (sparse M2048), +0.02 (sparse M4096) and +0.11 pp (dense) relative to the adopted settings ([LR-002 analysis.md](../LR-002/analysis.md)), so it was never worse. (LR-002's report gives this range as "cost −0.02 to +0.30", a known slip.)
3. **Not re-tuned at the new lrs:** k = 512 / 1024, initialization, the MSE weight and the linear weight decay were all chosen at lr 1e-3.
4. **M, k and capacity confounded:** M4096/k1024 doubles the available atoms, the active atoms and the head parameters relative to M2048/k512, and there is no dense M4096 or parameter-matched control, so A3/A4 cannot separate them.
5. **What the CIs cover:** seed-to-seed variation (split, initialization, batch order) on one fixed set of 10,000 test images. All results are **conditional on these test images**. The image-sampling component was not quantified, because per-image predictions were not saved (REVIEW-010 S-4). By REVIEW-010's rough estimate (an inference, not measured) it could be about 0.2–0.3 pp for one model pair, the same order as the A3/A4 effects and the B half-widths. A2 is the most fragile result. A follow-up without training could re-score the saved `best.pt` files, keep per-image correctness and run a paired bootstrap over images.
6. **Setting:** frozen ImageNet-pretrained ResNet-18 features, with only a new head trained on compressed images. This supports a downstream-head training-storage claim, not learning from compressed data alone. Storage savings say nothing about GPU memory or speed.
7. **Best-epoch selection** on validation slightly favors flat curves (LR-002 caveat 5). Medians of 11–15 show no head was cut off by the budget, though 2–3 runs per head selected epoch 19 or 20.
8. **Reproducibility note** (REVIEW-010 O-9): `results.csv` has no lr or epochs column. Per-run lr and epochs are in each `done.json` (and `settings.json`), so later analyses should join on `done.json`.

## Consequences for earlier claims and the write-up

The lr 1e-3 measurements themselves remain correct as measured (REVIEW-010 recomputed A1 +0.97 and A2 +0.41 on these seeds). What changes is the head-architecture generalization drawn from them (REVIEW-010 O-4).

| Earlier claim (lr 1e-3) | Status after RETEST-001 |
| --- | --- |
| Sparse beats linear by about 1 pp (GRID-002, SEED-001) | **Superseded:** lr-conditional; not supported at tuned lrs (+0.02, no detected difference). |
| Top-k sparsity adds about 0.4 pp (SEED-001) | **Superseded:** lr-conditional; at tuned lrs +0.09 pp, below 1 pp and not robust to multiplicity. |
| A bigger dictionary helps (MSWEEP-001, +0.46) | **Holds, small:** M4096/k1024 beats M2048/k512 by +0.63 pp (robust, below 1 pp). M, k and parameter count are not separated. |
| No head reduces the compression drop | **Still holds** (exploratory): at L090 the dictionary heads lose 0.5–0.7 pp more than linear, as in the lr 1e-3 runs. |
| Level 80 costs about 0.5–1.2 pp; level 90 about 1.5–2.3 pp | **Confirmed at tuned lrs:** mean 0.5–0.85 pp at L080 (CIs below 1 pp for linear and dense, up to 1.08–1.17 pp for the sparse heads); 1.5–2.2 pp at L090 (meaningful). |

The **storage finding** stands and is now the main result. Training on JPEG q20 copies (50.3:1 vs raw pixels; 89.8% fewer bytes than the stored ImageNet files, including stripped metadata) costs a **mean of 0.5–0.85 pp** of ImageNetV2 accuracy. The 95% CIs lie below 1 pp for the linear and dense heads but reach 1.08–1.17 pp for the sparse heads. q10 (86:1) costs 1.5–2.2 pp. All of this is conditional on the fixed ImageNetV2 test set and frozen pretrained ResNet-18 features.

The **sparse-dictionary head** does not beat a properly tuned linear head. The only robust head difference is the M4096/k1024 configuration's +0.6 pp, below the tolerance, and the design cannot attribute it to M, k or parameter count.

## Checks run

- `analyze.py` ran without assertion failures: 84 runs, 20-epoch histories, head/level/seed labels in `done.json` match the paths, 21 rows per head and evaluation, no duplicate test rows, 12 tests in the family, 21 earlier lr 1e-3 cells per head for the exploratory comparison.
- `results.csv` has 140 data rows (28 runs at source × 1 evaluation + 56 compressed runs × 2 evaluations).
- `last.pt` is deleted in every run directory; `best.pt` and `done.json` are kept (3.7 GB, not committed).
- No user `src/` file or earlier artifact was changed.
- Revision 1: `analyze.py` now asserts that `results.csv` has exactly 140 rows with no duplicate (run_dir, eval_level), source runs included. It passes, and both outputs are byte-identical to the original run. REVIEW-010 independently found the same 140 rows and no duplicates.

## Revision log

- **Revision 1 (2026-11-26), for [REVIEW-010](../REVIEW-010/report.md)** (verdict: revise, wording only; every number reproduced):
  - **S-1:** the storage headline now gives the mean cost and the CI qualifier (sparse heads' CIs reach 1.08–1.17 pp).
  - **S-2:** A3/A4 no longer say "capacity effect" or "larger dictionary". M, k and parameter count are named as confounded; capacity is labeled an inference.
  - **S-3:** "reverses the earlier impression" is replaced. The L090 cost difference replicates a SEED-001 pattern, is consistent with "no head adds compression tolerance" and is framed as the dictionary heads' edge shrinking.
  - **S-4:** caveat 5 now says the results are conditional on the 10,000 test images and the image-sampling component was not quantified, with the reviewer's rough estimate and a bootstrap follow-up.
  - **O-1:** duplicate-row assertion added (outputs unchanged). **O-3:** sparse M2048's lr departure stated. **O-4:** "withdrawn" → "superseded: lr-conditional". **O-5:** gain attributed to the lr and budget change. **O-6:** the validation column is labeled not like-for-like. **O-7:** metadata qualifier on 89.8%. **O-8:** runs at epoch 19–20 mentioned. **O-9:** noted in caveat 8.
  - **O-2:** no action; the disclosure stays.
  - **Process slip (coordinator):** I edited `analyze.py` and `report.md` before the reviewer had recorded its `revise` verdict. The harness then refused that verdict ("Artifact changed since submission"). RETEST-001 was therefore moved back with `recover` (the reason is in the ledger), reclaimed and resubmitted. The `revise` verdict is documented in REVIEW-010's report and in the recover note, not as a `review-revise` ledger event.
