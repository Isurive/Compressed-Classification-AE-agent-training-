# SEED-001: more seeds for sparse vs dense vs linear

**Agent:** `claude-orient-20261122-a`. **Run:** 2026-11-25 23:37 to 2026-11-26 00:05 (28 min, 63 runs, no errors).
**Approved by the user:** "ok can we get started on that then?", after the explanation of what more seeds means.
**Revision 1** (2026-11-26): addresses [REVIEW-006](../REVIEW-006/report.md) S-1 to S-3 and O-1 to O-3. The log is at the end.

**Pre-registration.** The design and the decision rule were written in [task.json](task.json) and added to the ledger before the first run: ledger add at 06:37:09Z, first run at 06:37:32Z (verified by REVIEW-006). [analyze.py](analyze.py) → [analysis.md](analysis.md) / [analysis.json](analysis.json) implements that rule. **It also prints one comparison that was not pre-registered (dense − linear)**, which is reported below as exploratory.

## Question

ABL-001 found that top-k sparsity, combined with the feature-MSE loss, gave +0.46 pp over the dense head. GRID-002 found sparse ahead of linear by under 1 pp at most levels. With 3 seeds, neither could be separated from noise. Are these gaps real?

## Design (as pre-registered)

| Item | Setting |
| --- | --- |
| Heads | **sparse**: M 2048, k 512. **dense**: M 2048, k 2048. Both Xavier fc, He encoder, loss CE + feature MSE (the ABL-001 `abl-mse` weights 1/0/0). **linear**: Xavier, weight decay 1e-4 (GRID-002). |
| Training | Same as GRID-002: 10 epochs, batch 128, Adam 1e-3, `best.pt` by validation top-1 then CE, the cached GRID-001 features (read only). |
| Levels | source, 80 (q20, 50.3:1 against raw), 90 (q10) |
| Seeds | **new seeds 45–51** (7 seeds). Each sets the split, the initialization and the batch order. |
| Test | Original ImageNetV2 (10k images), once per run, after training |
| Pre-registered comparisons | **sparse − dense** and **sparse − linear**, per level and pooled (the per-seed mean over the 3 levels). That is 8 tests. No other comparison is confirmatory. |
| Statistics | Paired differences; mean, sd, two-sided paired-t 95% CI (df 6), sign count |
| Decision rule | **Detected:** the CI excludes 0. **Meaningful:** detected and mean ≥ 1.0 pp (the project tolerance). |
| Secondary analysis | All 10 seeds (42–51), adding the ABL-001 `abl-mse` and GRID-002 linear runs. REVIEW-006 verified their settings are identical. |

## Primary result (pre-registered; new seeds 45–51; ImageNetV2-source top-1; pp)

| Comparison | source | L80 | L90 | **Pooled** | Verdict |
| --- | --- | --- | --- | --- | --- |
| sparse − dense | 0.53 [0.23, 0.82], 6/7 | 0.34 [0.06, 0.62], 6/7 | 0.37 [0.14, 0.59], 7/7 | **0.41 [0.29, 0.53], 7/7** | **detected, not meaningful** |
| sparse − linear | 1.23 [0.92, 1.54], 7/7 | 0.97 [0.43, 1.52], 7/7 | 0.70 [0.37, 1.03], 7/7 | **0.97 [0.78, 1.15], 7/7** | detected at every level; meaningful at source only (borderline: CI lower bound 0.92) |

Each cell shows the mean paired difference, the 95% CI, and the number of seeds where the difference is positive.

**Multiple comparisons.** All 8 pre-registered tests stay detected under a Holm correction; the largest p is 0.024 (sparse − dense, L80), from REVIEW-006. The pooled sparse − dense result has t = 8.25 with df 6.

**Mean accuracies (new seeds):**

| Head | source | L80 | L90 |
| --- | --- | --- | --- |
| sparse | 48.31 | 47.39 | 46.13 |
| dense | 47.79 | 47.05 | 45.76 |
| linear | 47.08 | 46.42 | 45.43 |

**Secondary (all 10 seeds, pre-registered):**

| Comparison | Pooled [95% CI] | Seeds positive |
| --- | --- | --- |
| sparse − dense | 0.39 [0.29, 0.50] | 10/10 |
| sparse − linear | 0.93 [0.81, 1.06] | 10/10 |

Full tables: [analysis.md](analysis.md).

## Exploratory (not pre-registered; do not treat as confirmatory)

| Analysis | Result | Source |
| --- | --- | --- |
| dense − linear | 0.70 / 0.63 / 0.33 pp at source / L80 / L90; pooled 0.55 [0.34, 0.77], 7/7. Under a Holm correction over all 12 printed tests, L90 only just passes (p = 0.0495). | `analysis.md` |
| sparse − dense gap, source → L90 | +0.16 pp [−0.18, 0.50]: **no change detected** | REVIEW-006 |
| sparse − linear gap, source → L90 | narrows by 0.53 pp [0.15, 0.91] | REVIEW-006 |
| dense − linear gap, source → L90 | narrows by 0.37 pp [0.08, 0.66] | REVIEW-006 |

## What this settles

1. **Mild top-k sparsity (k = 512 of 2048) with the reconstruction loss adds a real but small benefit, about 0.4 pp over the dense head.**
   - Positive in every seed's pooled mean (7/7).
   - Positive at every level on average. Per-level sign counts: 6/7, 6/7, 7/7.
   - Below the 1 pp project tolerance, so by the pre-declared rule it is *detected but not meaningful*.
   - This resolves the ABL-001 "+0.46, unresolved" item: the effect exists, and it is small.
   - Scope: stronger sparsity was worse in validation tuning (Xavier k = 32: 53.3%; k = 128: 58.0%; k = 512: 60.0%). k was chosen from 3 candidates while dense had none, a small budget asymmetry.
2. **The sparse head beats the linear head by about 1 pp on average.** It reaches the 1 pp tolerance only on uncompressed training images (1.23 pp; CI lower bound 0.92, so borderline). At levels 80 and 90 the gap is 0.7–1.0 pp.
3. **No evidence that sparsity adds compression tolerance.** No change in the sparse − dense gap between source and L90 was detected, consistent with a roughly constant ~0.4 pp benefit. The advantage of the wide heads over linear *does* narrow toward L90, but only in exploratory analysis.

## Caveats

- **Loss terms:** the sparse/dense heads here use MSE only, not the dependency/coherence terms. ABL-001 found those terms have no measurable effect.
- **Test reuse:** same test set as before. The seeds are new, but ImageNetV2 has informed the design of earlier follow-up experiments. The new-seed primary analysis limits this, but does not remove it.
- **Claim scope as before:** frozen pretrained backbone; only the head is trained on compressed images; Xavier-initialized head.

## Checks

- **Runs:** 63/63 `done.json` written; `results.csv` has 105 rows (21 source runs × 1 evaluation + 42 compressed runs × 2); no tracebacks. Logs: [grid_log.txt](grid_log.txt), [summary_log.txt](summary_log.txt).
- **Splits:** one split digest per seed across all three heads; the splits are distinct across seeds 42–51.
- **Integrity** (acceptance criterion 4):
  - the 8 user `src/` files are unchanged against `GRID-001/src_hashes_before.txt`;
  - there is no `src/__pycache__`;
  - every earlier registered artifact matches the ledger (the two COORD-001 files are superseded by COORD-003);
  - `harness.py check` is ok.
- **Code:** [src/run_seeds.py](../../../src/run_seeds.py) is new. It reuses `run_grid.py` and `run_ablation.py` unchanged.
- **Disk:** `last.pt` was deleted for these runs.

## Revision log

- **Revision 1** (2026-11-26, after REVIEW-006):
  - S-1: per-level sign counts added; the "every seed and level" wording corrected.
  - S-2: dense − linear moved to an exploratory section; the pre-registered family is stated as 8 tests, with the Holm result.
  - S-3: the compression-trend claim relabeled exploratory; no change detected for sparse − dense.
  - O-1: sparsity claim scoped to k = 512 of 2048.
  - O-2: integrity line added.
  - O-3: t = 8.25.
  - No numbers changed, and nothing was re-run.
