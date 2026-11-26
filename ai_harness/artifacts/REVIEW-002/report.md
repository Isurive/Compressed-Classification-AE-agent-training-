# REVIEW-002: independent review of GRID-001, GRID-002 and ABL-001

Reviewer `reviewer-20261125-a`, who did not author the reviewed work. Date 2026-11-25. All inputs were read-only. There was no training; the only computation was CPU inference for the re-scores. The scripts and their outputs are in `recompute/`.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, the author of the reviewed work) saved it here verbatim, with no edits to content, on 2026-11-25.

## Verdicts
- **GRID-001: revise.** Numbers, bytes, test isolation and pairing are all verified. The M-sweep interpretation is unsupported (G1-1).
- **GRID-002: revise.** "Dictionary head beats linear, established" overstates the 1 pp rule (G2-1).
- **ABL-001: revise.** The headline attribution to the MSE loss does not follow (A-1 and A-2).

No finding is blocking. Each revise needs only a correction to wording or disclosure. Severity scale:
- **Blocking:** invalidates results.
- **Should-fix:** an unsupported claim that must be corrected before acceptance.
- **Optional:** an improvement.

## 1. Recomputation (all matched)
- **`analyze.py` copies:** run from `recompute/<TASK>/` with ROOT set to `parents[5]`. Their `analysis.md` and `analysis.json` are byte-identical to the submitted files for all three tasks.
- **Accuracy tables (`independent_check.py`, which does not use the authors' scripts):** every cell matches to 0.01 pp in GRID-001 §3 and the §5 means, GRID-002 §3 and §4, and ABL-001 §2 and §3.
  - GRID-002 sparse minus linear: source 1.19, L20 1.57, L60 0.88, L80 0.99, L85 0.90, L90 0.47, L95 0.39.
  - ABL-001 MSE effect: +1.99 (sparse) and +1.39 (dense). MLP minus linear: −1.00.
- **`done.json` against `results.csv`:** 198 + 135 + 108 rows, 0 mismatches.
- **Checkpoints and test scores:**
  - Six `best.pt` files match the csv on epoch, validation top-1, digest, level, M, k and tensor shapes.
  - The `last.pt` settings of GRID-001 sparse L80 seed 43 are k 512, init `auto`, loss weights 1/0.1/0.05.
  - Two linear heads re-scored on the ImageNetV2 features give 0.5235 and 0.4680, equal to the csv.
- **Freeze rules:** recomputed from `tune_results.json`, they reproduce both `tuned.json` files.
- **Bytes, streamed from `manifest.csv` (350k rows):** every level and seed matches GRID-001 §4. Seed 42 at q20 is 615,578,103 B: R_raw 50.3, R_existing 9.81, 89.8% saving, 2.3% header. Training plus validation at q20 gives 683,922,845 B, equal to the whole pool.

## 2. Integrity (all passed)
- **Test-set isolation:**
  - `cmd_tune` (`run_grid.py`, lines 271–310) only calls `train_one` on the pool features. `tuned.json` was written at 17:54:38, before the grid started at 17:54:49.
  - `cmd_freeze` (`run_grid_xavier.py`, lines 33–45) reads only `tune_results.json`. Its file was written at 21:46:08, before the grid started at 21:46:13.
  - The ABL-001 variants were in `task.json` at 05:34:13Z. The run started at 05:34:45Z.
- **Splits:**
  - Each seed has one digest across all three files. Rebuilding the split with `training_splits` gives the same digest.
  - The splits are 45k training and 5k validation (5 per class) from the ImageNet-val pool, with no overlap between them.
  - ImageNetV2 is the test set only. It has 0 files whose sha256 also appears in the pool, and its feature digest is distinct.
- **Loss weights, from the per-epoch logs** (loss − ce − w·mse, which should equal the remaining penalty terms):

  | Variant | Epochs | Residual |
  | --- | --- | --- |
  | abl-mse | 300 | ±2e-8 |
  | abl-none | 300 | ±8e-9 |
  | GRID-002 full loss | 480 | 0.069–0.081 |

- **Hashes and source files:** artifact hashes match for 22/22, 12/12 and 11/11 artifacts, and `check` reports ok. All 8 user `src/` files are unchanged. There is no `__pycache__`.

## 3. GRID-001
**G1-1, should-fix (headline 5, line 12; §5, lines 114–125). The M-sweep result is confounded by early checkpoint selection.**
- **Evidence:**
  - M=1024 picked epoch 1 in 9/9 runs at L80, L90 and L95. Its source runs picked epochs 5, 1 and 6.
  - M=512 picked epoch 1 in all 3 source runs, and epochs 1–3 at the compressed levels.
  - M=4096 picked epochs 4–10.
  - Validation curves:
    - M1024 L80 seed 42 goes from 0.6242 at epoch 1 down to 0.6040.
    - M512 source seed 42 goes from 0.6422 down to 0.6136.
    - M4096 L80 seed 42 rises to 0.6514.
- **Implication:**
  - With the `auto` init, these checkpoints are close to the original ImageNet classifier after one epoch. So "accuracy improves with M" is confounded.
  - The "noisy, low baseline" explanation misses the cause, although "not evidence of robustness" still holds.
  - k=512 is 100% of the atoms at M=512 and 50% at M=1024.
- **Repair:**
  - Add best epochs per M.
  - Mark the M effect as unresolved under `auto`.
  - Optionally propose an M sweep with Xavier init and a k that is sparse at every M.

**G1-2, optional (§4, line 107). R_raw is not strictly RGB.** R_raw counts 899 grayscale images as 1 channel. Counting every image as RGB gives 50.8 instead of 50.3, so the reported figure is conservative. Fix the definition.

**G1-3, optional. Metadata stripping.** R_existing and the saving include stripping metadata from the source files. Say so.

**G1-4, optional (§2, line 32; §8, line 146).** `history.json` exists only for linear runs.

**G1-5, optional. Duplicate images in the pool.**
- 29 groups of byte-identical pool images carry conflicting labels.
- 3, 3 and 7 of these groups straddle training and validation for seeds 42, 43 and 44.
- This affects only validation selection.

**Supported:**
- §3 and the tolerance table.
- The scope and inherited-classifier disclosures, and the note that k=512 is weak sparsity.
- The tuning budget disclosure (6/2/6).
- The "98%" discussion. Relative retention at L80 is sparse 98.46%, dense 97.85% and linear 97.76%, with appropriate caveats.
- The pixel check (3500/3500).
- The matched-level test is labeled separately.

## 4. GRID-002
**G2-1, should-fix (line 8; line 110). The 1 pp claim is overstated.**
- **Evidence:**
  - "Benefit … over linear is [established]" and "the only difference that clears [1 pp]" conflict with the measured gaps. The gaps are at least 1 pp only at source and L20; elsewhere they are 0.88, 0.99, 0.90, 0.47 and 0.39.
  - ABL-001's mean over 5 levels is 0.78.
  - The 21/21 positive pairs come from 3 splits, with per-seed means of 0.76, 0.80 and 0.79.
- **Repair:** say the sign is consistent but the gap reaches 1 pp only at source and L20.

**G2-2, optional (line 91).** Label "the reversal is explained by inheritance" as an inference; it was not tested.

**G2-3, optional (line 119).** `history.json` is kept only for the 24 linear runs. For the 48 dictionary runs, history and loss weights exist only in `grid_log.txt`, and `best.pt` holds no settings.

**G2-4, optional.** The linear head's learning rate and epochs were not tuned. 9 of 24 linear runs peaked at epoch 9 or 10, but the curves are flat (±0.5 pp). Add this to the budget caveat (3/1/3).

**Supported:**
- Validation-only freeze.
- Splits match GRID-001.
- The §3 and §4 numbers.
- Sparse vs dense is correctly called not established.
- No added compression tolerance.
- Scope (the MSE target is the same compressed image's features).

## 5. ABL-001
**A-1, should-fix (line 14; §6.1, line 88). The attribution does not follow.**
- **Evidence:**
  - Within the wide heads, MSE adds +1.99 (sparse) and +1.39 (dense). This is clear.
  - Against linear:
    - The MLP without MSE is −1.00.
    - dense-mse minus linear is +0.39 (12/15; per-seed 0.12, 0.44, 0.60).
    - sparse-mse minus dense-mse is +0.46 (13/15).
    - sparse-full minus linear is +0.78.
  - "0.8–1.6 pp gain" is GRID-002's range for source–L85. On ABL-001's own levels the gain is 0.39–1.19.
- **Implication:** MSE is necessary to offset the width penalty. The residual advantage over linear is below 1 pp and partly comes from sparsity together with MSE.
- **Repair:** restate the answer and §6.1 accordingly.

**A-2, should-fix (§4, lines 71–76). Alternative explanations are missing.** Add:
- generic regularization: the MLP had no weight decay or dropout, while linear's weight decay was tuned;
- code-scale anchoring by the unit-norm atoms;
- training dynamics: CE-only runs peak at epochs 4–6, MSE runs at 6–9, with an untuned learning rate and 10 epochs.

Scope the claim to "versus an unregularized CE-only head of the same width".

**A-3, optional (line 31). The 1e-6 term is misquantified.**
- In `abl-none`, training feature MSE is 1.04–3.12, so the weighted term is about 1e-6 to 3e-6, not 2e-7. CE is 0.019–3.5.
- The decoder barely trains. Adam still gives it updates of normal size, but these do not affect predictions.
- So "CE-only in effect" is correct; only the number needs fixing.

**A-4, optional.** The "15/15" counts cover 3 splits. Report per-seed means: MSE effect 1.98, 2.09 and 1.90 (sparse) and 1.42, 1.36 and 1.38 (dense).

**A-5, optional (line 84).** `runs/ABL-001` has no `history.json`. The loss weights are not in `results.csv`; they are in `done.json`, `settings.json` and the log.

**Supported:**
- The variants match the task spec.
- The §2 and §3 numbers.
- The dependency and coherence terms have about zero effect.
- Sparsity alone has no detectable effect.
- There is no gain in tolerance to compression.

## 6. Cross-cutting (optional)
ImageNetV2 was reused adaptively: GRID-002 was designed from GRID-001's test results, and ABL-001 from GRID-002's. No settings were tuned on it. Before final claims, confirm the remaining effects under 1 pp with more seeds.

## 7. Not checked
- Training was not re-run.
- Stored variants were not decoded beyond the authors' pixel check.
- Tune-run checkpoints were not inspected.
- The metadata bytes in the source files were not quantified.

## Addendum (2026-11-25, after REVIEW-005)

**R-1: correction to G2-1.**
- G2-1 gave per-seed means of 0.76, 0.80 and 0.79. Those numbers are sparse-full − linear averaged over ABL-001's **5 levels** (source, 60, 80, 90, 95).
- The 21 positive pairs cited in G2-1 come from GRID-002's **7 levels** (source through 95). Over those 7 levels the per-seed means are **0.95, 0.90 and 0.89 (mean 0.91)**.
- I verified this from `runs/GRID-002/results.csv`, ImageNetV2-source rows: 21/21 pairs positive over 7 levels, and 15/15 over 5 levels.
- The conclusion of G2-1 is unchanged. Both sets of means are below 1 pp, and the per-level gap reaches 1 pp only at source (1.19) and level 20 (1.57).
- The revised GRID-002 report (headline 1) repeats the 5-level figures beside the 21-pair count. That is a citation-level slip I introduced. It does not change the accepted verdict.

**R-2: outcome of the re-review.** All three reports were revised (only `report.md` changed, and the hashes were re-verified). On re-review I recorded **accept** for each:
- **GRID-001:** G1-1 resolved. The M effect is marked unresolved, and the best-epoch table matches `results.csv`. G1-2 to G1-5 handled.
- **GRID-002:** G2-1 to G2-4 resolved or handled.
  - Nit: §7 says the reviewer "confirmed" the loss weights 1/0.1/0.05. I confirmed only that the regularizer residual is nonzero (0.069–0.081). The exact values follow from the `Settings` defaults in the code.
- **ABL-001:** A-1 to A-5 resolved or handled.
  - "MSE is necessary / needed to beat linear" holds only among the variants tested; no tuned weight decay was tried.
  - The epochs 4–6 vs 6–9 contrast holds for the sparse heads. The dense CE-only head peaks at epochs 5–9.
  - "Does nothing" would read better as "no detectable effect".
- **Warning:** the ledger *summary* string for ABL-001 still carries the withdrawn claim ("the feature-MSE auxiliary loss explains the sparse head's gain"). It should be corrected, or kept out of BOARD and synthesis.

R-3 (remaining overstatements) is left to the coordinator's erratum.
