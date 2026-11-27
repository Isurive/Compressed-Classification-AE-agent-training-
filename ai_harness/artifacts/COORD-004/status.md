# Accepted experiment results, as of 2026-11-27 (successor to COORD-002/status.md)

**Coordinator:** `claude-orient-20261122-a`. **Requested by the user:** "Yes, refresh status" (2026-11-26). **Revision 2 (2026-11-27, task COORD-005, user: "Yes, update them")** adds the weight-decay check WD-001: section B5, the ABL-001 and compression-tolerance rows of C, D, E2b, E3 and F. Revision 3 (COORD-006) is wording only.
**Replaces:** [COORD-002/status.md](../COORD-002/status.md) (2026-11-25) as the authoritative one-page list. That file is hash-locked and stays unchanged as history; its errata A–C are carried forward below.
**Source of truth:** the accepted `report.md` of each task cited. Where this page and a report differ, the report wins, **except** for the errata in section E, which apply to the reports.

## A. Setting (all results)

- **Model:** frozen ImageNet-pretrained ResNet-18; only a new head is trained on 512-d pooled features (cached once per compression level, GRID-001).
- **Training data:** 45k images of the 50k ImageNet-1K validation pool, all JPEG-compressed at one level; 5k matched-compression validation images pick `best.pt` (top-1, then CE). The seed sets the split, init and batch order.
- **Test:** original ImageNetV2 matched-frequency (10,000 images), scored once per run after training; never used for tuning or checkpoint choice.
- **Heads:** sparse dictionary (M atoms, top-k active, CE + feature MSE), dense (k = M), linear.
- **Tolerance:** only differences ≥ 1.0 pp count as meaningful (working tolerance from the user-approved REASON-002 plan; not finalized).
- **Scope:** a downstream-head training-storage claim, not learning from compressed data alone.

## B. Headline results (use these in the write-up)

### B1. Storage (GRID-001 §4; COORD-002 item 1, unchanged)

The full-file bytes of the 45k training set at level 80 (JPEG quality 20), headers counted:
- 0.47 bpp;
- **50.3:1 against uncompressed pixels**, at each image's actual channel count (50.8 if all were RGB);
- **9.8:1 against the stored ImageNet files**, an 89.8% saving that includes stripped metadata.

The stored files decode to exactly the training pixels (3,500 of 3,500 checked). Level 90 (q10) is 86:1 against raw pixels (94.0% smaller than the stored files).

### B2. Accuracy cost of compressed training data

At tuned learning rates ([RETEST-001](../RETEST-001/report.md), pre-registered, seeds 45–51, accepted after [REVIEW-010](../REVIEW-010/report.md)). Paired drop against the same head trained on original images, ImageNetV2-source top-1:

| Head | Level 80 cost (95% CI) | Level 90 cost (95% CI) |
| --- | --- | --- |
| sparse M2048 k512 | 0.85 [0.52, 1.17] | 2.19 [1.91, 2.46] |
| sparse M4096 k1024 | 0.79 [0.50, 1.08] | 2.18 [1.81, 2.54] |
| dense M2048 | 0.69 [0.41, 0.96] | 2.04 [1.60, 2.48] |
| linear | 0.53 [0.16, 0.89] | 1.51 [1.06, 1.95] |

- **Level 80:** every cost is detected and its mean is under 1 pp. This is not an equivalence result: the CIs stay under 1 pp for linear and dense but reach 1.08–1.17 pp for the sparse heads.
- **Level 90:** a meaningful (≥ 1 pp) cost for every head.
- **Levels 95 and 100** were measured only at lr 1e-3 (GRID-001, GRID-002, ABL-001): 2.7–5.0 pp and 7.7–18.7 pp. Not re-tested at tuned rates.
- **The 98%-at-50:1 target:** relative to each head's own uncompressed accuracy, level 80 retains 98.3–98.9% (RETEST-001 means). In absolute terms the heads reach about 48.4–49.1% at level 80 (48.9–49.9% on uncompressed data) against 57.28% for the untouched original classifier, so the target is not met in absolute terms.

### B3. Head comparisons at tuned learning rates (RETEST-001; supersedes all lr-1e-3 head comparisons)

Every head at a validation-chosen lr (dictionary heads 1e-4, linear 3e-4) and 20 epochs; for sparse M2048 this is a disclosed post-hoc departure from LR-002's adopted 3e-4 at 10 epochs, never worse on validation (+0.28 pp; RETEST-001 caveat 2); ImageNetV2-source top-1, pooled over levels source/80/90, paired by seed (n = 7):

| Comparison | Mean (pp) | 95% CI | Seeds positive | Holm-adjusted | Verdict |
| --- | --- | --- | --- | --- | --- |
| sparse M2048 − linear | +0.02 | [−0.25, +0.29] | 4/7 | not detected | **no detected difference** |
| sparse M2048 − dense M2048 | +0.09 | [+0.002, +0.17] | 5/7 | not detected (p 0.094) | small, below 1 pp, not robust to multiplicity |
| sparse M4096 − sparse M2048 | +0.63 | [+0.47, +0.80] | 7/7 | detected | robust, below 1 pp |
| sparse M4096 − linear | +0.65 | [+0.45, +0.85] | 7/7 | detected | robust, below 1 pp |

- **Mean accuracy (source / L080 / L090, %):** sparse M2048 49.29 / 48.45 / 47.10; sparse M4096 49.90 / 49.11 / 47.72; dense 49.10 / 48.42 / 47.06; linear 48.94 / 48.42 / 47.43.
- **The sparse-dictionary head (M2048/k512) is no better than a tuned linear head** (no detected difference; not an equivalence test), and top-k sparsity has no practically relevant effect at equal width (k = 512 of M = 2048, with MSE).
- **The M4096/k1024 head is the most accurate**, but below the 1 pp tolerance, and M, k and the parameter count double together (no dense-M4096 or parameter-matched control), so the cause is not identified. Capacity is the simplest explanation (inference).
- **Compression tolerance (exploratory):** no head reduces the compression cost. At level 90 the dictionary heads lose 0.5–0.7 pp more than linear (uncorrected), as already seen at lr 1e-3 (SEED-001 §3 / REVIEW-006), so their small edge over linear shrinks toward level 90.

### B4. Learning rate ([LR-001](../LR-001/report.md), [LR-002](../LR-002/report.md), validation only; confirmed on test by RETEST-001)

- Adam lr 1e-3, used by GRID-001 through MSWEEP-001, was poor for every Xavier-initialized head tested (sparse M2048, sparse M4096, dense M2048, linear); GRID-001's inherited-fc heads were never lr-checked. The dictionary heads diverge late in training at 1e-3 (unstable in 4/6–6/6 cells; 0 at the lower rates).
- The linear head was under-tuned by about 2 pp: 3e-4 for 20 epochs beat 1e-3 for 10 by +2.14 pp on validation.
- On ImageNetV2 the lr-and-budget change raised every head: sparse M2048 +1.01, sparse M4096 +1.18, dense +1.33, linear **+1.95** pp (RETEST-001, exploratory, same seeds).

### B5. Can tuned weight decay replace the reconstruction loss? ([WD-001](../WD-001/report.md); added in revision 2)

Pre-registered; phase-A selection on validation only, locked in WD-001S before any ImageNetV2 run; accepted after [REVIEW-013](../REVIEW-013/report.md) (selection lock) and [REVIEW-014](../REVIEW-014/report.md) (results), both cross-reviewed.

- **Design.** The two 2048-wide heads trained with cross-entropy only (reconstruction weight 1e-6, effectively off, as in ABL-001's CE-only setting), with coupled L2 weight decay through Adam on the encoder and classifier weights and biases (not the centre or decoder atoms):
  - dense-CE (M2048, k2048) and sparse-CE (M2048, k512);
  - each chose lr, weight decay and budget from 3 lr × 4 wd × {10, 20} epochs on validation (seeds 42–44, source and L090): dense-CE lr 1e-4, wd 1e-3, 20 ep; sparse-CE lr 3e-5, wd 1e-3, 20 ep;
  - tested on ImageNetV2 (seeds 45–51; source, L080, L090) against RETEST-001's runs of the MSE-trained heads and the linear head on the same seeds, levels and splits.
- **Result** (ImageNetV2-source top-1, pooled over levels as the per-seed mean of the per-level differences, paired by seed, n = 7, Holm over 4 tests):

| Test | Mean (pp) | 95% CI | Seeds positive | Holm p | Verdict |
| --- | --- | --- | --- | --- | --- |
| W1 dense-CE+wd − dense-MSE | −0.29 | [−0.41, −0.17] | 0/7 | 0.0023 | detected; below 1.0 and 0.5 pp |
| W2 sparse-CE+wd − sparse-MSE | −1.36 | [−1.52, −1.20] | 0/7 | < 0.0001 | **meaningful** (1.0 and 0.5 pp) |
| W3 dense-CE+wd − linear | −0.35 | [−0.57, −0.14] | 0/7 | 0.0063 | detected; below 1.0 and 0.5 pp |
| W4 sparse-CE+wd − linear | −1.34 | [−1.63, −1.05] | 0/7 | < 0.0001 | **meaningful** (1.0 and 0.5 pp) |

- **Mean accuracy (source / L080 / L090, %):** dense-CE+wd 48.90 / 48.17 / 46.66; sparse-CE+wd 47.95 / 47.10 / 45.70 (compare B3).
- **Dense head:** tuned weight decay comes close to the reconstruction loss but not all the way: a detected gap below both tolerances. This outcome was **not one of the two pre-declared readings** ("not detected or positive"; "negative and meaningful"), so "close, but not all the way" is a post-hoc description.
- **Sparse head:** within the pre-registered grid and 20-epoch budget, the reconstruction loss gives a meaningful benefit that tuned weight decay does not (the pre-declared reading). The gap is about the same at every level (−1.34, −1.34, −1.40 pp; descriptive, not an equivalence test).
- **Neither weight-decay head beats linear.** Per level, sparse-CE+wd − linear is −0.99 / −1.31 / −1.73 and dense-CE+wd − linear −0.04 / −0.25 / −0.78 pp (descriptive, uncorrected; dense CIs include 0 at source and L080).
- **Caveats:**
  - **sparse-CE may be under-trained:** it was selected at the grid's smallest lr with the 20-epoch budget binding, and 16/21 test runs picked epoch 19 or 20 (dense-CE: median 11, 0/21). A longer budget or an lr between 3e-5 and 1e-4 might narrow W2 and W4; that would be a new, post-hoc experiment.
  - **dense-CE's setting came from an exact validation tie** (lr 1e-4 vs 3e-5, 18100/30000 each), broken by the pre-registered lower-CE rule.
  - **Tuning effort favours the CE heads** (24 candidates each, against an lr-only check for the MSE heads and a linear weight decay set at lr 1e-3), so it does not explain why they fall short.
  - **Form and scope:** coupled L2, not AdamW; M = 2048 only; one frozen backbone.
- **Exploratory (uncorrected):** compression costs of the CE heads are 0.73 / 2.24 pp (dense, L080 / L090) and 0.85 / 2.25 pp (sparse). Against the MSE heads the cost differences are within ±0.21 pp with every CI including 0: **no detected change** (not an equivalence result; e.g. dense at L090 [−0.51, +0.92]). Against linear both CE heads lose +0.74 pp more at L090 (sparse CI [+0.33, +1.16]; dense [−0.01, +1.48]). On top-5, dense-CE+wd is 0.05–0.16 pp below dense-MSE and sparse-CE+wd 0.72–0.84 pp below sparse-MSE; on top-1 at the training level the gaps are 0.12–0.30 pp (dense) and 1.47–1.70 pp (sparse).

## C. Earlier results: status after RETEST-001

The lr-1e-3 measurements below remain correct **as measured**. What changes is the head-architecture generalization drawn from them.

| Result (task) | Status |
| --- | --- |
| Inherited classifier: linear head best when heads start from the original ImageNet fc (GRID-001) | Holds as measured at lr 1e-3; not re-tested at tuned rates. |
| Xavier init: sparse ahead of linear at every level from source to 95, ≥ 1 pp only at source and 20 (GRID-002) | **Superseded:** lr-conditional; +0.02 pp at tuned rates (RETEST-001). |
| Inherited fc worth 3.3–6.8 pp up to level 90 (GRID-002) | Holds as measured at lr 1e-3; not re-tested. lr-conditional on the Xavier side: probably smaller at tuned rates, where the Xavier heads gained 1.0–2.0 pp (inference; different seeds, inherited-fc heads not re-tuned). |
| A 2048-wide head with CE only is 1.0 pp worse than linear; the feature-MSE loss adds +1.4 (dense) / +2.0 (sparse) pp; dependency/coherence regularizers ≈ 0 (ABL-001) | Holds as measured at lr 1e-3, but the comparison with linear used the under-tuned linear head. "MSE is what lets a wide head beat linear" is **superseded**: at tuned rates the MSE-trained sparse M2048 head shows no detected difference from linear (+0.02 pp; RETEST-001). **Revision 2:** the CE-only wide heads were re-tested at tuned rates with tuned weight decay (WD-001, B5). With weight decay, dense CE-only is 0.29 pp below its MSE version and 0.35 pp below linear (both detected, below tolerance); sparse CE-only stays 1.36 pp below its MSE version and 1.34 pp below linear (meaningful, within the grid and 20-epoch budget). So for the sparse head the MSE loss still matters; for neither head does it make the wide head beat linear. |
| Sparse − dense +0.41 pp [0.29, 0.53], sparse − linear +0.97 pp [0.78, 1.15] (SEED-001, pre-registered, seeds 45–51) | **Superseded:** lr-conditional; +0.09 and +0.02 pp on the same seeds at tuned rates (RETEST-001). |
| Bigger dictionary at k = M/4 helps: +1.14 pp per doubling of M (512 → 4096), +0.46 pp for 4096 vs 2048 (MSWEEP-001, pre-registered) | **Holds in direction.** At tuned rates 4096 vs 2048 is +0.63 pp. M = 512 and 1024 were not re-run at tuned rates; the lr-1e-3 sweep was unstable for M ≥ 2048. M, k and parameters are confounded throughout. |
| No head type or loss term meaningfully reduced the compression drop (GRID-001, GRID-002, ABL-001, SEED-001) | **Still holds for the four heads re-tested at tuned rates** (exploratory, uncorrected; RETEST-001; see B3). Loss-term variants (ABL-001) not re-tested. **Revision 2:** replacing the MSE loss with tuned weight decay shows no detected change in the compression cost of either 2048-wide head (WD-001, exploratory; all CIs include 0; not an equivalence result). The dependency/coherence variants remain untested at tuned rates. |

## D. Unresolved

- **Image-sampling error.** All test CIs cover seed variation on one fixed set of 10,000 images; per-image correctness was not saved, so the image-sampling component is not quantified (REVIEW-010 S-4). Rough estimate (inference): 0.2–0.3 pp per model pair. The most fragile result is sparse − dense.
- **M vs k vs parameter count** in the M4096 result: no fixed-k arm or dense-M4096 control.
- **Final accuracy tolerance:** 1.0 pp is the working tolerance of the approved REASON-002 plan; it has not been finalized.
- **Settings chosen at lr 1e-3 and not re-tuned:** k = 512 (tuned) / 1024 (set by MSWEEP-001's k = M/4 rule), initialization, the MSE weight and the linear weight decay.
- **Test reuse.** ImageNetV2 has been scored many times, and its results informed which experiment was designed next; no setting was tuned on it.
- **Claim scope.** No result shows learning from compressed data alone (frozen pretrained backbone).
- ~~**Whether a weight-decay-tuned wide head would match the MSE head** (not run).~~ **Answered in revision 2** by WD-001 (B5): close but not all the way for the dense head (−0.29 pp, detected, below tolerance; post-hoc description); not for the sparse head within the grid and 20-epoch budget (−1.36 pp, meaningful; possibly under-trained).
- **New from WD-001:** whether sparse-CE+wd is under-trained (lr edge, budget binds; a longer budget or an lr between 3e-5 and 1e-4 would be a new, post-hoc experiment); decoupled weight decay (AdamW) or decay on all parameters; M other than 2048. The MSE heads (weight decay 0) and the linear head (1e-4, chosen at lr 1e-3) were not weight-decay tuned.
- **Citation check** of "Diverse Dictionary Learning" in `src/` (LIT-002, never claimed).

## E. Errata

### E1. Carried forward from COORD-002 (sections A–C there)

- **Stale ledger summaries:** the `summary` fields of ABL-001 and GRID-002 carry withdrawn wording. Do not quote ledger summaries in synthesis. This now also applies to GRID-002, ABL-001, SEED-001 and MSWEEP-001 summaries that state lr-1e-3 head comparisons as findings.
- **Corrections to accepted reports** (REVIEW-005 R-1, R-3): GRID-002 per-seed sparse − linear means are 0.95 / 0.90 / 0.89 (not 0.76 / 0.80 / 0.79); the GRID-001 ranking explanation is an untested inference; inherited fc worth 3.3–6.8 pp (not 3.5); "50.3:1 against raw pixels at native channel count" (not "raw RGB"); GRID-001 best-epoch ranges 5–10 (sparse) and 1–10 (dense); ABL-001 dense-mse peaks 5–8 and dense CE-only 5–9; GRID-002's reviewer confirmed only that the regularizers were active, not their weights.
- **ENG-004 pilot** scores are selection-biased (probably at most 1–2 pp). **"50:1 impossible at 32/64 px"** holds firmly at 32×32; at 64×64 only for Pillow's standard headers.

Full wording: [COORD-002/status.md](../COORD-002/status.md) sections A–C.

### E2. New, found on 2026-11-26 (in reviews, or disclosed by the author; none changes a decision)

| Report | Says | Correct | Found by |
| --- | --- | --- | --- |
| SEED-001 | 8 pre-registered tests survive correction | True; if all 12 printed tests are corrected together, sparse − dense at L80 only just passes (0.0235 vs 0.025) | `reviewer-20261126-e` (event 115), cross-review of REVIEW-006 |
| MSWEEP-001 task.json | a 10-test family | Its own rule defines 13; the analysis used 13, and the same tests are significant either way | the author's own disclosure in the MSWEEP-001 report, confirmed by REVIEW-007 |
| LR-001 (8.22; copied from REVIEW-008) and REVIEW-008 §3 / S-2 | dense source − L090 at lr 1e-3 is 8.22 pp; ties keep the earlier epoch; sparse M2048 plateaus | 8.21 pp; 2 of 4 tied maxima kept the later epoch (all follow "top-1 then CE"); the sparse M2048 plateau holds only within the 10-epoch budget | `reviewer-20261126-h` |
| LR-002 | switching dictionary heads to 20 epochs has a validation "cost of −0.02 to +0.30 pp" | A **gain** of +0.02 (M4096), +0.11 (dense), +0.28 (sparse M2048) pp (exact values stated in RETEST-001 revision 1) | `reviewer-20261126-i` |
| REVIEW-009 | the linear lr fix "removes about 1.2–1.5 pp"; 4 exact-tie windows | The sparse head's lr also changed, so the fix is not the linear head's alone; there are 2 more tie windows in the LR-001 histories | `reviewer-20261126-j` |
| REVIEW-010 | the lr-1e-3 cost differences +0.53 and +0.37 are new | They were already in SEED-001's report (from REVIEW-006); only sparse M4096 − linear (+0.23) is new | `reviewer-20261126-l` |

### E2b. New, found on 2026-11-27 (revision 2; wording only, none changes a number or a decision)

| Report | Says | Correct | Found by |
| --- | --- | --- | --- |
| WD-001 §5, last bullet | top-5 and training-level evaluation "show the same ordering: the CE heads are at or slightly below the MSE heads" | Fits dense only: dense-CE+wd is 0.05–0.16 pp below on top-5 and 0.12–0.30 pp on training-level top-1; sparse-CE+wd is 0.72–0.84 and 1.47–1.70 pp below | `reviewer-20261127-t` (REVIEW-014 O-1) |
| WD-001 §3 / §5 | sparse-CE − linear per-level gaps not listed, though §5 relies on them | −0.99 (source), −1.31 (L080), −1.73 (L090); all CIs exclude 0 | REVIEW-014 O-2 |
| WD-001 §3 | "the sparse gap is the same at every level" | "about the same (−1.34 to −1.40 pp; descriptive, not an equivalence test)" | REVIEW-014 O-3 |
| WD-001 §9 | the sparse conclusion without its qualifier | "within the pre-registered grid and 20-epoch budget" (see §4) | REVIEW-014 O-4 |
| WD-001 §7 | "Bit-identity of that specific run was not separately verified" | Also: the interrupted attempt's 8 logged epochs are identical to the rerun's epochs 1–8 (partial evidence of determinism) | REVIEW-014 O-5; confirmed by `reviewer-20261127-u` |
| WD-001 §5 | "Swapping MSE for weight decay does not change how much compression costs" | "no detected change" (the cost differences are not significant, e.g. dense at L090 [−0.51, +0.92]; not an equivalence result) | `reviewer-20261127-u`, cross-review of REVIEW-014 (event 197) |
| WD-001S | the selection script was written "before any phase-A result was read" | Last saved 07:53Z, after 1 of 144 runs had finished (not a selected setting); the rule was fixed by task.json at 07:50Z, so it could not steer the selection. WD-001 §7 already states this | `reviewer-20261127-r` (REVIEW-013 O-1), confirmed by `reviewer-20261127-s` |
| REVIEW-013 | "I have not submitted REVIEW-013" | True when written; the reviewer submitted afterwards at the coordinator's request, after checking the saved transcription | `reviewer-20261127-s`, cross-review of REVIEW-013 |
| REVIEW-014, REVIEW-015 | "Not submitted (left for the coordinator)"; "I claimed REVIEW-015 and have not submitted it" | The same pattern: true when written; ledger submits by `reviewer-20261127-t` (20:49:50Z) and `reviewer-20261127-v` (22:45:13Z) after each checked the saved transcription | the coordinator's own note (revision 3; REVIEW-015 O-3) |

B5 above already uses the corrected wording. The WD-001 report itself is hash-locked and unchanged.

### E3. Process notes

- `reviewer-20261126-i` accepted LR-002's revision 46 s after resubmission, after verifying hashes and post-hoc numbers against its own earlier recompute. The revised text was not independently re-checked, but `reviewer-20261126-j`'s cross-review of REVIEW-009 (event 143) recomputed the same post-hoc numbers two minutes later.
- **RETEST-001:** the coordinator edited `analyze.py` (04:55:12Z) before REVIEW-010 was even submitted (04:56:39Z; found by `reviewer-20261126-l`), and finished `report.md` revision 1 (04:57:44Z) before the reviewer's `revise` verdict could be recorded, so the verdict was refused as a ledger event. RETEST-001 was recovered (event 150, whose reason records the revise), reclaimed (151), resubmitted (152) and accepted (153). `analysis.md` and `analysis.json` are byte-identical across the revision.
- **WD-001A (revision 2):** the first attempt at WD-001's selection lock was added to the ledger but refused at claim, because its write paths overlapped the active WD-001. It stays `pending`, with no owner, work or folder, and should be ignored; the harness cannot block or close a pending task. WD-001S replaced it. Lesson: a new task's write paths may not overlap an active task's.
- **WD-001 overnight stop (revision 2):** the user stopped phase B overnight after 29 of 42 runs. The one partial run (sparse-CE L090 seed 49) was deleted and re-run from scratch; the resume is logged, and REVIEW-014 verified the 29 completed runs and the rerun.

## F. Where to look

- **Plain-language account:** [WORK_SUMMARY.md](../../../WORK_SUMMARY.md).
- **Confirmatory head and cost results:** [RETEST-001 report](../RETEST-001/report.md) and [analysis.md](../RETEST-001/analysis.md).
- **Weight decay instead of the reconstruction loss:** [WD-001 report](../WD-001/report.md) and [analysis.md](../WD-001/analysis.md), read with E2b.
- **Draft write-up:** [WRITE-001/report_draft.md](../WRITE-001/report_draft.md).
- **Settled choices:** [DECISIONS.md](../../DECISIONS.md), "Update, 2026-11-27" and "Current state, 2026-11-26".

## Revision log

- **Revision 1 (2026-11-26), for [REVIEW-011](../REVIEW-011/report.md)** (verdict revise, wording only; no number changed):
  - **S-1:** E3 now separates the `analyze.py` edit (before REVIEW-010 was submitted, found by `reviewer-20261126-l`) from the `report.md` revision (before the verdict).
  - **S-2:** "ties" replaced by "no detected difference" in section C (ABL-001 row) and in RESEARCH_STATUS.md.
  - **S-3:** the compression-tolerance row is limited to the four re-tested heads, labeled exploratory; loss-term variants not re-tested.
  - **S-4:** the tolerance is marked as the working tolerance, not finalized, in A and D, and in DECISIONS.md "Still open".
  - **O-1** absolute accuracy at level 80; **O-2** sparse M2048's post-hoc lr; **O-3** sparsity scope (k = 512 of M = 2048, with MSE); **O-4** inherited-fc row and "every Xavier-initialized head"; **O-5** "chosen" not "tuned"; **O-6** E2 attributions; **O-7** LR-002 post-hoc numbers re-checked by `reviewer-20261126-j`; **O-8** DECISIONS "unchanged" list and RESEARCH_STATUS COORD-002 row. WORK_SUMMARY.md (outside this task) is updated separately.
- **Revision 2 (2026-11-27, task [COORD-005](../COORD-005/task.json); user: "Yes, update them")**, after WD-001 was accepted (REVIEW-013, REVIEW-014, both cross-reviewed):
  - title date; new **B5** (W1–W4, readings, caveats, exploratory results with the corrected wording);
  - section C: the ABL-001 row (CE-only wide heads re-tested with tuned weight decay) and the compression-tolerance row;
  - section D: the weight-decay question marked answered; new open items from WD-001;
  - new **E2b** (wording slips found in REVIEW-013, REVIEW-014 and their cross-reviews); E3 notes on WD-001A and the overnight stop; F links.
  - Sections A, B1–B4, E1, E2 and the rest of C and D are unchanged.
  - Accepted after [REVIEW-015](../REVIEW-015/report.md), cross-reviewed by `reviewer-20261127-w`.
- **Revision 3 (2026-11-27, task [COORD-006](../COORD-006/task.json))**, wording only, no number changed, for REVIEW-015 and its cross-review:
  - **O-1:** the D "answered" line keeps its qualifiers;
  - **O-3:** the REVIEW-014 parenthetical in E2b's REVIEW-013 row moved to its own row, credited to the coordinator, with REVIEW-015's same line added;
  - **O-7:** B5 "Design" gives the CE-only reconstruction weight (1e-6, effectively off);
  - **cross-review (b):** the C ABL-001 row says the dense gaps are "detected, below tolerance".
