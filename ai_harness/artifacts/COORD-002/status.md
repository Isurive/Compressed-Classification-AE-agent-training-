# Accepted experiment results, as of 2026-11-25 (revision 2)

**Coordinator:** `claude-orient-20261122-a`.
**Source of truth:** the revised `report.md` files of GRID-001, GRID-002 and ABL-001.
- **First round:** `reviewer-20261125-a` returned *revise* in [REVIEW-002](../REVIEW-002/report.md).
- **Acceptance:** after one revision, the same reviewer accepted all three. Ledger events 59–61 hold the acceptances and their notes; the REVIEW-002 report itself shows only the first round.

Where this note and those reports differ, the reports win, **except** for the corrections in the erratum below, which apply to them.

Revision 2 addresses REVIEW-005 ([report](../REVIEW-005/report.md)): C-1 to C-6, R-1 and R-3. The revision log is at the end.

## Erratum (read before using ledger summaries or quoting the reports)

### A. Stale ledger summaries

The `summary` field that `py ai_harness/harness.py show <TASK>` prints for two done tasks still carries withdrawn wording. The same text is in their submit events. `BOARD.md` does not show summaries. How it happened:
- The author's own revision resubmissions kept the old sentences. For ABL-001 (event 58), the author prefixed the old summary instead of rewriting it.
- Both tasks are now done and cannot be resubmitted.

| Task | Stale text in the ledger summary | Authoritative wording |
| --- | --- | --- |
| ABL-001 | "the feature-MSE auxiliary loss explains the sparse head's gain" | [ABL-001 report](../ABL-001/report.md) §0 "Answer": in a 2048-wide head, the MSE loss offsets the width penalty. The resulting advantage over linear is **under 1 pp** (mean 0.78 pp over the levels tested). Neither MSE alone nor sparsity is individually established as the cause. |
| GRID-002 | "Inherited fc worth 3.7-6.8 pp; explains GRID-001 ranking" | Xavier heads score **3.3–6.8 pp lower** than inherited-fc heads up to level 90 (see C below). That inheritance explains GRID-001's ranking is an **untested inference** ([GRID-002 report](../GRID-002/report.md) §4). |

Do not quote ledger summaries in synthesis.

### B. Corrections to accepted reports (found by REVIEW-005, R-1 and R-3)

These reports are done and hash-locked, so the corrections live here.

| Report and location | Says | Correct |
| --- | --- | --- |
| GRID-002 headline 1 (line 20); REVIEW-002 G2-1 | per-seed mean sparse − linear gaps "0.76, 0.80 and 0.79" | Over GRID-002's 7 levels (source–95), they are **0.95, 0.90 and 0.89**. The quoted figures are ABL-001's 5-level subset. The conclusion is unchanged: every per-seed mean is under 1 pp. |
| GRID-002 headline 4 (line 25) | "the GRID-001 ranking … came from the linear head inheriting the most" (stated as fact) | Untested inference, as GRID-002 §4 says. |
| GRID-002 §4 (line 97) | inherited classifier worth "3.5–6.8 pp" up to level 90 | **3.3–6.8 pp** (minimum 3.26: sparse, level 20). |
| GRID-001 headline 1 (line 8) | "50.3:1 against raw RGB" | 50.3:1 against **raw pixels at each image's native channel count** (GRID-001 §4). It would be 50.8:1 if all images were counted as RGB. |
| GRID-001 §6 (line 141) | sparse/dense best epochs "5–7" | 5–7 is the mode. The ranges are 5–10 for sparse and 1–10 for dense. |
| ABL-001 §4 (line 92) | MSE runs peak at epochs "6–9"; CE-only runs at "4–6" | sparse-mse 6–9; **dense-mse 5–8**; sparse CE-only 4–6; **dense CE-only 5–9** (REVIEW-002 addendum). So the "slower training" contrast holds for sparse heads only. |
| GRID-002 §7 | the reviewer "confirmed" the loss weights 1 / 0.1 / 0.05 | The reviewer confirmed only that the regularizer terms were active (nonzero residual). The values come from the `Settings` defaults. |

### C. Related corrections from the other reviews (REVIEW-003, REVIEW-004)

- **ENG-004 pilot.** Its "held-out" 5,000 images are also the set used to select `best.pt`, so its scores are selection-biased; the bias is probably at most 1–2 pp. The report was revised to say so.
- **"50:1 impossible at 32/64 px" (ENG-001).** This holds firmly at 32×32. At 64×64 it holds only for Pillow's standard JPEG headers; a hand-built minimal header might fit (REVIEW-003 E1-1, an estimate, not measured).

## What the team can now state, measured and reviewer-verified

Setting for all three experiments:
- Frozen ImageNet-pretrained ResNet-18; only the head is trained.
- Trained on 45k ImageNet-val images, all JPEG-compressed at one level.
- Tested on original ImageNetV2 (10k images); 3 seeds, paired splits.

1. **Storage** (GRID-001 §4). The full-file bytes of the 45k training set at level 80 (JPEG quality 20) are:
   - 0.47 bpp;
   - **50.3:1 against uncompressed pixels**, using each image's actual channel count (50.8 if all were RGB);
   - **9.8:1 against the stored ImageNet files**, an 89.8% saving that includes stripped metadata;
   - headers counted.

   The stored files decode to exactly the training pixels (3,500 of 3,500 checked).
2. **Accuracy cost of compressed training data** (GRID-001 §3, GRID-002 §3, ABL-001 §3), paired drop against the same head trained on originals:

   | Training level | Cost |
   | --- | --- |
   | 80 | 0.5–1.2 pp |
   | 90 | 1.5–2.3 pp |
   | 95 | 2.7–5.0 pp |
   | 100 | 7.7–18.7 pp |

   These hold for every head type tested. Heads that inherit the ImageNet classifier lose less at levels 95–100.
3. **Inherited classifier and head ranking** (GRID-001 §3, §6; GRID-002 §3, §4).
   - **With the inherited fc:** the linear head is best at every level.
   - **With Xavier init (no inherited classifier):**
     - sparse is ahead of linear at every level from source to 95. The sign is consistent, but the gap is **≥ 1 pp only at source (1.19) and level 20 (1.57)**; elsewhere it is 0.39–0.99 pp.
     - **The sparse vs dense gap (0.2–0.8 pp) and the dense vs linear gap (0.1–0.9 pp) are under 1 pp and not established.**
     - **Linear is best at level 100.**
   - **Size of the effect:** Xavier heads score 3.3–6.8 pp lower than inherited-fc heads up to level 90, and more beyond.
   - **Reference:** no trained head beats the untouched original classifier (57.28%).
4. **What makes the dictionary head work** (ABL-001, Xavier init):

   | Change | Effect |
   | --- | --- |
   | 2048 width with CE only | −1.0 pp vs linear |
   | Add the feature-MSE loss | +1.4 pp (dense), +2.0 pp (sparse) |
   | Add the dependency/coherence regularizers | ≈ 0 |
   | Top-k sparsity | +0.46 pp with MSE, −0.14 pp without |

   The sparsity effect is under 1 pp and **unresolved**. Alternatives not ruled out (ABL-001 §4): generic regularization, code-scale anchoring, training dynamics.
5. **Compression tolerance.** No head type or loss term reduced the accuracy drop from compression by a meaningful (≥ 1 pp) or consistent-across-levels amount. Small, level-specific differences exist, e.g. sparse retained 0.42 ± 0.07 pp more than dense at level 95 in GRID-001.
6. **The 98%-at-50:1 target.** It is not met in absolute terms: trained heads reach about 46–54% on ImageNetV2 vs 57% for the original classifier. Relative to each head's own uncompressed accuracy, level 80 retains about 97.7–98.9%, within seed noise of 98%.

## Unresolved

- **Dictionary size M.** The GRID-001 M sweep is confounded by epoch-1 checkpoints. A clean sweep would need Xavier init and k scaled to M.
- **Sparsity, sparse vs dense with MSE (+0.46 pp).** Under test in SEED-001 (7 new seeds, pre-registered); not yet reported.
- **Whether a weight-decay-tuned MLP would match the MSE gain.**
- **Untuned learning rate and epochs.** The linear head peaked late in 9 of 24 GRID-002 runs.
- **Test reuse.** ImageNetV2 results informed the design of later experiments. Confirm effects under 1 pp with fresh seeds before final claims.
- **Claim scope.** All results are for a head on a frozen pretrained backbone; none show learning from compressed data alone.

## Shared files

`ai_harness/DECISIONS.md` and `RESEARCH_STATUS.md` are stale (REVIEW-003 C1-1). A separate coordinator task updates them and points to this note.

## Revision log

- **Revision 2 (2026-11-25, after REVIEW-005):**
  - C-1: qualified the Xavier ranking.
  - C-2: corrected the erratum's account of the stale summaries (cause, location) and added the GRID-002 summary.
  - C-3: 3.5 → 3.3 pp, with non-causal wording.
  - C-4: level-95 cost 2.7–5.0 pp; heads 46–54%.
  - C-5: acceptances cited as ledger events 59–61.
  - C-6: item 5 wording.
  - Added erratum section B for R-1 and R-3, and section C for related corrections from REVIEW-003 and REVIEW-004.
