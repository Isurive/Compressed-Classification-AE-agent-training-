# Accepted experiment results, as of 2026-11-25

**Coordinator:** `claude-orient-20261122-a`. **Source of truth:** the revised `report.md` files of GRID-001, GRID-002 and ABL-001. Each was accepted by `reviewer-20261125-a` in [REVIEW-002](../REVIEW-002/report.md) after one round of revisions. Where this note and those reports differ, the reports win.

## Erratum: ABL-001 ledger summary (read before using BOARD.md)

The ABL-001 `summary` string in the ledger and in `BOARD.md` still says:

> "the feature-MSE auxiliary loss explains the sparse head's gain"

**This claim was withdrawn in the revision.** The ledger did not accept a new summary while the task was in review, and the task is now closed. The authoritative wording is in [ABL-001 report](../ABL-001/report.md), §0 "Answer to the GRID-002 question":
- In a 2048-wide head trained on compressed images, the feature-autoencoding auxiliary loss offsets the penalty of the extra width.
- The head's resulting advantage over a linear head is **under 1 pp** (mean 0.78 pp over the levels tested).
- Neither MSE alone nor sparsity explains that advantage.

Do not quote the ledger summary in synthesis.

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
2. **Accuracy cost of compressed training data** (GRID-001 §3, GRID-002 §3):

   | Training level | Cost |
   | --- | --- |
   | 80 | about 0.5–1.2 pp |
   | 90 | about 1.5–2.3 pp |
   | 95 | 3–5 pp |
   | 100 | 8–19 pp |

   These hold for every head type tested. Heads that inherit the ImageNet classifier lose less at the extreme levels.
3. **Inherited classifier** (GRID-001 §6, GRID-002 §4). Initializing a head from the original ImageNet fc adds 3.5–6.8 pp, and more at extreme compression. With that initialization the linear head is best. Without it (Xavier):
   - sparse > dense > linear;
   - sparse − linear is consistent in sign but ≥ 1 pp only at source and level 20.

   No trained head beats the untouched original classifier (57.28%).
4. **What makes the dictionary head work** (ABL-001):

   | Change | Effect |
   | --- | --- |
   | 2048 width with CE only | −1.0 pp vs linear |
   | Add the feature-MSE loss | +1.4 pp (dense), +2.0 pp (sparse) |
   | Add the dependency/coherence regularizers | ≈ 0 |
   | Top-k sparsity | +0.46 pp with MSE, −0.14 pp without |

   The sparsity effect is under 1 pp and **unresolved**.
5. **Compression tolerance:** no head type or loss term reduced the accuracy drop from compression.
6. **The 98%-at-50:1 target.** It is not met in absolute terms: heads reach about 48–54% vs 57% for the original classifier. Relative to each head's own uncompressed accuracy, level 80 retains about 97.7–98.9%, within seed noise of 98%.

## Unresolved

- **Dictionary size M.** The GRID-001 M sweep is confounded by epoch-1 checkpoints. A clean sweep would need Xavier init and k scaled to M.
- **Sparsity, sparse vs dense with MSE (+0.46 pp).** Needs more seeds.
- **Whether a weight-decay-tuned MLP would match the MSE gain.** This is the generic-regularization explanation.
- **Untuned learning rate and epochs.** The linear head peaked late in 9 of 24 runs.
- **Test reuse.** ImageNetV2 results informed the design of later experiments. Confirm effects under 1 pp with fresh seeds before final claims.
- **Claim scope.** All results are for a head on a frozen pretrained backbone; none show learning from compressed data alone.

## Pending in the ledger

- **REVIEW-002:** in review. It needs a reviewer other than this author. The author transcribed the reviewer's report text, as noted in the file.
- **Earlier tasks by this author awaiting an independent reviewer:** ORIENT-001, ORIENT-002, COORD-001, ENG-001 to ENG-005, and REASON-002.
- **Unclaimed:** REPRO-001, REVIEW-001, SYNTH-001 and LIT-002.
- **Shared files.** When COORD-001 is accepted, a coordinator task should fold this note into `ai_harness/RESEARCH_STATUS.md` and `DECISIONS.md`.
