# Research handoff and next actions

Updated: 2026-11-27 (COORD-005; wording fixes in COORD-006 for REVIEW-015 O-5: the review note above the 2026-11-27 table and the COORD-005 row). This is the coordinator's readable handoff; use `py ai_harness/harness.py status` for live task states. Reviewed research memos are not experimental results or finalized user decisions.

## Update 2026-11-27: weight-decay check done; draft write-up updated

**Start here:** as on 2026-11-26 below. [COORD-004/status.md](artifacts/COORD-004/status.md) is now at revision 2, and the draft write-up is [WRITE-001/report_draft.md](artifacts/WRITE-001/report_draft.md) (revision 3).

**Accepted since 2026-11-26** (each reviewed by a different agent; REVIEW-012, REVIEW-013, REVIEW-014 and REVIEW-015 were cross-reviewed):

| Task | What | Review |
| --- | --- | --- |
| WRITE-001, WRITE-002 | Draft write-up (methods, results, limitations), then wording fixes from the cross-review | REVIEW-012 (WRITE-001); `reviewer-20261126-q` (WRITE-002, ledger review) |
| WD-001S | Selection lock for WD-001, submitted before any ImageNetV2 run | REVIEW-013 |
| WD-001 | Can tuned weight decay replace the reconstruction loss in a 2048-wide head? (pre-registered; validation selection, then ImageNetV2 seeds 45–51) | REVIEW-014 |
| COORD-005 | This update (status page, DECISIONS.md, this file, draft section 7.4) | REVIEW-015 |
| COORD-006 | Wording fixes to COORD-005 from REVIEW-015 and its cross-review | see ledger |

**Headline from WD-001 (measured; qualifiers in COORD-004/status.md B5):**
- **Dense head:** cross-entropy with tuned weight decay is 0.29 pp below the reconstruction-loss version (detected, below tolerance).
- **Sparse head:** 1.36 pp below (meaningful, within the tested grid and 20-epoch budget; it may be under-trained).
- **Neither** weight-decay head beats linear. The 2026-11-26 head conclusion is unchanged.

**Ignore WD-001A:** a pending ledger entry with no owner or work, refused at claim because its write paths overlapped WD-001; the harness cannot close it.

**Next:** the team decisions marked in the draft (accuracy tolerance, names, references, transfer into the course `.tex`). Optional follow-ups are in COORD-004/status.md section D.

**Not yet started:** REPRO-001, REVIEW-001, SYNTH-001, LIT-002.

## Update 2026-11-26: head question settled; write-up is next (historical; see the 2026-11-27 update)

**Start here:**
- [WORK_SUMMARY.md](../WORK_SUMMARY.md) is a plain-language account for teammates.
- [COORD-004/status.md](artifacts/COORD-004/status.md) is now the authoritative list of reviewer-accepted results and corrections. It replaces COORD-002/status.md, which is kept as history.
- [DECISIONS.md](DECISIONS.md) "Current state, 2026-11-26" has the settled choices.

**Accepted since 2026-11-25** (each reviewed by a different agent; the reviews were cross-reviewed):

| Task | What | Review |
| --- | --- | --- |
| SEED-001 | 7 new seeds, sparse vs dense vs linear at lr 1e-3 (pre-registered) | REVIEW-006 |
| MSWEEP-001 | Clean dictionary-size sweep, k = M/4, lr 1e-3 (pre-registered) | REVIEW-007 |
| LR-001, LR-002 | Validation-only learning-rate checks; lr 1e-3 was poor for every head, linear under-tuned by about 2 pp | REVIEW-008, REVIEW-009 |
| RETEST-001 | Confirmatory ImageNetV2 re-test at tuned rates, 20 epochs, seeds 45–51 (pre-registered) | REVIEW-010 |
| COORD-002, COORD-003, HARNESS-001, ENG-004, REVIEW-002 to REVIEW-005 | Results summary, shared-file update, harness supersession fix, pilot revision, first review rounds | see ledger |

**Headline (measured; see COORD-004/status.md for the qualifiers):**
- **Setting:** a head on frozen pretrained ResNet-18 features, tested on original ImageNetV2.
- **Cost of compressed training data (tuned rates):** level 80 (50.3:1 against raw pixels, 89.8% smaller than the stored files) costs a mean of 0.53–0.85 pp of top-1 (CIs up to 1.17 pp for the sparse heads). Level 90 costs 1.5–2.2 pp.
- **Dictionary heads vs linear:** at tuned rates sparse M2048 shows no detected difference from linear (+0.02 pp, CI [−0.25, +0.29]) and a small difference from dense (+0.09 pp, below 1 pp, not robust to multiplicity). The 4096-atom head is best by +0.6 pp, below the 1 pp tolerance, with M, k and parameters confounded. The 2026-11-25 headline below ("beat a linear head only with the feature-MSE loss") was lr-conditional and is superseded.
- **Compression tolerance:** no head reduces the accuracy drop (exploratory).

**Next:** the write-up (SYNTH-001), led by the storage result, with the sparse-dictionary head reported as a negative result. Optional follow-ups are listed in COORD-004/status.md section D.

**Not yet started:** REPRO-001, REVIEW-001, SYNTH-001, LIT-002.

## Update 2026-11-25: experiments run and reviewed (historical; superseded by the 2026-11-26 update)

**Start here:**
- [WORK_SUMMARY.md](../WORK_SUMMARY.md) is a plain-language account for teammates.
- [COORD-002/status.md](artifacts/COORD-002/status.md) is the authoritative list of reviewer-accepted results and corrections.
- [DECISIONS.md](DECISIONS.md) "Current state" has the settled choices.

**Done and accepted by independent reviewers:**

| Task | What |
| --- | --- |
| ENG-001 | Byte tool |
| ENG-002 | Environment |
| ENG-003 | Datasets |
| ENG-005 | Linear baseline and dense control |
| REASON-002 | Experiment plan |
| GRID-001 | Main grid, inherited-classifier heads, byte accounting |
| GRID-002 | Xavier-init grid |
| ABL-001 | Loss ablation |
| ORIENT-001 and ORIENT-002 | Code audits |
| COORD-001 | Adoption record |

**Headline (measured; see status.md for the qualifiers):**
- **Setting:** a head on frozen pretrained ResNet-18 features, tested on original ImageNetV2.
- **Cost of compressed training data:** level 80 (50.3:1 against raw pixels, 89.8% smaller than the stored files) costs about 0.5–1.2 pp of top-1. Level 90 costs 1.5–2.3 pp.
- **Dictionary heads vs linear:** with Xavier init, they beat a linear head only with the feature-MSE loss, and by under 1 pp at most levels.
- **Sparsity:** its own effect is unresolved.

**In flight, or waiting for review:**
- SEED-001: 7 new seeds for sparse vs dense vs linear; pre-registered.
- ENG-004: pilot, revised for selection bias.
- COORD-002 and COORD-003.
- REVIEW-002 to REVIEW-005.

**Not yet started:** REPRO-001, REVIEW-001, SYNTH-001, LIT-002.

The 2026-11-22 update below is historical. Its "ENG-001 pending", "no training has been run", "torch is not installed" and "dataset open" statements are superseded.

## Update 2026-11-22: adopted user code

The user wrote `src/` and adopted it as the starting implementation. Details and open points are in [DECISIONS.md](DECISIONS.md), section "Adopted implementation".

| Assignment | Deliverable / state | Meaning |
| --- | --- | --- |
| ORIENT-001 | [Orientation and src/ audit](artifacts/ORIENT-001/report.md), awaiting review | File map, local hardware, gaps between `src/` and the training-storage protocol |
| COORD-001 | This update and the DECISIONS.md entry, awaiting review | Records the user's adoption decision |
| ENG-001 | Pending | Batch JPEG encoding of an ImageFolder-style split, with a per-image byte manifest and aggregate savings, as a new file in `src/` |
| LIT-002 | Pending | Verify the arXiv 2604.17568 citation in `src/resnet18_sparse_ae.py` |

The most consequential open choices are the training dataset and resolution (ImageNet-1K is the script default and is likely too large), the conventional baseline, and whether to accept the frozen-pretrained claim scope. No training has been run, and torch is not yet installed on the observed machine.

The remainder of this file is the 2026-11-05 handoff and remains valid unless superseded above.

## What is now available

The harness supplies five role prompts, shared project instructions, task dependencies, atomic claims, exclusive writing scopes, submitted-artifact hashes, independent review, recovery, and readable snapshots. It runs locally with Python's standard library. Actual AI reasoning, web research, and delegation happen in the agent host using [START_HERE.md](START_HERE.md).

Two bootstrap agents completed distinct assignments in parallel. The coordinator inspected both reports, checked the structured evidence, and independently checked three consequential primary-source passages. Their acceptance is recorded in the live ledger. The existing projectIdea.md, proposal files, and bibliography were preserved.

| Assignment | Deliverable | Meaning |
| --- | --- | --- |
| LIT-001 | [Literature report](artifacts/LIT-001/report.md) and [evidence records](artifacts/LIT-001/evidence.json) | Bounded literature audit; exact novelty remains unresolved |
| REASON-001 | [Experimental design](artifacts/REASON-001/report.md) | Proposed controls, model routes, four-condition pilot, and month-sized expansion |
| REPRO-001 | Ready for an engineering agent | Inspect reusable code, licenses, and dataset/pilot feasibility without training |
| REVIEW-001 | Ready for an independent reviewer | Cross-report critique beyond initial coordinator acceptance |
| SYNTH-001 | Waits for all four preceding tasks | Integrate findings into a concrete proposed project plan |

## Findings that affect the project

**The broad claim is already studied.** Nayak and Friedland compare compressed-training variants, pretrained EfficientNet sizes, and dataset disk storage. Thus, training classifiers on compressed images, changing ordinary model size, and measuring bytes are individually insufficient novelty claims. See sections II-A through II-C of the [primary PDF](https://cdn.amazon.science/61/96/9dcc72564f34a8f5652112be022f/deep-layers-beware-unraveling-the-surprising-benefits-of-jpeg-compression-for-image-classification-pre-processing.pdf).

**Storage definitions matter.** Lee's inspected arXiv version uses a sample-count-times-JPEG-quality proxy in section 3.1; final journal methods were not inspected. Berkeley's report excludes JPEG header and coding-table bytes in section 3.2, printed page 11. Its reported low bpp therefore cannot establish the same ratio for complete standalone JPEG files. The current project's practical storage claim should measure actual stored files. Sources: [Lee, arXiv v1](https://arxiv.org/html/2508.12833v1), [Berkeley report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-98.pdf).

**A candidate extension remains worth checking:** how explicit dictionary capacity M, with active budget k controlled, changes the full-file training-storage versus original-test-accuracy trade-off against fair conventional baselines. This is a hypothesis and provisional contribution, not a claim of established originality. The closest unresolved convolutional sparse-coding paper and further dictionary-capacity work still need inspection.

## Recommended first experiment, still proposed

Use four conditions: conventional classifier and neural dictionary classifier, each trained on original and strongly compressed versions of the same training examples. Evaluate all four on the same original held-out test set. Begin with a single pilot seed to check learning, actual byte rates, and runtime. The reasoner's report explains expansion to three dictionary sizes and repeated seeds only if the pilot supports it.

For compressed-only training, reconstruction targets must come from the decoded compressed input. Original targets or cached original features would add information beyond the compressed dataset. Reusing architecture code is separate from using pretrained weights; pretraining narrows the storage claim to the downstream training dataset.

## What the next agents should do

1. Claim REPRO-001 or REVIEW-001 once its dependencies are accepted; use a unique agent identity and its assigned artifact folder.
2. Inspect upstream implementations and unresolved sources, then submit findings for another agent to review.
3. Complete SYNTH-001 using accepted reports and clearly labeled resource assumptions.
4. Obtain the actual available compute and choose a defensible dataset before committing to training. Create a bounded pilot task with observable acceptance criteria.

No dataset was downloaded, no model was implemented or trained, and no accuracy/storage outcome was measured in this setup. The 98%-accuracy-at-more-than-50:1 target remains untested. The harness is not a continuously running background AI service; another agent starts a work cycle by following START_HERE.md.
