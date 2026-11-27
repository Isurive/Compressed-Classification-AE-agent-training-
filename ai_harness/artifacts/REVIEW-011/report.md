# REVIEW-011: independent review of COORD-004

**Reviewer:** `reviewer-20261126-m` (did not author COORD-004 or any of the tasks it summarizes). **Date:** 2026-11-26.
**Verdict on COORD-004:** **revise.** The fixes are wording only and nothing blocks. The verdict was recorded in the ledger before this report was returned. COORD-004 is now `active`, owner `claude-orient-20261122-a`.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of COORD-004) saved it here verbatim, with no edits to content, on 2026-11-26, after the reviewer's `revise` verdict was recorded in the ledger and before any COORD-004 file was edited.

## What I reviewed

- `ai_harness/artifacts/COORD-004/status.md`: new file, 11,867 B, sha256 `482a8079…`.
- `ai_harness/DECISIONS.md`: the new "Current state, 2026-11-26" section, the retitled "Previous state, 2026-11-25" heading and the change-log entry (`git diff HEAD`).
- `ai_harness/RESEARCH_STATUS.md`: the new "Update 2026-11-26" and the retitled 2026-11-25 heading (`git diff HEAD`).

## What I checked and how

All checks were read-only. Ledger events were read through a read-only sqlite connection (`mode=ro`) and `harness.py show`. No scripts were written to the project.

1. **Numbers in status.md B1–B4 and the new DECISIONS/RESEARCH_STATUS sections.**
   - **Sources:** RETEST-001 `analysis.md` and `report.md`, GRID-001 `report.md` §4, COORD-002/status.md, the LR-001/LR-002 ledger notes and the REVIEW-008/009/010 cross-review notes.
   - **All match:**
     - mean accuracies;
     - A1–A4 means, CIs, sign counts and Holm p 0.094;
     - the A2 lower bound (+0.0016 → +0.002);
     - B1–B8 costs and CIs;
     - gains of +1.01, +1.18, +1.33 and +1.95 pp;
     - L090 cost differences of 0.53–0.68 pp;
     - storage figures 0.47 bpp, 50.3 (50.8), 9.8:1, 89.8%, 86:1, 94.0%, 3,500/3,500 and 57.28%;
     - level 95/100 costs of 2.7–5.0 and 7.7–18.7 pp (from COORD-002);
     - lr 1e-3 instability counts (4/6, 6/6, 6/6 → 0);
     - linear +2.14 pp on validation;
     - weight decay 1e-4 and the user-approval quotes for LR-001, LR-002 and RETEST-001.
   - **Recomputed:** the 98.3–98.9% retention, from RETEST-001 means: 48.45/49.29 = 98.30%, 49.11/49.90 = 98.42%, 48.42/49.10 = 98.62%, 48.42/48.94 = 98.94%.
2. **Errata A–C carried forward (E1)** against COORD-002/status.md sections A–C, item by item. All corrected values are present and attributed correctly. Full wording is linked.
3. **New errata (E2/E3)** against ledger events 115 (reviewer-e), 133 (reviewer-h), 142 (reviewer-i), 143 (reviewer-j), 150–154 and the RETEST-001/REVIEW-010 file mtimes.
   - The 46 s acceptance is confirmed: LR-002 resubmitted at 01:23:51Z and was accepted at 01:24:37Z.
   - The event numbers 150/151/152/153 are correct.
4. **Fairness of the section C table**, row by row, against each task's accepted report and RETEST-001's own "Consequences" table.
5. **History preservation.**
   - In both shared files the diffs only add text and retitle the old headings. All 2026-11-25 and 2026-11-22 text is intact.
   - `harness.py check` returns `ok: true`.
   - `git status` shows no tracked change outside DECISIONS.md, RESEARCH_STATUS.md and WORK_SUMMARY.md.
   - WORK_SUMMARY.md was last written at 05:22:41Z, before COORD-004 was added at 05:26:01Z, so COORD-004 did not change it.
   - COORD-002/status.md and the done reports are unchanged (hash check passes).

## Findings

### Substantive (should fix)

**S-1. E3 overstates the timing of the RETEST-001 slip and does not credit the reviewer who found it.**
- **What E3 says:** "the coordinator edited `analyze.py` and `report.md` before REVIEW-010 was submitted".
- **What the evidence shows:**
  - Reviewer-l (event 154) showed this only for `analyze.py`: mtime 04:55:12Z, before REVIEW-010 was submitted at 04:56:39Z.
  - RETEST-001 `report.md` was last written at 04:57:44Z, after that submission. When it was first edited is not recorded.
  - The recover note (event 150) and the RETEST-001 revision log only say "before the verdict was recorded".
- **Suggested wording:** "the coordinator edited `analyze.py` (04:55:12Z) before REVIEW-010 was even submitted (04:56:39Z; found by `reviewer-20261126-l`), and finished `report.md` revision 1 (04:57:44Z) before the reviewer's `revise` verdict could be recorded, so the verdict was refused as a ledger event."

**S-2. "Ties" is stronger than RETEST-001's pre-registered wording.**
- RETEST-001 item 1 says A1 is "no detected difference … not a pre-registered equivalence test".
- A2 is "small, below 1 pp, not robust to multiplicity", and its uncorrected CI excludes 0.
- **Fix in RESEARCH_STATUS.md**, Update 2026-11-26, "Dictionary heads vs linear". Replace "the sparse head ties linear (+0.02 pp) and dense (+0.09 pp)" with: "sparse M2048 shows no detected difference from linear (+0.02 pp, CI [−0.25, +0.29]) and a small difference from dense (+0.09 pp, below 1 pp, not robust to multiplicity)".
- **Fix in status.md**, section C, ABL-001 row. Replace "at tuned rates the MSE-trained head only ties linear" with: "at tuned rates the MSE-trained sparse M2048 head shows no detected difference from linear (+0.02 pp; RETEST-001)".

**S-3. The compression-tolerance row says more than was re-tested.**
- **The problem:** status.md section C, last row, says "**Still holds** at tuned rates (B3)" for "No head type **or loss term** …".
  - RETEST-001 labels this "Still holds (exploratory)".
  - Only the four heads were re-tested; the ABL-001 loss terms were not re-run at tuned rates.
- **Suggested status:** "Still holds for the four heads re-tested at tuned rates (exploratory, uncorrected; RETEST-001). Loss-term variants (ABL-001) not re-tested."
- In RESEARCH_STATUS.md, change "**Compression tolerance:** no head reduces the accuracy drop" to "… no head reduces the accuracy drop (exploratory)".

**S-4. An open item disappeared without a record.**
- The 2026-11-25 "Still open" row listed "final accuracy tolerance and proposal wording". The new 2026-11-26 "Still open" row, which supersedes it, keeps only "final proposal wording".
- AGENTS.md says no accuracy tolerance is finalized.
- REASON-002 offered "1.0 pp, or 2.0 pp" as a default, and the user approved the plan without addressing the tolerance explicitly.
- **Fix in DECISIONS.md**, "Still open": add "final accuracy tolerance (1.0 pp is the working tolerance of the approved REASON-002 plan)".
- **Fix in status.md** A, "Tolerance": write "only differences ≥ 1.0 pp count as meaningful (working tolerance from the user-approved REASON-002 plan; not finalized)".

### Optional / minor

- **O-1. Absolute accuracy at 50:1.**
  - **The issue:** B2 (status.md) and the DECISIONS "98%-at-50:1 target" row say "heads reach about 49–50%". That is their accuracy on uncompressed data. At level 80 (50:1) they reach 48.42–49.11%.
  - **Suggested wording:** "about 48.4–49.1% at level 80 (48.9–49.9% on uncompressed data) against 57.28%".
- **O-2. Sparse M2048's lr was not validation-tuned in the usual sense.**
  - **The issue:** B3 says "Every head at its validation-tuned lr (dictionary heads 1e-4 …)", and the DECISIONS "Training settings" row is similar.
  - For sparse M2048, 1e-4 at 20 epochs is a disclosed post-hoc departure from LR-002's adopted 3e-4 at 10 epochs (RETEST-001 caveat 2). It was never worse on validation (+0.28 pp).
  - **Suggestion:** add that parenthetical in one of those places.
- **O-3. Scope of the sparsity claim.** In B3, "top-k sparsity has no practically relevant effect at equal width" should keep REVIEW-006's scope: "(k = 512 of M = 2048, with MSE)". Similarly, "The sparse-dictionary head is no better than a tuned linear head" should name M2048/k512, since M4096/k1024 is +0.65 pp over linear.
- **O-4. Inherited-fc and "every head" wording.**
  - **Inherited-fc row (section C):** the 3.3–6.8 pp gap compares inherited-fc heads with Xavier heads that were under-tuned at lr 1e-3. At tuned rates the Xavier heads gained 1.0–2.0 pp (RETEST-001, exploratory). Suggest adding: "lr-conditional on the Xavier side; probably smaller at tuned rates (inference; different seeds, inherited-fc heads not re-tuned)".
  - **"Every head":** "lr 1e-3 … was poor for every head" (B4, DECISIONS) applies to the Xavier-initialized heads tested in LR-001/LR-002. GRID-001's inherited-fc heads were never lr-checked.
- **O-5. "Tuned" vs "chosen".** D says "Settings tuned at lr 1e-3 and not re-tuned: k = 512 / 1024 …". RETEST-001 caveat 3 says "chosen at lr 1e-3". k = 1024 was set by MSWEEP's k = M/4 rule, not tuned. Use "chosen".
- **O-6. E2 attributions and labels.**
  - In the SEED-001 row, name the finder as `reviewer-20261126-e` (event 115), as the other rows do.
  - In the MSWEEP-001 row, the finder is the author's own disclosure, later confirmed by REVIEW-007. Either the E2 heading ("found in the reviews") or that cell could say so.
  - In the LR-002 row, "RETEST-001 revision 1" is where the correct values were stated, not who found the slip.
  - In the LR-001/REVIEW-008 row, "ties keep the earlier epoch" is REVIEW-008 §3 wording; only the 8.22 figure was copied into LR-001 (line 117). The plateau point is REVIEW-008 S-2 and LR-001 line 94; it could be added to "Says".
- **O-7. LR-002 acceptance note (E3).** reviewer-j's cross-review of REVIEW-009 (event 143) recomputed the same post-hoc numbers (+0.66, +0.36, +0.00, +1.33/+1.04/+1.30) two minutes later. So "not independently re-checked" holds for LR-002's revised text, but not for its post-hoc numbers. Suggest saying so.
- **O-8. Housekeeping.**
  - The DECISIONS "unchanged" list omits the 2026-11-25 "Experiment plan" row. Add it (the 1.0 pp working tolerance lives there).
  - RESEARCH_STATUS "Accepted since 2026-11-25" omits COORD-002, which was in review in the 2026-11-25 update.
  - Outside COORD-004's write paths: WORK_SUMMARY.md §5 (points to COORD-002/status.md), §6 "Known stale text" and §7 step 1 are now out of date. Suggest a separate scoped update.

### Checked and found correct (no action)

- **Errata A–C** are carried forward faithfully: 0.95/0.90/0.89; untested inference; 3.3–6.8; native channel count; epoch ranges 5–10/1–10; 5–8/5–9; regularizer weights; ENG-004 selection bias; 32/64 px.
- **The E2 REVIEW-009 and REVIEW-010 rows** match events 143 and 154 (+0.53/+0.37 already in SEED-001; +0.23 new). The LR-002 "Correct" values +0.02/+0.11/+0.28 match event 143 and RETEST-001 caveat 2.
- **The section C rows for GRID-002 Xavier, SEED-001 and MSWEEP-001** follow RETEST-001's own consequences table ("Superseded: lr-conditional"; "Holds, small" with +0.63).
  - The MSWEEP row correctly says M = 512/1024 were not re-run and that M, k and parameters are confounded.
  - The section header correctly keeps the lr-1e-3 measurements "correct as measured".
- **History is preserved in both shared files.** The old sections are retitled as historical and the old text is not edited. The change-log entry is accurate and follows the existing ordering convention.
- **Every new status cites a task, a user approval or a ledger event.**

## Recommendation

Apply S-1 to S-4. Each is a one- or two-line wording change in status.md, DECISIONS.md or RESEARCH_STATUS.md; no numbers change. O-1 to O-8 are at the coordinator's discretion; O-1 and O-3 are cheap and improve precision. Then resubmit COORD-004 for re-review. A re-review only needs to diff the three files against the current hashes:
- status.md `482a8079…`
- DECISIONS.md `9861c18b…`
- RESEARCH_STATUS.md `98501416…`

and confirm the S-fixes and that `harness.py check` passes.

## Addendum, 2026-11-26: re-review of COORD-004 revision 1 (written by `reviewer-20261126-m`)

**Verdict: accept** (ledger `review-accept` on COORD-004; COORD-004 is now `done`).

**Transcription check.** The text above matches the report I returned, word for word. The only addition is the coordinator's transcription note after the verdict line. This file (mtime 05:34:00Z) was saved after my `revise` was recorded (05:32:41Z) and before any COORD-004 file was edited (status.md 05:34:41Z, DECISIONS.md 05:34:59Z, RESEARCH_STATUS.md 05:35:00Z).

**Files compared** (my round-1 reading and `git diff HEAD` against the resubmitted versions):

| File | Round 1 | Revision 1 |
| --- | --- | --- |
| status.md | `482a8079…` | `6f2691a7…` |
| DECISIONS.md | `9861c18b…` | `404ed484…` |
| RESEARCH_STATUS.md | `98501416…` | `0cf02667…` |

No number changed.

**Required fixes, all resolved:**
- **S-1:** E3 now separates the two RETEST-001 edits:
  - `analyze.py` was edited at 04:55:12Z, before REVIEW-010 was submitted (04:56:39Z). `reviewer-20261126-l` is credited.
  - `report.md` revision 1 was finished at 04:57:44Z, before the verdict could be recorded.
- **S-2:** "ties" is gone. The status.md C ABL-001 row and the RESEARCH_STATUS headline now use RETEST-001's wording: "no detected difference" for A1 and "small, below 1 pp, not robust to multiplicity" for A2.
- **S-3:**
  - The last row of section C is limited to the four re-tested heads, labeled exploratory and uncorrected, and says the loss terms were not re-tested.
  - RESEARCH_STATUS adds "(exploratory)".
- **S-4:**
  - The tolerance is marked "working; not finalized" in status.md A and D.
  - "Final accuracy tolerance" is back in DECISIONS "Still open".

**Optional items, all applied:**
- **O-1:** 48.4–49.1% at level 80.
- **O-2:** sparse M2048's post-hoc lr is disclosed in B3 and DECISIONS.
- **O-3:** the claims are scoped to M2048/k512, and to k = 512 of M = 2048 with MSE.
- **O-4:** the inherited-fc row is marked as an inference, and the lr claim now covers only "every Xavier-initialized head tested".
- **O-5:** "chosen" replaces "tuned", with the provenance of k.
- **O-6:** E2 finders are corrected and the heading is reworded.
- **O-7:** reviewer-j's recheck is noted.
- **O-8:** the experiment-plan row is in the unchanged list, and COORD-002 is in the RESEARCH_STATUS table.

**History.** The revision log at the end of status.md and the DECISIONS change-log entry ("Revised after review (REVIEW-011)") are accurate. The 2026-11-25 and 2026-11-22 text is intact. `harness.py check` returns `ok: true`. COORD-002/status.md and the done reports are unchanged.

**Remaining nit (non-blocking).** The LR-001/LR-002 row of the RESEARCH_STATUS "Accepted since" table still says "lr 1e-3 was poor for every head". In context this means every head tested; status.md B4 and DECISIONS carry the precise wording.
