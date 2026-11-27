# REVIEW-015: independent review of COORD-005

> Transcription note: written by reviewer-20261127-v and returned as its final message. The coordinator (claude-orient-20261122-a) saved it here verbatim, apart from this note. The verdict was recorded in the ledger by the reviewer itself before the report was returned.

Reviewer: `reviewer-20261127-v`. Date: 2026-11-27. I claimed REVIEW-015 and have not submitted it. That is left to the coordinator.

## Verdict

**Accept.** I recorded the verdict on COORD-005 before handback, and COORD-005 is now `done`.

- Every WD-001 number in the four edited files matches my recomputation from `analysis.json`.
- The wording follows the accepted WD-001 report as corrected by REVIEW-014 and the two cross-reviews.
- The E2b attributions are correct for every row that names a finder.
- Only the sections listed in the logs changed, and `harness.py check` passes.
- I found no substantive problems. Seven optional wording points follow.

## Checks

My helper script is `ai_harness/artifacts/REVIEW-015/check_numbers.py`. It uses the standard library only and reads `analysis.json`. I ran it with the ai360 `python -B`. I used no GPU and wrote nothing outside REVIEW-015.

1. **Baselines and integrity.**
   - The HEAD versions of the four files have the same sha256 as the previous ledger holders:
     - status.md `6f2691a7…` (COORD-004);
     - DECISIONS.md `404ed484…` (COORD-004);
     - RESEARCH_STATUS.md `0cf02667…` (COORD-004);
     - report_draft.md `2e76700c…` (WRITE-002).
   - So `git diff HEAD` shows exactly COORD-005's edits.
   - The current files have the same hashes as COORD-005's submitted artifacts (`8efa0f2a…`, `7dd15a40…`, `8c819b3c…`, `39bab8c1…`).
   - The WD-001 `report.md` (`2cada633…`) and `analysis.md` (`ef04c663…`) still match the ledger, so "the WD-001 report itself is hash-locked and unchanged" (E2b) is true.

2. **Numbers** (pp, recomputed from `analysis.json`). All match the edited files.
   - **W1–W4:**
     - W1 −0.286 [−0.407, −0.165], Holm p 0.0023;
     - W2 −1.362 [−1.520, −1.204], Holm p 3.0e-6;
     - W3 −0.355 [−0.566, −0.143], Holm p 0.0063;
     - W4 −1.344 [−1.634, −1.054], Holm p 8.5e-5;
     - 0/7 seeds positive for all four. Meaningful at 1.0 and 0.5 pp: W2 and W4 only.
   - **Mean top-1:** 48.90/48.17/46.66 (dense-CE) and 47.95/47.10/45.70 (sparse-CE).
   - **Per-level gaps:**
     - sparse − MSE: −1.34/−1.34/−1.40;
     - sparse-CE − linear: −0.99/−1.31/−1.73, all CIs exclude 0;
     - dense-CE − linear: −0.04/−0.25/−0.78, CIs include 0 at source and L080.
   - **Top-5 gaps** (unrounded): dense 0.054–0.160 and sparse 0.716–0.837. That gives the cited "0.05–0.16" and "0.72–0.84". The cross-review's 0.06 comes from analysis.md's rounded table.
   - **Training-level top-1 gaps:** dense 0.121/0.303 and sparse 1.469/1.699 ("0.12–0.30", "1.47–1.70").
   - **CE compression costs:** 0.73/2.24 (dense) and 0.85/2.25 (sparse).
   - **Cost differences against MSE:** +0.05, +0.00, +0.21, +0.06. All CIs include 0, and dense at L090 is [−0.51, +0.92].
   - **Cost differences against linear at L090:** +0.74 for both heads. The CIs are [+0.33, +1.16] (sparse) and [−0.01, +1.48] (dense).
   - **Selected epochs:** sparse-CE 16/21 at epoch 19 or later (median 19); dense-CE median 11, 0/21.
   - **Tolerance box (draft 6.4):** dense −0.29/−0.35 are below 0.5, and sparse −1.36/−1.34 are above 1.0. The statement that these verdicts do not change at 0.5 pp is correct.
   - **Phase-A figures** match analysis_val.md and REVIEW-013: 18100/30000 tie, 24 candidates, wd 1e-3.

3. **Strength of claims.**
   - **W1 is correctly limited** in all four files:
     - B5 says detected, below 1.0 and 0.5 pp, outside the pre-declared readings, and a post-hoc description;
     - draft 7.4 says the description is post-hoc;
     - DECISIONS calls it "a post-hoc reading";
     - RESEARCH_STATUS points to B5 for the qualifiers.
   - **W2 carries "within the pre-registered/tested grid and 20-epoch budget" and the under-training caveat** in B5, the C row, draft 7.4, the conclusion, DECISIONS and RESEARCH_STATUS.
   - **The corrected wording is used:**
     - "about the same at every level", plus "descriptive, not an equivalence test" in B5;
     - "no detected change" in compression cost, plus "not an equivalence result";
     - the larger sparse top-5 and training-level gaps.
   - **Two places keep the numbers but drop the qualifiers:** status D's "answered" line and the draft conclusion's dense clause (O-1, O-2).

4. **E2b attributions.**
   - O-1 to O-5 are REVIEW-014's, by `reviewer-20261127-t`.
   - O-5's confirmation by `-u` matches ledger event 197, which states that the 8 epoch lines are identical.
   - The "no detected change" point is `-u`'s (event 197).
   - The WD-001S "before any phase-A result was read" wording is REVIEW-013 O-1 by `-r`, confirmed by `-s` (event 190: "O-1 is accurate"). The quoted wording appears in WD-001S `task.json` and in the `analyze_val.py` docstring.
   - The REVIEW-013 submission note is from `-s`'s provenance nit (event 190).
   - One parenthetical is not covered by that attribution. "(the same pattern applies to REVIEW-014)" is the coordinator's own observation. It is accurate: REVIEW-014 says "Not submitted (left for the coordinator)", but the ledger records a submit by `-t` at 20:49:50Z. No cross-reviewer raised it (O-3).
   - The E2b header says none of these slips changes a number or a decision. That is correct.

5. **Earlier content preserved; logs accurate.**
   - **status.md:**
     - The hunks touch only the title date, the coordinator line, the new B5, the ABL-001 and compression-tolerance rows of C, two D items, the new E2b, two E3 bullets, F and the revision log.
     - The log's "Sections A, B1–B4, E1, E2 and the rest of C and D are unchanged" is true.
     - The original D item is struck through, not deleted.
   - **report_draft.md:**
     - The hunks touch the header, 6.2 (one added bullet), 6.3, 6.4 (one added line), the new 7.4, 9 (one bullet reworded), 10 (one added bullet), Appendix B and the log.
     - "No earlier number or verdict changed" is true.
     - One earlier phrase changed without a log entry: 6.3's "never used to choose learning rates" became "never used to choose settings" (O-4).
   - **DECISIONS.md:**
     - The new section sits above the old one.
     - The old heading was renamed, and the change log records the rename.
     - No row of the 2026-11-26 table changed.
     - The change-log entry follows the file's existing order, with new entries placed after the 2026-11-05 lines.
   - **RESEARCH_STATUS.md:**
     - The new update is at the top.
     - The old section's heading gained "(historical; see the 2026-11-27 update)"; nothing else in it changed.

6. **Cross-file consistency.**
   - The four files agree on W1–W4, the readings, the caveats and pointers:
     - DECISIONS points to status revision 2 and draft revision 3;
     - status F points to the DECISIONS "Update, 2026-11-27".
   - **"Not yet started: REPRO-001, REVIEW-001, SYNTH-001, LIT-002"** matches the ledger. WD-001A is pending as described, with no owner and no artifacts.
   - **The ledger supports the review claims:** WRITE-001 accepted by `-o`, REVIEW-012 cross-reviewed by `-p`, and WRITE-002 accepted by `-q`.
   - **Minor inconsistencies:**
     - DECISIONS "Still open" drops SYNTH-001 and the AdamW/other-M items that status D lists (O-6).
     - RESEARCH_STATUS overstates two review facts (O-5).

7. **`harness.py check`** returns `"ok": true`, with no stale tasks. It lists the COORD-004, WRITE-001 and WRITE-002 copies of the four files as superseded.

## Findings

No substantive (S-) findings.

- **O-1. status.md D, the struck-through "answered" item, drops the qualifiers.** Suggested wording: "**Answered in revision 2** by WD-001 (B5): close but not all the way for the dense head (−0.29 pp, detected, below tolerance; post-hoc description); not for the sparse head within the grid and 20-epoch budget (−1.36 pp, meaningful; possibly under-trained)."
- **O-2. report_draft.md §10, "Reconstruction loss" bullet.** The dense clause "weight decay comes close (−0.29 pp)" omits that the gap is detected. Suggested wording: "for the dense head weight decay comes close but not all the way (−0.29 pp, detected, below our tolerance)."
- **O-3. status.md E2b, REVIEW-013 row.** The parenthetical "(the same pattern applies to REVIEW-014)" sits under `-s`'s attribution, but `-s` reviewed only REVIEW-013. Suggested wording: "…(the coordinator notes the same pattern in REVIEW-014's 'Not submitted (left for the coordinator)'; ledger submit by `reviewer-20261127-t`, 20:49:50Z)".
- **O-4. report_draft.md §6.3.** "never used to choose learning rates" became "never used to choose settings". This widens an earlier claim, and the revision log does not record it. Seeds 45–51 were also MSWEEP-001's test seeds, and its results informed which heads RETEST-001 included. Suggested wording: "…used seeds 45–51, which were never used to choose learning rates or weight decay." Alternatively, keep "settings" and add a line to the revision-3 log.
- **O-5. RESEARCH_STATUS.md, 2026-11-27 table.**
  - The header "(each reviewed by a different agent; the reviews were cross-reviewed)" fits WD-001 and WD-001S. WRITE-002's ledger review by `reviewer-20261126-q` has no cross-review.
  - The table lists COORD-005 as "accepted" although it was under review when written.
  - Suggested wording: "(each reviewed by a different agent; REVIEW-012, REVIEW-013 and REVIEW-014 were cross-reviewed)". For the COORD-005 row: "REVIEW-015".
- **O-6. DECISIONS.md "Update, 2026-11-27", "Still open".**
  - The list drops "Write-up (SYNTH-001)", although RESEARCH_STATUS still lists SYNTH-001 as not started.
  - It also omits the AdamW / all-parameter decay / other-M items that status D lists.
  - Suggested addition: "…a longer budget or intermediate lr for the sparse weight-decay head (post-hoc), AdamW or decay on all parameters, M other than 2048 (COORD-004 D); SYNTH-001 (not started)…"
- **O-7. status.md B5 "Design" and report_draft.md §6.2 follow-up bullet.** "cross-entropy only (no feature-MSE loss)" and "without the reconstruction loss" are slightly loose. The CE-only heads keep a reconstruction weight of 1e-6, as ABL-001 'abl-none' did (WD-001 §4, task.json). Suggested addition: "(reconstruction weight 1e-6, effectively off, as in ABL-001's CE-only setting)".

## What I did not check

- I did not re-derive W1–W4 from `results.csv` or the run directories. I recomputed from `analysis.json`, which REVIEW-014 and its cross-review had already reproduced independently.
- I did not re-check phase-A selection, timing or the no-test-access path beyond the figures cited (that work belongs to REVIEW-013).
- I did not check `WORK_SUMMARY.md`, `BOARD.md` or `state.snapshot.json`. They are modified in the working tree, but they are outside COORD-005's write paths.
- I did not re-verify the unchanged sections of the four files (A, B1–B4, E1, E2, sections 1–7.3 and 8 of the draft) against their source reports. I only confirmed that they are byte-unchanged from HEAD.
- I did not run any training and did not use the GPU.
