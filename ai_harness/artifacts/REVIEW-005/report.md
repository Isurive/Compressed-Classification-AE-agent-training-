# REVIEW-005: independent review of REVIEW-002 and COORD-002

Reviewer `reviewer-20261125-d`. Date 2026-11-25. I did not author or review any of the work under review.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of COORD-002 and of the experiments REVIEW-002 covers) saved it here verbatim, with no edits to content, on 2026-11-25.

All inputs were read-only. There was no training, GPU use, download or install. My own spot-check script ran with `py -I` from the session scratchpad and read only `runs/*/results.csv`, `runs/GRID-001/jpeg_variants/manifest.csv` and `ai_harness/artifacts/GRID-001/msweep_log.txt`.

Severity scale:
- **Blocking:** invalidates results.
- **Should-fix:** a claim or number that does not match the evidence and must be corrected before acceptance.
- **Optional:** an improvement.

## Verdicts

- **REVIEW-002: revise.** The review is sound and its recomputation evidence exists and matches, with one exception. One supporting number in G2-1 is mislabeled (R-1), and the error has propagated into the accepted GRID-002 report. The conclusion is unaffected. The repair is a short addendum.
- **COORD-002: revise.** Every number except one traces to the accepted reports. But the Xavier ranking is stated without the reports' qualifiers (C-1). The erratum also misstates why the stale summary exists and where it appears, and it misses a second stale summary (C-2). The known "3.5–6.8 pp" error is confirmed; I rate it optional on its own (C-3).

No finding is blocking.

---

## Part 1: REVIEW-002

### What I verified

| Check | Result |
| --- | --- |
| Artifact hashes, 15 REVIEW-002 artifacts (my own sha256 against the ledger) | 15/15 match. `harness.py check` reports ok. |
| `recompute/<TASK>/analysis.md` and `analysis.json` vs submitted `artifacts/<TASK>/` copies | Byte-identical for all three tasks. |
| `recompute/<TASK>/analyze.py` vs submitted `analyze.py` | Each differs only in `ROOT = ...parents[5]` (was `parents[3]`), commented as the REVIEW-002 copy. This is correct for the deeper folder. |
| `stdout.txt` vs `analysis.md` | Identical for GRID-002 and ABL-001. For GRID-001 it is 2 bytes longer (trailing newline only). |
| `independent_check.py` | It is self-contained, does not reuse the authors' analyze.py, and has a correct `ROOT` (`parents[4]`). I did not re-run it, because it needs torch and writes next to itself. |
| Accuracy tables (my script, from `results.csv`) | Every GRID-001 §3, GRID-002 §3/§4 and ABL-001 §2/§3 value that REVIEW-002 quotes matches to 0.01 pp. This includes sparse−linear 1.19/1.57/0.88/0.99/0.90/0.47/0.39; MSE +1.99/+1.39; width −1.00; dense-mse−linear +0.39 (12/15); sparse-mse−dense-mse +0.46 (13/15); sparse-full−linear +0.78; and relative retention at L80. |
| G1-1 evidence: M-sweep best epochs | Matches `results.csv` exactly. M1024: 5,1,6 at source and 1,1,1 at L80, L90 and L95. M512: epochs 1–3. M4096: epochs 4–10. |
| G1-1 validation curves (parsed from `msweep_log.txt`) | M1024 L80 s42: 62.42 at epoch 1, then 60.40 at epoch 10, with no later epoch above epoch 1. M512 source s42: 64.22 → 61.36. M4096 L80 s42 peaks at 65.14. All match. |
| G2-4: GRID-002 linear best epoch ≥ 9 | 9 of 24. Matches. |
| Bytes, whole pool (streamed from `manifest.csv`, 350k rows) | Source 6,706,296,736 B. q20 683,922,845 B, R_raw 50.22. Counted as all-RGB it would be 50.74. Matches REVIEW-002 §1 and GRID-001 §4. I did not recompute the seed-42 45k subset, which needs the split code. |
| Transcription | The reviewer's original message is not available to me. Instead I compared report.md with the ledger notes the reviewer wrote itself (events 52–54 and 59–61). Every should-fix finding and every key number agrees, so I found no sign of transcription drift. |

### Findings

**R-1, should-fix. The G2-1 per-seed means are mislabeled.**
- **Location:** REVIEW-002 `report.md` §4 G2-1 evidence, bullet 3. The same text is in ledger note seq 53, so the error is the reviewer's, not the transcriber's.
- **Evidence:**
  - The report says "The 21/21 positive pairs come from 3 splits, with per-seed means of 0.76, 0.80 and 0.79."
  - Those three values are per-seed means of sparse-full − linear over **ABL-001's 5 levels** (source, 60, 80, 90, 95). They are printed under "ABL-001 cells" in `recompute/independent_check_output.txt` §4, and their mean is the 0.78 quoted one bullet earlier.
  - Over the **7 GRID-002 levels** that give the 21 pairs (source to 95), the per-seed means are **0.95, 0.90 and 0.89** (mean 0.91), from my recompute of `runs/GRID-002/results.csv`.
- **Implication:**
  - The conclusion still holds, because every per-seed mean is under 1 pp, and the wrong figures understate the gap.
  - But a number in the record does not match its stated basis.
  - It was copied verbatim into the accepted **GRID-002 report line 20** and accepted in re-review (seq 60). GRID-002 is done and cannot be revised through the ledger.
- **Repair:**
  - Add a dated addendum to REVIEW-002 `report.md`: over GRID-002's 7 levels the per-seed means are 0.95/0.90/0.89; 0.76/0.80/0.79 is ABL-001's 5-level subset.
  - Have the coordinator carry the same erratum for GRID-002 line 20, e.g. in COORD-002 or the later shared-file fold-in.

**R-2, optional. The report does not record the acceptance round.**
- **Location:** REVIEW-002 `report.md` "Verdicts".
- **Evidence:** The report records only the first-round *revise* verdicts. The three *accept* re-reviews (seq 59–61, at 07:18:02–04Z, about 80 s after the resubmissions) exist only as ledger notes. Those notes carry the residual nits and the "MATERIAL for the coordinator" warning about the ABL-001 summary.
- **Implication:** Anyone who follows COORD-002's link ("accepted ... in REVIEW-002") finds only revise verdicts. The acceptance criteria are still met, because the verdicts are recorded with notes.
- **Repair:** In the same addendum as R-1, summarize the re-review: accepted, the residual nits, and the ledger-summary warning.

**R-3, optional. The re-review missed some leftover overstatements.**
None of these changes a conclusion, and all are in done tasks. Route them through a coordinator erratum.
- **GRID-002 headline 4, line 25:** "So the GRID-001 ranking (linear first) came from the linear head inheriting the most." This is still stated as fact. §4 now labels the same claim an untested inference, so G2-2 was fixed only in §4. The GRID-002 ledger summary also says "explains GRID-001 ranking".
- **GRID-001 headline 1, line 8:** still "50.3:1 against raw RGB". §4 now defines R_raw with native channel counts (G1-2), so the headline is inconsistent with §4.
- **ABL-001 §4, line 92:** "MSE runs at 6–9". dense-mse selected epochs 5–8 (`results.csv`). The re-review corrected only the CE-only dense range.
- **GRID-001 §6, line 141:** "sparse/dense at epochs 5–7". Actual grid ranges are 5–10 for sparse and 1–10 for dense (one dense run at epoch 1). 5–7 is the mode, not the range.

**Soundness (no finding).** REVIEW-002's substantive findings hold up:
- G1-1, the epoch-1 confound in the M sweep;
- G2-1, the 1 pp overstatement;
- A-1/A-2, the attribution and the missing alternatives.

Each matches the data I checked. The integrity claims I could check (pairing, row counts, tune before test, bytes) agree. The revision logs in all three reports list every REVIEW-002 finding. I confirmed in the report text that each should-fix item (G1-1, G2-1, A-1, A-2) was addressed, apart from the leftovers above.

---

## Part 2: COORD-002 (`ai_harness/artifacts/COORD-002/status.md`)

### Claim trace

| COORD-002 claim | Source in accepted reports | Status |
| --- | --- | --- |
| Erratum quotes "the feature-MSE auxiliary loss explains the sparse head's gain" | ABL-001 ledger summary (seq 49 and 58) | Exact quote. |
| Authoritative wording: MSE offsets the width penalty; advantage < 1 pp (mean 0.78); neither MSE alone nor sparsity explains it | ABL-001 §0 "Answer", lines 16–29 | Supported. The report says "neither part is individually established", which is slightly weaker than "neither explains". Acceptable. |
| Setting: frozen ResNet-18, 45k compressed ImageNet-val, original ImageNetV2 10k, 3 seeds, paired | GRID-001 §1–2; GRID-002 §1, §7; ABL-001 §1 | Supported. |
| 1. L80: 0.47 bpp; 50.3:1 (native channels; 50.8 if RGB); 9.8:1; 89.8% including stripped metadata; headers counted; 3,500/3,500 | GRID-001 §4 (lines 101, 107–109), §7 | Supported. I recomputed 50.22/50.74 for the whole pool. |
| 2. L80 cost 0.5–1.2 pp; L90 1.5–2.3 pp | GRID-001 §3; GRID-002 §3; ABL-001 §3 drops | Supported. Range 0.51–1.21 and 1.54–2.26. |
| 2. L95 cost "3–5 pp" | Same | Slightly off: the minimum is 2.70 (GRID-001 linear). See C-4. |
| 2. L100 cost 8–19 pp | GRID-001 §3; GRID-002 §3 | Supported. Range 7.65–18.73. |
| 2. Inherited heads lose less at extreme levels | GRID-002 headline 3; §3 vs GRID-001 §3 | Supported. |
| 3. Inherited fc "adds 3.5–6.8 pp, and more at extreme compression" | GRID-002 §4, line 97 | The number is wrong in the source too. See C-3. |
| 3. With inheritance, linear is best | GRID-001 headline 2 | Supported. |
| 3. Xavier: "sparse > dense > linear" | GRID-002 §4 ranking table (line 103) | Stated more strongly than the reports' qualified findings. See C-1. |
| 3. sparse − linear consistent in sign, ≥ 1 pp only at source and L20 | GRID-002 headline 1, §6 | Supported. |
| 3. No trained head beats 57.28% | GRID-001 headline 4; GRID-002 headline 4 | Supported. The best single run is 54.22%. |
| 4. Width −1.0; MSE +1.4/+2.0; dep/coh ≈ 0; sparsity +0.46 with MSE and −0.14 without; unresolved | ABL-001 §0, §3, §4 | Supported. All values recomputed. |
| 5. No head type or loss term reduced the compression drop | GRID-001 headline 2; GRID-002 headline 2; ABL-001 headline 5 | Supported in substance. See C-6 for wording. |
| 6. Not met in absolute terms; heads "about 48–54%" vs 57% | GRID-001 §6; GRID-002 §6 | The range omits the 46–47% heads. See C-4. |
| 6. L80 relative retention 97.7–98.9%, within seed noise of 98% | GRID-001 §6 (98.5/97.9/97.8); GRID-002 §6 (98.5/97.7/98.9) | Supported. |
| Unresolved list: M, sparsity +0.46, wd-tuned MLP, untuned lr/epochs (9/24), test reuse, claim scope | GRID-001 §5; ABL-001 §4, §6; GRID-002 §6, §8 | Supported. |
| Pending list | `harness.py status` at the time of writing | Matches. |

### Findings

**C-1, should-fix. The Xavier ranking is stated without its qualifiers.**
- **Location:** status.md item 3, "Without it (Xavier): sparse > dense > linear".
- **Evidence:**
  - GRID-002 headline 1 says the sparse − dense edge is 0.2–0.8 pp and the benefit of top-k sparsity is "**not established**".
  - The dense − linear gaps are 0.80, 0.85, 0.59, 0.21, 0.57, 0.24 and 0.12 pp from source to level 95, all under the pre-declared 1 pp rule (my recompute). No report claims dense > linear as a finding.
  - At level 100, linear beats both (GRID-002 headline 2 and §3).
  - COORD-002 qualifies only the sparse − linear gap.
- **Implication:** This note is meant to be what "the team can now state". A synthesis author could quote the three-way ranking as established, when only the sparse − linear sign is consistent.
- **Repair:** For example: "sparse is ahead of linear at every level from source to 95 (consistent sign; ≥ 1 pp only at source and L20). The sparse vs dense and dense vs linear gaps are under 1 pp and not established. Linear is best at level 100."

**C-2, should-fix. The erratum misstates the cause and location of the stale summary, and misses a second one.**
- **Location:** status.md "Erratum", paragraphs 1–2.
- **Evidence:**
  1. **BOARD.md does not contain the quoted text.** "The ABL-001 `summary` string in the ledger and in `BOARD.md` still says ...". `BOARD.md` (regenerated 07:18Z) has no task summaries; a grep for "explains" or "MSE" finds nothing. The stale text is visible only via `harness.py show ABL-001` (seq 49, 58, and the `summary` field).
  2. **The ledger was not the obstacle.** "The ledger did not accept a new summary while the task was in review." After the revise verdict (seq 54) the task returned to `active`. The author's own resubmission at 07:16:47Z (seq 58) **did** set a new summary: it added the prefix "Revised per REVIEW-002 (report.md only; no numbers changed)" and kept the withdrawn sentence. `harness.py` sets `summary` on every submit.
  3. **GRID-002 has a stale summary too.** Its ledger summary (seq 57) still says "Inherited fc worth 3.7-6.8 pp; explains GRID-001 ranking". The "explains" is the untested inference that REVIEW-002 G2-2 asked to label. The erratum does not mention it.
- **Implication:** The erratum's core job, quoting the stale text and pointing to the authoritative wording, is done. But its account of how the stale text arose is inaccurate. It sends readers to the wrong place (BOARD.md). And it leaves one stale summary unflagged.
- **Repair:**
  - Say the stale text is in the ledger `summary` and submit events, as seen through `show`/`status`, not in BOARD.md.
  - Say it was carried forward in the revision resubmission (seq 58), and that the task is now done and cannot be resubmitted.
  - Add the GRID-002 summary to the erratum.
  - Optionally add the R-1 correction for GRID-002 line 20.

**C-3, optional (but fix with C-1/C-2). "3.5–6.8 pp" should be "3.3–6.8 pp".**
- **Location:** status.md item 3. Origin: GRID-002 §4, line 97 ("near-constant 3.5–6.8 pp up to level 90").
- **Evidence:** Paired Xavier − inherited differences up to level 90 (my recompute, matching `independent_check_output.txt` §4):

  | Head | Range | Notes |
  | --- | --- | --- |
  | sparse | 3.26–3.86 | **minimum 3.26 at L20**; L85 is 3.45 |
  | dense | 3.61–4.25 | |
  | linear | 6.05–6.80 | **maximum 6.80 at source** |

  The true range is **3.3–6.8**, so the known error is confirmed. The GRID-002 ledger summary's "3.7–6.8" uses source-level values only.
- **Severity:** Optional on its own.
  - It is a 0.2 pp error at the low end of a range whose spread (3.3 to 6.8) is driven by head type.
  - No conclusion depends on it: "a lot, and more at extreme compression" stands.
  - COORD-002 faithfully copied an accepted report, so this is not an overclaim by the coordinator.
  - But it is a known-wrong number in a document meant to be authoritative, so correct it in this revision and note the GRID-002 line 97 origin.
- **Also worth noting:** "adds" is causal wording. Between the grids, the linear head also changed weight decay (0 → 1e-4), though tuning shows that effect is about 0.1 pp on validation. Prefer "Xavier heads score 3.3–6.8 pp lower".

**C-4, optional. Two ranges are slightly too narrow.**
- **Location:** status.md item 2 (L95 row) and item 6.
- **Evidence:**
  - The L95 cost minimum is 2.70 (GRID-001 linear), not 3. Write "about 3–5" or "2.7–5.0".
  - "Heads reach about 48–54%" leaves out GRID-002 linear (47.15 at source) and the ABL-001 CE-only heads (46.16–46.20). Write "about 46–54%".

**C-5, optional. The acceptance link points to the wrong record.**
- **Location:** status.md header, "accepted by `reviewer-20261125-a` in REVIEW-002".
- **Evidence:** The linked REVIEW-002 report records only revise verdicts. The acceptances are ledger events 59–61 (see R-2).
- **Repair:** Cite those events.

**C-6, optional. Item 5's wording is slightly absolute.**
- **Location:** item 5, "no head type or loss term reduced the accuracy drop".
- **Evidence:** GRID-001 §3 records a small but consistent sparse-over-dense retention at L95 (0.42 ± 0.07 pp). GRID-002 records 0.38 ± 0.10 at L80.
- **Repair:** "none reduced it by a meaningful (≥ 1 pp) or consistent-across-levels amount", which matches GRID-001's own wording.

### Acceptance criteria for COORD-002

| Criterion | Status |
| --- | --- |
| States only conclusions present in the accepted reports, with links | Mostly met. C-1 drops qualifiers, and C-3 repeats a source error. |
| Erratum quotes the stale summary and points to the authoritative wording | Met, but the cause and location are misstated and GRID-002 is missing (C-2). |
| Lists what remains unresolved and the pending reviews | Met. |

---

## Checks run

1. `py ai_harness/harness.py claim REVIEW-005 ...`; `show` for REVIEW-002, COORD-002, GRID-001, GRID-002, ABL-001 and REVIEW-005; `check` (ok, 23 tasks); `status`.
2. sha256 of all REVIEW-002 (15) and COORD-002 (2) artifacts against the ledger: all match.
3. Byte comparison of each `recompute/<TASK>/analysis.{md,json}` with the submitted copy, plus a `diff` of the analyze.py copies.
4. My own spot-check script, kept in the session scratchpad and not submitted, run with `py -I`. It recomputes from the three `results.csv` files:
   - per-cell means, paired drops and relative retention;
   - Xavier − inherited per level;
   - per-seed sparse − linear over 7 and over 5 levels;
   - sign counts;
   - M-sweep and grid best epochs;
   - all ABL-001 effects, drops and best-epoch ranges.
5. Parsed `GRID-001/msweep_log.txt` per-epoch validation curves for the three runs cited in G1-1.
6. Streamed `runs/GRID-001/jpeg_variants/manifest.csv` (350k rows) for whole-pool bytes and R_raw by native-channel and RGB counting.
7. Read `harness.py` submit/review logic and searched `BOARD.md` to test the erratum's statements.

## Not checked

- The REVIEW-002 claims that need torch or the split code:
  - `best.pt` contents;
  - linear re-scores (0.5235/0.4680);
  - split-digest reconstruction;
  - sha256 isolation of ImageNetV2 from the pool;
  - the seed-42 45k byte subset;
  - per-epoch loss-weight residuals from the GRID-002/ABL-001 logs.

  Their outputs exist in `independent_check_output.txt` and are internally consistent with everything I did recompute.
- The reviewer's original final message, so transcription fidelity was checked only against the reviewer's own ledger notes.
- Training was not re-run, and no stored variants were decoded.
