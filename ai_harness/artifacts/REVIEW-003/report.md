# REVIEW-003: Independent review of ORIENT-001, ORIENT-002, COORD-001 and ENG-001

**Reviewer:** `reviewer-20261125-b` (did not author any of the reviewed work). **Date:** 2026-11-25 (local).

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of the reviewed tasks) saved it here verbatim, with no edits to content, on 2026-11-25.

**Scope:** read-only outside this folder. No training, GPU jobs, downloads or installs. All temporary outputs went to the session scratch directory and were deleted.

Each task is judged against its own acceptance criteria as of its submission date. Later work (ENG-002 to ENG-005, GRID-001, GRID-002, ABL-001) is used only to flag statements that would now mislead a reader.

## Verdict summary

| Task | Verdict | Blocking | Should-fix | Optional |
| --- | --- | --- | --- | --- |
| ORIENT-001 | **accept** | 0 | 0 | 3 |
| ORIENT-002 | **accept** | 0 | 0 | 3 |
| COORD-001 | **accept** | 0 | 1 (follow-up for the coordinator, not a defect of the task) | 0 |
| ENG-001 | **accept** | 0 | 0 | 7 |

Severity scale: blocking / should-fix / optional.

---

## 1. ORIENT-001: Codebase orientation and audit of `src/` (2026-11-22)

**Verdict: accept.**

Acceptance criteria:
- **File map covers every non-generated project file and its status.** Met, with one small omission (O1-1).
- **Environment facts labeled as observed commands; unexecuted checks labeled.** Met (§3 table plus "Unexecuted" paragraph).
- **src/ audit separates code facts from inferences and lists concrete gaps.** Met. §4 uses `[code]` and `[inference]` labels, and lists five encoder gaps and seven model-fit points.
- **No files outside the task directory modified.** Met in net effect; see O1-2.

**Limit on verification.** `src/compress_jpeg.py` and `src/resnet18_sparse_ae.py` were revised by the user on 2026-11-23. Git history starts on 2026-11-25. So the exact 2026-11-22 text cannot be recovered. I checked every ORIENT-001 code claim against the current files. All still hold for the standalone files:
- level-to-quality mapping `round(95 − 94·level/100)` and `optimize=True` (`compress_jpeg.py:51-52`);
- EXIF transpose, alpha flattened onto white, 16-bit gray to 8-bit, ICC kept when the mode is unchanged (`compress_jpeg.py:21-50`);
- top-k then ReLU, so at most k active (`resnet18_sparse_ae.py:42-46`);
- unit-norm atoms, learnable center, optional residual decoder, dependency and coherence losses (`:52-133`);
- frozen backbone with BatchNorm in eval mode, frozen original `fc` applied to the reconstruction (`:144-185`);
- `--ce-weight` default 0, checkpoint chosen by validation feature MSE, ImageFolder train/val (`:277-end`).

The other facts check out:
- `references.bib` has exactly the 7 listed keys.
- The proposal `.tex` still contains `Group Name [EDIT]` and `Member 1..3`.
- The Zheng et al. arXiv 2604.17568 citation is in the docstring (`resnet18_sparse_ae.py:20-21`).

### Findings

**O1-1 (optional). The file map omits `ai_harness/.gitignore`.**
- **Location:** ORIENT-001 `report.md` §2.
- **Evidence:** `ai_harness/.gitignore` is dated 2026-11-05 and is not generated. It is the file that makes `state.sqlite3` "gitignored", as §2 itself says. No other non-generated file that existed on 2026-11-22 is missing. `ai_harness/__pycache__/` is generated, so leaving it out is fine.
- **Implication:** negligible.
- **Repair:** none needed. Optionally add it to any future file map.

**O1-2 (optional). The syntax check wrote and then deleted `src/__pycache__`.**
- **Location:** §3 "Syntax" row; §6.
- **Evidence:** the report says `py -m py_compile src/*.py` created `src/__pycache__` and that the author deleted it. The net change is nil, and it was disclosed.
- **Implication:** it touched a user-owned folder during a read-only task. No harm resulted.
- **Repair:** in future, compile into a temp location, or use `python -B -c "import ast; ast.parse(...)"`. Project practice now is `python -B`.

**O1-3 (optional). Dated statements have no forward pointer.**
- **Location:** §1 "Author unknown"; §3 "torch no"; §4.2 items 2, 5, 6; §2 file map.
- **Evidence:** these were all superseded later:
  - authorship by COORD-001;
  - the training entry point (now `main.py` with CE on, a new fc on the code, and `best.pt` chosen by top-1), the dataset (the 50k val pool) and the evaluation by ORIENT-002 §4;
  - the torch install by ENG-002 (`ai360` env);
  - the file map by later files (`main.py`, `sparse_workflow.py`, `image_data.py`, `baseline_head.py`, `run_*.py`, `Datasets/`, `runs/`, `WORK_SUMMARY.md`).
- **Implication:** the report is dated and says it describes 2026-11-22, so it is not wrong. However, `DECISIONS.md` still links to "ORIENT-001 section 4" as *the* code audit, with no mention of ORIENT-002. A reader following that link gets the superseded picture.
- **Repair:** do not edit the hashed report. The fix belongs in the coordinator follow-up under C1-1: point the audit link to ORIENT-002 as well.

---

## 2. ORIENT-002: Audit of the expanded `src/` workflow (2026-11-23)

**Verdict: accept.**

Acceptance criteria:
- **Pipeline summary traced to file/line locations.** Met. I checked every cited line range against the current files, which have been unchanged since 2026-11-23 23:06:
  - loss at `sparse_workflow.py:241-258`, with the target taken from the same input at `:251`;
  - `LatentClassifier` at `:72-99`;
  - fc initialization at `:175-191` (folded, Xavier or PyTorch default);
  - same level for train and validation at `:392-393`;
  - `best.pt` chosen by (top-1, −CE) at `:454-457`;
  - `random_backbone` at `:49`, used only by `main.py check` (`main.py:194` is the only reference);
  - stratified holdout at `image_data.py:187-197`, k-fold at `:200-216`, balanced limits at `:219-237`, fingerprint at `:256-261`;
  - the 10 subcommands in `main.py`.

  The README claims are also accurate: 45k/5k split, 10-fold CV, and ImageNetV2 MatchedFrequency as the independent test (`src/README.md:18, 40-49, 156-159`). The `--baseline` flag does use the original ImageNet fc (`main.py:146-150`).
- **Gaps separated from code facts.** Met: §1-2 are facts, §3 is labeled inference.
- **Superseded statements listed.** Met (§4).
- **Unexecuted checks labeled.** Met (§2 "Unexecuted").

**Reproduced the key measured claim.** I rebuilt the 16-image synthetic tree in scratch, ran `build_jpeg_dataset.build` at levels 0/20/50/80/100, and compared each stored file decoded to RGB with `compress_jpeg.model_image(image, level)`. I also compared byte counts against `write_jpeg`. Results:
- **80 of 80 identical**, and `write_jpeg` is **0 to 200 B larger**.
- Both results are the same under Pillow 12.1.0 (`py` 3.14) and Pillow 13.0.0 (`ai360`).

This matches ORIENT-002 §2 exactly. GRID-001 later confirmed it at scale (3,500 of 3,500). The §3 inference that 50:1 against raw RGB is reachable at native ImageNet resolution was also confirmed later: GRID-001 measured 50.3:1 at level 80.

### Findings

**O2-1 (optional). The claim scope omits the inherited ImageNet classifier.**
- **Location:** §3 gap 3.
- **Evidence:** §1 correctly notes that fc can be "folded" from the original ImageNet fc (`initialize_fc`, `sparse_workflow.py:185-190`). `--fc-init auto` folds whenever an ImageNet class mapping exists. But §3 gap 3 discusses only the frozen backbone when it gives the supported claim ("the sparse head and classifier can be trained on compressed images"). A folded fc starts the classifier from weights learned on original ImageNet-train images. That is a second channel of original-image information.
- **Implication:** GRID-001 later found this mattered a great deal: inherited-fc heads mostly measured damage to a pretrained classifier. GRID-002 (Xavier) was needed to isolate it. A reader of ORIENT-002 alone could underestimate the issue. Later accepted reports and `WORK_SUMMARY.md` state it explicitly, so the current risk is low. The README's own main command uses `--fc-init xavier`.
- **Repair:** none required for this task. Future audits should list initialization from pretrained classifier weights next to backbone pretraining in the claim scope.

**O2-2 (optional). The pixel-identity check is not saved as a script or command.**
- **Location:** §2, first row.
- **Evidence:** the report gives the result (80/80; 16 images × 5 levels) but not the code or the scratch location. GRID-001 later saved `pixel_check.py`; ORIENT-002 did not.
- **Implication:** another agent cannot replay the check directly. I re-implemented it and got the same result, so the claim stands.
- **Repair:** none now. Future measured claims should save the script beside the report.

**O2-3 (optional). The proposed task IDs in §5 differ from the ledger.**
- **Location:** §5.
- **Evidence:** ORIENT-002 proposed three tasks:
  - ENG-003 as the conventional baseline (in the ledger, ENG-003 is the dataset download and the baseline is ENG-005);
  - ENG-004 as the `build_jpeg_dataset` byte run (in the ledger, ENG-004 is the first training run and the byte run happened inside GRID-001);
  - ENG-002 as the torch install, which does match the ledger.
  The "PyTorch available? No" row and the "no ImageNet data found" note are also now superseded by ENG-002 and ENG-003.
- **Implication:** these are labeled proposals, but a reader could look up the wrong report.
- **Repair:** none in the hashed file. The coordinator follow-up (C1-1) can note the mapping.

---

## 3. COORD-001: Record user adoption of `src/` (2026-11-22)

**Verdict: accept.**

Artifacts: `ai_harness/DECISIONS.md` and `ai_harness/RESEARCH_STATUS.md`. Both still match their submitted SHA-256 hashes, so **neither file has been updated since 2026-11-22**. The task folder `artifacts/COORD-001/` is empty. That is consistent with the spec, whose deliverables are the two shared files.

Acceptance criteria:
- **Decision recorded with user provenance and date.**
  - Met: the "Confirmed user direction" row and the "Adopted implementation (user decision, 2026-11-22)" section, which quotes "adopt it".
  - I cannot see the user conversation, so the quote is accepted as recorded.
- **Implications separated from still-open choices.**
  - Met: "What adoption settles" and "What adoption does not settle" are separate lists. The second covers dataset, claim scope, baseline, evaluation, M/k grid, loss weights, checkpoint criterion, the byte pipeline and the unverified citation.
  - An editing rule for `src/` was added, which is useful.
  - These statements match ORIENT-001 and the code as of 2026-11-22.
- **No unrelated content removed or rewritten.**
  - Largely verified. `RESEARCH_STATUS.md` keeps the 2026-11-05 handoff under an explicit "remainder ... remains valid unless superseded" line. `DECISIONS.md` keeps the 2026-11-05 tables, harness decisions, reviewed findings and change log, with additions marked (2026-11-22).
  - No pre-COORD-001 copy exists to diff against: git starts on 2026-11-25 and the harness stores no earlier hash. So this check rests on internal consistency, not a diff.

### Findings

**C1-1 (should-fix, follow-up for the coordinator; not a defect of COORD-001). `DECISIONS.md` and `RESEARCH_STATUS.md` are stale and now mislead.**
- **Location:** `DECISIONS.md` "Adopted implementation" and change log; `RESEARCH_STATUS.md` "Update 2026-11-22" table.
- **Evidence.** `AGENTS.md` tells every agent to read `DECISIONS.md` first. Yet as of 2026-11-25 it still says:
  - `resnet18_sparse_ae.py` "frozen original `fc` classifies the reconstruction", "cross-entropy is off by default", "checkpoint criterion (currently validation feature MSE)". The adopted entry point is now `main.py`, where CE weight is 1, a new fc reads the sparse code, and `best.pt` is chosen by top-1 (ORIENT-002 §4).
  - The training dataset is open, and "ImageNet-1K default is a large download". In fact the ImageNet-1K val pool and ImageNetV2 were downloaded (ENG-003) and used in GRID-001, GRID-002 and ABL-001.
  - The conventional baseline is open. In fact `baseline_head.py` exists (ENG-005) and was used in the grids.
  - "The current tool encodes single images by quality factor". In fact `build_jpeg_dataset.py` exists (ENG-001) and measured the 50k pool (GRID-001 §4).
  - "torch is not installed". In fact the `ai360` env exists (ENG-002).
  - The audit link points only to ORIENT-001 §4.

  `RESEARCH_STATUS.md` still lists ENG-001 as "Pending" and says "No training has been run". COORD-002 recorded accepted results only in `artifacts/COORD-002/status.md` and did not touch these files.
- **Implication:** a new agent or teammate following `AGENTS.md` will get a materially wrong picture of the dataset, baseline, entry point, environment and results status. The likely cause is procedural: the two files are hash-locked as COORD-001 artifacts while it is in review.
- **Repair:**
  - After COORD-001 is accepted, open a new coordinator task that updates `DECISIONS.md` and `RESEARCH_STATUS.md` with provenance.
  - The update should record `main.py` and its modules as the adopted entry point; the dataset choice (ImageNet-1K val pool 45k/5k, ImageNetV2 for test only); the environment; the baseline; ENG-001 done; and a pointer to `COORD-002/status.md` for accepted results.
  - It should point the code-audit link to both ORIENT-001 and ORIENT-002.
  - Keep COORD-001's text as history.
  - This is not a reason to revise COORD-001, whose own scope was met.

---

## 4. ENG-001: `src/build_jpeg_dataset.py` byte-manifest encoder (2026-11-22)

**Verdict: accept.**

Acceptance criteria:
- **Script runs on synthetic test images; output and manifest saved in the task folder.** Met. The task folder has `pilot/out/` (q90/50/20/10/5) and `pilot/out_levels/` (q95/48/20/1), each with `manifest.csv`, `summary.json` and JPEGs, plus `cli_output.txt`.
  - I checked every manifest row against the file on disk: 80/80 sizes match in `out/` and 64/64 in `out_levels/`.
  - Level→quality mapping is correct: 0→95, 50→48, 80→20, 100→1.
- **Full-file bytes, bpp and aggregate savings computed as in REASON-001 §4.** Met.
  - Per image, `bpp = 8·jpeg_bytes/(W·H)` and `raw_bytes = channels·W·H` (`build_jpeg_dataset.py:200-204`).
  - In aggregate, `R_existing = Σsource/Σjpeg` and `savings = 100·(1 − Σjpeg/Σsource)` are ratios of totals (`:141-162`), with bpp and size distributions.
  - Headers and payload are split at the first SOS marker (`:70-88`).
  - The test asserts the formulas against the files on disk.
- **`src/compress_jpeg.py` and `src/resnet18_sparse_ae.py` unmodified.** Accepted as reported, but I cannot independently verify it now; see E1-7. The tool imports `_jpeg_pixels` and does not restructure the user's file.
- **Unexecuted checks labeled.** Met ("Unexecuted / limits": real dataset, attainable rates, `optimize=False`, progressive or abbreviated streams, timing).

**Re-ran tests.** The test uses `tempfile.mkdtemp`, so I pointed `TMP`/`TEMP` at scratch and ran `-B`. Results:
- **7/7 pass** under `py` 3.14 / Pillow 12.1.0 (2.8 s);
- **7/7 pass** under `ai360` 3.12 / Pillow 13.0.0 (1.6 s).

No `__pycache__` was created in `src/` or the artifact folder, and the temp directory was empty afterwards.

**Recomputed the report's table** from `pilot/out/manifest.csv`. Mean header, payload and file bytes and R_raw for 32×32 q90/q5 and 64×64 q50/q5 match exactly:
- 32×32: 310/352/662/4.6 at q90 and 274/37/310/9.9 at q5;
- 64×64: 298/340/638/19.3 at q50 and 275/62/337/36.4 at q5.

The `cli_output.txt` totals match `summary.json`. RGB header range is 271–329 B; the report says "about 275–325".

The arithmetic is also correct: 3072/50 = 61 B, 12288/50 = 246 B, and √(275·50/3) ≈ 68 px. Later use confirms the tool works at scale: GRID-001 ran it on 50k real images × 7 levels.

### Findings

**E1-1 (optional). The 64×64 impossibility is stated more broadly than measured.**
- **Location:** `report.md` "Finding that affects the 50:1 target", first bullet.
- **Evidence:** the measured header (271–329 B) is for Pillow's standard layout: JFIF APP0, two DQT tables and four optimized DHT tables. A hand-built baseline JPEG could drop APP0, share one quantization table and share Huffman tables. Its header could be roughly 190 B, which would leave about 55 B of payload under the 246 B budget at 64×64. The q5 payload measured 62 B. So "cannot be reached" is firm at 32×32 (61 B is less than any color header) but only holds at 64×64 for this encoder's standard headers. The "Not checked" list does cover alternative stream options.
- **Implication:** low now, since the dataset is ImageNet at native resolution. But the "impossible at 32/64 px" phrasing has propagated into team notes.
- **Repair:** in any reuse, say "with this encoder's standard headers" for 64×64.

**E1-2 (optional). Labels, manifest and filesystem overhead are not stated.**
- **Location:** `build_jpeg_dataset.py:24-27` docstring; `summary.json`.
- **Evidence:** REASON-001 §4 asks to "state whether unchanged labels/manifests are included". The summary does not say so. Also, `jpeg_bytes` is the logical file size (`len(encoded)`), but the docstring calls it "the complete file size on disk". Allocated size on disk differs for small files (cluster or MFT-resident storage).
- **Implication:** the figures are correct as logical full-file bytes, but the definition is underspecified for a storage claim.
- **Repair:** add one line to the summary and docstring: "logical file bytes; labels via folder names; manifest, directory and filesystem allocation overhead not counted".

**E1-3 (optional). `R_existing` mixes stripped metadata with recompression.**
- **Location:** `aggregate()` (`:141-162`).
- **Evidence:** variants are metadata-stripped, but the sources keep theirs, so `R_existing` includes the bytes saved by stripping. REVIEW-002 (G1-3) had to add this caveat to GRID-001.
- **Repair:** optionally record each source's metadata bytes, or add a "strip-only" re-encode at source quality, so the two effects can be separated.

**E1-4 (optional). Pillow version is recorded per run, not per row.**
- **Location:** `MANIFEST_FIELDS` (`:56-60`).
- **Evidence:** the spec lists the Pillow version among the per-image manifest fields. It is recorded once in `summary.json` (`pillow_version`). That is adequate because one run uses one Pillow version.
- **Repair:** none needed, or add a column.

**E1-5 (optional). One test claim is overstated.**
- **Location:** `report.md` "Metadata is stripped ... A test confirms the source had all three and the outputs have none".
- **Evidence:** `test_build_jpeg_dataset.py:61-62` asserts only that the outputs lack `icc_profile`, `exif` and `comment`. No test asserts the source has them. I checked directly: `pilot/source/train/cat/oriented_jpeg_source.jpg` carries all three and EXIF orientation 6, and the outputs carry only JFIF keys. So the property holds, but the test's coverage is overstated.
- **Repair:** add a source-side assertion to the test, or reword.

**E1-6 (optional). The CMYK path is mislabeled (latent).**
- **Location:** `:54-55`, `:129-131`, `:193`, `:205`.
- **Evidence:** `_jpeg_pixels` passes CMYK through unchanged. For a 4-layer JPEG, `JpegImagePlugin.get_sampling` returns −1, which the tool labels "none (grayscale)". `raw_bytes` then uses 4 channels. The GRID-001 manifest has no CMYK sources (343,707 RGB and 6,293 L rows), so no reported number is affected.
- **Repair:** map −1 by mode ("n/a (CMYK)"), and decide whether CMYK raw bytes should count 3 or 4 channels.

**E1-7 (optional). The "unmodified user files" evidence is truncated.**
- **Location:** `report.md` "User files untouched" line.
- **Evidence:** only truncated hashes are given (`674F290B…3C8F61`, `B05E8448…621906`). The user legitimately revised both files on 2026-11-23, and the current hashes are `46805bf2…76590` and `0982737c…f22c`. The 2026-11-22 state therefore cannot be re-verified. Nothing suggests ENG-001 edited them: its write paths and ledger record cover only `src/build_jpeg_dataset.py`.
- **Repair:** for future tasks, save full before/after hashes to a file, as ENG-005 did with `src_hashes_before.txt`.

Also noted, not scored: `summary.json` records an absolute `source_root` that includes the local Windows user profile path. The repository now has a GitHub remote. Consider recording a project-relative path.

---

## 5. Checks actually run

| Check | Command or method | Result |
| --- | --- | --- |
| Ledger hash integrity | `py ai_harness/harness.py check` (before and after recording verdicts) | `{"ok": true, "stale": []}` both times |
| Submitted hashes by hand | `sha256sum` on all 12 registered artifacts of the four tasks | All match the ledger, including `DECISIONS.md` and `RESEARCH_STATUS.md`, which are therefore unchanged since 2026-11-22 |
| ENG-001 tests | `py -B -m unittest discover -s ai_harness/artifacts/ENG-001 -p test_build_jpeg_dataset.py -v`, and the same with `ai360\python.exe -B`; `TMP`/`TEMP` set to scratch | 7/7 OK under both; no `__pycache__` in `src/` or artifacts; scratch temp empty afterwards |
| ENG-001 pilot table | Python recomputation from `pilot/out/manifest.csv`; file sizes vs manifest for `out/` and `out_levels/` | Table reproduced exactly; 80/80 and 64/64 sizes match |
| ENG-001 metadata | Pillow inspection of the pilot source JPEG and q90/q5 outputs | Source has ICC, EXIF (orientation 6) and comment; outputs have JFIF keys only; size rotated to 32×40 |
| ORIENT-002 pixel identity | Scratch script: `make_synthetic` → `build` at levels 0/20/50/80/100 → compare with `model_image` and `write_jpeg` sizes | 80/80 identical; byte delta 0–200 B; both Pillow 12.1.0 and 13.0.0 |
| ORIENT-001/002 code claims | Read `compress_jpeg.py`, `resnet18_sparse_ae.py`, `sparse_workflow.py`, `image_data.py`, the relevant `main.py` sections and `src/README.md` | All cited line ranges and behaviors confirmed (§1, §2) |
| File-map facts | `ls` of the project and `ai_harness`; `references.bib` keys; proposal placeholders | Confirmed; only `ai_harness/.gitignore` missing |
| CMYK exposure | `awk` tally of `runs/GRID-001/jpeg_variants/manifest.csv` (read-only) | No CMYK sources; 899 grayscale images × 7 levels |

## 6. What I could not check

- **The 2026-11-22 text of `compress_jpeg.py` and `resnet18_sparse_ae.py`.** They were revised on 2026-11-23, and no earlier copy exists. ORIENT-001 claims were checked against the current files, and ENG-001's "unmodified" claim against the ledger record only.
- **A diff of `DECISIONS.md` and `RESEARCH_STATUS.md` against their 2026-11-05 versions.** No earlier copy or hash exists.
- **The user's "adopt it" statement.** I cannot see the conversation; it is accepted as recorded.
- **`main.py check` and any torch path.** Not needed for these tasks, and not run, in keeping with the no-GPU, no-training scope.
- **Whether a minimal hand-built JPEG reaches 50:1 at 64×64 (E1-1).** This is an estimate from JPEG segment sizes, not a measurement.
