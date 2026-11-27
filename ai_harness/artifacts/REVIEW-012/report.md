# REVIEW-012: independent review of WRITE-001

**Reviewer:** `reviewer-20261126-o` (did not write WRITE-001). **Date:** 2026-11-26.
**Reviewed:** `ai_harness/artifacts/WRITE-001/report_draft.md` (21,545 B, sha256 906a05c7…dd6a) and `proposed_references.bib` (3,714 B, sha256 7dab6166…fab3), against the WRITE-001 and REVIEW-012 task specs.
**Verdict: revise.** Recorded in the ledger before this report (`review WRITE-001 --verdict revise`; WRITE-001 is back with `claude-orient-20261122-a`). Every fix is a wording fix. **No number in the draft is wrong.** Nine substantive issues need fixing: they are overstatements, an incorrect method statement, an incorrect account of where the learning rates came from, and an inconsistent tolerance box.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of WRITE-001) saved it here verbatim, with no edits to content, on 2026-11-26, after the reviewer's `revise` verdict was recorded in the ledger and before any WRITE-001 file was edited.

## 1. What I checked and how

Everything was read-only. I ran no training, used no GPU and loaded no checkpoints. I used one stdlib script in the scratchpad (`chk.py`) and read `analysis.json` with `py -I`.

| Check | Method | Result |
| --- | --- | --- |
| Mean-accuracy table (7.2), 12 confirmatory tests (7.2 costs, 7.3 comparisons) | Compared with RETEST-001 `analysis.md` and `report.md`, and `analysis.json` (unrounded) | All match. A2's lower bound is +0.0016 pp, so "+0.002" is right. |
| Retention 98.3–98.9% | Recomputed from the means: 98.94 / 98.62 / 98.30 / 98.42% | Correct |
| Gains from the lr change (+1.95; +1.01 to +1.33) | RETEST-001 analysis, exploratory section | Correct |
| SEED-001 +0.97 [0.78, 1.15] 7/7 and +0.41 | SEED-001 report, lines 31–32 | Correct |
| LR-001 instability "4–6 of 6" | LR-001 table: 4/6, 6/6, 6/6 | Correct |
| Linear under-tuned "about 2 pp" | LR-002: +2.14 [1.84, 2.44] | Correct |
| Storage table (7.1) | GRID-001 `analysis.md` "Bytes, seed 42, train45k" and report §4 | Every cell matches: 6,040,232,194 B; 615,578,103 B; 360,039,910; 238,198,791; 131,331,537; bpp; R_raw; savings. The level-80 header size of 14,262,918 B supports "about 14 MB". |
| 3,500/3,500 pixel check; 57.28%; 29 duplicate groups; no ImageNetV2 hash in the pool | GRID-001 report, `pixel_check_output.txt`, REVIEW-004 | Correct, with wording points in O-3 and O-4 |
| Levels 95/100 ranges, 1.0 pp tolerance and its origin | COORD-004 B2 and A; REASON-002 `plan.md` line 76 | Correct |
| JPEG quality formula | `src/compress_jpeg.py` line 49: `round(95 - 94*level/100)`, `optimize=True`. Levels 80, 90, 95, 100 give quality 20, 10, 6, 1. | Correct |
| Loss | `src/sparse_workflow.py` `training_pass`: `mse = F.mse_loss(reconstruction, features)` on the same (compressed) input features, `loss = 1*mse + 1*ce`. RETEST uses `abl-mse` (dependency and coherence 0). | Correct |
| Architecture and parameter counts | `resnet18_sparse_ae.py`, `sparse_workflow.LatentClassifier`, `run_retest.py`, `run_grid.py` (batch 128, holdout 0.10, Adam with wd only for linear) | Linear 513,000. M2048: 3,100,136 (encoder 512×2048 + bias, centre 512, fc 2048×1000 + bias). M4096: 6,198,760. So 0.51 / 3.10 / 6.20 M are correct. Decoder atoms are unit-norm (`F.normalize`) and used only in training. He init on the encoder, Xavier on fc. Top-k then ReLU. |
| Hardware and timings | WORK_SUMMARY §1 (3080 Ti 16 GB, i9-12900H, 32 GB, Py 3.12, torch 2.11+cu128, torchvision 0.26). GRID-001 (31 min for 8 levels × 60k). RETEST (84 runs in 48 min). | Correct, except O-5 |
| Course files unchanged | `git status` / `git diff HEAD` on `AI360_Project_Proposal_Template.tex`, `references.bib`, `projectIdea.md`, `src/` show nothing. SHA-256 of the 8 user `src/` files equals GRID-001 `src_hashes_before.txt`. | Unchanged |
| Placeholders | Names/IDs table, tolerance box, timeline and division (Appendix A) | Present (see O-15 for the template tables) |
| Citations | Every † key is in `proposed_references.bib`. Non-† keys `nayak2023deep`, `lee2026storage`, `maudgalya2021learned` exist in `references.bib`. "Diverse Dictionary Learning" is not cited. The four uninspected entries are named but not characterized. | OK |
| Related work against LIT-001 `report.md` and `evidence.json` | Line-by-line comparison | Mostly faithful. See S-6, O-9, O-10. |

## 2. Substantive findings (must fix)

**S-1. The headline storage result is overstated in the conclusion and the introduction.**
- **Where:** §10 says: "A classifier head trained on them loses on average under 1 pp of ImageNetV2 accuracy (0.53–0.85 pp)."
- **Problem:** this leaves out the qualifier that REVIEW-010 S-1 required in RETEST-001: the sparse heads' confidence intervals reach 1.08–1.17 pp. It also leaves out the frozen-backbone scope and the fact that the 89.8% saving includes stripped metadata.
- **Fix:** replace the first bullet of §10 with: "**Storage.** Storing the training images as JPEG quality 20 shrinks them about 50-fold against raw pixels (50.3:1). That is 89.8% smaller than the original files, a saving that includes metadata stripped on re-encoding. A classifier head trained on them, on frozen ImageNet-pretrained ResNet-18 features, loses on average 0.53–0.85 pp of ImageNetV2 accuracy. The 95% CIs stay below 1 pp for the linear and dense heads but reach 1.08–1.17 pp for the sparse heads, so this is not an equivalence result. At quality 10 (86:1) it loses 1.5–2.2 pp."
- **Introduction:** §2 currently asks "if a model is trained only on heavily JPEG-compressed copies of its training images". Only the head is trained, and the backbone learned from original ImageNet. Replace with: "if a classifier head on top of a frozen pretrained network is trained only on heavily JPEG-compressed copies of its training images".

**S-2. Where the learning rates came from (§6.2) is stated wrongly.**
- **Where:** §6.2 says: "These were chosen on validation data only, in two pre-registered learning-rate checks (LR-001, LR-002). For sparse M2048 the 20-epoch, 1e-4 setting departs from LR-002's 10-epoch choice; this departure is disclosed, and it was never worse on validation."
- **Problem:** LR-002's pre-registered rule adopted **10 epochs for all three dictionary heads**: sparse M2048 at 3e-4, M4096 at 1e-4 and dense at 1e-4. The 20-epoch, 1e-4 setting for every dictionary head is the post-hoc equal-budget proposal from REVIEW-009 S-1 (RETEST-001 caveat 2; COORD-004 B3). So all three dictionary heads depart from LR-002 in epochs, and sparse M2048 departs in lr as well.
- **Fix:** "The linear setting (3e-4, 20 epochs) is the one adopted by the pre-registered validation-only check LR-002. For the dictionary heads, LR-002's rule adopted 10 epochs (sparse M2048 at 3e-4; M4096 and dense at 1e-4). The final 20-epoch, lr 1e-4 setting is a post-hoc equal-budget choice, disclosed before the confirmatory test. It departs from LR-002 in epochs for all three, and in lr for sparse M2048. On validation it was never worse: +0.28 (sparse M2048), +0.02 (M4096) and +0.11 pp (dense)."

**S-3. The M4096 margin is wrong at level 90, and the conclusion brings back withdrawn wording.**
- **Where:** §7.3 says: "The 4096-atom head is the most accurate head at every level, by about 0.6 pp."
- **Problem:** its margin over the next-best head is +0.61 at source and +0.66 at L080. At L090 the next-best head is linear, and the margin is only **+0.29 [−0.02, +0.61]** (RETEST-001 analysis, per-level table). Against linear the margins are +0.96 / +0.69 / +0.29.
- **Fix:** "The 4096-atom head has the highest mean at every level. Pooled over levels it leads the 2048-atom head by 0.63 pp and the linear head by 0.65 pp (7/7 seeds). Its lead over linear shrinks from +0.96 pp at source to +0.29 pp at level 90, where the CI includes 0."
- **Conclusion:** §10 says "Only a larger dictionary is slightly more accurate (+0.6 pp, below our tolerance)." REVIEW-010 S-2 removed "larger dictionary" wording from RETEST-001, because M, k and the parameter count double together. Replace with: "Only the M4096/k1024 head is slightly more accurate (+0.6 pp pooled, below our tolerance). It doubles the available atoms, the active atoms and the parameters at once, so the cause is not identified."

**S-4. The head comparisons drop their qualifiers.**
- §7.3 says: "the sparse-dictionary head (M2048/k512) is no better than a linear classifier." Add: "(+0.02 pp, CI [−0.25, +0.29]: no detected difference; this is not an equivalence test)". COORD-004 B3 includes this qualifier.
- §7.3 says: "Removing sparsity at the same width (dense) changes accuracy by less than 0.1 pp." This states a point estimate as a fact. Replace with: "Sparse minus dense at the same width is +0.09 pp on average (CI [+0.002, +0.17]), not robust to multiplicity. We find no practically relevant effect of top-k sparsity (k = 512 of M = 2048, with the MSE loss)."
- §10 says "At properly tuned learning rates it does not beat a linear classifier". Append "(no detected difference)".

**S-5. The discussion's account of the lr artifact (§8, §10) is stronger than the accepted reports.**
- **Wrong comparison:** §8 says "every head was re-trained at a validation-chosen learning rate with the same 20-epoch budget". The earlier runs used **10 epochs**, so the gains mix the lr change with the 10→20-epoch change (RETEST-001 exploratory section, and caveat 2). Replace with: "When every head was re-trained at a validation-chosen learning rate and a common 20-epoch budget (the earlier runs used 10 epochs) and re-tested on ImageNetV2, every head improved: linear by +1.95 pp and the dictionary heads by +1.0 to +1.3 pp (exploratory; the gain mixes the lr and budget changes). The gap closed because the linear head gained about 0.95 pp more than the sparse head."
- **Overstated framing:** COORD-004 E2 says the sparse head's lr also changed, so the fix is not the linear head's alone. The accepted label is "superseded: lr-conditional", not "false". Change the §8 title to "Discussion: a head advantage that depended on the learning rate". Change §10 "Method" to: "An apparent 1-pp advantage of the sparse head at a shared learning rate (Adam 1e-3, 10 epochs) did not survive validation-chosen learning rates. Mainly, the linear baseline had been under-tuned. A pre-registered re-test on ImageNetV2 confirmed this." Keep the sentence "The earlier measurements were correct as measured…".

**S-6. Part of the "How our work relates" list contradicts LIT-001.**
- **Where:** "We add: complete-file byte accounting, headers included; an original held-out test set (ImageNetV2); pre-registered, multi-seed comparisons; …"
- **Problem:** LIT-001 says Nayak and Friedland "measures storage on disk" ("Nayak already measures disk storage"). The draft itself says this two paragraphs earlier. Lee (preprint) "tests on original images", and Maudgalya "evaluates original test images". LIT-001 did not record whether earlier work used multiple seeds. So presenting these as additions over [nayak2023deep, lee2026storage, maudgalya2021learned] is not supported.
- **Fix:** "Within this setting, our design uses: complete-file byte accounting with headers counted (unlike Lee et al.'s quality-factor proxy and Maudgalya's payload-only rate; Nayak and Friedland also measure on-disk storage); a separately collected, never-compressed test set (ImageNetV2); seed-paired comparisons with the design and decision rule recorded before the runs; and a controlled test of whether an explicit sparse-dictionary head changes the trade-off. The last item is the incremental question; the others are methodological choices, not claimed firsts."

**S-7. The 0.5 pp tolerance consequences (§6.4) are internally inconsistent.**
- **Problem:** the box gives a CI caveat only for the dense and linear costs. Under the pre-registered rule (meaningful = detected and mean ≥ tolerance), all six of A3, A4 and B1–B4 become meaningful, and all are Holm-detected, so all become "findings". But the A3 CI [0.467, 0.796] and the A4 CI [0.450, 0.848] dip below 0.5 just as B3 [0.41, 0.96] and B4 [0.16, 0.89] do. Only B1 [0.52, 1.17] and B2 [0.504, 1.08] stay above 0.5. The box also omits that the exploratory L090 extra costs of the dictionary heads over linear (+0.68, +0.67, and dense +0.53) would exceed 0.5.
- **Fix:** "If you choose 0.5 pp: under the pre-registered rule, six Holm-detected results become meaningful findings. These are the 4096-atom head's advantage over linear (+0.65) and over the 2048-atom head (+0.63), and all four level-80 costs (sparse M2048 0.85, sparse M4096 0.79, dense 0.69, linear 0.53). Only the sparse M2048 and M4096 level-80 costs have CIs entirely above 0.5. The others' CIs dip below it. The exploratory finding that the dictionary heads lose 0.5–0.7 pp more than linear at level 90 would also reach the tolerance. The level-90 results, A1 and A2 are unchanged. Rewrite §7.2, §7.3, §8 and §10, including the 'sub-1-pp' wording, which becomes 'a meaningful loss for every head'."

**S-8. One method statement in Limitations is wrong.**
- **Where:** §9 says: "One codec and one resolution: Pillow JPEG at 224-pixel crops."
- **Problem:** images are JPEG-encoded at native resolution. `build_jpeg_dataset.py` says "Pixels are never resized". `compress_jpeg.model_image` encodes the full image. The bpp and ratios in §7.1 are at native size, and the 224-px resize and crop happen only after decoding (`image_data.ImageView`).
- **Fix:** "One codec and one setting: Pillow 13.0.0 JPEG with 4:2:0 chroma subsampling, encoded at each image's native resolution. The model sees 224-px centre crops of the decoded images. Other codecs, resolutions or encoders were not tested."

**S-9. The 98% target is reinterpreted as met.**
- **Where:** §9 says: "The proposal's stretch target of '98% accuracy at over 50:1' is met only in relative terms (98.3–98.9% retention at 50.3:1)."
- **Problem:** `projectIdea.md` §6 records the user's target as **98% absolute accuracy** and states that absolute 98%, 98% retention and a pp loss "are different criteria". COORD-004 B2 says the target "is not met in absolute terms".
- **Fix:** "The proposal's stretch target of 98% absolute accuracy above 50:1 is not met. At level 80 (50.3:1) the heads reach about 48.4–49.1% on ImageNetV2, and the original classifier reaches 57.28%. A different criterion, retention relative to each head's own uncompressed training, gives 98.3–98.9% at level 80."

## 3. Optional and minor findings

- **O-1 (§2):** "a combination of a few learned 'atoms'". With k = 512 of 2048, a quarter of the atoms are active. Use "a sparse combination of learned 'atoms' (k of M active; here 512 of 2048)".
- **O-2 (§2 Scope, §4):** "the test set is always the original ImageNetV2 images" / "Test images are never compressed". RETEST-001 also scored ImageNetV2 compressed to the training level (secondary, exploratory). Add: "All claims use the original ImageNetV2 images. A secondary score on ImageNetV2 compressed to the training level was recorded but is not used."
- **O-3 (§4):** "29 duplicate images with conflicting labels. They affect validation only." These are 29 *groups* of byte-identical images (REVIEW-004 E3-1). Use the GRID-001 wording: "29 groups of byte-identical images with conflicting labels; a few groups span the train/validation split for each seed, which affects only validation-based checkpoint selection, not the ImageNetV2 test."
- **O-4 (§4):** "A full comparison of file hashes found none of its 10,000 images in our training pool." Use "found no byte-identical copy of any of its 10,000 images in our training pool (near-duplicates were not checked)".
- **O-5 (§5):** "about 330 images/s" comes from ENG-004's run. The GRID-001 extraction (480,000 images in 31 min) works out to about 258 images/s, and it includes JPEG encoding as well as decoding and resizing. Use "about 260 images/s; CPU-bound JPEG encode/decode and resizing" or drop the rate.
- **O-6 (§6.3):** "See section 9 for how this changed our conclusions" should say section 8. "Each experiment was re-derived by a separate reviewer agent" overstates (COORD-004 E3: LR-002's revised text was accepted 46 s after resubmission). Use "independently reviewed, with numbers recomputed from logs, before acceptance".
- **O-7 (§6.3):** LR-001 and LR-002 pre-registered switch rules, not test families, and RETEST's 20-epoch dictionary budget was post hoc (see S-2). Use "the design and decision rule (and, for SEED-001, MSWEEP-001 and RETEST-001, the test family)".
- **O-8 (§6.2):** the "not re-tuned" list omits the MSE weight (1) and the linear weight decay (1e-4), which §9 and COORD-004 D list. Make the two lists match.
- **O-9 (§3, Lee):** "testing on original images" was verified only for arXiv v1. Move it inside the "In the preprint version we inspected…" sentence.
- **O-10 (§3):** `makhzani2014ksparse` was not inspected in LIT-001, but the section header says it describes only inspected works. Either add "(not inspected in our audit)" or say the header covers prior work on compressed training only.
- **O-11 (§3):** Wei et al. is cited by DOI only. If it stays, add a bib entry marked unverified (abstract seen only through a third-party reproduction), or keep it as a footnote.
- **O-12 (§9):** missing limitation: "Storage savings say nothing about GPU memory or training speed; with cached features, training compute is the same at every level" (RETEST-001 caveat 6; AGENTS.md).
- **O-13 (§7.3):** for the L090 extra cost, add REVIEW-010 S-3's caveat: "this should not be read as the linear head being more tolerant: its smaller absolute cost goes with lower source accuracy". "If anything" understates that the two sparse heads' extra cost is detected (uncorrected).
- **O-14 (§8):** "essentially unchanged" should give the numbers: lr 1e-3 gave about 0.5–1.2 pp (L080) and 1.5–2.3 pp (L090); tuned rates give 0.53–0.85 and 1.51–2.19 (RETEST-001 consequences table).
- **O-15 (template):** the template requires ≤4 pages and asks that every table be completed, including the Timeline and Division of Work tables. Add empty placeholder tables with the template's columns to Appendix A, and note that the proposal version must be cut to 4 pages.
- **O-16 (§6.1):** for completeness, the encoder subtracts a learned centre (initialized to the training-feature mean) before projection. The dense head keeps ReLU. The training-only decoder adds 1.05 M (M2048) or 2.10 M (M4096) parameters. The counts given are correct.
- **O-17 (§9):** "roughly 1,300 images per class" should be "about 1,280 on average (1.28M images; ENG-004/ENG-005)".
- **O-18 (draft header):** "every number comes from accepted, independently reviewed artifacts" is not quite true. The hardware and timing figures come from WORK_SUMMARY.md and ENG-004, and the parameter counts from `src/`. Add "or from `src/` and WORK_SUMMARY.md".
- **O-19 (§7.2):** the level-95/100 ranges pool inherited-fc (GRID-001), Xavier (GRID-002) and ABL-001 heads. Say "across the heads tested then".
- **O-20 (§9):** "our rough estimate is about 0.2–0.3 pp" should be "REVIEW-010's rough estimate (an inference, not measured)".

## 4. proposed_references.bib

The file is correctly flagged as verify-before-use, and every † key used in the draft is present. Verification from repository evidence:

- **`tang2020dictionary`:** the title, arXiv ID 2005.10940 and "accepted to TNNLS" note match LIT-001 `report.md` and `evidence.json`. The author list ("Tang, Hao and others") **cannot be verified** from the repository, because LIT-001 recorded no authors. The entry itself already flags this. If the paper is cited, prefer the published TNNLS version once it is verified.
- **`recht2019imagenet`:** the title, venue (ICML) and year 2019 match `Datasets/DATASETS.md` line 21. The author list, PMLR volume 97 and pages 5389–5400 cannot be verified from the repository.
- **`russakovsky2015imagenet`, `he2016deep`, `he2015delving`, `kingma2015adam`, `glorot2010understanding`, `holm1979simple`, `makhzani2014ksparse`:** no repository evidence. They are consistent with my background knowledge, but I could not browse, so they are unverified. The least certain detail is whether `makhzani2014ksparse` should be an ICLR 2014 conference paper or arXiv:1312.5663.
- **Repository evidence and attributions:** nothing in the repository contradicts any entry. I found no invented citation and no misattribution. The usage is also appropriate: ResNet-18 is cited to He 2016, the inits to Glorot and He 2015, Adam to Kingma, Holm to Holm, and the datasets to Russakovsky and Recht.

## 5. Recommendation

Revise, wording only: fix S-1 to S-9. Take the O-items at the author's discretion, though O-2, O-3, O-6 and O-12 are cheap and worth doing. After revision, a re-review should check:
- the new §6.2, §6.4, §7.3, §8 and §10 text against RETEST-001 caveat 2 and the per-level table;
- that §3 now matches LIT-001.

No re-computation is needed. The tables, parameter counts, formula and loss description can stay as they are.

## Addendum: re-review of WRITE-001 revision 1 (2026-11-26, `reviewer-20261126-o`)

**Reviewed:** `report_draft.md` revision 1 (26,923 B, sha256 f2233633…2b918). `proposed_references.bib` is unchanged (sha256 7dab6166…fab3, the same as in the first review).
**Verdict: accept.** Recorded in the ledger (`review WRITE-001 --verdict accept`; WRITE-001 is now `done`).

**Text above:** I confirmed that the text above matches the report I returned. The only addition is the coordinator's transcription note.

**Checks of the substantive fixes:**
- **S-1:** the §2 question now says "classifier head on top of a frozen pretrained network". The §10 storage bullet carries the CI qualifier (1.08–1.17 pp), the frozen-backbone scope and the metadata note. Fixed.
- **S-2:** §6.2 now matches RETEST-001 caveat 2 and LR-002. LR-002 adopted 10 epochs for all three dictionary heads (sparse M2048 at 3e-4, M4096 and dense at 1e-4). The 20-epoch, 1e-4 setting is post hoc, and the validation differences are +0.28 / +0.02 / +0.11. Fixed.
- **S-3:** §7.3 margins match the RETEST-001 per-level table: +0.96 pp at source and +0.29 pp at L090 against linear, with the L090 CI including 0. The pooled +0.63 and +0.65 are also correct. The §10 wording no longer says "larger dictionary". Fixed.
- **S-4:** the "not an equivalence test" qualifier and the sparse − dense figure +0.09 [+0.002, +0.17] are restored. Fixed.
- **S-5:** §8 now says the earlier runs used 10 epochs and that the gain mixes the lr and budget changes. The 0.95 pp difference in gains matches RETEST-001 (0.97 − 0.02). The title and the "Method" bullet are no longer framed as a "false result". Fixed.
- **S-6:** the §3 list is now methodological choices and agrees with LIT-001: Nayak measures disk storage, Lee uses a quality-factor proxy, and Maudgalya excludes headers. Fixed.
- **S-7:** the 0.5 pp box agrees with `analysis.json`. Six results become meaningful (A3, A4, B1–B4). Only the B1 and B2 CIs stay above 0.5, and the L090 extra cost is included. Fixed.
- **S-8:** the method now states native-resolution encoding with 4:2:0 subsampling and 224-px crops taken after decoding, consistent with `build_jpeg_dataset.py`. Fixed.
- **S-9:** the 98% target is stated as not met in absolute terms, consistent with `projectIdea.md` §6 and COORD-004 B2, and retention is presented as a different criterion. Fixed.

**Other checks:**
- **Optional items:** O-1 to O-20 were applied (O-11 deliberately not), and I checked each against its source. The figures are 260 img/s (480k in 31 min), 1.05 M and 2.10 M decoder parameters, 1,280 images per class on average, the old 0.5–1.2 and 1.5–2.3 pp costs, and dense +0.53. All are correct.
- **New problems:** the rewrite changed no number and introduced no new overstatement.
- **Course files:** `git diff HEAD` on the `.tex`, `references.bib`, `projectIdea.md` and `src/` is empty, so they are still unchanged.

**Non-blocking notes:**
- (a) In §3, "not inspected in our audit beyond an abstract" fits Tang et al. But `makhzani2014ksparse` was not inspected at all. Suggested text: "(background; Tang et al. abstract only; Makhzani and Frey not inspected)".
- (b) In §3, the Wei et al. venue "IEEE GRSL" is inferred from the DOI prefix (LGRS). LIT-001 does not state the venue, so it should be verified with the DOI.
- (c) The `proposed_references.bib` verification status in §4 above still applies.
