# REVIEW-004: Independent review of ENG-002, ENG-003, ENG-004, ENG-005 and REASON-002

**Reviewer:** `reviewer-20261125-c` (did not author any of the reviewed work). **Date:** 2026-11-25.

> **Transcription note.** The reviewer's file-write tool was blocked, so it returned this text in its final message. The coordinator session (`claude-orient-20261122-a`, author of the reviewed tasks) saved it here verbatim, with no edits to content, on 2026-11-25.

**Scope:** read-only review. I ran no training, GPU jobs, downloads or installs. The only computation was CPU work: manifest analysis, re-hashing a file sample, checkpoint inspection, split recomputation and a 150-image JPEG re-encode. My verification script is `C:\Users\ericw\AppData\Local\Temp\claude\5a605a89fa322016\2139bfc6-af5b-461d-bfca-a7c21622894d\scratchpad\verify_data.py` (temporary; it is not in the project).
**Basis:** each task is judged against its own acceptance criteria as of submission. Statements that are now outdated count as should-fix only if they would mislead a reader today.

## Summary of verdicts

| Task | Verdict | Blocking | Should-fix | Optional |
| --- | --- | --- | --- | --- |
| ENG-002 PyTorch env and `main.py check` | **accept** | 0 | 0 | 2 |
| ENG-003 ImageNet-1K val and ImageNetV2 download | **accept** | 0 | 0 | 4 |
| ENG-004 first 45k/5k pilot | **revise** | 0 | 1 | 1 |
| ENG-005 linear baseline and dense control | **accept** | 0 | 0 | 4 |
| REASON-002 experiment plan | **accept** | 0 | 1 (already remedied downstream) | 2 |

`py ai_harness/harness.py check` returned `{"ok": true, "tasks": 23, "stale": []}`, so every submitted artifact hash still verifies.

---

## ENG-002: PyTorch environment and `main.py check`. Verdict: accept

The acceptance criteria are met:
- The env name, Python, torch, torchvision and CUDA versions are recorded.
- The three `check` outputs are saved verbatim.
- No `src/` edits are evident.
- The one download, the pretrained ResNet-18 weights, is stated.

**Verified:**
- Running the env's own Python gives Python 3.12.16, torch 2.11.0+cu128, torchvision 0.26.0+cu128, Pillow 13.0.0, numpy 2.5.4, cuDNN 91900, `cuda.is_available()` True, and "NVIDIA GeForce RTX 3080 Ti Laptop GPU". All of these match the report's table.
- Conda-meta shows all 20 conda packages came from `conda.anaconda.org/conda-forge`, which supports the "conda-forge only" claim.
- `resnet18-f37072fd.pth` is in the torch hub cache and is 46,830,571 bytes, which matches "46.8 MB".
- `check_cpu.txt`, `check_cuda.txt` and `check_cuda_pretrained.txt` each contain `"result": "PASS"`, `max_active: 32`, `frozen_backbone: true`, `parameters_unchanged: true`, `decoder_free_reload_matches: true` and `exit 0`.
- `bench_output.json` matches the throughput table: 7.73 ms/img, 129.4 img/s, 2,657 img/s, 901 MiB, 126.9 steps/s and 215 MiB.
- `bench.py` measures what the report says it measures. It uses synthetic inputs, and the report states this.

**Findings:**
- **E2-1 (optional).**
  - *Location:* the end of each `check_*.txt` file.
  - *Evidence:* `bc: command not found` / `wall  s`.
  - *Implication:* cosmetic only. The report explains it and gives the exit codes.
  - *Repair:* none needed.
- **E2-2 (optional).**
  - *Location:* report section "Practical recommendations".
  - *Evidence:* "Keep ImageNet data outside the OneDrive folder". The user later chose `Final Project/Datasets` (ENG-003).
  - *Implication:* the advice is outdated but labeled as a recommendation, so it does not mislead.
  - *Repair:* none needed.

## ENG-003: dataset download. Verdict: accept

All four acceptance criteria are met and independently verified. This was a sample-based check, as instructed.

| Check | Result |
| --- | --- |
| Manifest rows | val 50,000; V2 10,000 |
| Manifest byte sums | val 6,706,296,736; V2 1,244,630,473. Both match the report and DATASETS.md exactly. |
| Files on disk (folder walk) | val: 50,000 files in 1,000 folders, exactly 50 in each. V2: 10,000 files in 1,000 folders, exactly 10 in each. No stray top-level files. |
| Disk path set = manifest path set | true for both datasets |
| Random sample of 200 per dataset (seed 20261125): re-hash from disk | 200/200 SHA-256 and 200/200 sizes match the manifest for each dataset. 200/200 have JPEG magic bytes. 200/200 decode fully with Pillow. |
| Val labels vs the official table | 0 of 50,000 mismatches. Each `ILSVRC2012_val_N` folder WNID was compared with row N−1 of `src/imagenet1k_val_labels.txt.gz`. IDs cover 1…50,000 exactly. |
| WNID order and V2 folders | `src/imagenet1k_wnids.txt` is sorted (canonical order). V2 folders are exactly 0…999. |
| ImageNetV2 vs the pool, byte-identical overlap | **0** of 10,000 V2 SHA-256s appear in the val manifest. This is a full manifest comparison, made trustworthy by the 200-file disk samples. |
| Duplicates within V2 | 0 |
| Duplicates within the val pool | 29 SHA-256 values occur more than once, and all 29 carry conflicting labels (see E3-1) |
| Split compatibility | `image_data.training_splits(…, holdout 0.1, seed 42)` gives 45,000 / 5,000. ceil(45,000/128) = 352 batches; × 10 = 3,520 steps. This matches the `--plan-only` output in the report. |

The extractor (`extract_datasets.py`) copies bytes without decoding or re-encoding. It rejects unsafe names, non-JPEG data, duplicate names and WNID/label disagreements. That supports the claim that bytes were written unchanged from the source archives. The manifest hashes were computed from the in-memory source bytes, and the files on disk match them.

**Findings:**
- **E3-1 (optional).**
  - *Location:* `Datasets/DATASETS.md` and the ENG-003 report.
  - *Evidence:* 29 groups of byte-identical val images have conflicting labels. This is a known property of ImageNet val. Neither file mentions it. GRID-001 `report.md` line 147 and `WORK_SUMMARY.md` do record it (3, 3 and 7 groups span train and validation for seeds 42, 43 and 44).
  - *Implication:* AGENTS.md asks that variants of one image not leak across splits. The effect is negligible (validation-based selection only; ImageNetV2 is unaffected), but the dataset README is where a future user would look.
  - *Repair:* the coordinator could add one line to DATASETS.md under a scoped task. ENG-003 does not need a revision.
- **E3-2 (optional).**
  - *Location:* DATASETS.md, layout `imagenetv2-matched-frequency/<0-999>/<sha1>.jpeg`.
  - *Evidence:* in 0 of 200 sampled files does the file name equal the SHA-1 of the file bytes.
  - *Implication:* the name is the upstream identifier, not a content hash. Someone could mistake it for an integrity check.
  - *Repair:* optionally write "upstream hash-style name". Use the manifest SHA-256 for integrity.
- **E3-3 (optional).**
  - *Location:* the ENG-003 report, "All 50,000 labels match `src/imagenet1k_val_labels.txt.gz`".
  - *Evidence:* the script that ran this comparison is not among the artifacts. `extract_datasets.py` only checks the WNID embedded in the HF file name. I reproduced the claim independently: 0 mismatches.
  - *Repair:* none needed now. Future verifications should save their check script.
- **E3-4 (optional, now resolved).**
  - *Location:* the ENG-003 report, "Byte-identical overlap … was not checked".
  - *Evidence:* this review and GRID-001 both find zero overlap.
  - *Repair:* none needed. The statement is accurate as of submission and does not mislead.

## ENG-004: first 45k/5k pilot. Verdict: revise (one should-fix, no blocking)

**Verified from `train_log.txt`, `eval_log.txt`, the JSON and the checkpoints (loaded read-only with `weights_only=True`):**

| Report claim | Recomputed | OK |
| --- | --- | --- |
| Total 3 min 35 s | 14:35:13 → 14:38:48 = 215 s | yes |
| Startup 8 s | 14:35:13 → 14:35:21 | yes |
| 45k extraction 137 s, about 330 img/s | 14:35:21 → 14:37:38 = 137 s; 45,000/137 = 328 img/s; 328/129.4 = 2.5× ENG-002 | yes |
| 5k extraction 33 s | 14:37:38 → 14:38:11 | yes |
| 10 epochs 36 s | 14:38:11 → 14:38:47 | yes |
| Evaluation 71 s | `exit 0 in 71 s` | yes |
| `best.pt` epoch 8 | `best.pt` epoch = 8. Epoch 8 val top-1 0.5710 is the maximum of the 10 epochs. | yes |
| Level 20 top-1 57.1, top-5 80.2; source 57.6 / 80.2 | JSON 0.5710/0.8022; 0.5760/0.8018 | yes |
| Original fc 69.3/88.7 and 70.0/88.9 | JSON `backbone_*` | yes |
| Level 20 = quality 76 | `round(95 − 94·0.20)` = 76 (`compress_jpeg.py` line 51) | yes |
| Seed 42, M 2048, k 32, he/xavier, lr 1e-3, batch 128, cache memory | `last.pt` settings | yes |
| Train top-1 99.8%; val MSE 0.59 → 0.36 | epoch 10 0.99836; 0.5865 → 0.3554 | yes |
| ImageNetV2 untouched | both commands use `--data ../Datasets/imagenet-1k/val` only | yes |

The report clearly labels itself a single-seed pilot with no tuning and says it is not a research result.

**Findings:**
- **E4-1 (should-fix).**
  - *Location:* the ENG-004 report, section "Held-out results (5,000 images, single seed, pilot)", and the submit summary ("Held-out top-1 57.6% source / 57.1% level 20").
  - *Evidence:* I recomputed `evaluation_split` from the checkpoint protocol. The evaluated 5,000 images are **exactly** the validation set used to choose `best.pt`: set equality is true, the overlap with the 45k training paths is 0, and there are 5 images per class. `best.pt` is the argmax of validation top-1 over 10 epochs on those images. The level-20 "held-out" top-1 (0.5710) is identical to the epoch-8 selection score. The report never says the reported split was also the selection split.
  - *Implication:* the numbers are slightly optimistic. Validation top-1 ranged from 55.0 to 57.1% over epochs 2–10, so the bias is probably at most about 1–2 points. "Held-out" without that caveat overstates their independence. ENG-005 §5 and REASON-002 §4 later disclose this bias for this exact run, but ENG-004 is cited on its own (for example in `WORK_SUMMARY.md` step 1), and the review brief asks each pilot report to state it.
  - *Repair:* add one sentence or an addendum file to `ai_harness/artifacts/ENG-004/`: "The 5,000 held-out images are also the validation set used to select `best.pt` (epoch 8 of 10), so these numbers are selection-biased and slightly optimistic. See ENG-005 §5." Then resubmit. No rerun is needed.
- **E4-2 (optional).**
  - *Location:* Diagnostics, "99.7% of atoms used".
  - *Evidence:* the `best.pt` epoch 8 value is 99.76%. The final epoch is 99.66%, and the source evaluation is 99.85%.
  - *Implication:* negligible.
  - *Repair:* optionally say which epoch the figure comes from.

## ENG-005: baseline head and dense control. Verdict: accept

**Acceptance criteria:**
- **Existing `src/` files unmodified.** `src_hashes_before.txt` lists 7 files. Today's `sha256sum` matches all 7 exactly (checked 2026-11-25, after GRID-001/002 and ABL-001).
- **Identical split digest.** Loading `runs/ENG-004/he_45k/best.pt`, `runs/ENG-005/dense_k2048_45k/best.pt` and `runs/ENG-005/linear_45k/best.pt` read-only gives the same `protocol` for all three: single-pool holdout, holdout 0.1, seed 42, val_limit 0, digest `2bbb411f2b9dcda3f5372d8956415e4bf89d31e6bc9f26f7ef39119dcb4b858f`. Recomputing `training_splits` independently gives the same digest.
- **Same epochs, batch, lr, seed, level and selection rule; differences listed.** Confirmed from the logs, the checkpoint settings and the code:
  - `baseline_head.py` uses the same `torch.manual_seed(seed)` before the split, `seed + epoch` shuffles, `maybe_cache`/`loader`/`batch_features`, the (top-1, −CE) selection rule and the same `evaluation_split`.
  - The dense `best.pt` has atoms = k = 2048.
  - The differences (head architecture and loss: linear uses CE only) are given in the run table.
- **Single-seed pilot.** The report labels it single-seed, and §5 states the selection-on-reported-split bias explicitly.

**Numbers recomputed from the logs and JSON:**
- Best epochs: dense 5 (0.6302, the maximum) and linear 5 (0.6162, the maximum).
- Dense 63.0/63.2, top-5 85.1. Linear 61.6/62.0, top-5 84.8. Matches the JSON.
- Train top-1 at the final epoch: dense 94.9% (peak 99.5% at epoch 7); linear 92.8%.
- Dense active atoms about 829; 897 at epoch 8, 715 at epoch 10.
- Dense − sparse = 5.9 / 5.6 pp. Dense − linear = 1.4 / 1.2 pp. Source → level 20 costs 0.5 (sparse), 0.2 (dense) and 0.4 (linear) pp.
- Run times: 4 min 55 s (linear) and 4 min 46 s (dense).
- The standalone evaluation reproduces the training-time validation score exactly (linear 0.6162, dense 0.6302), which shows the evaluation path is consistent.

**Findings:**
- **E5-1 (optional).**
  - *Location:* the report, "Hashes before match the hashes after for all seven user files".
  - *Evidence:* the seven include `src/build_jpeg_dataset.py` (agent-authored, ENG-001), so six are user files plus one agent file.
  - *Repair:* wording only.
- **E5-2 (optional).**
  - *Location:* Observation 4, "All learned heads stay 7–13 points below the original fc".
  - *Evidence:* dense is 6.3 (level 20) and 6.8 (source) points below.
  - *Repair:* "6–13".
- **E5-3 (optional).**
  - *Location:* `src/baseline_head.py`, `--weight-decay`.
  - *Evidence:* it uses `torch.optim.Adam(weight_decay=…)`, which is coupled L2, not decoupled AdamW.
  - *Implication:* this is not used in ENG-005, but it matters when interpreting the later linear-head weight-decay tuning.
  - *Repair:* document it in the help text if it is reused.
- **E5-4 (optional).**
  - *Location:* the run table and §1 ("isolates the top-k sparsity").
  - *Evidence:* the controls differ in more than sparsity. The linear head has about 0.5M trainable parameters. The dictionary heads have about 3.1M (encoder 512×2048 + fc 2048×1000) plus MSE supervision. The k = M control still applies ReLU, leaving about 830 of 2048 atoms active.
  - *Implication:* the report already says "ReLU (all atoms)" and "about 830 atoms active". The capacity difference is not quantified, which matters for the M/k and capacity confound rule in AGENTS.md. ABL-001 later addresses it.
  - *Repair:* none needed for acceptance.

## REASON-002: experiment plan. Verdict: accept

**Acceptance criteria:**
- **Grid, controls and primary metric explicit and traceable.** §3 gives the heads, traced to ENG-005. §4 gives the protocol, and §7 the primary metric (ImageNetV2-source top-1 against training bytes), with drop and interaction pre-declared as REASON-001 §5 asks. Met.
- **Test set never used for selection.** §4: selection on the 5k validation set; ImageNetV2 is evaluated once after freezing; validation scores are diagnostic only. Met. GRID-001 §"Test set isolation" confirms the execution followed this.
- **Runtime and disk estimates derived from measured pilot timings.** About 3–4 min per level comes from ENG-004/005 (170 s and about 240 s per 50k pass), and about 40 s per head run from ENG-004's 36 s for 10 epochs. Met. GRID-001's actual phase times (features about 31 min, grid about 39 min) fall within the plan's 1.5–2 h total.
- **User decisions separate from defaults.** §8 lists the five decisions with defaults in bold. Met.

**Byte sample spot-check:**
- `byte_sample.txt` is arithmetically consistent. For example, at level 80: 12.6 KiB × 8192 / 205,859 px = 0.501 bpp; 24/0.501 = 47.9; 126.5/12.6 = 10.04.
- I re-encoded 150 random val images (seed 7) with `compress_jpeg.write_jpeg`. Level 20: 1.50 bpp, R_raw 16.0, R_existing 3.15. Level 80: 0.498 bpp, R_raw 48.2, R_existing 9.47. Both agree with the plan's sample (1.53/15.6/3.28 and 0.50/47.9/10.0) within sampling error.
- At level 100 my sample gives R_raw 186 against the plan's 229. A likely reason is that `write_jpeg` keeps ICC profiles, which dominate files of about 3 KiB, while the plan's sample stripped metadata. My sample is also smaller.
- The plan says final numbers come from the full pool. GRID-001's full-pool `bytes_log.txt` gives level 80 = 50.2:1 R_raw and 89.8% saving, which supersedes the sample.

**Findings:**
- **R2-1 (should-fix as of submission; already remedied downstream, so no revision is required).**
  - *Location:* plan §8 decision 1 (the recommended tuning) against §1 ("a **newly trained** head") and §3 ("`--fc-init xavier`").
  - *Evidence:* the recommended tuning grid includes `--fc-init auto`, which folds the original ImageNet fc into the head. The plan does not flag that this adds clean-data (original-ImageNet) supervision to every head and narrows the claim. AGENTS.md requires this kind of extra information to be disclosed.
  - The plan also says "Each model gets a comparable search budget", but it specifies 6 sparse candidates and 3 linear candidates and leaves dense unspecified. Execution used 6, 2 and 6 for sparse, dense and linear (GRID-001 lines 40–44).
  - *Implication:* when executed, tuning chose `auto`/`original` for every model, so GRID-001 mostly measured fine-tuning damage to an inherited classifier.
  - *Why no revision:* GRID-001 lines 11, 17, 141 and 148 disclose both problems prominently, and GRID-002 runs the Xavier alternative. Both are accepted. A reader following the plan → GRID-001 chain is not misled.
  - *Repair (optional):* if the plan is reused for a new grid, state that `auto`/`original` init is clean-data head initialization and changes the claim, and specify per-model budgets.
- **R2-2 (optional).**
  - *Location:* plan §2, "measured on 1,000 random ImageNet-val images (seed 0)".
  - *Evidence:* the script or command that produced `byte_sample.txt` is not among the artifacts.
  - *Implication:* the sample cannot be exactly reproduced. The full-pool measurement in GRID-001 supersedes it, so the impact is low.
  - *Repair:* none needed.
- **R2-3 (optional).**
  - *Location:* plan §6, "ImageNetV2 features: 9 × about 45 s".
  - *Evidence:* ENG-004/005 evaluation took about 35–45 s per 5,000 images, so about 70–90 s per 10,000-image ImageNetV2 pass would be expected.
  - *Implication:* this is a minor underestimate inside a total that held up.
  - *Repair:* none needed.

---

## Checks I actually ran

1. `py ai_harness/harness.py check`: ok, 23 tasks, none stale.
2. `harness.py show` for all five tasks, to get the specs, acceptance criteria and artifact lists.
3. ENG-002: printed the versions with the env's own Python; read the conda-meta channel URLs (all conda-forge); confirmed pyarrow 26.0.1 and huggingface_hub 2.4.0 are present (disclosed in ENG-003); checked the size of the cached `resnet18-f37072fd.pth`; read all six artifacts.
4. ENG-003: wrote `verify_data.py` in my scratchpad and ran it with the env's Python and `-B`.
   - Inputs: the manifest counts and sums, a folder walk, and the disk path set compared with the manifest.
   - Sample: 200 random files per dataset, each re-hashed (SHA-256), size-checked, magic-checked and decoded.
   - Comparisons: all 50,000 val labels against the official table; V2 against the pool by manifest SHA-256; duplicates within each dataset; whether V2 file names are SHA-1 hashes of their bytes.
5. ENG-004/005: recomputed every timing from log timestamps and every reported metric from the logs and JSON.
   - Loaded four checkpoints read-only with `torch.load(..., weights_only=True)` on the CPU to check the epoch, protocol, digest, settings and atoms/k.
   - Recomputed `training_splits` and `evaluation_split` from `src/image_data.py` (file listing only, no images loaded). Confirmed the 45k/5k split, the digest, zero train/val overlap, and that the evaluated set equals the selection set.
6. `sha256sum` of the current `src/*.py` and `README.md` against `ENG-005/src_hashes_before.txt`: all seven match.
7. Read `src/baseline_head.py` in full, plus the relevant parts of `sparse_workflow.py`, `resnet18_sparse_ae.py` (`sparse_codes`) and `compress_jpeg.py`.
8. REASON-002: checked the arithmetic in `byte_sample.txt`; re-encoded 150 random val images at levels 20, 80 and 100 on the CPU with `write_jpeg`; compared the results with GRID-001's `bytes_log.txt`; read GRID-001's report to see how the plan was executed.
9. Recorded the five verdicts with `harness.py review`. ENG-002, ENG-003, ENG-005 and REASON-002 are now `done`. ENG-004 returned to `active` under its owner for the E4-1 fix.

## What I could not check

- **Hugging Face provenance.** I did not check the revisions, the parquet file hashes or the ImageNetV2 archive SHA-256, because downloads were out of scope. Byte fidelity to the upstream source is supported by the extractor's design and by the manifest/disk agreement, not by an upstream per-image hash.
- **A full re-hash of all 60,000 files.** By instruction, I checked 400 files.
- **Exact reproduction of accuracies by re-running inference.** That needs GPU-scale feature extraction over 5,000 images. Instead I relied on three things: training-time validation scores equal standalone evaluation scores (an independent code path), checkpoint metadata, and an exact split recomputation.
- **Whether `src/` was edited during ENG-002/003/004, before ENG-005 recorded its "before" hashes.** The current hashes match that list, and no evidence suggests edits.
- **ENG-002's GPU benchmark numbers.** I did not rerun them (no GPU jobs). I checked them only against `bench_output.json`.
- **The plan's 1,000-image byte sample.** It cannot be reproduced exactly because no script was saved. My independent 150-image sample is consistent at levels 20 and 80.
