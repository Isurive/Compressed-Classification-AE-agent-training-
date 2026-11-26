# Current decisions and unresolved choices

Updated: 2026-11-25 (COORD-003); earlier text from 2026-11-22 (COORD-001) is kept below as history. Coordinator-owned integration file. Cite the user, a task report, or a measured run when changing a status. A reviewed report is evidence, not automatic user approval of its suggested design.

## Current state, 2026-11-25 (read this first)

This section supersedes the 2026-11-22 "Adopted implementation" and "Open choices" sections wherever they conflict.

| Item | Status | Provenance |
| --- | --- | --- |
| Entry point | `src/main.py` with `sparse_workflow.py`, `image_data.py`, `compress_jpeg.py` and `resnet18_sparse_ae.py`. Frozen ImageNet ResNet-18 → top-k sparse code (M atoms, k active) → a **new** fc on the code. Loss: CE + feature MSE + dependency + coherence. `best.pt` is chosen by validation top-1, then CE. | User expanded `src/` on 2026-11-23; audit in [ORIENT-002](artifacts/ORIENT-002/report.md), which supersedes parts of [ORIENT-001](artifacts/ORIENT-001/report.md) §4 |
| Training/validation data | ImageNet-1K **val** (50k images) as the pool: 45k train / 5k validation (`--holdout 0.10`); the seed sets the split | The agent downloaded the data on 2026-11-24 with the user's approval and Hugging Face login, into `Final Project/Datasets`, the location the user chose ([ENG-003](artifacts/ENG-003/report.md), accepted) |
| Test data | ImageNetV2 matched-frequency (10k images), **test only**; never used for tuning or checkpoint selection. **Caveat:** its results did inform which follow-up experiment was designed next (GRID-001 → GRID-002 → ABL-001 → SEED-001). Confirm effects under 1 pp with fresh seeds before final claims (COORD-002 "Unresolved"). | `src/README.md`; ENG-003; GRID-001 protocol |
| Environment | conda env `ai360` (Python 3.12, torch 2.11.0+cu128); one RTX 3080 Ti Laptop GPU is sufficient | User approved the install ([ENG-002](artifacts/ENG-002/report.md), accepted) |
| Conventional baseline | `src/baseline_head.py`: a linear head on the same frozen features | [ENG-005](artifacts/ENG-005/report.md), accepted |
| Byte measurement | `src/build_jpeg_dataset.py`: full-file bytes, headers counted, manifest per image | [ENG-001](artifacts/ENG-001/report.md), accepted; full pool measured in GRID-001 §4 |
| Experiment plan | Tune on validation only, then a grid of 8 levels × 3 heads × 3 seeds, an M sweep, and a 1.0 pp tolerance | User approved [REASON-002](artifacts/REASON-002/plan.md) on 2026-11-24 ("tune first", outputs in `Final Project/runs`, include the M sweep) |
| Claim scope | A **head** trained on compressed images over a frozen pretrained backbone. Not learning from compressed data alone. `--fc-init auto`/`original` also inherits the original ImageNet classifier and must be disclosed. | GRID-001 §1, GRID-002 §1 |
| Accepted results | See [COORD-002/status.md](artifacts/COORD-002/status.md). It includes an erratum for stale ledger summaries and corrected numbers in the done reports. | GRID-001, GRID-002 and ABL-001 were accepted in REVIEW-002. COORD-002 revision 2 was accepted on re-review. As of this edit, REVIEW-002 to REVIEW-005 are themselves under cross-review; check `harness.py status` for current states. |
| 98%-at-50:1 target | **Tested; not met in absolute terms.** Level 80 is 50.3:1 against raw pixels, but heads reach about 46–54% top-1 against the original classifier's 57%. Relative to each head's own uncompressed accuracy, retention is about 97.7–98.9%, within seed noise of 98%. | COORD-002 item 6 |
| Pretrained vs random initialization | **Decided for the current experiments:** the backbone is pretrained and frozen. The head is Xavier-initialized in GRID-002, ABL-001 and SEED-001, and initialized from the original classifier in GRID-001 (disclosed). End-to-end or from-scratch training is not done; it would be a separate decision for the user. | GRID-001 §1; GRID-002 §1 |
| Still open | Effect of sparsity (SEED-001 running); effect of M; a weight-decay-tuned wide head; final accuracy tolerance and proposal wording; the citation check (LIT-002) | — |
| Repository | Public GitHub repo created by the user, `Isurive/Compressed-Classification-AE-agent-training-`. Data, checkpoints and features are git-ignored; commits use the GitHub noreply email. | User, 2026-11-25 |

## Confirmed user direction

| Item | Status and provenance |
| --- | --- |
| Course project with three people, approximately one month | User requirement; see projectIdea.md |
| Neural-network component | User requirement |
| Interest in learned dictionaries and number of available atoms | User's chosen research direction |
| Smaller stored training datasets using heavily compressed images | Latest user clarification; preserve this emphasis |
| Incremental contribution acceptable | User explicitly permits similarity to existing work |
| Multi-agent coordination harness in Final Project | User request, 2026-11-05 |
| `src/compress_jpeg.py` and `src/resnet18_sparse_ae.py` are the project's adopted starting implementation | User decision, 2026-11-22: the user wrote both files and said "adopt it". Recorded by COORD-001. See "Adopted implementation" below. |

## Targets and recommendations, not finalized commitments

- *(Superseded 2026-11-25: tested, see "Current state".)* 98% accuracy above 50:1 compression is a user-proposed target whose feasibility is untested.
- JPEG is the current proposed codec. Multiple codecs are not necessary for the initial month-sized scope.
- A common original held-out test set is the recommended main evaluation for compressed-training replacement.
- Available atoms M and active atoms k should be controlled separately.
- Reusing an existing architecture and framework losses/optimizers is recommended and is accommodated by the course template.
- *(Superseded 2026-11-25 for the current experiments: see "Current state".)* Random initialization versus pretrained weights remains unresolved; the supported claim must match the choice. The adopted `src/resnet18_sparse_ae.py` uses a frozen ImageNet-pretrained ResNet-18, so results from it support a downstream/fine-tuning-data claim unless a from-scratch variant is added (2026-11-22).
- The exact originality claim remains provisional. Broad compressed-training/storage ideas already have prior work.

## Open choices

Dataset, native/standardized image resolution, available GPU and compute budget, compression-rate targets that can actually be encoded, dictionary implementation, loss/objective, M/k values, accuracy tolerance, seed count, and final proposal wording.

"Dictionary implementation" now has an adopted starting point (below); its M/k values, losses, and training dataset are still open.

Resolve through a small feasibility pilot after the design and resource needs are concrete. Continue independent literature and reasoning work while missing details remain open. Do not turn every routine implementation choice into an approval gate.

## Adopted implementation (user decision, 2026-11-22). Historical: see "Current state" above for what has since changed

The user authored and adopted `src/`. Code audit: [ORIENT-001](artifacts/ORIENT-001/report.md) section 4. That report was written before authorship was known; its "author unknown" note is superseded by this entry.

**What adoption settles**

- `src/compress_jpeg.py` is the starting JPEG encoder. It is Pillow-based, sets compression by quality factor, and handles one image per call.
- `src/resnet18_sparse_ae.py` is the starting neural dictionary component: a top-k sparse autoencoder with unit-norm atoms on frozen torchvision ResNet-18 pooled features. M (`n_atoms`) and k are set separately, and the frozen original `fc` classifies the reconstruction.
- Agents build on these files instead of proposing a replacement architecture from scratch. REPRO-001 should treat them as the primary route and survey alternatives only as fallbacks or baselines.

**What adoption does not settle (still open)**

- Training dataset and resolution. The script's ImageNet-1K default is a large download and needs a resource plan and the user's OK. A smaller dataset may be chosen.
- Claim scope. The backbone is pretrained and frozen, so the storage claim covers the data used to train the sparse head or fine-tune, unless a from-scratch variant is added.
- Conventional baseline, evaluation protocol, M/k grid, loss weights (cross-entropy is off by default), and checkpoint criterion (currently validation feature MSE).
- The dataset-level compression pipeline and byte manifest. The current tool encodes single images by quality factor.
- The docstring citation "Zheng et al., Diverse Dictionary Learning, arXiv 2604.17568" is unverified. Verify it before it enters the proposal or `references.bib`.

**Editing rule:** the user edits `src/` directly. Agents edit `src/` only under a claimed task whose `write_paths` cover the specific files. Keep a new script in a new file rather than restructuring the user's files without asking.

## Harness decisions

- Local Python standard-library CLI with a SQLite event/task ledger; no paid model API dependency.
- Current agent host handles actual LLM execution and parallel delegation. The CLI handles coordination and durable evidence.
- One ledger writer location on one computer, even though the directory is synced with OneDrive.
- The original proposal files, bibliography, and projectIdea.md are preserved during harness creation.
- Bootstrap literature and reasoning agents used exclusive task directories. Their claims, submissions, and independent coordinator acceptance are recorded in the ledger.

## Reviewed findings, not finalized design choices

- LIT-001 documents prior work on compressed training, ordinary model capacity, and disk storage. Those broad ideas alone are not a sufficient novelty claim. The explicit dictionary-capacity extension remains provisional, with a close sparse-coding paper unresolved.
- Whole-file bytes are essential to this project's storage claim: inspected studies sometimes use a quality-factor proxy or omit JPEG overhead. See the evidence and primary-source links in RESEARCH_STATUS.md.
- REASON-001 supplies a proposed four-condition pilot and controls for M/k, pretraining, clean targets, and comparable evaluation. Acceptance confirms the memo is a useful research handoff; it does not finalize its suggested architecture, dataset, or thresholds.

## Change log

- 2026-11-05: Initialized from the user conversation and current files. No dataset/model/hardware choice finalized; no training results exist.
- 2026-11-05: Literature and reasoning reports completed by separate agents and accepted by the coordinator after artifact/source inspection. Engineering feasibility and further cross-report review are next; final synthesis waits for those tasks.
- 2026-11-25 (COORD-003): added "Current state" at the top.
  - **Recorded:** the expanded `src/` entry point, the datasets, the environment, the baseline, the approved plan, the claim scope (including the inherited classifier), the repository, and a pointer to the accepted results.
  - **Marked superseded:** the 2026-11-22 notes on the dataset, torch, the CE default, the checkpoint criterion, and the single-image encoder.
  - **Revised after review (REVIEW-003 reviewer):**
    - The old 98%-target and initialization lines are marked superseded.
    - The test-reuse caveat is added.
    - ENG-003 is attributed correctly: an agent download, approved by the user.
    - The review states are flagged.
- 2026-11-22: User wrote `src/compress_jpeg.py` and `src/resnet18_sparse_ae.py` and adopted them as the starting implementation (COORD-001). Observed local hardware: one RTX 3080 Ti Laptop GPU with 16 GB; torch is not installed (ORIENT-001 section 3; one member's machine only). Dataset, baseline, and claim scope remain open. Follow-ups queued: ENG-001 (dataset JPEG pipeline and byte manifest) and LIT-002 (verify the arXiv 2604.17568 citation).
