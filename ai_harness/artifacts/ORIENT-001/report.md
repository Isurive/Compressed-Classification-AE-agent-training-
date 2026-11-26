# ORIENT-001: Codebase orientation and audit of untracked `src/` code

**Agent:** `claude-orient-20261122-a` (Claude Code, desktop app). **Role:** engineer. **Date:** 2026-11-22 (local time).
**Status:** Submitted for independent review. Read-only audit. No installs, downloads, training, or edits outside this folder.

Purpose: give any agent joining after 2026-11-22 a quick, accurate picture of what exists, what changed since the 2026-11-05 bootstrap, and what the new `src/` code does and does not yet support. Read this after `AGENTS.md` and `ai_harness/START_HERE.md`.

## 1. What changed since the bootstrap (2026-11-05)

- **New, untracked:** `src/compress_jpeg.py` and `src/resnet18_sparse_ae.py`, last modified 2026-11-22 22:55–22:56. No harness task owns `src/`, and no report, `DECISIONS.md` entry, or `README.md` layout mentions it. **Author unknown** (the user or another agent). Treat `src/` as someone else's active work: do not edit it without the user's say-so or a task that assigns it.
- The ledger has no other activity since 2026-11-05. REPRO-001, REVIEW-001 and SYNTH-001 are still `pending`.
- `DECISIONS.md` and `RESEARCH_STATUS.md` are dated 2026-11-05 and do not yet reflect `src/`.

## 2. File map

| Path | What it is | Status / who may edit |
| --- | --- | --- |
| `AGENTS.md` | Shared rules for all agents (research standards, ownership) | Read first. Coordinator only. |
| `projectIdea.md` | Historical handoff of the planning conversation (2026-11-04) | Historical context. Preserve. |
| `AI360_Project_Proposal_Template.tex` / `.pdf` | Course proposal template, ≤4 pages excl. references | **Still unfilled** (`Group Name [EDIT]`, `Member 1..3`). Edit only under an explicit assignment. |
| `references.bib` | 7 entries: Nayak 2023, Friedland 2020, Lee 2026, Kuchnik 2021, Maudgalya 2021, Pistono 2020, Yang 2024 | User's bibliography. Preserve. |
| `src/compress_jpeg.py` | Single-image JPEG encoder CLI (Pillow) | Untracked new code. See §4. |
| `src/resnet18_sparse_ae.py` | Top-k sparse dictionary autoencoder on frozen ResNet-18 features (PyTorch) | Untracked new code. See §4. |
| `ai_harness/START_HERE.md`, `README.md` | Join prompts and CLI protocol | Reference. |
| `ai_harness/DECISIONS.md`, `RESEARCH_STATUS.md` | Coordinator-owned decisions and synthesis | Coordinator only. |
| `ai_harness/harness.py`, `test_harness.py`, `VALIDATION.md` | Ledger CLI (stdlib + SQLite) and its tests | Infrastructure. |
| `ai_harness/tasks.json` | Bootstrap seed tasks only, not the live board | Use `add` for new tasks. |
| `ai_harness/state.sqlite3` | **Live ledger** (gitignored) | Write only through the CLI. |
| `ai_harness/BOARD.md`, `state.snapshot.json` | Generated exports, may be stale | Regenerate with `snapshot`. Never hand-edit. |
| `ai_harness/roles/*.md`, `templates/*.md` | Role instructions and report templates | Reference. |
| `ai_harness/artifacts/LIT-001/` | Literature audit + `evidence.json` (accepted) | Done. |
| `ai_harness/artifacts/REASON-001/` | Experiment design memo (accepted) | Done. |
| `ai_harness/artifacts/ORIENT-001/` | This report | Submitted. |

## 3. Local environment (observed 2026-11-22, read-only commands)

| Check | Command | Result |
| --- | --- | --- |
| Python | `py --version` | Python 3.14.2 |
| Packages | `importlib.util.find_spec` | `PIL` yes, `numpy` yes, **`torch` no, `torchvision` no** |
| GPU | `nvidia-smi --query-gpu=...` | NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16384 MiB, driver 610.60 |
| CPU / RAM | `Get-CimInstance` | Intel Core i9-12900H, 31.7 GB RAM |
| Syntax | `py -m py_compile src/*.py` | Both files compile (this does not import torch). I deleted the `src/__pycache__` this created. |

**Unexecuted:** no import or runtime test of `src/resnet18_sparse_ae.py` (torch missing). No check of CUDA/PyTorch wheel compatibility with Python 3.14. A separate venv with a supported Python version may be simpler. Installing torch is a new dependency and needs the user's go-ahead (AGENTS.md).

This is **one team member's machine**. Other members' hardware and any course or cloud GPU are still unknown. This answers part of the REPRO-001 "hardware information needed" question, but it is not a team-wide compute decision.

## 4. Audit of `src/`

Labels: **[code]** means read directly from the source. **[inference]** means my reasoning, still to be checked. Nothing here was run.

### 4.1 `src/compress_jpeg.py`

- **[code]** Maps `--compression-level 0..100` linearly to Pillow JPEG quality `round(95 − 0.94·level)` (0 → q95, 100 → q1). Uses `optimize=True`. Applies EXIF transpose. Flattens alpha onto white. Converts 16-bit gray to 8-bit. Keeps the source ICC profile when the color mode is unchanged. Writes atomically through a temp file.
- **[code]** Processes one image per call. Prints input → output byte counts. Writes no manifest or per-image record.
- **Gaps against the storage protocol (AGENTS.md; REASON-001 §4):**
  1. Compression is set by **quality factor**, not a byte or bpp target. That is acceptable as a first grid, but every result must report the measured full-file bytes and bpp, never the "level".
  2. No dataset-level batch mode. No manifest recording image ID, label, split, W×H, mode, source bytes, JPEG bytes, bpp, quality, or encoder version. Nothing computes aggregate `R_existing` / savings.
  3. **[inference]** The ICC profile and EXIF handling can add or drop metadata bytes. For a storage claim, metadata policy should be explicit and identical across conditions (for example, always strip it).
  4. Chroma subsampling is left at Pillow's default and not recorded. It should be fixed and reported.
  5. **[inference]** Re-encoding a source that is already JPEG creates a compression chain. REASON-001 §4 says to generate every variant from one reference representation.

### 4.2 `src/resnet18_sparse_ae.py`

What it is **[code]**:
- `SparseDictionaryAE`: a linear encoder gives M scores. Top-k keeps k, then ReLU, so the codes are nonnegative with **at most k** active per image (fewer if some top-k scores are ≤0). A decoder with unit-norm atoms reconstructs, and there is an optional nonlinear residual decoder. A learnable `center` offset is included. M (`n_atoms`) and k are separate arguments, which satisfies the "M ≠ k" standard.
- Extra losses: `dependency_loss` (an L1/L2 sparsity proxy on decoder Jacobian columns) and `coherence_loss` (the largest squared cosine between atoms).
- `ResNet18SparseAE`: an ImageNet-pretrained ResNet-18, **frozen** (BatchNorm in eval mode). The SAE reconstructs the 512-d pooled feature, and the **frozen original `fc`** classifies the reconstruction. Optionally, a trainable copy of `layer4` is added.
- Training (`main`): ImageNet-1K ImageFolder, loss = feature MSE + 0.1·dependency + 0.05·coherence. **Cross-entropy is off by default** (`--ce-weight 0`). The checkpoint is chosen by best validation **feature MSE**, not accuracy. The validation set's labels are read through ImageFolder, and val is used for checkpoint selection.
- The docstring cites "Zheng et al., Diverse Dictionary Learning, https://arxiv.org/abs/2604.17568". **This citation is unverified.** It is not in `references.bib` or LIT-001. A literature agent should confirm it exists and says what the code claims before it is cited.

How it fits the project question **[inference, needs review]**:
1. **It is a genuine dictionary component.** It has an explicit D, sparse codes, a reconstruction role, and separate M and k. That meets the AGENTS.md bar better than a CNN-width change would.
2. **It is not yet a compressed-training experiment.** Nothing connects it to `compress_jpeg.py` or to stored JPEG training files. It trains on whatever ImageFolder contains.
3. **Pretraining narrows the claim.** The backbone and `fc` are frozen ImageNet weights learned from the original images. Training this SAE on compressed images supports at most a downstream claim like "the sparse head can be fit on compressed data". It does not show that a classifier can learn from compressed data alone (AGENTS.md; REASON-001 §6.6).
4. **Clean-target check.** The reconstruction target is the frozen backbone's feature of the *same input batch*. If that batch is compressed, the target comes from compressed pixels, which is allowed. However, `--finetune-layer4` trains `layer4` toward the frozen `layer4` output. That is fine for compressed inputs, but any variant that computes targets from originals would add hidden clean supervision and must be disclosed.
5. **The dataset is out of scope for a month.** ImageNet-1K (~1.28M train images, roughly 140+ GB) is a large download and needs a resource plan under AGENTS.md. REASON-001 and LIT-001 point to smaller datasets (CIFAR-10, Tiny ImageNet). At 32×32, the frozen ImageNet ResNet-18 transforms resize to 224, so storage at native resolution and the model's input resolution must be reported separately.
6. **The baseline is missing.** REASON-001 §2 needs a conventional classifier trained under the same data conditions (original vs. compressed) with a comparable budget. The default evaluation compares reconstructed-feature accuracy against the frozen model's own accuracy. That measures SAE fidelity, not the compression-storage trade-off.
7. **Selection criterion.** Picking checkpoints by validation feature MSE while reporting top-1 is a choice the team should make deliberately. Whatever is chosen, the final test set must be separate from the selection split. ImageNet "val" is commonly used as the test set.

## 5. Suggested next steps (proposals, not decisions)

1. **Ask the user** who wrote `src/` and whether it is the chosen model route. If yes, the coordinator should record that in `DECISIONS.md` with provenance and create a task that owns `src/`.
2. **REPRO-001 remains ready.** It can now take this machine's GPU facts as an input. It should still check DDLCN and any other dictionary implementation, and propose at most two dataset/resolution options.
3. A small **ENG data-pipeline task**: batch-encode a fixed split into stored JPEG files with a manifest (ID, label, split, dims, source bytes, JPEG bytes, bpp, quality, subsampling, metadata policy), plus aggregate savings. It could reuse `compress_jpeg.py` logic. It needs a chosen dataset first.
4. A **LIT task** to verify arXiv 2604.17568 (Zheng et al.) before anyone cites it.
5. REVIEW-001 and SYNTH-001 are unchanged and still pending.

## 6. Completion evidence

Files read: everything in §2 except `harness.py` internals, the PDFs, and the full `LIT-001/evidence.json`. Commands run: those in §3, plus `harness.py status/check/add/claim`. I modified nothing outside this folder, except deleting the `src/__pycache__` directory that my syntax check created. I did not edit `src/`, the proposal, the bibliography, `projectIdea.md`, or the shared decision files.
