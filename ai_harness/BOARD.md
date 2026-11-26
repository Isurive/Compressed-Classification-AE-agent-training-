# Research task board

Generated: 2026-11-25T07:18:55+00:00

Snapshot only. Run `py ai_harness/harness.py status` for current state.

| Task | Role | State | Owner | Dependencies | Title |
| --- | --- | --- | --- | --- | --- |
| LIT-001 | literature | done | paper-miner-bootstrap | — | Audit closest compressed-training and dictionary-learning papers |
| REASON-001 | reasoning | done | reasoner-bootstrap | — | Design the minimum controlled training-storage experiment |
| REPRO-001 | engineer | pending | — | — | Audit reusable model implementations and pilot requirements |
| REVIEW-001 | reviewer | pending | — | LIT-001, REASON-001 | Independently challenge evidence and experimental reasoning |
| SYNTH-001 | coordinator | pending | — | LIT-001, REASON-001, REPRO-001, REVIEW-001 | Integrate a concrete month-sized project recommendation |
| ORIENT-001 | engineer | review | claude-orient-20261122-a | — | Codebase orientation and audit of untracked src/ code |
| COORD-001 | coordinator | review | claude-orient-20261122-a | — | Record user adoption of src/ as the starting implementation |
| ENG-001 | engineer | review | claude-orient-20261122-a | — | Dataset-level JPEG encoding with a byte manifest |
| LIT-002 | literature | pending | — | — | Verify the Diverse Dictionary Learning citation used in src/ |
| ORIENT-002 | engineer | review | claude-orient-20261122-a | — | Audit the user's expanded src/ workflow (main.py, sparse_workflow.py, image_data.py) |
| ENG-002 | engineer | review | claude-orient-20261122-a | — | Install PyTorch environment and run main.py check |
| ENG-003 | engineer | review | claude-orient-20261122-a | — | Download ImageNet-1K val and ImageNetV2 matched-frequency into Datasets/ |
| ENG-004 | engineer | review | claude-orient-20261122-a | — | First real training run: timing and sanity check on the 45k/5k ImageNet-val split |
| ENG-005 | engineer | review | claude-orient-20261122-a | — | Conventional baseline head and dense control, compared with the sparse run |
| REASON-002 | reasoning | review | claude-orient-20261122-a | — | Plan the main compressed-training experiment grid |
| GRID-001 | engineer | done | claude-orient-20261122-a | — | Run the REASON-002 experiment: tuning, main grid, M sweep, byte accounting |
| GRID-002 | engineer | done | claude-orient-20261122-a | — | Xavier-init grid: heads learned without the inherited ImageNet classifier |
| ABL-001 | engineer | done | claude-orient-20261122-a | — | Ablation: is the sparse head's gain over linear from top-k sparsity, the auxiliary loss, or width? |
| REVIEW-002 | reviewer | review | reviewer-20261125-a | — | Independent review of GRID-001, GRID-002 and ABL-001 experiments |
| COORD-002 | coordinator | review | claude-orient-20261122-a | — | Record accepted experiment results (GRID-001, GRID-002, ABL-001) and the ABL-001 ledger-summary erratum |

## Submitted evidence

- **LIT-001 (done)**: `ai_harness/artifacts/LIT-001/report.md`, `ai_harness/artifacts/LIT-001/evidence.json`
- **REASON-001 (done)**: `ai_harness/artifacts/REASON-001/report.md`
- **ORIENT-001 (review)**: `ai_harness/artifacts/ORIENT-001/report.md`
- **COORD-001 (review)**: `ai_harness/DECISIONS.md`, `ai_harness/RESEARCH_STATUS.md`
- **ENG-001 (review)**: `src/build_jpeg_dataset.py`, `ai_harness/artifacts/ENG-001/report.md`, `ai_harness/artifacts/ENG-001/make_synthetic.py`, `ai_harness/artifacts/ENG-001/test_build_jpeg_dataset.py`, `ai_harness/artifacts/ENG-001/pilot/out/manifest.csv`, `ai_harness/artifacts/ENG-001/pilot/out/summary.json`, `ai_harness/artifacts/ENG-001/pilot/cli_output.txt`
- **ORIENT-002 (review)**: `ai_harness/artifacts/ORIENT-002/report.md`
- **ENG-002 (review)**: `ai_harness/artifacts/ENG-002/report.md`, `ai_harness/artifacts/ENG-002/check_cpu.txt`, `ai_harness/artifacts/ENG-002/check_cuda.txt`, `ai_harness/artifacts/ENG-002/check_cuda_pretrained.txt`, `ai_harness/artifacts/ENG-002/bench.py`, `ai_harness/artifacts/ENG-002/bench_output.json`
- **ENG-003 (review)**: `ai_harness/artifacts/ENG-003/report.md`, `ai_harness/artifacts/ENG-003/extract_datasets.py`, `Datasets/DATASETS.md`, `Datasets/_manifests/imagenet-1k-val.csv`, `Datasets/_manifests/imagenetv2-matched-frequency.csv`
- **ENG-004 (review)**: `ai_harness/artifacts/ENG-004/report.md`, `ai_harness/artifacts/ENG-004/train_log.txt`, `ai_harness/artifacts/ENG-004/eval_log.txt`, `ai_harness/artifacts/ENG-004/heldout_source_vs_20.json`
- **ENG-005 (review)**: `src/baseline_head.py`, `ai_harness/artifacts/ENG-005/report.md`, `ai_harness/artifacts/ENG-005/src_hashes_before.txt`, `ai_harness/artifacts/ENG-005/linear_train_log.txt`, `ai_harness/artifacts/ENG-005/dense_train_log.txt`, `ai_harness/artifacts/ENG-005/linear_eval_log.txt`, `ai_harness/artifacts/ENG-005/dense_eval_log.txt`, `ai_harness/artifacts/ENG-005/linear_heldout.json`, `ai_harness/artifacts/ENG-005/dense_heldout.json`
- **REASON-002 (review)**: `ai_harness/artifacts/REASON-002/plan.md`, `ai_harness/artifacts/REASON-002/byte_sample.txt`
- **GRID-001 (done)**: `ai_harness/artifacts/GRID-001/report.md`, `ai_harness/artifacts/GRID-001/analysis.md`, `ai_harness/artifacts/GRID-001/analysis.json`, `ai_harness/artifacts/GRID-001/analyze.py`, `ai_harness/artifacts/GRID-001/pixel_check.py`, `ai_harness/artifacts/GRID-001/pixel_check_output.txt`, `ai_harness/artifacts/GRID-001/bytes_log.txt`, `ai_harness/artifacts/GRID-001/features_log.txt`, `ai_harness/artifacts/GRID-001/tune_log.txt`, `ai_harness/artifacts/GRID-001/grid_log.txt`, `ai_harness/artifacts/GRID-001/msweep_log.txt`, `ai_harness/artifacts/GRID-001/reference_log.txt`, `ai_harness/artifacts/GRID-001/summary_log.txt`, `ai_harness/artifacts/GRID-001/src_hashes_before.txt`, `src/run_grid.py`, `runs/GRID-001/results.csv`, `runs/GRID-001/summary.csv`, `runs/GRID-001/tune_results.json`, `runs/GRID-001/tuned.json`, `runs/GRID-001/reference_original_fc.json`, `runs/GRID-001/jpeg_variants/summary.json`, `runs/GRID-001/jpeg_variants/manifest.csv`
- **GRID-002 (done)**: `ai_harness/artifacts/GRID-002/report.md`, `ai_harness/artifacts/GRID-002/analysis.md`, `ai_harness/artifacts/GRID-002/analysis.json`, `ai_harness/artifacts/GRID-002/analyze.py`, `ai_harness/artifacts/GRID-002/task.json`, `ai_harness/artifacts/GRID-002/freeze_log.txt`, `ai_harness/artifacts/GRID-002/grid_log.txt`, `ai_harness/artifacts/GRID-002/summary_log.txt`, `src/run_grid_xavier.py`, `runs/GRID-002/results.csv`, `runs/GRID-002/summary.csv`, `runs/GRID-002/tuned.json`
- **ABL-001 (done)**: `ai_harness/artifacts/ABL-001/report.md`, `ai_harness/artifacts/ABL-001/analysis.md`, `ai_harness/artifacts/ABL-001/analysis.json`, `ai_harness/artifacts/ABL-001/analyze.py`, `ai_harness/artifacts/ABL-001/task.json`, `ai_harness/artifacts/ABL-001/grid_log.txt`, `ai_harness/artifacts/ABL-001/summary_log.txt`, `src/run_ablation.py`, `runs/ABL-001/results.csv`, `runs/ABL-001/summary.csv`, `runs/ABL-001/settings.json`
- **REVIEW-002 (review)**: `ai_harness/artifacts/REVIEW-002/report.md`, `ai_harness/artifacts/REVIEW-002/recompute/independent_check.py`, `ai_harness/artifacts/REVIEW-002/recompute/independent_check_output.txt`, `ai_harness/artifacts/REVIEW-002/recompute/ABL-001/analyze.py`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-001/analyze.py`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-002/analyze.py`, `ai_harness/artifacts/REVIEW-002/recompute/ABL-001/analysis.md`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-001/analysis.md`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-002/analysis.md`, `ai_harness/artifacts/REVIEW-002/recompute/ABL-001/analysis.json`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-001/analysis.json`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-002/analysis.json`, `ai_harness/artifacts/REVIEW-002/recompute/ABL-001/stdout.txt`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-001/stdout.txt`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-002/stdout.txt`
- **COORD-002 (review)**: `ai_harness/artifacts/COORD-002/status.md`, `ai_harness/artifacts/COORD-002/task.json`
