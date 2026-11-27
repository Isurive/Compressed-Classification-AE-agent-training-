# Research task board

Generated: 2026-11-27T22:50:08+00:00

Snapshot only. Run `py ai_harness/harness.py status` for current state.

| Task | Role | State | Owner | Dependencies | Title |
| --- | --- | --- | --- | --- | --- |
| LIT-001 | literature | done | paper-miner-bootstrap | — | Audit closest compressed-training and dictionary-learning papers |
| REASON-001 | reasoning | done | reasoner-bootstrap | — | Design the minimum controlled training-storage experiment |
| REPRO-001 | engineer | pending | — | — | Audit reusable model implementations and pilot requirements |
| REVIEW-001 | reviewer | pending | — | LIT-001, REASON-001 | Independently challenge evidence and experimental reasoning |
| SYNTH-001 | coordinator | pending | — | LIT-001, REASON-001, REPRO-001, REVIEW-001 | Integrate a concrete month-sized project recommendation |
| ORIENT-001 | engineer | done | claude-orient-20261122-a | — | Codebase orientation and audit of untracked src/ code |
| COORD-001 | coordinator | done | claude-orient-20261122-a | — | Record user adoption of src/ as the starting implementation |
| ENG-001 | engineer | done | claude-orient-20261122-a | — | Dataset-level JPEG encoding with a byte manifest |
| LIT-002 | literature | pending | — | — | Verify the Diverse Dictionary Learning citation used in src/ |
| ORIENT-002 | engineer | done | claude-orient-20261122-a | — | Audit the user's expanded src/ workflow (main.py, sparse_workflow.py, image_data.py) |
| ENG-002 | engineer | done | claude-orient-20261122-a | — | Install PyTorch environment and run main.py check |
| ENG-003 | engineer | done | claude-orient-20261122-a | — | Download ImageNet-1K val and ImageNetV2 matched-frequency into Datasets/ |
| ENG-004 | engineer | done | claude-orient-20261122-a | — | First real training run: timing and sanity check on the 45k/5k ImageNet-val split |
| ENG-005 | engineer | done | claude-orient-20261122-a | — | Conventional baseline head and dense control, compared with the sparse run |
| REASON-002 | reasoning | done | claude-orient-20261122-a | — | Plan the main compressed-training experiment grid |
| GRID-001 | engineer | done | claude-orient-20261122-a | — | Run the REASON-002 experiment: tuning, main grid, M sweep, byte accounting |
| GRID-002 | engineer | done | claude-orient-20261122-a | — | Xavier-init grid: heads learned without the inherited ImageNet classifier |
| ABL-001 | engineer | done | claude-orient-20261122-a | — | Ablation: is the sparse head's gain over linear from top-k sparsity, the auxiliary loss, or width? |
| REVIEW-002 | reviewer | done | reviewer-20261125-a | — | Independent review of GRID-001, GRID-002 and ABL-001 experiments |
| COORD-002 | coordinator | done | claude-orient-20261122-a | — | Record accepted experiment results (GRID-001, GRID-002, ABL-001) and the ABL-001 ledger-summary erratum |
| REVIEW-003 | reviewer | done | reviewer-20261125-b | — | Independent review of setup tasks: ORIENT-001, ORIENT-002, COORD-001, ENG-001 |
| REVIEW-004 | reviewer | done | reviewer-20261125-c | — | Independent review of environment, data and pilot tasks: ENG-002, ENG-003, ENG-004, ENG-005, REASON-002 |
| REVIEW-005 | reviewer | done | reviewer-20261125-d | — | Independent review of REVIEW-002 and COORD-002 |
| SEED-001 | engineer | done | claude-orient-20261122-a | — | More seeds: is the sparse-vs-dense (and sparse-vs-linear) gap real? |
| COORD-003 | coordinator | done | claude-orient-20261122-a | — | Bring DECISIONS.md and RESEARCH_STATUS.md up to date (REVIEW-003 C1-1) |
| HARNESS-001 | engineer | done | claude-orient-20261122-a | — | Let a later accepted task supersede a shared-file artifact of an earlier done task |
| REVIEW-006 | reviewer | done | reviewer-20261125-d | — | Independent review of SEED-001 (more seeds, pre-registered) |
| MSWEEP-001 | engineer | done | claude-orient-20261122-a | — | Clean dictionary-size (M) sweep: does the number of available atoms matter? |
| REVIEW-007 | reviewer | done | reviewer-20261126-e | — | Independent review of MSWEEP-001 (clean dictionary-size sweep) |
| LR-001 | engineer | done | claude-orient-20261122-a | — | Validation-only learning-rate check for large dictionary heads (M >= 2048) |
| REVIEW-008 | reviewer | done | reviewer-20261126-g | — | Independent review of LR-001 (validation-only learning-rate check) |
| LR-002 | engineer | done | claude-orient-20261122-a | — | Validation-only lr/epoch extension: locate the best lr for all dictionary heads and the linear head |
| REVIEW-009 | reviewer | done | reviewer-20261126-i | — | Independent review of LR-002 (validation-only lr/epoch extension incl. linear head) |
| RETEST-001 | engineer | done | claude-orient-20261122-a | — | Confirmatory ImageNetV2 re-test of the heads at tuned learning rates and a common 20-epoch budget |
| REVIEW-010 | reviewer | done | reviewer-20261126-k | — | Independent review of RETEST-001 (confirmatory ImageNetV2 re-test at tuned lrs, common 20-epoch budget) |
| COORD-004 | coordinator | done | claude-orient-20261122-a | — | Refresh shared status after LR-001, LR-002 and RETEST-001 (successor to COORD-002/status.md; DECISIONS.md and RESEARCH_STATUS.md current state) |
| REVIEW-011 | reviewer | done | reviewer-20261126-m | — | Independent review of COORD-004 (shared status refresh after LR-001, LR-002 and RETEST-001) |
| WRITE-001 | coordinator | done | claude-orient-20261122-a | — | Draft the project write-up (methods, results, limitations) from accepted results |
| REVIEW-012 | reviewer | done | reviewer-20261126-o | — | Independent review of WRITE-001 (draft write-up) |
| WRITE-002 | coordinator | done | claude-orient-20261122-a | WRITE-001 | Wording fixes to the accepted write-up draft found in the cross-review of REVIEW-012 |
| WD-001 | engineer | done | claude-orient-20261122-a | — | Regularized wide head: can tuned weight decay replace the reconstruction loss? (validation selection, then ImageNetV2 test) |
| WD-001A | engineer | pending | — | — | WD-001 phase-A selection lock: submit selected.json before any ImageNetV2 run |
| WD-001S | engineer | done | claude-orient-20261122-a | — | WD-001 phase-A selection lock: submit selected.json before any ImageNetV2 run |
| REVIEW-013 | reviewer | done | reviewer-20261127-r | — | Independent review of WD-001S (phase-A selection lock for the weight-decay experiment WD-001) |
| REVIEW-014 | reviewer | done | reviewer-20261127-t | — | Independent review of WD-001 (weight-decay check: phase-B ImageNetV2 results, analysis and report) |
| COORD-005 | coordinator | done | claude-orient-20261122-a | WD-001, WD-001S, REVIEW-013, REVIEW-014, COORD-004, WRITE-002 | Add the accepted WD-001 weight-decay result to the shared status page, DECISIONS.md, RESEARCH_STATUS.md and the draft write-up |
| REVIEW-015 | reviewer | done | reviewer-20261127-v | — | Independent review of COORD-005 (WD-001 added to the status page, DECISIONS.md, RESEARCH_STATUS.md and the draft write-up) |
| COORD-006 | coordinator | done | claude-orient-20261122-a | COORD-005, REVIEW-015 | Wording fixes to COORD-005's shared-file update from REVIEW-015 and its cross-review |

## Submitted evidence

- **LIT-001 (done)**: `ai_harness/artifacts/LIT-001/report.md`, `ai_harness/artifacts/LIT-001/evidence.json`
- **REASON-001 (done)**: `ai_harness/artifacts/REASON-001/report.md`
- **ORIENT-001 (done)**: `ai_harness/artifacts/ORIENT-001/report.md`
- **COORD-001 (done)**: `ai_harness/DECISIONS.md`, `ai_harness/RESEARCH_STATUS.md`
- **ENG-001 (done)**: `src/build_jpeg_dataset.py`, `ai_harness/artifacts/ENG-001/report.md`, `ai_harness/artifacts/ENG-001/make_synthetic.py`, `ai_harness/artifacts/ENG-001/test_build_jpeg_dataset.py`, `ai_harness/artifacts/ENG-001/pilot/out/manifest.csv`, `ai_harness/artifacts/ENG-001/pilot/out/summary.json`, `ai_harness/artifacts/ENG-001/pilot/cli_output.txt`
- **ORIENT-002 (done)**: `ai_harness/artifacts/ORIENT-002/report.md`
- **ENG-002 (done)**: `ai_harness/artifacts/ENG-002/report.md`, `ai_harness/artifacts/ENG-002/check_cpu.txt`, `ai_harness/artifacts/ENG-002/check_cuda.txt`, `ai_harness/artifacts/ENG-002/check_cuda_pretrained.txt`, `ai_harness/artifacts/ENG-002/bench.py`, `ai_harness/artifacts/ENG-002/bench_output.json`
- **ENG-003 (done)**: `ai_harness/artifacts/ENG-003/report.md`, `ai_harness/artifacts/ENG-003/extract_datasets.py`, `Datasets/DATASETS.md`, `Datasets/_manifests/imagenet-1k-val.csv`, `Datasets/_manifests/imagenetv2-matched-frequency.csv`
- **ENG-004 (done)**: `ai_harness/artifacts/ENG-004/report.md`, `ai_harness/artifacts/ENG-004/train_log.txt`, `ai_harness/artifacts/ENG-004/eval_log.txt`, `ai_harness/artifacts/ENG-004/heldout_source_vs_20.json`
- **ENG-005 (done)**: `src/baseline_head.py`, `ai_harness/artifacts/ENG-005/report.md`, `ai_harness/artifacts/ENG-005/src_hashes_before.txt`, `ai_harness/artifacts/ENG-005/linear_train_log.txt`, `ai_harness/artifacts/ENG-005/dense_train_log.txt`, `ai_harness/artifacts/ENG-005/linear_eval_log.txt`, `ai_harness/artifacts/ENG-005/dense_eval_log.txt`, `ai_harness/artifacts/ENG-005/linear_heldout.json`, `ai_harness/artifacts/ENG-005/dense_heldout.json`
- **REASON-002 (done)**: `ai_harness/artifacts/REASON-002/plan.md`, `ai_harness/artifacts/REASON-002/byte_sample.txt`
- **GRID-001 (done)**: `ai_harness/artifacts/GRID-001/report.md`, `ai_harness/artifacts/GRID-001/analysis.md`, `ai_harness/artifacts/GRID-001/analysis.json`, `ai_harness/artifacts/GRID-001/analyze.py`, `ai_harness/artifacts/GRID-001/pixel_check.py`, `ai_harness/artifacts/GRID-001/pixel_check_output.txt`, `ai_harness/artifacts/GRID-001/bytes_log.txt`, `ai_harness/artifacts/GRID-001/features_log.txt`, `ai_harness/artifacts/GRID-001/tune_log.txt`, `ai_harness/artifacts/GRID-001/grid_log.txt`, `ai_harness/artifacts/GRID-001/msweep_log.txt`, `ai_harness/artifacts/GRID-001/reference_log.txt`, `ai_harness/artifacts/GRID-001/summary_log.txt`, `ai_harness/artifacts/GRID-001/src_hashes_before.txt`, `src/run_grid.py`, `runs/GRID-001/results.csv`, `runs/GRID-001/summary.csv`, `runs/GRID-001/tune_results.json`, `runs/GRID-001/tuned.json`, `runs/GRID-001/reference_original_fc.json`, `runs/GRID-001/jpeg_variants/summary.json`, `runs/GRID-001/jpeg_variants/manifest.csv`
- **GRID-002 (done)**: `ai_harness/artifacts/GRID-002/report.md`, `ai_harness/artifacts/GRID-002/analysis.md`, `ai_harness/artifacts/GRID-002/analysis.json`, `ai_harness/artifacts/GRID-002/analyze.py`, `ai_harness/artifacts/GRID-002/task.json`, `ai_harness/artifacts/GRID-002/freeze_log.txt`, `ai_harness/artifacts/GRID-002/grid_log.txt`, `ai_harness/artifacts/GRID-002/summary_log.txt`, `src/run_grid_xavier.py`, `runs/GRID-002/results.csv`, `runs/GRID-002/summary.csv`, `runs/GRID-002/tuned.json`
- **ABL-001 (done)**: `ai_harness/artifacts/ABL-001/report.md`, `ai_harness/artifacts/ABL-001/analysis.md`, `ai_harness/artifacts/ABL-001/analysis.json`, `ai_harness/artifacts/ABL-001/analyze.py`, `ai_harness/artifacts/ABL-001/task.json`, `ai_harness/artifacts/ABL-001/grid_log.txt`, `ai_harness/artifacts/ABL-001/summary_log.txt`, `src/run_ablation.py`, `runs/ABL-001/results.csv`, `runs/ABL-001/summary.csv`, `runs/ABL-001/settings.json`
- **REVIEW-002 (done)**: `ai_harness/artifacts/REVIEW-002/report.md`, `ai_harness/artifacts/REVIEW-002/recompute/independent_check.py`, `ai_harness/artifacts/REVIEW-002/recompute/independent_check_output.txt`, `ai_harness/artifacts/REVIEW-002/recompute/ABL-001/analyze.py`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-001/analyze.py`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-002/analyze.py`, `ai_harness/artifacts/REVIEW-002/recompute/ABL-001/analysis.md`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-001/analysis.md`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-002/analysis.md`, `ai_harness/artifacts/REVIEW-002/recompute/ABL-001/analysis.json`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-001/analysis.json`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-002/analysis.json`, `ai_harness/artifacts/REVIEW-002/recompute/ABL-001/stdout.txt`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-001/stdout.txt`, `ai_harness/artifacts/REVIEW-002/recompute/GRID-002/stdout.txt`
- **COORD-002 (done)**: `ai_harness/artifacts/COORD-002/status.md`, `ai_harness/artifacts/COORD-002/task.json`
- **REVIEW-003 (done)**: `ai_harness/artifacts/REVIEW-003/report.md`, `ai_harness/artifacts/REVIEW-003/task.json`
- **REVIEW-004 (done)**: `ai_harness/artifacts/REVIEW-004/report.md`, `ai_harness/artifacts/REVIEW-004/task.json`
- **REVIEW-005 (done)**: `ai_harness/artifacts/REVIEW-005/report.md`
- **SEED-001 (done)**: `ai_harness/artifacts/SEED-001/report.md`, `ai_harness/artifacts/SEED-001/analysis.md`, `ai_harness/artifacts/SEED-001/analysis.json`, `ai_harness/artifacts/SEED-001/analyze.py`, `ai_harness/artifacts/SEED-001/task.json`, `ai_harness/artifacts/SEED-001/grid_log.txt`, `ai_harness/artifacts/SEED-001/summary_log.txt`, `src/run_seeds.py`, `runs/SEED-001/results.csv`, `runs/SEED-001/summary.csv`, `runs/SEED-001/settings.json`
- **COORD-003 (done)**: `ai_harness/DECISIONS.md`, `ai_harness/RESEARCH_STATUS.md`, `ai_harness/artifacts/COORD-003/task.json`
- **HARNESS-001 (done)**: `ai_harness/harness.py`, `ai_harness/test_harness.py`, `ai_harness/artifacts/HARNESS-001/task.json`, `ai_harness/artifacts/HARNESS-001/test_output.txt`
- **REVIEW-006 (done)**: `ai_harness/artifacts/REVIEW-006/report.md`, `ai_harness/artifacts/REVIEW-006/task.json`
- **MSWEEP-001 (done)**: `ai_harness/artifacts/MSWEEP-001/report.md`, `ai_harness/artifacts/MSWEEP-001/analysis.md`, `ai_harness/artifacts/MSWEEP-001/analysis.json`, `ai_harness/artifacts/MSWEEP-001/analyze.py`, `ai_harness/artifacts/MSWEEP-001/task.json`, `ai_harness/artifacts/MSWEEP-001/grid_log.txt`, `ai_harness/artifacts/MSWEEP-001/summary_log.txt`, `src/run_msweep.py`, `runs/MSWEEP-001/results.csv`, `runs/MSWEEP-001/summary.csv`, `runs/MSWEEP-001/settings.json`
- **REVIEW-007 (done)**: `ai_harness/artifacts/REVIEW-007/report.md`, `ai_harness/artifacts/REVIEW-007/task.json`
- **LR-001 (done)**: `ai_harness/artifacts/LR-001/report.md`, `ai_harness/artifacts/LR-001/task.json`, `ai_harness/artifacts/LR-001/analyze.py`, `ai_harness/artifacts/LR-001/analysis.md`, `ai_harness/artifacts/LR-001/analysis.json`, `ai_harness/artifacts/LR-001/grid_log.txt`, `src/run_lrcheck.py`, `runs/LR-001/results.csv`, `runs/LR-001/settings.json`
- **REVIEW-008 (done)**: `ai_harness/artifacts/REVIEW-008/report.md`, `ai_harness/artifacts/REVIEW-008/task.json`
- **LR-002 (done)**: `ai_harness/artifacts/LR-002/report.md`, `ai_harness/artifacts/LR-002/task.json`, `ai_harness/artifacts/LR-002/analyze.py`, `ai_harness/artifacts/LR-002/analysis.md`, `ai_harness/artifacts/LR-002/analysis.json`, `ai_harness/artifacts/LR-002/posthoc.py`, `ai_harness/artifacts/LR-002/posthoc.md`, `ai_harness/artifacts/LR-002/grid_log.txt`, `src/run_lrcheck2.py`, `runs/LR-002/results.csv`, `runs/LR-002/settings.json`
- **REVIEW-009 (done)**: `ai_harness/artifacts/REVIEW-009/report.md`, `ai_harness/artifacts/REVIEW-009/task.json`
- **RETEST-001 (done)**: `ai_harness/artifacts/RETEST-001/task.json`, `ai_harness/artifacts/RETEST-001/analyze.py`, `ai_harness/artifacts/RETEST-001/analysis.md`, `ai_harness/artifacts/RETEST-001/analysis.json`, `ai_harness/artifacts/RETEST-001/grid_log.txt`, `ai_harness/artifacts/RETEST-001/report.md`, `src/run_retest.py`, `runs/RETEST-001/results.csv`, `runs/RETEST-001/settings.json`
- **REVIEW-010 (done)**: `ai_harness/artifacts/REVIEW-010/task.json`, `ai_harness/artifacts/REVIEW-010/report.md`, `ai_harness/artifacts/REVIEW-010/review010.py`
- **COORD-004 (done)**: `ai_harness/artifacts/COORD-004/task.json`, `ai_harness/artifacts/COORD-004/status.md`, `ai_harness/DECISIONS.md`, `ai_harness/RESEARCH_STATUS.md`
- **REVIEW-011 (done)**: `ai_harness/artifacts/REVIEW-011/task.json`, `ai_harness/artifacts/REVIEW-011/report.md`
- **WRITE-001 (done)**: `ai_harness/artifacts/WRITE-001/task.json`, `ai_harness/artifacts/WRITE-001/report_draft.md`, `ai_harness/artifacts/WRITE-001/proposed_references.bib`
- **REVIEW-012 (done)**: `ai_harness/artifacts/REVIEW-012/task.json`, `ai_harness/artifacts/REVIEW-012/report.md`
- **WRITE-002 (done)**: `ai_harness/artifacts/WRITE-002/task.json`, `ai_harness/artifacts/WRITE-001/report_draft.md`
- **WD-001 (done)**: `ai_harness/artifacts/WD-001/task.json`, `ai_harness/artifacts/WD-001/report.md`, `ai_harness/artifacts/WD-001/analyze_val.py`, `ai_harness/artifacts/WD-001/analyze.py`, `ai_harness/artifacts/WD-001/selected.json`, `ai_harness/artifacts/WD-001/analysis_val.md`, `ai_harness/artifacts/WD-001/analysis_val.json`, `ai_harness/artifacts/WD-001/analysis.md`, `ai_harness/artifacts/WD-001/analysis.json`, `ai_harness/artifacts/WD-001/results.csv`, `ai_harness/artifacts/WD-001/logs/sanity_log.txt`, `ai_harness/artifacts/WD-001/logs/val_log.txt`, `ai_harness/artifacts/WD-001/logs/test_log.txt`, `ai_harness/artifacts/WD-001/logs/settings_val.json`, `ai_harness/artifacts/WD-001/logs/settings_test.json`, `src/run_wdcheck.py`
- **WD-001S (done)**: `ai_harness/artifacts/WD-001S/task.json`, `ai_harness/artifacts/WD-001S/selected.json`, `ai_harness/artifacts/WD-001S/analysis_val.md`, `ai_harness/artifacts/WD-001S/analysis_val.json`, `ai_harness/artifacts/WD-001S/analyze_val.py`
- **REVIEW-013 (done)**: `ai_harness/artifacts/REVIEW-013/task.json`, `ai_harness/artifacts/REVIEW-013/report.md`, `ai_harness/artifacts/REVIEW-013/recompute.py`
- **REVIEW-014 (done)**: `ai_harness/artifacts/REVIEW-014/task.json`, `ai_harness/artifacts/REVIEW-014/report.md`, `ai_harness/artifacts/REVIEW-014/recompute.py`
- **COORD-005 (done)**: `ai_harness/artifacts/COORD-005/task.json`, `ai_harness/artifacts/COORD-004/status.md`, `ai_harness/artifacts/WRITE-001/report_draft.md`, `ai_harness/DECISIONS.md`, `ai_harness/RESEARCH_STATUS.md`
- **REVIEW-015 (done)**: `ai_harness/artifacts/REVIEW-015/task.json`, `ai_harness/artifacts/REVIEW-015/report.md`, `ai_harness/artifacts/REVIEW-015/check_numbers.py`
- **COORD-006 (done)**: `ai_harness/artifacts/COORD-006/task.json`, `ai_harness/artifacts/COORD-004/status.md`, `ai_harness/artifacts/WRITE-001/report_draft.md`, `ai_harness/DECISIONS.md`, `ai_harness/RESEARCH_STATUS.md`
