# ENG-004: First real training run (timing and sanity pilot)

**Agent:** `claude-orient-20261122-a`. **Role:** engineer. **Date:** 2026-11-24.
**Status:** Submitted for review. **This is a single-seed pilot with no tuning. It is not a research result.** ImageNetV2 was not touched. `src/` was not edited.

## What ran

Env `ai360` (Python 3.12.16, torch 2.11.0+cu128) on an RTX 3080 Ti Laptop GPU, run from `src/`:

```
python -B -u main.py train --data ../Datasets/imagenet-1k/val --holdout 0.10 --train-limit 0 --val-limit 0 \
  --epochs 10 --batch-size 128 --lr 0.001 --cache memory --encoder-init he --fc-init xavier \
  --workers 4 --device cuda --out ../runs/ENG-004/he_45k
python -B -u main.py evaluate --checkpoint ../runs/ENG-004/he_45k/best.pt --data ../Datasets/imagenet-1k/val \
  --limit 0 --levels source 20 --baseline --workers 4 --device cuda --json ../runs/ENG-004/he_45k/heldout_source_vs_20.json
```

This is the src/README.md 45k/5k command, plus `--workers 4 --device cuda`.
- Defaults: seed 42, compression level 20 (JPEG quality 76) for **both** train and validation, M = 2048, k = 32.
- Loss: CE + feature MSE + 0.1·dependency + 0.05·coherence.
- Backbone: frozen ImageNet-pretrained ResNet-18.
- Logs: [train_log.txt](train_log.txt) (timestamped) and [eval_log.txt](eval_log.txt). Metrics: [heldout_source_vs_20.json](heldout_source_vs_20.json).
- Checkpoints in `runs/ENG-004/he_45k/`: `best.pt` (57 MB, epoch 8) and `last.pt` (97 MB).

## Timing (measured from log timestamps)

| Stage | Wall time | Rate |
| --- | --- | --- |
| Startup and split | 8 s | |
| Feature extraction, 45,000 train images (JPEG level 20 + preprocessing on CPU, 4 workers; ResNet on GPU) | 137 s | about 330 images/s, about 2.5× the single-process benchmark in ENG-002 |
| Feature extraction, 5,000 validation images | 33 s | |
| 10 epochs of head training on cached features | 36 s | about 3.6 s/epoch |
| **Total training command** | **3 min 35 s** | |
| Evaluation, 2 levels × 5,000 images | 71 s | |

CPU preprocessing dominates, as ENG-002 predicted. One training condition costs about 3.5 min, so a grid of tens of runs is a matter of hours on this laptop.

## Held-out results (5,000 images, single seed, pilot)

| Model | Eval on level 20 | Eval on source (original) images |
| --- | --- | --- |
| Sparse-code classifier trained at level 20 (`best.pt`, epoch 8) | top-1 **57.1%**, top-5 80.2% | top-1 **57.6%**, top-5 80.2% |
| Original TorchVision ResNet-18 fc (`--baseline`, trained on ImageNet-train originals) | top-1 69.3%, top-5 88.7% | top-1 70.0%, top-5 88.9% |

**Selection bias (added 2026-11-25 after REVIEW-004, finding E4-1).** These 5,000 "held-out" images are the **same validation set used to select `best.pt`**. `best.pt` is the epoch (8 of 10) with the highest top-1 on exactly these images, and the level-20 score (57.1%) equals that selection score. So the trained head's numbers are selection-biased and slightly optimistic. Validation top-1 ranged from 55.0% to 57.1% over epochs 2–10, so the bias is probably at most about 1–2 points. The original-fc row is not affected, because it involves no selection. ENG-005 §5 notes the same bias, and the later GRID experiments avoid it by testing only on ImageNetV2.

Training top-1 reached 99.8% while validation top-1 plateaued at about 55–57% from epoch 2, with validation CE flat at about 1.9. Diagnostics: 32 active atoms per image, 99.7% of atoms used, and validation feature MSE fell from 0.59 to 0.36. The 99.7% is from the `best.pt` checkpoint (epoch 8; 99.76%). The final epoch was 99.66% (REVIEW-004, E4-2).

## Observations (inference, for review)

1. **The pipeline works end to end on real data.** The original fc scores 70.0% on source images, consistent with ResNet-18's published ImageNet top-1 of about 69.8%. This supports the label mapping and preprocessing.
2. **Level 20 barely matters here.** The drop from source to level 20 is about 0.5–0.7 points for both models. Level 20 is mild compression (about quality 76). The research question needs much stronger levels.
3. **The new head overfits.** With 45 images per class and a 2048×1000 Xavier-initialized fc, train accuracy reaches about 100% while validation sits 12–13 points below the original fc. The original fc is not a fair baseline either way: it was trained on 1.28M ImageNet-train originals. Natural next things to try include `--fc-init auto` (folded from the original fc), weight decay or early stopping, and smaller M. None of these were tried here, and tuning should use the validation split only.
4. **For the storage question,** this run is one cell (level 20, dictionary head). The REASON-001 / ORIENT-002 design still needs a conventional head trained on the same data, several stronger levels, measured bytes per level (`build_jpeg_dataset.py`), and repeated seeds.

## Housekeeping

- `runs/ENG-004/he_45k/.training.lock` (1 byte) remains after a clean exit. It is created by the user's `run_writer`, and I left it as is. It may be intentional, so check before reusing the folder.
- `runs/` is inside OneDrive. The checkpoints add about 150 MB per run.
