"""LR-001: validation-only learning-rate check for large dictionary heads (M >= 2048).

    python -B -u run_lrcheck.py grid      # 3 heads x 3 lr x 2 levels x seeds 42-44 = 54 runs, validation only

(run from src/, ai360 env). Design and analysis are pre-registered in ai_harness/artifacts/LR-001/task.json.

Heads: sparse M2048 k512, sparse M4096 k1024, dense M2048 k2048. Loss CE + feature MSE (reconstruction 1,
dependency 0, coherence 0), Xavier fc, He encoder, 10 epochs, batch 128, cached GRID-001 features (read only).
Only the Adam learning rate varies: 1e-3, 3e-4, 1e-4. NO test-set evaluation: the script never touches
ImageNetV2. Each done.json stores the full per-epoch history so instability can be measured without log parsing.
Reuses run_grid.py / run_ablation.py unchanged. Writes runs/LR-001.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import torch

import run_grid as rg
import run_ablation as ra  # importing sets rg.OUT to ABL-001; it is reset below before any use.
from sparse_workflow import Settings, train as train_dictionary

rg.OUT = rg.ROOT / "runs" / "LR-001"  # FEATURES stays runs/GRID-001/features (read only).
LEVELS = [None, 90]
SEEDS = [42, 43, 44]
LRS = [1e-3, 3e-4, 1e-4]
HEADS = {
    "sparse-M2048": {"model": "sparse", "atoms": 2048, "k": 512, "fc_init": "xavier", "weight_decay": 0.0},
    "sparse-M4096": {"model": "sparse", "atoms": 4096, "k": 1024, "fc_init": "xavier", "weight_decay": 0.0},
    "dense-M2048": {"model": "dense", "atoms": 2048, "k": 2048, "fc_init": "xavier", "weight_decay": 0.0},
}
MSE_ONLY = ra.VARIANTS["abl-mse"]
FIELDS = ["head", "model", "atoms", "k", "lr", "train_level", "seed", "split_digest", "best_epoch", "val_top1",
          "val_ce", "train_seconds"]


def lr_name(lr):
    return f"lr{lr:.0e}".replace("e-0", "e-")


def train_lr(head, cfg, lr, level, seed, device, run_dir):
    """run_ablation.train_variant with an explicit learning rate and the per-epoch history saved."""
    done = run_dir / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    if (run_dir / "best.pt").exists():
        raise ValueError(f"{run_dir} has a partial run (best.pt without done.json). Remove it and rerun.")
    part = rg.pool_split(seed)
    cache = rg.cache_for(part, level)
    start = time.time()
    print(f"[lr-check] {head} M={cfg['atoms']} k={cfg['k']} lr={lr:g} {rg.level_name(level)} seed {seed}", flush=True)
    settings = Settings(data=str(rg.POOL.resolve()), train_limit=0, val_limit=0, holdout=rg.HOLDOUT, seed=seed,
                        atoms=cfg["atoms"], k=cfg["k"], compression_level=level, batch_size=rg.BATCH, lr=lr,
                        encoder_init="he", fc_init=cfg["fc_init"], cache="memory", **MSE_ONLY)
    history = train_dictionary(settings, run_dir, rg.EPOCHS, 0, device, partitions=part, fold_cache=cache)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    summary = {"head": head, **cfg, **MSE_ONLY, "lr": lr, "train_level": rg.level_name(level), "seed": seed,
               "phase": "lr-check", "split_digest": best["protocol"]["digest"], "best_epoch": best["epoch"],
               "val_top1": best["validation"]["top1"], "val_ce": best["validation"]["ce"],
               "run_dir": run_dir.relative_to(rg.ROOT).as_posix(), "train_seconds": round(time.time() - start, 1),
               "history": [{"epoch": r["epoch"], "train_loss": r["train"]["loss"], "train_ce": r["train"]["ce"],
                            "val_top1": r["validation"]["top1"], "val_ce": r["validation"]["ce"]} for r in history]}
    done.write_text(json.dumps(summary, indent=2))
    last = run_dir / "last.pt"
    if last.exists():
        last.unlink()
    return summary


def cmd_grid(args, device):
    (rg.OUT / "settings.json").write_text(json.dumps({"heads": HEADS, "lrs": args.lrs, "dictionary_loss": MSE_ONLY,
                                                      "levels": [rg.level_name(x) for x in args.levels],
                                                      "seeds": args.seeds, "epochs": rg.EPOCHS, "batch": rg.BATCH,
                                                      "test_evaluation": "none (validation only)"}, indent=2))
    rows = []
    for seed in args.seeds:
        for level in args.levels:
            for head in args.heads:
                for lr in args.lrs:
                    run_dir = rg.OUT / head / lr_name(lr) / rg.level_name(level) / f"seed{seed}"
                    s = train_lr(head, HEADS[head], lr, level, seed, device, run_dir)
                    rows.append({k: s[k] for k in FIELDS})
    with open(rg.OUT / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {rg.OUT / 'results.csv'}", flush=True)


def main():
    def level_arg(text):
        return None if text == "source" else int(text)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("grid",))
    parser.add_argument("--levels", type=level_arg, nargs="+", default=LEVELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--lrs", type=float, nargs="+", default=LRS)
    parser.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rg.OUT.mkdir(parents=True, exist_ok=True)
    cmd_grid(args, device)


if __name__ == "__main__":
    main()
