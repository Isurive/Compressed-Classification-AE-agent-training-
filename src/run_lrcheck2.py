"""LR-002: validation-only lr/epoch extension for the dictionary heads and the linear head.

    python -B -u run_lrcheck2.py grid      # 54 dictionary runs + 24 linear runs, 20 epochs each, validation only

(run from src/, ai360 env). Design and analysis are pre-registered in ai_harness/artifacts/LR-002/task.json.

Dictionary heads (sparse M2048 k512, sparse M4096 k1024, dense M2048 k2048; CE + feature MSE 1/0/0, Xavier fc,
He encoder) at lr 1e-4, 3e-5, 1e-5; linear head (Xavier, weight decay 1e-4, CE) at lr 3e-3, 1e-3, 3e-4, 1e-4.
All runs train 20 epochs at a constant lr with per-epoch reseeding, so epochs 1-10 equal a 10-epoch run; the
10-epoch budget is read from the saved history. NO test-set evaluation: the script never touches ImageNetV2.
Reuses run_grid.py / run_ablation.py / sparse_workflow.py unchanged. Writes runs/LR-002.
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

rg.OUT = rg.ROOT / "runs" / "LR-002"  # FEATURES stays runs/GRID-001/features (read only).
EPOCHS = 20
LEVELS = [None, 90]
SEEDS = [42, 43, 44]
DICT_LRS = [1e-4, 3e-5, 1e-5]
LINEAR_LRS = [3e-3, 1e-3, 3e-4, 1e-4]
HEADS = {
    "sparse-M2048": {"model": "sparse", "atoms": 2048, "k": 512, "fc_init": "xavier", "weight_decay": 0.0},
    "sparse-M4096": {"model": "sparse", "atoms": 4096, "k": 1024, "fc_init": "xavier", "weight_decay": 0.0},
    "dense-M2048": {"model": "dense", "atoms": 2048, "k": 2048, "fc_init": "xavier", "weight_decay": 0.0},
    "linear": {"model": "linear", "atoms": 0, "k": 0, "fc_init": "xavier", "weight_decay": 1e-4},
}
MSE_ONLY = ra.VARIANTS["abl-mse"]
FIELDS = ["head", "model", "atoms", "k", "lr", "epochs", "train_level", "seed", "split_digest", "best_epoch",
          "val_top1", "val_ce", "train_seconds"]


def lr_name(lr):
    return f"lr{lr:.0e}".replace("e-0", "e-")


def lrs_for(head):
    return LINEAR_LRS if head == "linear" else DICT_LRS


def train_run(head, cfg, lr, level, seed, device, run_dir):
    """One 20-epoch run with an explicit lr; done.json stores the per-epoch history."""
    done = run_dir / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    if (run_dir / "best.pt").exists():
        raise ValueError(f"{run_dir} has a partial run (best.pt without done.json). Remove it and rerun.")
    part = rg.pool_split(seed)
    cache = rg.cache_for(part, level)
    start = time.time()
    print(f"[lr-check2] {head} M={cfg['atoms']} k={cfg['k']} lr={lr:g} epochs={EPOCHS} {rg.level_name(level)} "
          f"seed {seed}", flush=True)
    if cfg["model"] == "linear":
        ids = rg.reference_ids(part[0].classes, "auto")
        saved = rg.LR, rg.EPOCHS
        rg.LR, rg.EPOCHS = lr, EPOCHS  # train_linear reads these module globals at call time
        try:
            rg.train_linear(part, cache, ids, level, seed, cfg, run_dir, device)
        finally:
            rg.LR, rg.EPOCHS = saved
        raw = json.loads((run_dir / "history.json").read_text())
        history = [{"epoch": r["epoch"], "train_loss": None, "train_ce": None,
                    "val_top1": r["validation"]["top1"], "val_ce": r["validation"]["ce"]} for r in raw]
        weights = {}
    else:
        settings = Settings(data=str(rg.POOL.resolve()), train_limit=0, val_limit=0, holdout=rg.HOLDOUT, seed=seed,
                            atoms=cfg["atoms"], k=cfg["k"], compression_level=level, batch_size=rg.BATCH, lr=lr,
                            encoder_init="he", fc_init=cfg["fc_init"], cache="memory", **MSE_ONLY)
        raw = train_dictionary(settings, run_dir, EPOCHS, 0, device, partitions=part, fold_cache=cache)
        history = [{"epoch": r["epoch"], "train_loss": r["train"]["loss"], "train_ce": r["train"]["ce"],
                    "val_top1": r["validation"]["top1"], "val_ce": r["validation"]["ce"]} for r in raw]
        weights = MSE_ONLY
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    summary = {"head": head, **cfg, **weights, "lr": lr, "epochs": EPOCHS, "train_level": rg.level_name(level),
               "seed": seed, "phase": "lr-check2", "split_digest": best["protocol"]["digest"],
               "best_epoch": best["epoch"], "val_top1": best["validation"]["top1"], "val_ce": best["validation"]["ce"],
               "run_dir": run_dir.relative_to(rg.ROOT).as_posix(), "train_seconds": round(time.time() - start, 1),
               "history": history}
    done.write_text(json.dumps(summary, indent=2))
    last = run_dir / "last.pt"
    if last.exists():
        last.unlink()
    return summary


def cmd_grid(args, device):
    (rg.OUT / "settings.json").write_text(json.dumps(
        {"heads": HEADS, "dictionary_lrs": DICT_LRS, "linear_lrs": LINEAR_LRS, "epochs": EPOCHS, "batch": rg.BATCH,
         "dictionary_loss": MSE_ONLY, "levels": [rg.level_name(x) for x in LEVELS], "seeds": SEEDS,
         "test_evaluation": "none (validation only)"}, indent=2))
    rows = []
    for seed in args.seeds:
        for level in args.levels:
            for head in args.heads:
                for lr in (args.lrs or lrs_for(head)):
                    run_dir = rg.OUT / head / lr_name(lr) / rg.level_name(level) / f"seed{seed}"
                    s = train_run(head, HEADS[head], lr, level, seed, device, run_dir)
                    rows.append({k: s[k] for k in FIELDS})
    out = rg.OUT / ("results.csv" if args.lrs is None and set(args.heads) == set(HEADS) else "results_partial.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}", flush=True)


def main():
    def level_arg(text):
        return None if text == "source" else int(text)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("grid",))
    parser.add_argument("--levels", type=level_arg, nargs="+", default=LEVELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--lrs", type=float, nargs="+", default=None, help="override the per-head lr list")
    parser.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rg.OUT.mkdir(parents=True, exist_ok=True)
    cmd_grid(args, device)


if __name__ == "__main__":
    main()
