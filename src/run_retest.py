"""RETEST-001: confirmatory ImageNetV2 re-test of the heads at tuned lrs and a common 20-epoch budget.

    python -B -u run_retest.py grid      # 4 heads x 3 levels x seeds 45-51 (84 runs), ImageNetV2 test after each run

(run from src/, ai360 env). Design and analysis are pre-registered in ai_harness/artifacts/RETEST-001/task.json.

Heads: sparse M2048 k512, sparse M4096 k1024, dense M2048 k2048 (CE + feature MSE 1/0/0, Xavier fc, He encoder,
Adam lr 1e-4) and linear (Xavier, weight decay 1e-4, CE, Adam lr 3e-4). Every run trains 20 epochs at a constant
lr with per-epoch reseeding; best.pt is the epoch with the best validation top-1 (then lower CE). After training,
best.pt is scored once on ImageNetV2 at source (primary) and at the training level (secondary), via run_grid.test_run.
Reuses run_grid.py / run_ablation.py / sparse_workflow.py unchanged. Writes runs/RETEST-001.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import run_grid as rg
import run_ablation as ra  # importing sets rg.OUT to ABL-001; it is reset below before any use.
from sparse_workflow import Settings, train as train_dictionary

rg.OUT = rg.ROOT / "runs" / "RETEST-001"  # FEATURES stays runs/GRID-001/features (read only).
PHASE = "retest"
EPOCHS = 20
LEVELS = [None, 80, 90]
SEEDS = [45, 46, 47, 48, 49, 50, 51]
HEADS = {
    "sparse-M2048": {"model": "sparse", "atoms": 2048, "k": 512, "fc_init": "xavier", "weight_decay": 0.0, "lr": 1e-4},
    "sparse-M4096": {"model": "sparse", "atoms": 4096, "k": 1024, "fc_init": "xavier", "weight_decay": 0.0, "lr": 1e-4},
    "dense-M2048": {"model": "dense", "atoms": 2048, "k": 2048, "fc_init": "xavier", "weight_decay": 0.0, "lr": 1e-4},
    "linear": {"model": "linear", "atoms": 0, "k": 0, "fc_init": "xavier", "weight_decay": 1e-4, "lr": 3e-4},
}
MSE_ONLY = ra.VARIANTS["abl-mse"]


def train_run(head, cfg, level, seed, device, run_dir):
    """One 20-epoch run at the head's lr; done.json stores the summary and the per-epoch history."""
    done = run_dir / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    if (run_dir / "best.pt").exists():
        raise ValueError(f"{run_dir} has a partial run (best.pt without done.json). Remove it and rerun.")
    lr = cfg["lr"]
    part = rg.pool_split(seed)
    cache = rg.cache_for(part, level)
    start = time.time()
    print(f"[retest] {head} M={cfg['atoms']} k={cfg['k']} lr={lr:g} epochs={EPOCHS} {rg.level_name(level)} "
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
    summary = {"head": head, **cfg, **weights, "epochs": EPOCHS, "train_level": rg.level_name(level),
               "seed": seed, "phase": PHASE, "split_digest": best["protocol"]["digest"],
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
        {"heads": HEADS, "epochs": EPOCHS, "batch": rg.BATCH, "dictionary_loss": MSE_ONLY,
         "levels": [rg.level_name(x) for x in LEVELS], "seeds": SEEDS,
         "test": "ImageNetV2 matched-frequency, eval at source and at the training level, once per run"}, indent=2))
    done_rows = rg.recorded()
    for seed in args.seeds:
        for level in args.levels:
            for head in args.heads:
                run_dir = rg.OUT / head / rg.level_name(level) / f"seed{seed}"
                summary = train_run(head, HEADS[head], level, seed, device, run_dir)
                rg.test_run(summary, level, device, done_rows)
    print("grid finished", flush=True)


def main():
    def level_arg(text):
        return None if text == "source" else int(text)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("grid",))
    parser.add_argument("--levels", type=level_arg, nargs="+", default=LEVELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rg.OUT.mkdir(parents=True, exist_ok=True)
    cmd_grid(args, device)


if __name__ == "__main__":
    main()
