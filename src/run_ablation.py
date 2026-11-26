"""ABL-001: which part of the sparse-dictionary head gives its gain over the linear head (GRID-002)?

    python -B -u run_ablation.py grid       # 2 heads x 2 loss variants x 5 levels x seeds 42-44
    python -B -u run_ablation.py summary

(run from src/, ai360 env)

Every setting is GRID-002's frozen Xavier configuration (runs/GRID-002/tuned.json): M 2048, k 512 (sparse) or
k 2048 (dense), fc-init xavier, encoder-init he, 10 epochs, batch 128, Adam lr 1e-3. Only the loss weights change:

  phase `abl-mse`   reconstruction 1.0, dependency 0, coherence 0      (feature-MSE auxiliary only)
  phase `abl-none`  reconstruction 1e-6, dependency 0, coherence 0     (CE only; 1e-6 because Settings.validate
                                                                       requires a positive weight)

The full-loss cells (reconstruction 1.0, dependency 0.1, coherence 0.05) and the linear head come from GRID-002.
Dense + `abl-none` is a width-matched ReLU MLP, 512 -> 2048 -> 1000, on centered features.
Reuses run_grid.py unchanged (features, splits, scoring, results format). Writes runs/ABL-001.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import run_grid as rg
from sparse_workflow import Settings, train as train_dictionary

rg.OUT = rg.ROOT / "runs" / "ABL-001"  # FEATURES stays runs/GRID-001/features (read only).
TUNED = rg.ROOT / "runs" / "GRID-002" / "tuned.json"
LEVELS = [None, 60, 80, 90, 95]
VARIANTS = {
    "abl-mse": {"reconstruction_weight": 1.0, "dependency_weight": 0.0, "coherence_weight": 0.0},
    "abl-none": {"reconstruction_weight": 1e-6, "dependency_weight": 0.0, "coherence_weight": 0.0},
}


def train_variant(model_cfg, weights, level, seed, phase, device, run_dir):
    """run_grid.train_one for dictionary heads, with explicit loss weights."""
    done = run_dir / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    if (run_dir / "best.pt").exists():
        raise ValueError(f"{run_dir} has a partial run (best.pt without done.json). Remove it and rerun.")
    part = rg.pool_split(seed)
    cache = rg.cache_for(part, level)
    start = time.time()
    print(f"[{phase}] {model_cfg['model']} M={model_cfg['atoms']} k={model_cfg['k']} {rg.level_name(level)} "
          f"seed {seed} {json.dumps(weights)}", flush=True)
    cfg = Settings(data=str(rg.POOL.resolve()), train_limit=0, val_limit=0, holdout=rg.HOLDOUT, seed=seed,
                   atoms=model_cfg["atoms"], k=model_cfg["k"], compression_level=level, batch_size=rg.BATCH,
                   lr=rg.LR, encoder_init="he", fc_init=model_cfg["fc_init"], cache="memory", **weights)
    train_dictionary(cfg, run_dir, rg.EPOCHS, 0, device, partitions=part, fold_cache=cache)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    summary = {**model_cfg, **weights, "train_level": rg.level_name(level), "seed": seed, "phase": phase,
               "split_digest": best["protocol"]["digest"], "best_epoch": best["epoch"],
               "val_top1": best["validation"]["top1"], "val_ce": best["validation"]["ce"],
               "run_dir": run_dir.relative_to(rg.ROOT).as_posix(), "train_seconds": round(time.time() - start, 1)}
    done.write_text(json.dumps(summary, indent=2))
    return summary


def cmd_grid(args, device):
    tuned = json.loads(TUNED.read_text())
    (rg.OUT / "settings.json").write_text(json.dumps({"base": tuned, "variants": VARIANTS,
                                                      "levels": [rg.level_name(x) for x in args.levels],
                                                      "seeds": args.seeds}, indent=2))
    done_rows = rg.recorded()
    for seed in args.seeds:
        for level in args.levels:
            for phase, weights in VARIANTS.items():
                for name in ("sparse", "dense"):
                    cfg = tuned[name]
                    run_dir = rg.OUT / phase / name / rg.level_name(level) / f"seed{seed}"
                    summary = train_variant(cfg, weights, level, seed, phase, device, run_dir)
                    last = run_dir / "last.pt"
                    if last.exists() and (run_dir / "done.json").exists():
                        last.unlink()
                    rg.test_run(summary, level, device, done_rows)


def main():
    def level_arg(text):
        return None if text == "source" else int(text)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("grid", "summary"))
    parser.add_argument("--levels", type=level_arg, nargs="+", default=LEVELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=rg.SEEDS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rg.OUT.mkdir(parents=True, exist_ok=True)
    {"grid": cmd_grid, "summary": rg.cmd_summary}[args.phase](args, device)


if __name__ == "__main__":
    main()
