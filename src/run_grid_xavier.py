"""GRID-002: the GRID-001 grid with every head initialized independently (fc-init xavier).

    python -B -u run_grid_xavier.py freeze    # derive xavier settings from GRID-001 validation tuning
    python -B -u run_grid_xavier.py grid      # 3 models x 8 levels x seeds 42-44, then ImageNetV2 test
    python -B -u run_grid_xavier.py summary

(run from src/, ai360 env)

GRID-001 tuning chose fc-init `auto`/`original` for every model, so every head there started from the
original ImageNet classifier. This wrapper removes that inherited clean-data information. It reuses
run_grid.py unchanged: the same cached features (runs/GRID-001/features, read only), splits, training
code, scoring and results format. Only the output folder and the frozen settings differ.

Settings rule (pre-declared in GRID-001/run_grid.py cmd_tune, here restricted to xavier candidates):
highest mean 5k-validation top-1 over levels source and 90 (seed 42), ties broken by lower mean CE.
Candidates come from runs/GRID-001/tune_results.json; no new tuning, and ImageNetV2 is not read by `freeze`.
To limit disk use inside OneDrive, last.pt of each finished dictionary run is deleted (best.pt kept).
"""
from __future__ import annotations

import argparse
import json

import torch

import run_grid as rg

GRID001 = rg.ROOT / "runs" / "GRID-001"
rg.OUT = rg.ROOT / "runs" / "GRID-002"  # run_grid looks OUT up at call time; FEATURES stays GRID-001.
INITS = {"sparse": "xavier", "dense": "xavier", "linear": "xavier"}


def cmd_freeze(args, device):
    table = json.loads((GRID001 / "tune_results.json").read_text())
    frozen = {}
    for model, init in INITS.items():
        rows = [r for r in table if r["model"] == model and r["fc_init"] == init]
        best = max(rows, key=lambda r: (r["mean_val_top1"], -r["mean_val_ce"]))
        frozen[model] = {k: best[k] for k in ("atoms", "fc_init", "k", "model", "weight_decay")}
        print(f"{model}: {len(rows)} xavier candidates -> {json.dumps(best)}", flush=True)
    path = rg.OUT / "tuned.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError(f"{path} already exists with different settings; refusing to overwrite.")
    path.write_text(json.dumps(frozen, indent=2))
    print("Frozen settings:", json.dumps(frozen), flush=True)


def cmd_grid(args, device):
    tuned = rg.load_tuned()
    done_rows = rg.recorded()
    for seed in args.seeds:
        for level in args.levels:
            for cfg in rg.model_configs(tuned):
                if cfg["model"] not in args.models:
                    continue
                run_dir = rg.OUT / "grid" / cfg["model"] / rg.level_name(level) / f"seed{seed}"
                summary = rg.train_one(cfg, level, seed, "grid-xavier", device, run_dir)
                last = run_dir / "last.pt"
                if last.exists() and (run_dir / "done.json").exists():
                    last.unlink()
                rg.test_run(summary, level, device, done_rows)


def main():
    def level_arg(text):
        return None if text == "source" else int(text)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("freeze", "grid", "summary"))
    parser.add_argument("--levels", type=level_arg, nargs="+", default=rg.LEVELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=rg.SEEDS)
    parser.add_argument("--models", nargs="+", default=["sparse", "dense", "linear"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rg.OUT.mkdir(parents=True, exist_ok=True)
    {"freeze": cmd_freeze, "grid": cmd_grid, "summary": rg.cmd_summary}[args.phase](args, device)


if __name__ == "__main__":
    main()
