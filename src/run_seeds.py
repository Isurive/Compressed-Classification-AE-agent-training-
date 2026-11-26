"""SEED-001: more seeds for sparse vs dense (both CE + feature MSE) and linear, at levels source/80/90.

    python -B -u run_seeds.py grid       # 3 models x 3 levels x seeds 45-51 (63 runs), then ImageNetV2 test
    python -B -u run_seeds.py summary

(run from src/, ai360 env). Design and analysis are pre-registered in ai_harness/artifacts/SEED-001/task.json.

Settings are exactly the ones already used:
  sparse / dense: GRID-002 tuned.json (Xavier fc, M 2048, k 512 / 2048) with ABL-001 `abl-mse` loss weights
                  (reconstruction 1, dependency 0, coherence 0), trained with run_ablation.train_variant.
  linear:         GRID-002 tuned.json (Xavier, weight decay 1e-4), trained with run_grid.train_one.
Reuses run_grid.py / run_ablation.py unchanged; only the output folder and seeds differ. Writes runs/SEED-001.
"""
from __future__ import annotations

import argparse
import json

import torch

import run_grid as rg
import run_ablation as ra  # importing sets rg.OUT to ABL-001; it is reset below before any use.

rg.OUT = rg.ROOT / "runs" / "SEED-001"  # FEATURES stays runs/GRID-001/features (read only).
TUNED = rg.ROOT / "runs" / "GRID-002" / "tuned.json"
LEVELS = [None, 80, 90]
SEEDS = [45, 46, 47, 48, 49, 50, 51]
MSE_ONLY = ra.VARIANTS["abl-mse"]


def cmd_grid(args, device):
    tuned = json.loads(TUNED.read_text())
    (rg.OUT / "settings.json").write_text(json.dumps({"base": tuned, "dictionary_loss": MSE_ONLY,
                                                      "levels": [rg.level_name(x) for x in args.levels],
                                                      "seeds": args.seeds}, indent=2))
    done_rows = rg.recorded()
    for seed in args.seeds:
        for level in args.levels:
            for name in ("sparse", "dense", "linear"):
                cfg = tuned[name]
                run_dir = rg.OUT / name / rg.level_name(level) / f"seed{seed}"
                if name == "linear":
                    summary = rg.train_one(cfg, level, seed, "seed-linear", device, run_dir)
                else:
                    summary = ra.train_variant(cfg, MSE_ONLY, level, seed, "seed-mse", device, run_dir)
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
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rg.OUT.mkdir(parents=True, exist_ok=True)
    {"grid": cmd_grid, "summary": rg.cmd_summary}[args.phase](args, device)


if __name__ == "__main__":
    main()
