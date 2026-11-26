"""MSWEEP-001: clean dictionary-size sweep, M in {512, 1024, 4096} with k = M/4 (M = 2048 reused from SEED-001).

    python -B -u run_msweep.py grid       # 3 sizes x 3 levels x seeds 45-51 (63 runs), then ImageNetV2 test
    python -B -u run_msweep.py summary

(run from src/, ai360 env). Design and analysis are pre-registered in ai_harness/artifacts/MSWEEP-001/task.json.

Every setting equals SEED-001's sparse head except atoms/k: Xavier fc, He encoder, loss CE + feature MSE
(reconstruction 1, dependency 0, coherence 0), 10 epochs, batch 128, Adam 1e-3, cached GRID-001 features.
Reuses run_grid.py / run_ablation.py unchanged; only the output folder and sizes differ. Writes runs/MSWEEP-001.
"""
from __future__ import annotations

import argparse
import json

import torch

import run_grid as rg
import run_ablation as ra  # importing sets rg.OUT to ABL-001; it is reset below before any use.

rg.OUT = rg.ROOT / "runs" / "MSWEEP-001"  # FEATURES stays runs/GRID-001/features (read only).
TUNED = rg.ROOT / "runs" / "GRID-002" / "tuned.json"
LEVELS = [None, 80, 90]
SEEDS = [45, 46, 47, 48, 49, 50, 51]
ATOMS = [512, 1024, 4096]  # 2048 is reused from runs/SEED-001 (sparse)
ACTIVE_FRACTION = 4  # k = M / 4, the share of active atoms at the tuned M = 2048, k = 512 point
MSE_ONLY = ra.VARIANTS["abl-mse"]


def config(atoms):
    base = json.loads(TUNED.read_text())["sparse"]
    return {**base, "atoms": atoms, "k": atoms // ACTIVE_FRACTION}


def cmd_grid(args, device):
    configs = {m: config(m) for m in args.atoms}
    (rg.OUT / "settings.json").write_text(json.dumps({"configs": {str(m): c for m, c in configs.items()},
                                                      "dictionary_loss": MSE_ONLY,
                                                      "levels": [rg.level_name(x) for x in args.levels],
                                                      "seeds": args.seeds, "reused": "M2048 from runs/SEED-001 sparse"},
                                                     indent=2))
    done_rows = rg.recorded()
    for seed in args.seeds:
        for level in args.levels:
            for atoms, cfg in configs.items():
                run_dir = rg.OUT / f"M{atoms}" / rg.level_name(level) / f"seed{seed}"
                summary = ra.train_variant(cfg, MSE_ONLY, level, seed, "msweep-clean", device, run_dir)
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
    parser.add_argument("--atoms", type=int, nargs="+", default=ATOMS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rg.OUT.mkdir(parents=True, exist_ok=True)
    {"grid": cmd_grid, "summary": rg.cmd_summary}[args.phase](args, device)


if __name__ == "__main__":
    main()
