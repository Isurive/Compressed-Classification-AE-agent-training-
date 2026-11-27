"""WD-001: can tuned weight decay replace the reconstruction loss in a wide head?

    python -B -u run_wdcheck.py sanity   # wd 0, lr 1e-3, 10 ep: must reproduce ABL-001 abl-none dense source seed 42
    python -B -u run_wdcheck.py val      # phase A: validation only, 144 runs (no test data)
    python -B -u run_wdcheck.py test     # phase B: selected config per head, seeds 45-51 x 3 levels, ImageNetV2 test

(run from src/, ai360 env). Design and analysis are pre-registered in ai_harness/artifacts/WD-001/task.json.

Heads: dense-CE (M2048 k2048) and sparse-CE (M2048 k512), cross-entropy only (ABL-001 'abl-none' weights:
reconstruction 1e-6, dependency 0, coherence 0), Xavier fc, He encoder, batch 128, constant lr, 20 epochs.

Weight decay: sparse_workflow.train builds `torch.optim.Adam([*sae.parameters(), *model.fc.parameters()], lr=...)`
with no weight decay, and the user's src/ files are not edited. During each call this wrapper temporarily
replaces sparse_workflow.new_parts (to see the model) and torch.optim.Adam (to split the same parameters into two
groups): coupled L2 `weight_decay=wd` on the deployed head's linear layers (encoder weight+bias, classifier
weight+bias), 0 on the learned centre and the training-only decoder atoms. It asserts that the parameters passed
are exactly the expected ones, logs the groups, and restores both names in a finally block.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import torch

import run_grid as rg
import run_ablation as ra  # importing sets rg.OUT to ABL-001; it is reset below before any use.
import sparse_workflow as sw
from sparse_workflow import Settings

rg.OUT = rg.ROOT / "runs" / "WD-001"  # FEATURES stays runs/GRID-001/features (read only).
ART = rg.ROOT / "ai_harness" / "artifacts" / "WD-001"
EPOCHS = 20
CE_ONLY = ra.VARIANTS["abl-none"]
HEADS = {
    "dense-CE": {"model": "dense", "atoms": 2048, "k": 2048, "fc_init": "xavier"},
    "sparse-CE": {"model": "sparse", "atoms": 2048, "k": 512, "fc_init": "xavier"},
}
VAL_LRS = [3e-4, 1e-4, 3e-5]
VAL_WDS = [0.0, 1e-3, 1e-2, 1e-1]
VAL_LEVELS, VAL_SEEDS = [None, 90], [42, 43, 44]
TEST_LEVELS, TEST_SEEDS = [None, 80, 90], [45, 46, 47, 48, 49, 50, 51]


def tag(lr, wd):
    return f"lr{lr:.0e}_wd{wd:.0e}".replace("e-0", "e-").replace("e+0", "e")


def train_ce(head, lr, wd, epochs, level, seed, device, run_dir, phase):
    """One CE-only dictionary-head run with weight decay on the deployed linear layers."""
    done = run_dir / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    if (run_dir / "best.pt").exists():
        raise ValueError(f"{run_dir} has a partial run (best.pt without done.json). Remove it and rerun.")
    cfg = HEADS[head]
    part = rg.pool_split(seed)
    cache = rg.cache_for(part, level)
    print(f"[{phase}] {head} M={cfg['atoms']} k={cfg['k']} lr={lr:g} wd={wd:g} epochs={epochs} "
          f"{rg.level_name(level)} seed {seed}", flush=True)
    settings = Settings(data=str(rg.POOL.resolve()), train_limit=0, val_limit=0, holdout=rg.HOLDOUT, seed=seed,
                        atoms=cfg["atoms"], k=cfg["k"], compression_level=level, batch_size=rg.BATCH, lr=lr,
                        encoder_init="he", fc_init=cfg["fc_init"], cache="memory", **CE_ONLY)
    holder, groups_log = {}, {}
    orig_new_parts, orig_adam = sw.new_parts, torch.optim.Adam

    def new_parts(c, n):
        parts = orig_new_parts(c, n)
        holder["model"], holder["sae"] = parts[0], parts[1]
        return parts

    def adam(params, lr, **kwargs):
        params = list(params)
        model, sae = holder["model"], holder["sae"]
        expected = [*sae.parameters(), *model.fc.parameters()]
        assert [id(p) for p in params] == [id(p) for p in expected], "unexpected optimizer parameters"
        decay_ids = {id(p) for p in (model.encoder.weight, model.encoder.bias, model.fc.weight, model.fc.bias)}
        decay = [p for p in params if id(p) in decay_ids]
        rest = [p for p in params if id(p) not in decay_ids]
        assert len(decay) == 4 and len(decay) + len(rest) == len(params) and not kwargs.get("weight_decay")
        groups_log.update(decay=[list(p.shape) for p in decay], no_decay=[list(p.shape) for p in rest], wd=wd)
        print(f"  wd groups: decay {groups_log['decay']} (wd={wd:g}); no decay {groups_log['no_decay']}", flush=True)
        return orig_adam([{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}],
                         lr=lr, **kwargs)

    start = time.time()
    sw.new_parts, torch.optim.Adam = new_parts, adam
    try:
        raw = sw.train(settings, run_dir, epochs, 0, device, partitions=part, fold_cache=cache)
    finally:
        sw.new_parts, torch.optim.Adam = orig_new_parts, orig_adam
    history = [{"epoch": r["epoch"], "train_loss": r["train"]["loss"], "train_ce": r["train"]["ce"],
                "train_top1": r["train"]["top1"], "val_top1": r["validation"]["top1"],
                "val_ce": r["validation"]["ce"]} for r in raw]
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    summary = {"head": head, **cfg, **CE_ONLY, "lr": lr, "weight_decay": wd, "epochs": epochs,
               "train_level": rg.level_name(level), "seed": seed, "phase": phase,
               "split_digest": best["protocol"]["digest"], "best_epoch": best["epoch"],
               "val_top1": best["validation"]["top1"], "val_ce": best["validation"]["ce"],
               "run_dir": run_dir.relative_to(rg.ROOT).as_posix(), "train_seconds": round(time.time() - start, 1),
               "wd_groups": groups_log, "history": history}
    done.write_text(json.dumps(summary, indent=2))
    last = run_dir / "last.pt"
    if last.exists():
        last.unlink()
    return summary


def cmd_sanity(args, device):
    run_dir = rg.OUT / "sanity" / "dense-CE_lr1e-3_wd0_10ep" / "source" / "seed42"
    s = train_ce("dense-CE", 1e-3, 0.0, 10, None, 42, device, run_dir, "wd-sanity")
    ref = json.loads((rg.ROOT / "runs/ABL-001/abl-none/dense/source/seed42/done.json").read_text())
    same = all(s[k] == ref[k] for k in ("split_digest", "best_epoch", "val_top1", "val_ce"))
    print(json.dumps({"new": {k: s[k] for k in ("split_digest", "best_epoch", "val_top1", "val_ce")},
                      "abl001": {k: ref[k] for k in ("split_digest", "best_epoch", "val_top1", "val_ce")},
                      "identical": same}, indent=2), flush=True)


def cmd_val(args, device):
    (rg.OUT / "settings_val.json").write_text(json.dumps(
        {"heads": HEADS, "loss": CE_ONLY, "lrs": VAL_LRS, "wds": VAL_WDS, "epochs": EPOCHS,
         "levels": [rg.level_name(x) for x in VAL_LEVELS], "seeds": VAL_SEEDS,
         "test_evaluation": "none (validation only)"}, indent=2))
    for seed in args.seeds or VAL_SEEDS:
        for level in VAL_LEVELS:
            for head in args.heads:
                for lr in VAL_LRS:
                    for wd in VAL_WDS:
                        run_dir = rg.OUT / "val" / head / tag(lr, wd) / rg.level_name(level) / f"seed{seed}"
                        train_ce(head, lr, wd, EPOCHS, level, seed, device, run_dir, "wd-val")
    print("phase A finished", flush=True)


def cmd_test(args, device):
    selected = json.loads((ART / "selected.json").read_text())
    (rg.OUT / "settings_test.json").write_text(json.dumps(
        {"heads": HEADS, "loss": CE_ONLY, "selected": selected, "levels": [rg.level_name(x) for x in TEST_LEVELS],
         "seeds": TEST_SEEDS, "test": "ImageNetV2 matched-frequency, eval at source and at the training level"},
        indent=2))
    done_rows = rg.recorded()
    for seed in args.seeds or TEST_SEEDS:
        for level in TEST_LEVELS:
            for head in args.heads:
                c = selected[head]
                run_dir = rg.OUT / "test" / head / rg.level_name(level) / f"seed{seed}"
                summary = train_ce(head, c["lr"], c["weight_decay"], c["epochs"], level, seed, device, run_dir,
                                   "wd-test")
                rg.test_run(summary, level, device, done_rows)
    print("phase B finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("sanity", "val", "test"))
    parser.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    rg.OUT.mkdir(parents=True, exist_ok=True)
    {"sanity": cmd_sanity, "val": cmd_val, "test": cmd_test}[args.phase](args, device)


if __name__ == "__main__":
    main()
