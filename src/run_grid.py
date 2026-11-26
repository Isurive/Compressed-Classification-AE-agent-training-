"""GRID-001: compressed-training experiment orchestrator (plan: ai_harness/artifacts/REASON-002/plan.md).

Phases, run in order from src/ with the ai360 environment:

    python run_grid.py features   # frozen ResNet-18 features of the 50k pool and ImageNetV2, once per level
    python run_grid.py tune       # seed 42, levels source/90, 5k validation only -> tuned.json
    python run_grid.py grid       # 8 levels x {sparse, dense, linear} x seeds 42-44, then ImageNetV2 test
    python run_grid.py msweep     # dictionary size M at the tuned k, levels source/80/90/95, seeds 42-44
    python run_grid.py summary    # mean +/- sd tables from results.csv

Nothing here changes how a model is trained. Sparse and dense models are trained by the
user's sparse_workflow.train through its cross-validation feature-cache path (FoldFeatureCache),
so the cached pool features replace per-run extraction. The linear head mirrors baseline_head.py.
The split for each seed comes from image_data.training_splits exactly as main.py makes it.

ImageNetV2 is read only by `grid` and `msweep`, after tuned.json is frozen. `tune` never reads it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset
from torchvision.models import resnet18

from baseline_head import FORMAT as LINEAR_FORMAT, FrozenFeatures, evaluate as evaluate_linear
from image_data import ImageView, evaluation_split, reference_ids, training_splits
from resnet18_sparse_ae import atomic_torch_save
from sparse_workflow import WEIGHTS, FoldFeatureCache, Settings, loader, read_checkpoint, restore_predictor, run_writer
from sparse_workflow import train as train_dictionary

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "Datasets" / "imagenet-1k" / "val"
TEST = ROOT / "Datasets" / "imagenetv2-matched-frequency"
OUT = ROOT / "runs" / "GRID-001"
FEATURES = OUT / "features"
LEVELS = [None, 20, 60, 80, 85, 90, 95, 100]
SEEDS = [42, 43, 44]
HOLDOUT, EPOCHS, BATCH, LR = 0.10, 10, 128, 1e-3
TUNE_LEVELS, TUNE_SEED = [None, 90], 42
MSWEEP_LEVELS, MSWEEP_ATOMS = [None, 80, 90, 95], [512, 1024, 2048, 4096]
RESULT_FIELDS = ["phase", "model", "atoms", "k", "fc_init", "weight_decay", "train_level", "seed", "split_digest",
                 "best_epoch", "val_top1", "val_ce", "eval_set", "eval_level", "top1", "top5", "ce", "run_dir", "train_seconds"]


def level_name(level):
    return "source" if level is None else f"L{level:03d}"


def keys_digest(source, indices):
    digest = hashlib.sha256()
    for i in indices:
        digest.update(f"{source.key(i)}\t{source.labels[i]}\n".encode())
    return digest.hexdigest()


@torch.no_grad()
def extract(model, source, indices, level, workers, device):
    features, labels = [], []
    view = ImageView(source, indices, level)
    for step, (images, target) in enumerate(loader(view, BATCH, workers, device), start=1):
        features.append(model.features(images.to(device, non_blocking=True)).float().cpu())
        labels.append(target)
        if step % 50 == 0:
            print(f"    {min(step * BATCH, len(view)):,}/{len(view):,}", flush=True)
    return torch.cat(features), torch.cat(labels)


@lru_cache(maxsize=None)
def pool_split(seed):
    """The exact split main.py makes for `--holdout 0.10 --train-limit 0 --val-limit 0 --seed seed`."""
    return training_splits(str(POOL.resolve()), 0, 0, HOLDOUT, seed)


def pool_features(level):
    path = FEATURES / f"pool_{level_name(level)}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python run_grid.py features")
    return torch.load(path, weights_only=True)


def test_features(level):
    path = FEATURES / f"test_{level_name(level)}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python run_grid.py features")
    return torch.load(path, weights_only=True)


def cmd_features(args, device):
    FEATURES.mkdir(parents=True, exist_ok=True)
    tr = pool_split(SEEDS[0])[0]
    classes, ids = tr.classes, reference_ids(tr.classes, "auto")
    pool_ids = range(len(tr.labels))
    pool_digest = keys_digest(tr, pool_ids)
    # entire=True: every ImageNetV2 image, relabeled into the pool's class order through the ImageNet IDs.
    test_source, test_idx = evaluation_split(str(TEST.resolve()), classes,
                                             {"folds": 0, "seed": SEEDS[0], "holdout": HOLDOUT},
                                             "single-pool holdout", 0, True, ids)
    test_digest = keys_digest(test_source, test_idx)
    model = FrozenFeatures().to(device)
    for level in args.levels:
        for kind, source, indices, digest in (("pool", tr, pool_ids, pool_digest),
                                              ("test", test_source, test_idx, test_digest)):
            path = FEATURES / f"{kind}_{level_name(level)}.pt"
            if path.exists():
                print(f"{path.name} exists, skipping", flush=True)
                continue
            start = time.time()
            print(f"Extracting {kind} features at {level_name(level)} ({len(indices):,} images)", flush=True)
            feats, labels = extract(model, source, indices, level, args.workers, device)
            atomic_torch_save({"features": feats, "labels": labels, "level": level, "keys_digest": digest,
                               "classes": classes, "seconds": time.time() - start}, path)
            print(f"  saved {path.name} in {time.time() - start:.0f} s", flush=True)


def cache_for(part, level):
    tr = part[0]
    data = pool_features(level)
    if data["keys_digest"] != keys_digest(tr, range(len(tr.labels))) or data["labels"].tolist() != tr.labels:
        raise ValueError(f"Cached pool features for {level_name(level)} do not match the current pool order.")
    cache = FoldFeatureCache()
    cache.data = TensorDataset(data["features"], data["labels"])
    return cache


def train_linear(part, cache, ids, level, seed, cfg, out, device):
    tr, ti, va, vi, split, digest = part
    train_data, val_data = cache.select(ti), cache.select(vi)
    torch.manual_seed(seed)
    teacher = resnet18(weights=WEIGHTS).fc.requires_grad_(False).to(device)
    head = nn.Linear(512, len(tr.classes)).to(device)
    with torch.no_grad():
        if cfg["fc_init"] == "original":
            rows = torch.as_tensor(ids, device=device)
            head.weight.copy_(teacher.weight[rows])
            head.bias.copy_(teacher.bias[rows])
        else:
            nn.init.xavier_uniform_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=cfg["weight_decay"])
    holder = SimpleNamespace(teacher=teacher, features=None)
    val_batches = loader(val_data, BATCH, 0, device)
    protocol = {"kind": split, "holdout": HOLDOUT, "seed": seed, "val_limit": 0, "digest": digest}
    history, best_score = [], (-1.0, float("-inf"))
    with run_writer(out):
        for epoch in range(1, EPOCHS + 1):
            torch.manual_seed(seed + epoch)
            head.train()
            for features, labels in loader(train_data, BATCH, 0, device, shuffle=True, seed=seed + epoch):
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(head(features), labels).backward()
                optimizer.step()
            metrics = evaluate_linear(holder, head, val_batches, device, ids)
            history.append({"epoch": epoch, "validation": metrics})
            score = (metrics["top1"], -metrics["ce"])
            if score > best_score:
                best_score = score
                atomic_torch_save({"format": LINEAR_FORMAT, "head": head.state_dict(), "classes": list(tr.classes),
                                   "reference_ids": ids, "compression_level": level, "epoch": epoch,
                                   "protocol": protocol, "validation": metrics,
                                   "settings": {"epochs": EPOCHS, "batch_size": BATCH, "lr": LR, **cfg}},
                                  out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))


def train_one(model_cfg, level, seed, phase, device, run_dir):
    """Train one configuration unless already complete. Returns the run summary dict."""
    done = run_dir / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    if (run_dir / "best.pt").exists():
        raise ValueError(f"{run_dir} has a partial run (best.pt without done.json). Remove it and rerun.")
    part = pool_split(seed)
    tr = part[0]
    ids = reference_ids(tr.classes, "auto")
    cache = cache_for(part, level)
    start = time.time()
    print(f"[{phase}] {model_cfg['model']} M={model_cfg['atoms']} k={model_cfg['k']} {level_name(level)} seed {seed}",
          flush=True)
    if model_cfg["model"] == "linear":
        train_linear(part, cache, ids, level, seed, model_cfg, run_dir, device)
    else:
        cfg = Settings(data=str(POOL.resolve()), train_limit=0, val_limit=0, holdout=HOLDOUT, seed=seed,
                       atoms=model_cfg["atoms"], k=model_cfg["k"], compression_level=level, batch_size=BATCH, lr=LR,
                       encoder_init="he", fc_init=model_cfg["fc_init"], cache="memory")
        train_dictionary(cfg, run_dir, EPOCHS, 0, device, partitions=part, fold_cache=cache)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    summary = {**model_cfg, "train_level": level_name(level), "seed": seed, "phase": phase,
               "split_digest": best["protocol"]["digest"], "best_epoch": best["epoch"],
               "val_top1": best["validation"]["top1"], "val_ce": best["validation"]["ce"],
               "run_dir": run_dir.relative_to(ROOT).as_posix(), "train_seconds": round(time.time() - start, 1)}
    done.write_text(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def score(logits_fn, data, device):
    feats, labels = data["features"], data["labels"]
    n, top1, top5, ce = len(labels), 0, 0, 0.0
    for i in range(0, n, 1024):
        f, y = feats[i:i + 1024].to(device), labels[i:i + 1024].to(device)
        logits = logits_fn(f)
        winners = logits.topk(5, dim=1).indices
        top1 += (winners[:, 0] == y).sum().item()
        top5 += (winners == y[:, None]).any(dim=1).sum().item()
        ce += F.cross_entropy(logits, y, reduction="sum").item()
    return {"top1": top1 / n, "top5": top5 / n, "ce": ce / n}


def predictor(run_dir, model, device):
    if model == "linear":
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        head = nn.Linear(512, len(state["classes"])).to(device)
        head.load_state_dict(state["head"])
        return head.eval()
    m = restore_predictor(read_checkpoint(run_dir / "best.pt"), device)
    return lambda f: m.fc(m.encode_features(f))


def append_results(rows):
    path = OUT / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def recorded():
    path = OUT / "results.csv"
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["phase"], r["run_dir"], r["eval_set"], r["eval_level"]) for r in csv.DictReader(f)}


def test_run(summary, level, device, done_rows):
    """Score one finished run on ImageNetV2 at source (primary) and at its training level (secondary)."""
    run_dir = ROOT / summary["run_dir"]
    fn = None
    rows = []
    for eval_level in dict.fromkeys([None, level]):
        key = (summary["phase"], summary["run_dir"], "imagenetv2", level_name(eval_level))
        if key in done_rows:
            continue
        fn = fn or predictor(run_dir, summary["model"], device)
        rows.append({**summary, "eval_set": "imagenetv2", "eval_level": level_name(eval_level),
                     **score(fn, test_features(eval_level), device)})
    append_results(rows)


def model_configs(tuned=None):
    if tuned is not None:
        return [tuned["sparse"], tuned["dense"], tuned["linear"]]
    return [{"model": "sparse", "atoms": 2048, "k": 32, "fc_init": "xavier", "weight_decay": 0.0}]


def cmd_tune(args, device):
    """Validation-only search, equal spirit across models. Rule (pre-declared in REASON-002 / GRID-001 report):
    pick the configuration with the highest mean 5k-validation top-1 over levels source and 90,
    ties broken by lower mean validation CE."""
    candidates = ([{"model": "sparse", "atoms": 2048, "k": k, "fc_init": init, "weight_decay": 0.0}
                   for k in (32, 128, 512) for init in ("xavier", "auto")]
                  + [{"model": "dense", "atoms": 2048, "k": 2048, "fc_init": init, "weight_decay": 0.0}
                     for init in ("xavier", "auto")]
                  + [{"model": "linear", "atoms": 0, "k": 0, "fc_init": init, "weight_decay": wd}
                     for wd in (0.0, 1e-4, 1e-3) for init in ("xavier", "original")])
    results = []
    for cfg in candidates:
        for level in TUNE_LEVELS:
            tag = f"{cfg['model']}_M{cfg['atoms']}_k{cfg['k']}_{cfg['fc_init']}_wd{cfg['weight_decay']:g}"
            run_dir = OUT / "tune" / tag / level_name(level)
            summary = train_one(cfg, level, TUNE_SEED, "tune", device, run_dir)
            last = run_dir / "last.pt"
            if last.exists():
                last.unlink()  # Tuning runs keep best.pt only (decided with the user to limit disk use).
            results.append(summary)
    by_cfg = defaultdict(list)
    for r in results:
        by_cfg[json.dumps({k: r[k] for k in ("model", "atoms", "k", "fc_init", "weight_decay")}, sort_keys=True)].append(r)
    table, tuned = [], {}
    for key, rs in by_cfg.items():
        cfg = json.loads(key)
        mean_top1 = statistics.fmean(r["val_top1"] for r in rs)
        mean_ce = statistics.fmean(r["val_ce"] for r in rs)
        table.append({**cfg, "mean_val_top1": mean_top1, "mean_val_ce": mean_ce,
                      **{f"val_top1_{r['train_level']}": r["val_top1"] for r in rs}})
        best = tuned.get(cfg["model"])
        if best is None or (mean_top1, -mean_ce) > (best[0], -best[1]):
            tuned[cfg["model"]] = (mean_top1, mean_ce, cfg)
    table.sort(key=lambda r: (r["model"], -r["mean_val_top1"]))
    (OUT / "tune_results.json").write_text(json.dumps(table, indent=2))
    frozen = {m: v[2] for m, v in tuned.items()}
    (OUT / "tuned.json").write_text(json.dumps(frozen, indent=2))
    for row in table:
        print(json.dumps(row), flush=True)
    print("Frozen settings:", json.dumps(frozen), flush=True)


def load_tuned():
    path = OUT / "tuned.json"
    if not path.exists():
        raise FileNotFoundError("Run `python run_grid.py tune` first; the grid uses frozen tuned settings.")
    return json.loads(path.read_text())


def cmd_grid(args, device):
    tuned = load_tuned()
    done_rows = recorded()
    for seed in args.seeds:
        for level in args.levels:
            for cfg in model_configs(tuned):
                if cfg["model"] not in args.models:
                    continue
                run_dir = OUT / "grid" / cfg["model"] / level_name(level) / f"seed{seed}"
                summary = train_one(cfg, level, seed, "grid", device, run_dir)
                test_run(summary, level, device, done_rows)


def cmd_msweep(args, device):
    tuned = load_tuned()
    done_rows = recorded()
    k = tuned["sparse"]["k"]
    for seed in args.seeds:
        for level in MSWEEP_LEVELS:
            for atoms in MSWEEP_ATOMS:
                if atoms == tuned["sparse"]["atoms"]:
                    continue  # Identical to the grid's sparse run; analysis reuses those rows.
                if k > atoms:
                    print(f"Skipping M={atoms}: tuned k={k} exceeds it.", flush=True)
                    continue
                cfg = {**tuned["sparse"], "atoms": atoms}
                run_dir = OUT / "msweep" / f"M{atoms}" / level_name(level) / f"seed{seed}"
                summary = train_one(cfg, level, seed, "msweep", device, run_dir)
                test_run(summary, level, device, done_rows)


def cmd_reference(args, device):
    """The original ImageNet fc on the same ImageNetV2 features (not trained here; a ceiling reference)."""
    teacher = resnet18(weights=WEIGHTS).fc.to(device).eval()
    tr = pool_split(SEEDS[0])[0]
    rows_ids = torch.as_tensor(reference_ids(tr.classes, "auto"), device=device)
    rows = []
    for level in LEVELS:
        rows.append({"phase": "reference", "model": "original_fc", "train_level": "imagenet-train", "eval_set": "imagenetv2",
                     "eval_level": level_name(level),
                     **score(lambda f: teacher(f).index_select(1, rows_ids), test_features(level), device)})
    (OUT / "reference_original_fc.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(json.dumps(r), flush=True)


def cmd_summary(args, device):
    with open(OUT / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["phase"], r["model"], r["atoms"], r["k"], r["train_level"], r["eval_set"], r["eval_level"])].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        t = [float(r["top1"]) for r in rs]
        out.append({"phase": key[0], "model": key[1], "atoms": key[2], "k": key[3], "train_level": key[4],
                    "eval_set": key[5], "eval_level": key[6], "n_seeds": len(rs),
                    "top1_mean": statistics.fmean(t), "top1_sd": statistics.stdev(t) if len(t) > 1 else 0.0,
                    "seeds": sorted(int(r["seed"]) for r in rs)})
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote {OUT / 'summary.csv'} ({len(out)} groups)")


def main():
    def level_arg(text):
        return None if text == "source" else int(text)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("features", "tune", "grid", "msweep", "reference", "summary"))
    parser.add_argument("--levels", type=level_arg, nargs="+", default=LEVELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=["sparse", "dense", "linear"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    {"features": cmd_features, "tune": cmd_tune, "grid": cmd_grid, "msweep": cmd_msweep,
     "reference": cmd_reference, "summary": cmd_summary}[args.phase](args, device)


if __name__ == "__main__":
    main()
