"""LR-001 analysis, implementing the rule pre-registered in task.json (validation only; no test data).

    python -B ai_harness/artifacts/LR-001/analyze.py   (from Final Project; standard library only)

Data: runs/LR-001/<head>/<lr>/<level>/seed<s>/done.json (54 runs, per-epoch history inside).
Metric: best-checkpoint validation top-1 on the 5k pool validation split. Cells = 2 levels x 3 seeds, paired by cell.
(1) lr* per head = highest mean validation top-1.
(2) Switch rule: replace 1e-3 by lr* only if lr* != 1e-3, mean(A(lr*) - A(1e-3)) >= 0.30 pp and >= 5/6 cells positive.
    Every lower lr vs 1e-3 is reported (6 comparisons, descriptive): mean, sd, paired-t 95% CI (df 5), sign count.
(3) Instability: a run is unstable if training loss rises > 5% between consecutive epochs; 'resolved' if <= 1/6.
    Also the validation top-1 drop from the best epoch to epoch 10 (median, max).
(4) Truncation flag: median best epoch at lr* == 10.
(5) Sanity: lr 1e-3 cells of sparse-M2048 / dense-M2048 vs ABL-001 abl-mse (same settings): |diff|, digests.
Exploratory: sparse M4096 - M2048 and sparse M2048 - dense M2048 gaps per lr, and at each head's lr*.
Writes analysis.md / analysis.json next to this file.
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "LR-001"
HEADS = ["sparse-M2048", "sparse-M4096", "dense-M2048"]
LRS = [1e-3, 3e-4, 1e-4]
LEVELS = ["source", "L090"]
SEEDS = [42, 43, 44]
CELLS = [(lv, s) for lv in LEVELS for s in SEEDS]
T975 = {5: 2.571}
SWITCH_PP, SWITCH_POS = 0.0030, 5
RISE = 1.05


def lr_name(lr):
    return f"lr{lr:.0e}".replace("e-0", "e-")


def paired(xs):
    n = len(xs)
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    half = T975[n - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci_low": m - half, "ci_high": m + half,
            "positive": sum(x > 0 for x in xs), "values": xs}


def unstable(hist):
    losses = [h["train_loss"] for h in hist]
    return any(b > a * RISE for a, b in zip(losses, losses[1:]))


def load():
    runs = {}
    for h in HEADS:
        for lr in LRS:
            for lv, s in CELLS:
                p = RUNS / h / lr_name(lr) / lv / f"seed{s}" / "done.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    # Integrity (added in revision 1, REVIEW-008 O-6): fields match the path, full 10-epoch history.
                    assert (r["head"], r["lr"], r["train_level"], r["seed"]) == (h, lr, lv, s), p
                    assert [e["epoch"] for e in r["history"]] == list(range(1, 11)), p
                    runs[(h, lr, lv, s)] = r
    return runs


def main():
    runs = load()
    missing = [f"{h} {lr_name(lr)} {lv} seed{s}" for h in HEADS for lr in LRS for lv, s in CELLS
               if (h, lr, lv, s) not in runs]
    if missing:
        raise SystemExit(f"{len(missing)} runs missing, e.g. {missing[:3]}")
    acc = {k: v["val_top1"] for k, v in runs.items()}
    digests = {s: {runs[k]["split_digest"] for k in runs if k[3] == s} for s in SEEDS}

    per = {}
    for h in HEADS:
        for lr in LRS:
            rs = [runs[(h, lr, lv, s)] for lv, s in CELLS]
            drops = [r["val_top1"] - r["history"][-1]["val_top1"] for r in rs]
            per[(h, lr)] = {"mean_val_top1": statistics.mean(r["val_top1"] for r in rs),
                            "unstable": sum(unstable(r["history"]) for r in rs),
                            "median_drop": statistics.median(drops), "max_drop": max(drops),
                            "median_best_epoch": statistics.median(r["best_epoch"] for r in rs),
                            "best_epochs": [r["best_epoch"] for r in rs]}

    decisions, comparisons = {}, {}
    for h in HEADS:
        best_lr = max(LRS, key=lambda lr: per[(h, lr)]["mean_val_top1"])
        for lr in LRS[1:]:
            comparisons[f"{h} {lr_name(lr)} - lr1e-3"] = paired(
                [acc[(h, lr, lv, s)] - acc[(h, 1e-3, lv, s)] for lv, s in CELLS])
        if best_lr == 1e-3:
            switch, diff = False, None
        else:
            diff = comparisons[f"{h} {lr_name(best_lr)} - lr1e-3"]
            switch = diff["mean"] >= SWITCH_PP and diff["positive"] >= SWITCH_POS
        decisions[h] = {"lr_star": best_lr, "switch": switch, "recommended_lr": best_lr if switch else 1e-3,
                        "instability_at_1e-3": per[(h, 1e-3)]["unstable"],
                        "instability_resolved_at": [lr for lr in LRS if per[(h, lr)]["unstable"] <= 1],
                        "truncation_flag_at_lr_star": per[(h, best_lr)]["median_best_epoch"] == 10}

    sanity = []
    for h, model in (("sparse-M2048", "sparse"), ("dense-M2048", "dense")):
        for lv, s in CELLS:
            p = ROOT / "runs" / "ABL-001" / "abl-mse" / model / lv / f"seed{s}" / "done.json"
            old = json.loads(p.read_text())
            new = runs[(h, 1e-3, lv, s)]
            sanity.append({"head": h, "level": lv, "seed": s, "abl001": old["val_top1"], "lr001": new["val_top1"],
                           "abs_diff": abs(old["val_top1"] - new["val_top1"]),
                           "same_best_epoch": old["best_epoch"] == new["best_epoch"],
                           "same_digest": old["split_digest"] == new["split_digest"]})

    expl = {}
    for lr in LRS:
        expl[f"sparse M4096 - M2048 @ {lr_name(lr)}"] = paired(
            [acc[("sparse-M4096", lr, lv, s)] - acc[("sparse-M2048", lr, lv, s)] for lv, s in CELLS])
        expl[f"sparse M2048 - dense M2048 @ {lr_name(lr)}"] = paired(
            [acc[("sparse-M2048", lr, lv, s)] - acc[("dense-M2048", lr, lv, s)] for lv, s in CELLS])
    st = {h: decisions[h]["lr_star"] for h in HEADS}
    expl["sparse M4096 - M2048 @ each lr*"] = paired(
        [acc[("sparse-M4096", st["sparse-M4096"], lv, s)] - acc[("sparse-M2048", st["sparse-M2048"], lv, s)]
         for lv, s in CELLS])
    expl["sparse M2048 - dense M2048 @ each lr*"] = paired(
        [acc[("sparse-M2048", st["sparse-M2048"], lv, s)] - acc[("dense-M2048", st["dense-M2048"], lv, s)]
         for lv, s in CELLS])

    res = {"runs": len(runs), "missing": missing, "one_split_per_seed": all(len(d) == 1 for d in digests.values()),
           "per_head_lr": {f"{h} {lr_name(lr)}": v for (h, lr), v in per.items()},
           "comparisons_vs_1e-3": comparisons, "decisions": decisions, "sanity_vs_abl001": sanity,
           "exploratory": expl}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    p = lambda v: f"{100 * v:.2f}"  # noqa: E731
    L = ["# LR-001 analysis (generated by analyze.py; validation only)", "",
         f"Runs: {len(runs)}/54. One split digest per seed across heads and lrs: **{res['one_split_per_seed']}**. "
         "Metric: best-checkpoint validation top-1 (%), 5k pool validation split, cells = source/L090 x seeds 42-44.", "",
         "## Per head and lr", "",
         "| Head | lr | mean val top-1 | unstable runs | median drop best->ep10 (pp) | max drop (pp) | median best epoch | best epochs |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (h, lr), e in per.items():
        L.append(f"| {h} | {lr:g} | {p(e['mean_val_top1'])} | {e['unstable']}/6 | {p(e['median_drop'])} | "
                 f"{p(e['max_drop'])} | {e['median_best_epoch']} | {e['best_epochs']} |")
    L += ["", "## Lower lr vs 1e-3 (paired by cell, descriptive)", "",
          "| Comparison | mean (pp) | sd | 95% CI (df 5) | positive |", "| --- | --- | --- | --- | --- |"]
    for k, e in comparisons.items():
        L.append(f"| {k} | {p(e['mean'])} | {p(e['sd'])} | [{p(e['ci_low'])}, {p(e['ci_high'])}] | {e['positive']}/6 |")
    L += ["", "## Pre-registered decisions", "",
          "| Head | lr* | switch from 1e-3? | recommended lr | unstable at 1e-3 | instability resolved at | truncation flag at lr* |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for h, d in decisions.items():
        L.append(f"| {h} | {d['lr_star']:g} | {'YES' if d['switch'] else 'no'} | {d['recommended_lr']:g} | "
                 f"{d['instability_at_1e-3']}/6 | {', '.join(f'{x:g}' for x in d['instability_resolved_at']) or 'none'} | "
                 f"{'FLAG' if d['truncation_flag_at_lr_star'] else 'ok'} |")
    L += ["", "## Sanity: lr 1e-3 vs ABL-001 abl-mse (identical settings)", "",
          "| Head | level | seed | ABL-001 | LR-001 | abs diff (pp) | same best epoch | same digest |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in sanity:
        L.append(f"| {e['head']} | {e['level']} | {e['seed']} | {p(e['abl001'])} | {p(e['lr001'])} | "
                 f"{p(e['abs_diff'])} | {e['same_best_epoch']} | {e['same_digest']} |")
    L += ["", "## Exploratory (not pre-registered decisions; validation only)", "",
          "| Gap | mean (pp) | 95% CI (df 5) | positive |", "| --- | --- | --- | --- |"]
    for k, e in expl.items():
        L.append(f"| {k} | {p(e['mean'])} | [{p(e['ci_low'])}, {p(e['ci_high'])}] | {e['positive']}/6 |")
    (HERE / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
