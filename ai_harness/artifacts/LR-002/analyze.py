"""LR-002 analysis, implementing the rule pre-registered in task.json (validation only; no test data).

    python -B ai_harness/artifacts/LR-002/analyze.py   (from Final Project; standard library only)

Data: runs/LR-002/<head>/<lr>/<level>/seed<s>/done.json (78 runs, 20-epoch history inside) and, for the dictionary
heads' lr 3e-4 candidate, runs/LR-001/<head>/lr3e-4/... (10-epoch history).
Selection within a budget n: the epoch <= n with the highest validation top-1, then lower CE (the best.pt rule).
Cells = source/L090 x seeds 42-44, paired by cell.
(1) c* = candidate (lr, budget) with the highest mean validation top-1.
(2) Baseline b: sparse M2048 3e-4@10, sparse M4096 1e-4@10, dense M2048 1e-4@10, linear 1e-3@10. Adopt c* only if
    c* != b, mean(A(c*) - A(b)) >= 0.30 pp and >= 5/6 cells positive; else keep b.
(3) Flags on the adopted setting: lower edge (dict 1e-5, linear 1e-4), upper edge (linear 3e-3), budget binds
    (adopted budget 20 and median selected epoch >= 19).
(4) Instability (dictionary heads): training loss rises > 5% between consecutive epochs, over 20 epochs.
(5) Sanity: 1e-4 first-10 selection vs LR-001 lr1e-4 (18 cells); linear 1e-3 first-10 vs GRID-002 grid linear (6 cells).
Exploratory: sparse M2048 - dense, sparse M2048 - linear, sparse M4096 - sparse M2048 at the adopted settings.
Writes analysis.md / analysis.json next to this file.
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "LR-002"
LR001 = ROOT / "runs" / "LR-001"
DICT = ["sparse-M2048", "sparse-M4096", "dense-M2048"]
HEADS = DICT + ["linear"]
DICT_LRS, LINEAR_LRS = [1e-4, 3e-5, 1e-5], [3e-3, 1e-3, 3e-4, 1e-4]
LEVELS, SEEDS = ["source", "L090"], [42, 43, 44]
CELLS = [(lv, s) for lv in LEVELS for s in SEEDS]
BUDGETS = [10, 20]
BASELINE = {"sparse-M2048": (3e-4, 10), "sparse-M4096": (1e-4, 10), "dense-M2048": (1e-4, 10), "linear": (1e-3, 10)}
T975 = {5: 2.571}
SWITCH_PP, SWITCH_POS, RISE = 0.0030, 5, 1.05


def lr_name(lr):
    return f"lr{lr:.0e}".replace("e-0", "e-")


def cname(c):
    return f"{c[0]:g}@{c[1]}ep"


def select(history, n):
    return max((e for e in history if e["epoch"] <= n), key=lambda e: (e["val_top1"], -e["val_ce"]))


def paired(xs):
    n = len(xs)
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    half = T975[n - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci_low": m - half, "ci_high": m + half,
            "positive": sum(x > 0 for x in xs), "values": xs}


def load():
    runs = {}
    for h in HEADS:
        for lr in (LINEAR_LRS if h == "linear" else DICT_LRS):
            for lv, s in CELLS:
                p = RUNS / h / lr_name(lr) / lv / f"seed{s}" / "done.json"
                r = json.loads(p.read_text())
                assert (r["head"], r["lr"], r["train_level"], r["seed"], r["epochs"]) == (h, lr, lv, s, 20), p
                assert [e["epoch"] for e in r["history"]] == list(range(1, 21)), p
                sel = select(r["history"], 20)
                assert (sel["epoch"], sel["val_top1"]) == (r["best_epoch"], r["val_top1"]), p  # matches best.pt
                runs[(h, lr, lv, s)] = r
    old = {}
    for h in DICT:
        for lv, s in CELLS:
            p = LR001 / h / "lr3e-4" / lv / f"seed{s}" / "done.json"
            r = json.loads(p.read_text())
            assert [e["epoch"] for e in r["history"]] == list(range(1, 11)), p
            old[(h, lv, s)] = r
    return runs, old


def main():
    runs, old = load()
    digests = {s: {r["split_digest"] for k, r in runs.items() if k[3] == s} for s in SEEDS}

    # Candidate accuracies: acc[h][(lr, budget)][(lv, s)] = (val_top1, epoch)
    acc = {h: {} for h in HEADS}
    for h in HEADS:
        for lr in (LINEAR_LRS if h == "linear" else DICT_LRS):
            for b in BUDGETS:
                acc[h][(lr, b)] = {}
                for lv, s in CELLS:
                    e = select(runs[(h, lr, lv, s)]["history"], b)
                    acc[h][(lr, b)][(lv, s)] = (e["val_top1"], e["epoch"])
        if h in DICT:
            acc[h][(3e-4, 10)] = {(lv, s): (old[(h, lv, s)]["val_top1"], old[(h, lv, s)]["best_epoch"]) for lv, s in CELLS}

    cand, decisions, vs_base = {}, {}, {}
    for h in HEADS:
        for c, cells in acc[h].items():
            cand[f"{h} {cname(c)}"] = {"mean_val_top1": statistics.mean(v[0] for v in cells.values()),
                                       "median_epoch": statistics.median(v[1] for v in cells.values()),
                                       "epochs": [cells[x][1] for x in CELLS]}
        base = BASELINE[h]
        for c in acc[h]:
            if c != base:
                vs_base[f"{h} {cname(c)} - {cname(base)}"] = paired(
                    [acc[h][c][x][0] - acc[h][base][x][0] for x in CELLS])
        best = max(acc[h], key=lambda c: statistics.mean(v[0] for v in acc[h][c].values()))
        if best == base:
            adopt, diff = False, None
        else:
            diff = vs_base[f"{h} {cname(best)} - {cname(base)}"]
            adopt = diff["mean"] >= SWITCH_PP and diff["positive"] >= SWITCH_POS
        adopted = best if adopt else base
        lrs = LINEAR_LRS if h == "linear" else [3e-4] + DICT_LRS
        med = statistics.median(v[1] for v in acc[h][adopted].values())
        flags = []
        if adopted[0] == min(lrs):
            flags.append("lower edge")
        if h == "linear" and adopted[0] == max(lrs):
            flags.append("upper edge")
        if adopted[1] == 20 and med >= 19:
            flags.append("budget binds")
        decisions[h] = {"best_candidate": cname(best), "baseline": cname(base), "adopt_best": adopt,
                        "adopted": cname(adopted), "adopted_lr": adopted[0], "adopted_budget": adopted[1],
                        "flags": flags, "located": not flags}

    instab = {}
    for h in DICT:
        for lr in DICT_LRS:
            n = 0
            for lv, s in CELLS:
                losses = [e["train_loss"] for e in runs[(h, lr, lv, s)]["history"]]
                n += any(b > a * RISE for a, b in zip(losses, losses[1:]))
            instab[f"{h} {lr_name(lr)}"] = n

    sanity = []
    for h in DICT:
        for lv, s in CELLS:
            o = json.loads((LR001 / h / "lr1e-4" / lv / f"seed{s}" / "done.json").read_text())
            r = runs[(h, 1e-4, lv, s)]
            e = select(r["history"], 10)
            sanity.append({"head": h, "level": lv, "seed": s, "ref": "LR-001", "old": o["val_top1"], "new": e["val_top1"],
                           "same": (o["val_top1"], o["best_epoch"], o["split_digest"]) == (e["val_top1"], e["epoch"], r["split_digest"])})
    for lv, s in CELLS:
        o = json.loads((ROOT / "runs" / "GRID-002" / "grid" / "linear" / lv / f"seed{s}" / "done.json").read_text())
        r = runs[("linear", 1e-3, lv, s)]
        e = select(r["history"], 10)
        sanity.append({"head": "linear", "level": lv, "seed": s, "ref": "GRID-002", "old": o["val_top1"], "new": e["val_top1"],
                       "same": (o["val_top1"], o["best_epoch"], o["split_digest"]) == (e["val_top1"], e["epoch"], r["split_digest"])})

    def at(h):
        d = decisions[h]
        return acc[h][(d["adopted_lr"], d["adopted_budget"])]

    expl = {}
    for a, b in (("sparse-M2048", "dense-M2048"), ("sparse-M2048", "linear"), ("sparse-M4096", "sparse-M2048")):
        expl[f"{a} - {b} @ adopted"] = paired([at(a)[x][0] - at(b)[x][0] for x in CELLS])

    res = {"runs": len(runs), "one_split_per_seed": all(len(d) == 1 for d in digests.values()), "candidates": cand,
           "vs_baseline": vs_base, "decisions": decisions, "instability_20ep": instab, "sanity": sanity,
           "sanity_all_same": all(x["same"] for x in sanity), "exploratory": expl}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    p = lambda v: f"{100 * v:.2f}"  # noqa: E731
    L = ["# LR-002 analysis (generated by analyze.py; validation only)", "",
         f"Runs: {len(runs)}/78. One split digest per seed: **{res['one_split_per_seed']}**. "
         f"Sanity reproductions ({len(sanity)} cells) all identical: **{res['sanity_all_same']}**. "
         "Metric: validation top-1 (%) of the selected epoch, 5k pool validation split, cells = source/L090 x seeds 42-44.", "",
         "## Candidates (mean over 6 cells)", "", "| Head | candidate | mean val top-1 | median selected epoch | selected epochs |",
         "| --- | --- | --- | --- | --- |"]
    for k, e in cand.items():
        h, c = k.split(" ", 1)
        L.append(f"| {h} | {c} | {p(e['mean_val_top1'])} | {e['median_epoch']} | {e['epochs']} |")
    L += ["", "## Every candidate vs baseline (paired by cell, descriptive)", "",
          "| Comparison | mean (pp) | sd | 95% CI (df 5) | positive |", "| --- | --- | --- | --- | --- |"]
    for k, e in vs_base.items():
        L.append(f"| {k} | {p(e['mean'])} | {p(e['sd'])} | [{p(e['ci_low'])}, {p(e['ci_high'])}] | {e['positive']}/6 |")
    L += ["", "## Pre-registered decisions", "", "| Head | best candidate | baseline | adopt best? | adopted | flags | located |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for h, d in decisions.items():
        L.append(f"| {h} | {d['best_candidate']} | {d['baseline']} | {'YES' if d['adopt_best'] else 'no'} | {d['adopted']} | "
                 f"{', '.join(d['flags']) or '-'} | {'yes' if d['located'] else 'NO'} |")
    L += ["", "## Instability over 20 epochs (dictionary heads; loss rise > 5%)", "", "| Head lr | unstable runs |", "| --- | --- |"]
    for k, n in instab.items():
        L.append(f"| {k} | {n}/6 |")
    L += ["", "## Sanity reproductions (first-10-epoch selection vs earlier runs)", "",
          "| Head | level | seed | reference | old | new | identical (top-1, epoch, digest) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in sanity:
        L.append(f"| {e['head']} | {e['level']} | {e['seed']} | {e['ref']} | {p(e['old'])} | {p(e['new'])} | {e['same']} |")
    L += ["", "## Exploratory (not pre-registered decisions; validation only; selected on the same cells)", "",
          "| Gap | mean (pp) | 95% CI (df 5) | positive |", "| --- | --- | --- | --- |"]
    for k, e in expl.items():
        L.append(f"| {k} | {p(e['mean'])} | [{p(e['ci_low'])}, {p(e['ci_high'])}] | {e['positive']}/6 |")
    (HERE / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
