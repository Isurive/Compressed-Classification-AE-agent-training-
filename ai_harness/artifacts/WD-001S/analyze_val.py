"""WD-001 phase A analysis (validation only), implementing the selection rule pre-registered in task.json.
Written while phase A ran, before any phase-A result was read.

    python -B ai_harness/artifacts/WD-001/analyze_val.py   (from Final Project; standard library only)

Per head (dense-CE, sparse-CE): candidates = lr {3e-4, 1e-4, 3e-5} x wd {0, 1e-3, 1e-2, 1e-1} x budget {10, 20}.
Selection within a budget n: epoch <= n with the highest validation top-1, then lower CE. Selected candidate: highest
mean validation top-1 over the 6 cells (source/L090 x seeds 42-44), ties by lower mean validation CE.
Flags: wd upper edge (1e-1); wd lower edge (selected wd 0 and the best wd-1e-3 candidate within 0.10 pp);
lr edge (3e-4 or 3e-5); budget binds (budget 20 and median selected epoch >= 19).
Descriptive / exploratory: selected vs wd 0 at the same lr and budget; CE heads vs the MSE heads' LR-002 runs
(lr 1e-4, 20 epochs) on the same cells; instability (training loss rises > 5% between consecutive epochs).
Writes selected.json, analysis_val.md and analysis_val.json next to this file.
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VAL = ROOT / "runs" / "WD-001" / "val"
HEADS = ["dense-CE", "sparse-CE"]
MSE_REF = {"dense-CE": "dense-M2048", "sparse-CE": "sparse-M2048"}
LRS = [3e-4, 1e-4, 3e-5]
WDS = [0.0, 1e-3, 1e-2, 1e-1]
BUDGETS = [10, 20]
CELLS = [(lv, s) for lv in ("source", "L090") for s in (42, 43, 44)]
T975 = 2.5706  # df 5
RISE = 1.05


def tag(lr, wd):
    return f"lr{lr:.0e}_wd{wd:.0e}".replace("e-0", "e-").replace("e+0", "e")


def select(history, n):
    return max((e for e in history if e["epoch"] <= n), key=lambda e: (e["val_top1"], -e["val_ce"]))


def paired(a, b):
    x = [100 * (i - j) for i, j in zip(a, b)]
    m = statistics.mean(x)
    h = T975 * statistics.stdev(x) / math.sqrt(len(x))
    return {"mean": m, "ci_low": m - h, "ci_high": m + h, "positive": sum(v > 0 for v in x), "n": len(x)}


def main():
    runs, digests = {}, {}
    for h in HEADS:
        for lr in LRS:
            for wd in WDS:
                for lv, s in CELLS:
                    d = json.loads((VAL / h / tag(lr, wd) / lv / f"seed{s}" / "done.json").read_text())
                    assert d["epochs"] == 20 and len(d["history"]) == 20 and d["lr"] == lr and d["weight_decay"] == wd
                    assert d["wd_groups"]["wd"] == wd and len(d["wd_groups"]["decay"]) == 4
                    runs[(h, lr, wd, lv, s)] = d
                    digests.setdefault(s, set()).add(d["split_digest"])
    assert len(runs) == 144

    cand, selected, flags, details = {}, {}, {}, {}
    for h in HEADS:
        rows = []
        for lr in LRS:
            for wd in WDS:
                for n in BUDGETS:
                    sel = [select(runs[(h, lr, wd, lv, s)]["history"], n) for lv, s in CELLS]
                    rows.append({"lr": lr, "weight_decay": wd, "epochs": n,
                                 "mean_val_top1": statistics.mean(e["val_top1"] for e in sel),
                                 "mean_val_ce": statistics.mean(e["val_ce"] for e in sel),
                                 "median_epoch": statistics.median(e["epoch"] for e in sel),
                                 "epochs_selected": [e["epoch"] for e in sel],
                                 "cells": [e["val_top1"] for e in sel]})
        cand[h] = rows
        best = max(rows, key=lambda r: (r["mean_val_top1"], -r["mean_val_ce"]))
        selected[h] = {"lr": best["lr"], "weight_decay": best["weight_decay"], "epochs": best["epochs"]}
        f = []
        if best["weight_decay"] == 1e-1:
            f.append("wd upper edge")
        if best["weight_decay"] == 0.0:
            b3 = max((r for r in rows if r["weight_decay"] == 1e-3), key=lambda r: r["mean_val_top1"])
            if best["mean_val_top1"] - b3["mean_val_top1"] <= 0.0010:
                f.append("wd lower edge")
        if best["lr"] in (3e-4, 3e-5):
            f.append("lr edge")
        if best["epochs"] == 20 and best["median_epoch"] >= 19:
            f.append("budget binds")
        flags[h] = f
        wd0 = next(r for r in rows if r["lr"] == best["lr"] and r["epochs"] == best["epochs"]
                   and r["weight_decay"] == 0.0)
        mse = []
        for lv, s in CELLS:
            d = json.loads((ROOT / "runs/LR-002" / MSE_REF[h] / "lr1e-4" / lv / f"seed{s}" / "done.json").read_text())
            mse.append(select(d["history"], 20)["val_top1"])
        details[h] = {"selected_minus_wd0_same_lr_budget": paired(best["cells"], wd0["cells"]),
                      "selected_minus_MSE_head_lr1e-4_20ep (LR-002)": paired(best["cells"], mse),
                      "wd0_minus_MSE_head_same (LR-002)": paired(wd0["cells"], mse)}
    instab = {f"{h} {tag(lr, wd)}": sum(any(b > a * RISE for a, b in zip(L, L[1:])) for L in
                                         ([e["train_loss"] for e in runs[(h, lr, wd, lv, s)]["history"]]
                                          for lv, s in CELLS))
              for h in HEADS for lr in LRS for wd in WDS}
    res = {"runs": len(runs), "one_split_per_seed": all(len(v) == 1 for v in digests.values()),
           "selected": selected, "flags": flags, "candidates": cand, "exploratory": details, "instability": instab}
    (HERE / "analysis_val.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (HERE / "selected.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    q = lambda v: f"{100 * v:.2f}"  # noqa: E731
    L = ["# WD-001 phase A analysis (generated by analyze_val.py; validation only)", "",
         f"Runs: {len(runs)}/144. One split digest per seed: **{res['one_split_per_seed']}**. Metric: validation top-1 "
         "of the selected epoch, 5k pool validation split, cells = source/L090 x seeds 42-44.", "",
         "## Selection (pre-registered rule)", "", "| Head | lr | weight decay | budget | mean val top-1 | flags |",
         "| --- | --- | --- | --- | --- | --- |"]
    for h in HEADS:
        b = max(cand[h], key=lambda r: (r["mean_val_top1"], -r["mean_val_ce"]))
        L.append(f"| {h} | {b['lr']:g} | {b['weight_decay']:g} | {b['epochs']} | {q(b['mean_val_top1'])} | "
                 f"{', '.join(flags[h]) or 'none'} |")
    L += ["", "## All candidates (mean val top-1 %, median selected epoch)", "",
          "| Head | lr | wd | 10 ep | 20 ep |", "| --- | --- | --- | --- | --- |"]
    for h in HEADS:
        for lr in LRS:
            for wd in WDS:
                r10, r20 = (next(r for r in cand[h] if r["lr"] == lr and r["weight_decay"] == wd and r["epochs"] == n)
                            for n in BUDGETS)
                L.append(f"| {h} | {lr:g} | {wd:g} | {q(r10['mean_val_top1'])} (ep {r10['median_epoch']}) | "
                         f"{q(r20['mean_val_top1'])} (ep {r20['median_epoch']}) |")
    L += ["", "## Exploratory (validation; selected on the same cells)", "",
          "| Head | comparison | mean (pp) | 95% CI (df 5) | positive |", "| --- | --- | --- | --- | --- |"]
    for h in HEADS:
        for k, e in details[h].items():
            L.append(f"| {h} | {k} | {e['mean']:+.2f} | [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}] | {e['positive']}/6 |")
    L += ["", "## Instability (runs of 6 with training loss rising > 5% between epochs)", "",
          "| head config | unstable |", "| --- | --- |"] + [f"| {k} | {v}/6 |" for k, v in instab.items()]
    (HERE / "analysis_val.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
