"""REVIEW-013: independent recomputation of the WD-001 phase-A selection (does not import analyze_val.py).

    python -B ai_harness/artifacts/REVIEW-013/recompute.py   (from Final Project; standard library only)

Reads runs/WD-001/val/<head>/<tag>/<level>/seed<s>/done.json (discovered by directory walk, not by a tag
function) and runs/LR-002/{dense,sparse}-M2048/lr1e-4/<level>/seed<s>/done.json. Uses integer correct counts
(val_top1 * 5000) for top-1 so that ties are exact, and Fractions for means.
"""
import json
import math
from fractions import Fraction
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "runs" / "WD-001" / "val"
NVAL = 5000
T5 = 2.5706

runs = {}
for p in sorted(VAL.glob("*/*/*/seed*/done.json")):
    head, tag, level, seed = p.parts[-5], p.parts[-4], p.parts[-3], int(p.parts[-2][4:])
    d = json.loads(p.read_text())
    assert d["head"] == head and d["seed"] == seed and d["train_level"] == level and d["phase"] == "wd-val"
    assert len(d["history"]) == 20 and [e["epoch"] for e in d["history"]] == list(range(1, 21))
    runs[(head, d["lr"], d["weight_decay"], level, seed)] = d
print("runs found:", len(runs))
heads = sorted({k[0] for k in runs})
lrs = sorted({k[1] for k in runs})
wds = sorted({k[2] for k in runs})
cells = sorted({(k[3], k[4]) for k in runs})
print("heads", heads, "lrs", lrs, "wds", wds, "cells", cells)
assert len(runs) == len(heads) * len(lrs) * len(wds) * len(cells) == 144

# checks on logged metadata
digests = {}
bad_groups = []
for k, d in runs.items():
    digests.setdefault(k[4], set()).add(d["split_digest"])
    g = d["wd_groups"]
    if not (g["wd"] == d["weight_decay"] and g["decay"] == [[2048, 512], [2048], [1000, 2048], [1000]]
            and g["no_decay"] == [[512], [2048, 512]]):
        bad_groups.append(k)
    exp = {"dense-CE": ("dense", 2048), "sparse-CE": ("sparse", 512)}[k[0]]
    assert (d["model"], d["k"], d["atoms"], d["fc_init"]) == (*exp, 2048, "xavier")
    assert (d["reconstruction_weight"], d["dependency_weight"], d["coherence_weight"]) == (1e-6, 0.0, 0.0)
    # best.pt-derived val_top1 must equal the 20-epoch history selection
    b = max(d["history"], key=lambda e: (e["val_top1"], -e["val_ce"]))
    assert b["epoch"] == d["best_epoch"] and b["val_top1"] == d["val_top1"], k
print("split digests per seed:", {s: len(v) for s, v in digests.items()}, {s: next(iter(v))[:12] for s, v in digests.items()})
print("runs with unexpected wd groups:", bad_groups)


def cnt(e):
    c = e["val_top1"] * NVAL
    assert abs(c - round(c)) < 1e-6
    return round(c)


def pick(history, n):
    return max((e for e in history if e["epoch"] <= n), key=lambda e: (cnt(e), -e["val_ce"]))


out = {}
for h in heads:
    rows = []
    for lr in lrs:
        for wd in wds:
            for n in (10, 20):
                sel = [pick(runs[(h, lr, wd, lv, s)]["history"], n) for lv, s in cells]
                rows.append(dict(lr=lr, wd=wd, n=n, correct=sum(cnt(e) for e in sel),
                                 ce=sum(Fraction(e["val_ce"]) for e in sel) / len(sel),
                                 ep=median(e["epoch"] for e in sel), eps=[e["epoch"] for e in sel],
                                 cells=[cnt(e) for e in sel]))
    rows.sort(key=lambda r: (-r["correct"], r["ce"]))
    print(f"\n== {h}: top candidates (mean top-1 %, mean CE, median epoch)")
    for r in rows[:5]:
        print(f"  lr {r['lr']:g} wd {r['wd']:g} {r['n']} ep: correct {r['correct']}/{NVAL * 6} = "
              f"{100 * r['correct'] / (NVAL * 6):.4f}%  CE {float(r['ce']):.5f}  ep {r['ep']} {r['eps']}")
    best = rows[0]
    ties = [r for r in rows if r["correct"] == best["correct"]]
    print("  exact top-1 ties at the max:", [(r["lr"], r["wd"], r["n"], round(float(r["ce"]), 5)) for r in ties])
    flags = []
    if best["wd"] == 1e-1:
        flags.append("wd upper edge")
    if best["wd"] == 0.0:
        b3 = max(r["correct"] for r in rows if r["wd"] == 1e-3)
        if (best["correct"] - b3) / (NVAL * 6) * 100 <= 0.10:
            flags.append("wd lower edge")
    if best["lr"] in (3e-4, 3e-5):
        flags.append("lr edge")
    if best["n"] == 20 and best["ep"] >= 19:
        flags.append("budget binds")
    print("  SELECTED:", dict(lr=best["lr"], weight_decay=best["wd"], epochs=best["n"]), "flags:", flags)
    # runner-up gap
    nxt = next(r for r in rows if r["correct"] < best["correct"])
    print(f"  next distinct top-1: lr {nxt['lr']:g} wd {nxt['wd']:g} {nxt['n']} ep, "
          f"gap {100 * (best['correct'] - nxt['correct']) / (NVAL * 6):.4f} pp")

    # exploratory paired CIs (df 5)
    def ci(a, b):
        x = [100 * (i - j) / NVAL for i, j in zip(a, b)]
        m = sum(x) / len(x)
        sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
        hw = T5 * sd / math.sqrt(len(x))
        return f"{m:+.2f} [{m - hw:+.2f}, {m + hw:+.2f}] {sum(v > 0 for v in x)}/{len(x)}"

    wd0 = next(r for r in rows if r["lr"] == best["lr"] and r["n"] == best["n"] and r["wd"] == 0.0)
    ref = "dense-M2048" if h == "dense-CE" else "sparse-M2048"
    mse = []
    for lv, s in cells:
        d = json.loads((ROOT / "runs/LR-002" / ref / "lr1e-4" / lv / f"seed{s}" / "done.json").read_text())
        assert len(d["history"]) >= 20
        mse.append(cnt(pick(d["history"], 20)))
    print("  selected - wd0 (same lr, budget):", ci(best["cells"], wd0["cells"]))
    print("  selected - MSE (LR-002 lr1e-4 20ep):", ci(best["cells"], mse))
    print("  wd0 - MSE:", ci(wd0["cells"], mse))
    out[h] = {"lr": best["lr"], "weight_decay": best["wd"], "epochs": best["n"]}

sel = json.loads((ROOT / "ai_harness/artifacts/WD-001S/selected.json").read_text())
print("\nselected.json matches recomputation:", sel == out, sel)
