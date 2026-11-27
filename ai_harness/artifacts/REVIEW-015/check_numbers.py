"""REVIEW-015 helper: recompute the WD-001 numbers that COORD-005 cites.

Reads ai_harness/artifacts/WD-001/analysis.json only (read-only); prints results.
Run from Final Project: python -B ai_harness/artifacts/REVIEW-015/check_numbers.py
"""
import json

d = json.load(open("ai_harness/artifacts/WD-001/analysis.json"))
pp = lambda x: 100 * x

print("== Confirmatory family (pp) ==")
for k, v in d["family"].items():
    print(f"{k}: mean {pp(v['mean']):+.3f} CI [{pp(v['ci_low']):+.3f}, {pp(v['ci_high']):+.3f}] "
          f"pos {v['positive']}/7 holm {v['p_holm']:.2e} det {v['holm_detected']} "
          f"m1 {v['meaningful_1.0pp']} m05 {v['meaningful_0.5pp']}")

print("== Per level (pp) ==")
for k, lv in d["per_level"].items():
    for lev, v in lv.items():
        inc0 = v["ci_low"] <= 0 <= v["ci_high"]
        print(f"{k} {lev}: {pp(v['mean']):+.3f} CI [{pp(v['ci_low']):+.3f}, {pp(v['ci_high']):+.3f}] includes0={inc0}")

ex = d["exploratory"]
print("== Mean top-1 (%) ==")
for h, lv in ex["mean_top1"].items():
    print(h, {k: round(pp(x), 2) for k, x in lv.items()})

print("== Top-5 gaps CE - MSE (pp, unrounded) ==")
for ce, mse in (("dense-CE", "dense-M2048"), ("sparse-CE", "sparse-M2048")):
    gaps = {lev: pp(ex["mean_top5"][mse][lev] - ex["mean_top5"][ce][lev]) for lev in ex["mean_top5"][ce]}
    print(ce, {k: round(g, 3) for k, g in gaps.items()}, "range", round(min(gaps.values()), 3), "-", round(max(gaps.values()), 3))

print("== Training-level top-1 gaps MSE - CE (pp, unrounded) ==")
for ce, mse in (("dense-CE", "dense-M2048"), ("sparse-CE", "sparse-M2048")):
    gaps = {lev: pp(ex["train_level_eval_top1"][mse][lev] - ex["train_level_eval_top1"][ce][lev])
            for lev in ex["train_level_eval_top1"][ce]}
    print(ce, {k: round(g, 3) for k, g in gaps.items()})

print("== Compression costs and differences (pp) ==")
for sec in ("compression_cost", "compression_cost_differences"):
    for k, v in ex[sec].items():
        inc0 = v["ci_low"] <= 0 <= v["ci_high"]
        print(f"{k}: {pp(v['mean']):+.3f} CI [{pp(v['ci_low']):+.3f}, {pp(v['ci_high']):+.3f}] includes0={inc0}")

print("== Selected epochs ==")
for h, v in d["selected_epochs"].items():
    print(h, "median", v["median"], ">=19:", v["ge_budget_minus_1"], "epochs", sorted(v["epochs"]))
