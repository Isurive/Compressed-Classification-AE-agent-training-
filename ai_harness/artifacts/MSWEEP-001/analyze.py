"""MSWEEP-001 analysis, implementing the rule pre-registered in task.json.

    python -B ai_harness/artifacts/MSWEEP-001/analyze.py   (from Final Project)

Data: runs/MSWEEP-001 (M = 512, 1024, 4096) and runs/SEED-001 phase seed-mse sparse (M = 2048), seeds 45-51,
ImageNetV2-source top-1, levels source/L080/L090. Paired by seed.
Confirmatory family (13 tests; task.json miscounts it as 10): A(M) - A(2048) for M in {512, 1024, 4096} per level + pooled (12), and the per-seed
slope of pooled accuracy on log2(M) (1). Detected = 95% CI excludes 0. Meaningful = detected and |mean| >= 1 pp
(slope: |slope| x 3 >= 1 pp). Holm-adjusted p-values reported alongside. Sanity: median best epoch per M (flag <= 2).
Writes analysis.md / analysis.json next to this file (copy the script elsewhere to experiment).
"""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LEVELS = ["source", "L080", "L090"]
SEEDS = list(range(45, 52))
SIZES = [512, 1024, 2048, 4096]
T975 = {6: 2.447}
TOL = 0.010


def t_two_sided_p(t, df):
    """Two-sided p-value of Student's t by Simpson integration of the density (no SciPy needed)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)  # noqa: E731
    a, n = abs(t), 20000
    h = a / n
    s = f(0) + f(a) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))
    return max(0.0, 1 - 2 * s * h / 3)


def stats(xs, scale=1.0):
    n = len(xs)
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    se = sd / math.sqrt(n)
    half = T975[n - 1] * se
    t = m / se if se > 0 else float("inf")
    detected = (m - half) > 0 or (m + half) < 0
    return {"n": n, "mean": m, "sd": sd, "ci_low": m - half, "ci_high": m + half, "t": t,
            "p": t_two_sided_p(t, n - 1), "positive": sum(x > 0 for x in xs), "detected": detected,
            "meaningful": detected and abs(m) * scale >= TOL, "values": xs}


def load():
    acc, epochs, settings = defaultdict(dict), defaultdict(list), {}
    digests = defaultdict(set)
    sources = [(ROOT / "runs/MSWEEP-001/results.csv", "msweep-clean", None),
               (ROOT / "runs/SEED-001/results.csv", "seed-mse", "sparse")]
    for path, phase, model in sources:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["phase"] != phase or (model and r["model"] != model) or r["eval_level"] != "source":
                continue
            if int(r["seed"]) not in SEEDS or r["train_level"] not in LEVELS:
                continue
            m = int(r["atoms"])
            acc[m][(r["train_level"], int(r["seed"]))] = float(r["top1"])
            epochs[m].append(int(r["best_epoch"]))
            digests[int(r["seed"])].add(r["split_digest"])
            settings[m] = {k: r[k] for k in ("atoms", "k", "fc_init", "weight_decay")}
    return acc, epochs, digests, settings


def main():
    acc, epochs, digests, settings = load()
    missing = [(m, lv, s) for m in SIZES for lv in LEVELS for s in SEEDS if (lv, s) not in acc[m]]
    pooled = {m: {s: statistics.mean(acc[m][(lv, s)] for lv in LEVELS) for s in SEEDS} for m in SIZES}
    family = {}
    for m in (512, 1024, 4096):
        for lv in LEVELS:
            family[f"M{m}-M2048 {lv}"] = stats([acc[m][(lv, s)] - acc[2048][(lv, s)] for s in SEEDS])
        family[f"M{m}-M2048 pooled"] = stats([pooled[m][s] - pooled[2048][s] for s in SEEDS])
    xs = [math.log2(m) for m in SIZES]
    xbar = statistics.mean(xs)
    slopes = [sum((x - xbar) * (pooled[m][s]) for x, m in zip(xs, SIZES)) / sum((x - xbar) ** 2 for x in xs)
              for s in SEEDS]
    family["slope per doubling (pooled)"] = stats(slopes, scale=3.0)
    # Holm adjustment over the 13-test family (task.json text says 10; its rule defines 13)
    order = sorted(family, key=lambda k: family[k]["p"])
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * family[key]["p"]))
        family[key]["p_holm"] = running
        family[key]["holm_detected"] = running < 0.05
    # Exploratory: per-level slopes
    expl = {}
    for lv in LEVELS:
        sl = [sum((x - xbar) * acc[m][(lv, s)] for x, m in zip(xs, SIZES)) / sum((x - xbar) ** 2 for x in xs)
              for s in SEEDS]
        expl[f"slope per doubling {lv}"] = stats(sl, scale=3.0)
    sanity = {m: {"median_best_epoch": statistics.median(epochs[m]), "min": min(epochs[m]), "max": max(epochs[m]),
                  "flag_le_2": statistics.median(epochs[m]) <= 2} for m in SIZES}
    means = {m: {lv: statistics.mean(acc[m][(lv, s)] for s in SEEDS) for lv in LEVELS} for m in SIZES}
    res = {"missing_cells": missing, "one_split_per_seed": all(len(d) == 1 for d in digests.values()),
           "settings": settings, "mean_accuracy": means, "confirmatory": family, "exploratory": expl,
           "sanity_best_epoch": sanity}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    p = lambda v: f"{100 * v:.2f}"  # noqa: E731
    L = ["# MSWEEP-001 analysis (generated by analyze.py)", "",
         f"Missing cells: {len(missing)}. One split digest per seed: **{res['one_split_per_seed']}**. "
         "Metric: ImageNetV2-source top-1 (%), seeds 45-51.", "",
         "## Mean accuracy", "", "| M (k) | " + " | ".join(LEVELS) + " | pooled |", "| --- |" + " --- |" * 4]
    for m in SIZES:
        L.append(f"| {m} ({m // 4}) | " + " | ".join(p(means[m][lv]) for lv in LEVELS)
                 + f" | {p(statistics.mean(pooled[m].values()))} |")
    L += ["", "## Confirmatory family (pre-registered rule; 13 tests -- task.json text miscounts as 10)", "",
          "| Test | Mean (pp) | sd | 95% CI | positive | p | Holm p | detected | meaningful |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, e in family.items():
        L.append(f"| {k} | {p(e['mean'])} | {p(e['sd'])} | [{p(e['ci_low'])}, {p(e['ci_high'])}] | {e['positive']}/{e['n']} | "
                 f"{e['p']:.4f} | {e['p_holm']:.4f} | {'yes' if e['detected'] else 'no'} | {'yes' if e['meaningful'] else 'no'} |")
    L += ["", "## Exploratory (not confirmatory)", "", "| Test | Mean (pp) | 95% CI | positive |", "| --- | --- | --- | --- |"]
    for k, e in expl.items():
        L.append(f"| {k} | {p(e['mean'])} | [{p(e['ci_low'])}, {p(e['ci_high'])}] | {e['positive']}/{e['n']} |")
    L += ["", "## Sanity check: best epoch per M (pre-declared; flag if median <= 2)", "",
          "| M | median | min | max | flag |", "| --- | --- | --- | --- | --- |"]
    for m, e in sanity.items():
        L.append(f"| {m} | {e['median_best_epoch']} | {e['min']} | {e['max']} | {'FLAG' if e['flag_le_2'] else 'ok'} |")
    (HERE / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
