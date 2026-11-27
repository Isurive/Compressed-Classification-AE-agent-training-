"""WD-001 phase B analysis (ImageNetV2), implementing the rule pre-registered in task.json.
Written while phase A ran, before any phase-A or phase-B result was read.

    python -B ai_harness/artifacts/WD-001/analyze.py   (from Final Project; standard library only)

Metric: ImageNetV2-source top-1 of best.pt, paired by seed 45-51 (n = 7, df = 6, t = 2.447).
Confirmatory family (4 tests, pooled over source/L080/L090 as the per-seed mean of per-level differences):
W1 dense-CE+wd - dense-MSE; W2 sparse-CE+wd - sparse-MSE; W3 dense-CE+wd - linear; W4 sparse-CE+wd - linear.
MSE heads and linear are RETEST-001's runs (same seeds, levels, splits). Detected = CI excludes 0; meaningful =
detected and |mean| >= tolerance, reported at 1.0 pp (working) and 0.5 pp; Holm over the 4 tests.
Checks: 42 runs with full histories, split digests equal SEED-001's, selected epochs, instability, wd groups.
Exploratory: per-level gaps, compression costs and their differences from the MSE heads, top-5, training-level eval.
Writes analysis.md / analysis.json next to this file.
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
NEW = ["dense-CE", "sparse-CE"]
OLD = ["dense-M2048", "sparse-M2048", "linear"]
T975 = 2.447
TOLS = (0.010, 0.005)
RISE = 1.05


def t_two_sided_p(t, df):
    """Two-sided p-value of Student's t by Simpson integration (as MSWEEP-001 / RETEST-001; no SciPy)."""
    if math.isinf(t):
        return 0.0
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)  # noqa: E731
    a, n = abs(t), 20000
    h = a / n
    s = f(0) + f(a) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))
    return max(0.0, 1 - 2 * s * h / 3)


def stats(xs):
    n = len(xs)
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    se = sd / math.sqrt(n)
    half = T975 * se
    t = m / se if se > 0 else float("inf")
    det = (m - half) > 0 or (m + half) < 0
    return {"n": n, "mean": m, "sd": sd, "ci_low": m - half, "ci_high": m + half, "t": t, "p": t_two_sided_p(t, n - 1),
            "positive": sum(x > 0 for x in xs), "detected": det,
            "meaningful_1.0pp": det and abs(m) >= TOLS[0], "meaningful_0.5pp": det and abs(m) >= TOLS[1], "values": xs}


def load(path, phase, head_part):
    acc = defaultdict(dict)  # (head, "source"|"train") -> {(level, seed): row}
    seen, rows = set(), 0
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["phase"] != phase:
            continue
        key = (r["run_dir"], r["eval_level"])
        assert key not in seen, ("duplicate test row", key)
        seen.add(key)
        rows += 1
        h = Path(r["run_dir"]).parts[head_part]
        cell = (r["train_level"], int(r["seed"]))
        if r["eval_level"] == "source":
            acc[(h, "source")][cell] = r
        if r["eval_level"] == r["train_level"]:
            acc[(h, "train")][cell] = r
    return acc, rows


def main():
    new, rows = load(ROOT / "runs/WD-001/results.csv", "wd-test", 3)
    assert rows == 2 * (7 + 2 * 14), rows
    old, _ = load(ROOT / "runs/RETEST-001/results.csv", "retest", 2)
    acc = {**new, **old}
    for h in NEW + OLD:
        for ev in ("source", "train"):
            assert len(acc[(h, ev)]) == 21, (h, ev, len(acc[(h, ev)]))
    top1 = {h: {k: float(r["top1"]) for k, r in acc[(h, "source")].items()} for h in NEW + OLD}

    selected = json.loads((HERE / "selected.json").read_text())
    s001 = defaultdict(set)
    for r in csv.DictReader(open(ROOT / "runs/SEED-001/results.csv", encoding="utf-8")):
        s001[int(r["seed"])].add(r["split_digest"])
    digests, epochs, instab, groups_ok, nruns = defaultdict(set), defaultdict(list), defaultdict(int), True, 0
    for h in NEW:
        for lv in LEVELS:
            for s in SEEDS:
                d = json.loads((ROOT / "runs/WD-001/test" / h / lv / f"seed{s}" / "done.json").read_text())
                c = selected[h]
                assert (d["lr"], d["weight_decay"], d["epochs"]) == (c["lr"], c["weight_decay"], c["epochs"])
                assert len(d["history"]) == d["epochs"]
                nruns += 1
                digests[s].add(d["split_digest"])
                epochs[h].append(d["best_epoch"])
                L = [e["train_loss"] for e in d["history"]]
                instab[h] += any(b > a * RISE for a, b in zip(L, L[1:]))
                groups_ok &= d["wd_groups"]["wd"] == c["weight_decay"] and len(d["wd_groups"]["decay"]) == 4
    digest_ok = all(digests[s] == s001[s] for s in SEEDS)

    def pooled(a, b):
        return stats([statistics.mean(top1[a][(lv, s)] - top1[b][(lv, s)] for lv in LEVELS) for s in SEEDS])

    fam = {"W1 dense-CE+wd - dense-MSE": pooled("dense-CE", "dense-M2048"),
           "W2 sparse-CE+wd - sparse-MSE": pooled("sparse-CE", "sparse-M2048"),
           "W3 dense-CE+wd - linear": pooled("dense-CE", "linear"),
           "W4 sparse-CE+wd - linear": pooled("sparse-CE", "linear")}
    order = sorted(fam, key=lambda k: fam[k]["p"])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * fam[k]["p"]))
        fam[k]["p_holm"] = run
        fam[k]["holm_detected"] = run < 0.05

    per_level = {f"{a} - {b}": {lv: stats([top1[a][(lv, s)] - top1[b][(lv, s)] for s in SEEDS]) for lv in LEVELS}
                 for a, b in (("dense-CE", "dense-M2048"), ("sparse-CE", "sparse-M2048"), ("dense-CE", "linear"),
                              ("sparse-CE", "linear"))}
    cost = {f"{h} source - {lv}": stats([top1[h][("source", s)] - top1[h][(lv, s)] for s in SEEDS])
            for h in NEW for lv in ("L080", "L090")}
    cost_diff = {f"cost({a}) - cost({b}) at {lv}": stats(
        [(top1[a][("source", s)] - top1[a][(lv, s)]) - (top1[b][("source", s)] - top1[b][(lv, s)]) for s in SEEDS])
        for lv in ("L080", "L090") for a, b in (("dense-CE", "dense-M2048"), ("sparse-CE", "sparse-M2048"),
                                                 ("dense-CE", "linear"), ("sparse-CE", "linear"))}
    means = {h: {lv: statistics.mean(top1[h][(lv, s)] for s in SEEDS) for lv in LEVELS} for h in NEW + OLD}
    top5 = {h: {lv: statistics.mean(float(acc[(h, "source")][(lv, s)]["top5"]) for s in SEEDS) for lv in LEVELS}
            for h in NEW + OLD}
    trainlvl = {h: {lv: statistics.mean(float(acc[(h, "train")][(lv, s)]["top1"]) for s in SEEDS)
                    for lv in ("L080", "L090")} for h in NEW + OLD}
    sel = {h: {"median": statistics.median(epochs[h]), "ge_budget_minus_1": sum(e >= selected[h]["epochs"] - 1
                                                                                 for e in epochs[h]),
               "le2": sum(e <= 2 for e in epochs[h]), "epochs": epochs[h]} for h in NEW}
    res = {"runs": nruns, "selected": selected, "split_digests_match_seed001": digest_ok, "wd_groups_ok": groups_ok,
           "selected_epochs": sel, "instability_runs": dict(instab), "family": fam, "holm_order": order,
           "per_level": per_level, "exploratory": {"mean_top1": means, "mean_top5": top5,
                                                   "train_level_eval_top1": trainlvl, "compression_cost": cost,
                                                   "compression_cost_differences": cost_diff}}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    p = lambda v: f"{100 * v:+.2f}"  # noqa: E731
    q = lambda v: f"{100 * v:.2f}"  # noqa: E731
    yn = lambda b: "yes" if b else "no"  # noqa: E731

    def row(name, e):
        return (f"| {name} | {p(e['mean'])} | {q(e['sd'])} | [{p(e['ci_low'])}, {p(e['ci_high'])}] | "
                f"{e['positive']}/{e['n']} | {e['p']:.4f} |")

    L = ["# WD-001 phase B analysis (generated by analyze.py)", "",
         f"Runs: {nruns}/42. Selected (phase A): {json.dumps(selected)}. Split digests equal SEED-001's: "
         f"**{digest_ok}**. wd groups as specified: **{groups_ok}**. Metric: ImageNetV2-source top-1; paired by seed "
         "45-51 (n = 7, df = 6). MSE heads and linear: RETEST-001 runs.", "",
         "## Mean accuracy (%)", "", "| Head | source | L080 | L090 |", "| --- | --- | --- | --- |"]
    L += [f"| {h} | " + " | ".join(q(means[h][lv]) for lv in LEVELS) + " |" for h in NEW + OLD]
    L += ["", "## CONFIRMATORY family (4 tests)", "",
          "| Test | Mean (pp) | sd | 95% CI | positive | p | Holm p | detected | meaningful @1.0 | meaningful @0.5 | "
          "Holm-detected |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, e in fam.items():
        L.append(row(k, e) + f" {e['p_holm']:.4f} | {yn(e['detected'])} | {yn(e['meaningful_1.0pp'])} | "
                 f"{yn(e['meaningful_0.5pp'])} | {yn(e['holm_detected'])} |")
    L += ["", "## Checks", "", "| Head | median selected epoch | at budget-1 or later | at epoch <= 2 | unstable runs |",
          "| --- | --- | --- | --- | --- |"]
    L += [f"| {h} | {sel[h]['median']} | {sel[h]['ge_budget_minus_1']}/21 | {sel[h]['le2']}/21 | {instab[h]}/21 |"
          for h in NEW]
    L += ["", "## SECONDARY: per-level differences (descriptive)", "",
          "| Comparison | Level | Mean (pp) | sd | 95% CI | positive | p |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for comp, by in per_level.items():
        for lv, e in by.items():
            L.append(row(f"{comp} | {lv}", e))
    L += ["", "## EXPLORATORY", "", "### Compression cost of the CE heads", "",
          "| Cost | Mean (pp) | sd | 95% CI | positive | p |", "| --- | --- | --- | --- | --- | --- |"]
    L += [row(k, e) for k, e in cost.items()]
    L += ["", "### Compression-cost differences", "", "| Comparison | Mean (pp) | sd | 95% CI | positive | p |",
          "| --- | --- | --- | --- | --- | --- |"] + [row(k, e) for k, e in cost_diff.items()]
    L += ["", "### Top-5 (source eval) and top-1 at the training level (%)", "",
          "| Head | top-5 source | top-5 L080 | top-5 L090 | top-1 train-level L080 | top-1 train-level L090 |",
          "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {h} | {q(top5[h]['source'])} | {q(top5[h]['L080'])} | {q(top5[h]['L090'])} | "
          f"{q(trainlvl[h]['L080'])} | {q(trainlvl[h]['L090'])} |" for h in NEW + OLD]
    (HERE / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
