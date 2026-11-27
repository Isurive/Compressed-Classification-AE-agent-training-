"""RETEST-001 analysis, implementing the rule pre-registered in task.json (written while the grid ran, before any
grid result was read beyond the two smoke-test rows).

    python -B ai_harness/artifacts/RETEST-001/analyze.py   (from Final Project; standard library only)

Metric: ImageNetV2-source top-1 of best.pt, paired by seed (45-51, n = 7, df = 6).
Confirmatory family (12 tests): A1 sparse M2048 - linear, A2 sparse M2048 - dense M2048, A3 sparse M4096 - sparse
M2048, A4 sparse M4096 - linear (pooled over source/L080/L090: per-seed mean of the per-level differences);
B1-B8 compression cost A(source) - A(L080) and A(source) - A(L090) per head.
'Detected' = 95% CI excludes 0; 'meaningful' = detected and |mean| >= 1.0 pp; Holm-adjusted p alongside.
Checks: 84 runs with 20-epoch histories; split digest per seed equals SEED-001's; selected-epoch distribution;
instability (dictionary heads; total training loss rises > 5% between consecutive epochs).
Exploratory (labeled): training-level ImageNetV2; top-5; gain over the lr 1e-3 / 10-epoch runs on the same seeds;
compression-cost differences between heads; LR-002 validation predictions.
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
RUNS = ROOT / "runs" / "RETEST-001"
HEADS = ["sparse-M2048", "sparse-M4096", "dense-M2048", "linear"]
DICT = HEADS[:3]
LEVELS = ["source", "L080", "L090"]
SEEDS = list(range(45, 52))
EPOCHS = 20
T975 = {6: 2.447}
TOL = 0.010
RISE = 1.05
# Earlier lr 1e-3 / 10-epoch runs on the same seeds and levels (exploratory comparison).
OLD = {"sparse-M2048": ("SEED-001", "seed-mse", "sparse", "2048"), "dense-M2048": ("SEED-001", "seed-mse", "dense", "2048"),
       "linear": ("SEED-001", "seed-linear", "linear", "0"), "sparse-M4096": ("MSWEEP-001", "msweep-clean", "sparse", "4096")}
# LR-002 posthoc.md, equal 20-epoch budget (validation, seeds 42-44, source/L090), for the exploratory comparison.
LR002_VAL = {"A1": 0.36, "A2": 0.00, "A3": 1.04}


def t_two_sided_p(t, df):
    """Two-sided p-value of Student's t by Simpson integration of the density (as MSWEEP-001; no SciPy)."""
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
    half = T975[n - 1] * se
    t = m / se if se > 0 else float("inf")
    detected = (m - half) > 0 or (m + half) < 0
    return {"n": n, "mean": m, "sd": sd, "ci_low": m - half, "ci_high": m + half, "t": t, "p": t_two_sided_p(t, n - 1),
            "positive": sum(x > 0 for x in xs), "detected": detected, "meaningful": detected and abs(m) >= TOL,
            "values": xs}


def head_of(run_dir):
    return Path(run_dir).parts[2]


def load_results():
    acc = defaultdict(dict)   # (head, eval) -> {(level, seed): row}
    seen, rows = set(), 0  # revision 1 (REVIEW-010 O-1): explicit duplicate check over every row, incl. source runs
    for r in csv.DictReader(open(RUNS / "results.csv", encoding="utf-8")):
        assert (r["run_dir"], r["eval_level"]) not in seen, ("duplicate test row", r["run_dir"], r["eval_level"])
        seen.add((r["run_dir"], r["eval_level"]))
        rows += 1
        assert r["phase"] == "retest" and r["eval_set"] == "imagenetv2", r
        ev = "source" if r["eval_level"] == "source" else "train"
        key = (head_of(r["run_dir"]), r["train_level"], int(r["seed"]))
        assert key[0] in HEADS and key[1] in LEVELS and key[2] in SEEDS, key
        if r["train_level"] == "source":
            acc[(key[0], "source")][key[1:]] = r
            acc[(key[0], "train")][key[1:]] = r
        else:
            assert (key[1:]) not in acc[(key[0], ev)], ("duplicate", key, ev)
            acc[(key[0], ev)][key[1:]] = r
    assert rows == 28 + 2 * 56, rows  # source runs: 1 row; compressed runs: source + training-level rows
    for h in HEADS:
        for ev in ("source", "train"):
            assert len(acc[(h, ev)]) == len(LEVELS) * len(SEEDS), (h, ev, len(acc[(h, ev)]))
    return acc


def load_old():
    old = defaultdict(dict)
    for h, (task, phase, model, atoms) in OLD.items():
        for r in csv.DictReader(open(ROOT / "runs" / task / "results.csv", encoding="utf-8")):
            if (r["phase"], r["model"], r["atoms"], r["eval_level"]) == (phase, model, atoms, "source") \
                    and r["train_level"] in LEVELS and int(r["seed"]) in SEEDS:
                old[h][(r["train_level"], int(r["seed"]))] = float(r["top1"])
        assert len(old[h]) == 21, (h, len(old[h]))
    return old


def seed001_digests():
    d = defaultdict(set)
    for r in csv.DictReader(open(ROOT / "runs" / "SEED-001" / "results.csv", encoding="utf-8")):
        d[int(r["seed"])].add(r["split_digest"])
    return d


def main():
    acc = load_results()
    top1 = {h: {k: float(r["top1"]) for k, r in acc[(h, "source")].items()} for h in HEADS}

    # Checks
    runs, digests, epochs, instab = 0, defaultdict(set), defaultdict(list), defaultdict(int)
    for h in HEADS:
        for lv in LEVELS:
            for s in SEEDS:
                d = json.loads((RUNS / h / lv / f"seed{s}" / "done.json").read_text())
                assert d["epochs"] == EPOCHS and len(d["history"]) == EPOCHS, (h, lv, s)
                assert d["head"] == h and d["train_level"] == lv and d["seed"] == s
                runs += 1
                digests[s].add(d["split_digest"])
                epochs[h].append(d["best_epoch"])
                if h in DICT:
                    losses = [e["train_loss"] for e in d["history"]]
                    instab[h] += any(b > a * RISE for a, b in zip(losses, losses[1:]))
    s1 = seed001_digests()
    digest_ok = all(len(digests[s]) == 1 and digests[s] == s1[s] for s in SEEDS)
    sel = {h: {"median": statistics.median(epochs[h]), "ge19": sum(e >= 19 for e in epochs[h]),
               "le2": sum(e <= 2 for e in epochs[h]), "epochs": epochs[h],
               "budget_binds": statistics.median(epochs[h]) >= 19} for h in HEADS}

    # Confirmatory family
    def pooled(a, b):
        return stats([statistics.mean(top1[a][(lv, s)] - top1[b][(lv, s)] for lv in LEVELS) for s in SEEDS])

    fam = {"A1 sparse-M2048 - linear": pooled("sparse-M2048", "linear"),
           "A2 sparse-M2048 - dense-M2048": pooled("sparse-M2048", "dense-M2048"),
           "A3 sparse-M4096 - sparse-M2048": pooled("sparse-M4096", "sparse-M2048"),
           "A4 sparse-M4096 - linear": pooled("sparse-M4096", "linear")}
    for i, lv in enumerate(("L080", "L090")):
        for j, h in enumerate(HEADS):
            fam[f"B{1 + 4 * i + j} {h} source - {lv}"] = stats([top1[h][("source", s)] - top1[h][(lv, s)] for s in SEEDS])
    assert len(fam) == 12
    order = sorted(fam, key=lambda k: fam[k]["p"])
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * fam[key]["p"]))
        fam[key]["p_holm"] = running
        fam[key]["holm_detected"] = running < 0.05
    for key, e in fam.items():
        if e["meaningful"] and e["holm_detected"]:
            e["wording"] = "finding"
        elif e["detected"]:
            e["wording"] = "small difference below the 1 pp tolerance or not robust to multiplicity"
        else:
            e["wording"] = "no detected difference"

    # Secondary: per-level head differences (descriptive)
    pairs = [("sparse-M2048", "linear"), ("sparse-M2048", "dense-M2048"), ("sparse-M4096", "sparse-M2048"),
             ("sparse-M4096", "linear"), ("dense-M2048", "linear")]
    per_level = {f"{a} - {b}": {lv: stats([top1[a][(lv, s)] - top1[b][(lv, s)] for s in SEEDS]) for lv in LEVELS}
                 for a, b in pairs}

    # Exploratory
    means = {h: {lv: statistics.mean(top1[h][(lv, s)] for s in SEEDS) for lv in LEVELS} for h in HEADS}
    top5 = {h: {lv: statistics.mean(float(acc[(h, "source")][(lv, s)]["top5"]) for s in SEEDS) for lv in LEVELS}
            for h in HEADS}
    trainlvl = {h: {lv: statistics.mean(float(acc[(h, "train")][(lv, s)]["top1"]) for s in SEEDS) for lv in LEVELS}
                for h in HEADS}
    old = load_old()
    gain = {h: {**{lv: stats([top1[h][(lv, s)] - old[h][(lv, s)] for s in SEEDS]) for lv in LEVELS},
                "pooled": stats([statistics.mean(top1[h][(lv, s)] - old[h][(lv, s)] for lv in LEVELS) for s in SEEDS])}
            for h in HEADS}
    old_gaps = {"A1": stats([statistics.mean(old["sparse-M2048"][(lv, s)] - old["linear"][(lv, s)] for lv in LEVELS) for s in SEEDS]),
                "A2": stats([statistics.mean(old["sparse-M2048"][(lv, s)] - old["dense-M2048"][(lv, s)] for lv in LEVELS) for s in SEEDS]),
                "A3": stats([statistics.mean(old["sparse-M4096"][(lv, s)] - old["sparse-M2048"][(lv, s)] for lv in LEVELS) for s in SEEDS])}
    cost_diff = {}
    for lv in ("L080", "L090"):
        for a, b in (("sparse-M2048", "linear"), ("sparse-M2048", "dense-M2048"), ("sparse-M4096", "linear")):
            cost_diff[f"cost({a}) - cost({b}) at {lv}"] = stats(
                [(top1[a][("source", s)] - top1[a][(lv, s)]) - (top1[b][("source", s)] - top1[b][(lv, s)]) for s in SEEDS])

    res = {"runs": runs, "split_digests_match_seed001": digest_ok, "selected_epochs": sel,
           "instability_runs": dict(instab), "family": fam, "holm_order": order, "per_level": per_level,
           "exploratory": {"mean_top1": means, "mean_top5": top5, "train_level_eval_top1": trainlvl,
                           "gain_over_lr1e-3": gain, "old_lr1e-3_gaps_same_seeds": old_gaps,
                           "compression_cost_differences": cost_diff, "lr002_validation_equal20": LR002_VAL}}
    (HERE / "analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    p = lambda v: f"{100 * v:+.2f}"  # noqa: E731
    q = lambda v: f"{100 * v:.2f}"  # noqa: E731
    yn = lambda b: "yes" if b else "no"  # noqa: E731

    def row(name, e):
        return (f"| {name} | {p(e['mean'])} | {q(e['sd'])} | [{p(e['ci_low'])}, {p(e['ci_high'])}] | "
                f"{e['positive']}/{e['n']} | {e['p']:.4f} |")

    L = ["# RETEST-001 analysis (generated by analyze.py)", "",
         f"Runs: {runs}/84 (20-epoch histories). Split digest per seed equals SEED-001's: **{digest_ok}**. "
         "Metric: ImageNetV2-source top-1 of best.pt; paired by seed 45-51 (n = 7, df = 6).", "",
         "## Mean accuracy (%)", "", "| Head | source | L080 | L090 |", "| --- | --- | --- | --- |"]
    L += [f"| {h} | " + " | ".join(q(means[h][lv]) for lv in LEVELS) + " |" for h in HEADS]
    L += ["", "## CONFIRMATORY family (12 tests)", "",
          "| Test | Mean (pp) | sd | 95% CI | positive | p | Holm p | detected | meaningful (≥ 1 pp) | Holm-detected | wording |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, e in fam.items():
        L.append(row(k, e) + f" {e['p_holm']:.4f} | {yn(e['detected'])} | {yn(e['meaningful'])} | "
                 f"{yn(e['holm_detected'])} | {e['wording']} |")
    L += ["", "## Checks", "", "| Head | median selected epoch | runs at epoch ≥ 19 | runs at epoch ≤ 2 | budget binds | "
          "unstable runs |", "| --- | --- | --- | --- | --- | --- |"]
    for h in HEADS:
        L.append(f"| {h} | {sel[h]['median']} | {sel[h]['ge19']}/21 | {sel[h]['le2']}/21 | {yn(sel[h]['budget_binds'])} | "
                 f"{instab[h] if h in DICT else 'n/a (no training loss logged)'} |")
    L += ["", "## SECONDARY: per-level head differences (descriptive, not in the family)", "",
          "| Comparison | Level | Mean (pp) | sd | 95% CI | positive | p |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for comp, by in per_level.items():
        for lv, e in by.items():
            L.append(row(f"{comp} | {lv}", e))
    L += ["", "## EXPLORATORY (not pre-registered as confirmatory)", "",
          "### Gain over the earlier lr 1e-3 / 10-epoch runs (same seeds and levels; SEED-001, MSWEEP-001)", "",
          "| Head | Level | Mean (pp) | sd | 95% CI | positive | p |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for h in HEADS:
        for lv, e in gain[h].items():
            L.append(row(f"{h} | {lv}", e))
    L += ["", "### Pooled head gaps: old lr 1e-3 runs vs this task vs LR-002 validation prediction (pp)", "",
          "| Gap | old lr 1e-3 (test, same seeds) | RETEST-001 (test) | LR-002 validation, equal 20 ep |",
          "| --- | --- | --- | --- |"]
    for k, name in (("A1", "A1 sparse-M2048 - linear"), ("A2", "A2 sparse-M2048 - dense-M2048"),
                    ("A3", "A3 sparse-M4096 - sparse-M2048")):
        L.append(f"| {name} | {p(old_gaps[k]['mean'])} | {p(fam[name]['mean'])} | {LR002_VAL[k]:+.2f} |")
    L += ["", "### Compression-cost differences between heads", "",
          "| Comparison | Mean (pp) | sd | 95% CI | positive | p |", "| --- | --- | --- | --- | --- | --- |"]
    L += [row(k, e) for k, e in cost_diff.items()]
    L += ["", "### ImageNetV2 top-5 (source eval) and top-1 at the training level (%)", "",
          "| Head | top-5 source | top-5 L080 | top-5 L090 | top-1 train-level L080 | top-1 train-level L090 |",
          "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {h} | {q(top5[h]['source'])} | {q(top5[h]['L080'])} | {q(top5[h]['L090'])} | "
          f"{q(trainlvl[h]['L080'])} | {q(trainlvl[h]['L090'])} |" for h in HEADS]
    (HERE / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
