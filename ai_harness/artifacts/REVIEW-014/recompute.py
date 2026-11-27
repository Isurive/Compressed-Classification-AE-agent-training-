"""REVIEW-014 independent recomputation of WD-001 phase B (does not import analyze.py).
Run from Final Project: python -B ai_harness/artifacts/REVIEW-014/recompute.py
"""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEVELS = ["source", "L080", "L090"]
SEEDS = range(45, 52)
T = 2.447


def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def tcdf_p(t, df=6):
    # two-sided p via regularized incomplete beta, continued fraction (independent of analyze.py's Simpson rule)
    x = df / (df + t * t)
    a, b = df / 2, 0.5

    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / d
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; c = 1 + aa / c; d = 1 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; c = 1 + aa / c; d = 1 / d; h *= d * c
        return h
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * betacf(a, b, x) / a
    return 1 - math.exp(lbt) * betacf(b, a, 1 - x) / b


def summ(x):
    m, s = mean(x), sd(x)
    se = s / math.sqrt(len(x))
    t = m / se
    return dict(mean=100 * m, sd=100 * s, lo=100 * (m - T * se), hi=100 * (m + T * se), pos=sum(v > 0 for v in x),
                t=t, p=tcdf_p(t))


def fmt(d):
    return (f"{d['mean']:+.3f} sd {d['sd']:.3f} CI [{d['lo']:+.3f}, {d['hi']:+.3f}] pos {d['pos']}/7 "
            f"t {d['t']:.3f} p {d['p']:.6f}")


# --- load test rows
src = {}  # head -> {(lv, seed): top1}
tr = defaultdict(dict)
t5 = defaultdict(dict)
rows_new = []
for r in csv.DictReader(open(ROOT / "runs/WD-001/results.csv", encoding="utf-8")):
    rows_new.append(r)
print("WD-001 rows:", len(rows_new), Counter(r["phase"] for r in rows_new))
keys = Counter((r["phase"], r["run_dir"], r["eval_set"], r["eval_level"]) for r in rows_new)
print("duplicate keys:", [k for k, v in keys.items() if v > 1])
per_run = Counter(r["run_dir"] for r in rows_new)
bad = [k for k, v in per_run.items() if v != (1 if k.split("/")[4] == "source" else 2)]
print("runs in csv:", len(per_run), "runs with wrong row count:", bad)
sel = json.loads((ROOT / "ai_harness/artifacts/WD-001/selected.json").read_text())
headmap = {("dense", "2048"): "dense-CE", ("sparse", "512"): "sparse-CE"}
for r in rows_new:
    h = r["run_dir"].split("/")[3]
    assert headmap[(r["model"], r["k"])] == h
    assert r["run_dir"] == f"runs/WD-001/test/{h}/{r['train_level']}/seed{r['seed']}"
    assert float(r["weight_decay"]) == sel[h]["weight_decay"] and r["eval_set"] == "imagenetv2"
    if r["eval_level"] == "source":
        src.setdefault(h, {})[(r["train_level"], int(r["seed"]))] = float(r["top1"])
        t5[h][(r["train_level"], int(r["seed"]))] = float(r["top5"])
    if r["eval_level"] == r["train_level"]:
        tr[h][(r["train_level"], int(r["seed"]))] = float(r["top1"])

lrmap = {}
for r in csv.DictReader(open(ROOT / "runs/RETEST-001/results.csv", encoding="utf-8")):
    if r["phase"] != "retest":
        continue
    h = r["run_dir"].split("/")[2]
    if h not in ("dense-M2048", "sparse-M2048", "linear"):
        continue
    d = json.loads((ROOT / r["run_dir"] / "done.json").read_text())
    lrmap.setdefault(h, set()).add((d["lr"], d["epochs"], d["weight_decay"]))
    if r["eval_level"] == "source":
        k = (r["train_level"], int(r["seed"]))
        assert k not in src.get(h, {})
        src.setdefault(h, {})[k] = float(r["top1"])
        t5[h][k] = float(r["top5"])
    if r["eval_level"] == r["train_level"]:
        tr[h][(r["train_level"], int(r["seed"]))] = float(r["top1"])
print("RETEST-001 comparison (lr, epochs, wd):", lrmap)
for h, d in src.items():
    assert len(d) == 21, (h, len(d))

print("\nMean top-1 (source eval):")
for h in src:
    print(f"  {h:13s}", " ".join(f"{100 * mean([src[h][(lv, s)] for s in SEEDS]):.3f}" for lv in LEVELS),
          "| top5", " ".join(f"{100 * mean([t5[h][(lv, s)] for s in SEEDS]):.3f}" for lv in LEVELS),
          "| train-level", " ".join(f"{100 * mean([tr[h][(lv, s)] for s in SEEDS]):.3f}" for lv in ("L080", "L090")))

pairs = [("W1", "dense-CE", "dense-M2048"), ("W2", "sparse-CE", "sparse-M2048"), ("W3", "dense-CE", "linear"),
         ("W4", "sparse-CE", "linear")]
fam = {}
print("\nConfirmatory family:")
for name, a, b in pairs:
    x = [mean([src[a][(lv, s)] - src[b][(lv, s)] for lv in LEVELS]) for s in SEEDS]
    fam[name] = summ(x)
    print(f"  {name} {a}-{b}: {fmt(fam[name])}")
order = sorted(fam, key=lambda k: fam[k]["p"])
run = 0
for i, k in enumerate(order):
    run = max(run, min(1, (4 - i) * fam[k]["p"]))
    print(f"  Holm {k}: p_adj {run:.6f} detected {run < 0.05}; CI excludes 0: "
          f"{fam[k]['lo'] > 0 or fam[k]['hi'] < 0}; meaningful@1.0 {abs(fam[k]['mean']) >= 1.0}; "
          f"@0.5 {abs(fam[k]['mean']) >= 0.5}")

print("\nPer level:")
for name, a, b in pairs:
    for lv in LEVELS:
        print(f"  {a}-{b} {lv}: {fmt(summ([src[a][(lv, s)] - src[b][(lv, s)] for s in SEEDS]))}")
print("\nCompression cost (source-trained - level-trained):")
for h in ("dense-CE", "sparse-CE", "dense-M2048", "sparse-M2048", "linear"):
    for lv in ("L080", "L090"):
        print(f"  {h} {lv}: {fmt(summ([src[h][('source', s)] - src[h][(lv, s)] for s in SEEDS]))}")
print("\nCost differences:")
for lv in ("L080", "L090"):
    for _, a, b in pairs:
        x = [(src[a][("source", s)] - src[a][(lv, s)]) - (src[b][("source", s)] - src[b][(lv, s)]) for s in SEEDS]
        print(f"  cost({a})-cost({b}) {lv}: {fmt(summ(x))}")

# --- done.json checks
s001 = defaultdict(set)
for r in csv.DictReader(open(ROOT / "runs/SEED-001/results.csv", encoding="utf-8")):
    s001[int(r["seed"])].add(r["split_digest"])
print("\nSEED-001 digests per seed (count):", {s: len(v) for s, v in s001.items()})
dirs = sorted(p.parent for p in (ROOT / "runs/WD-001/test").glob("*/*/seed*/done.json"))
alldirs = sorted(p for p in (ROOT / "runs/WD-001/test").glob("*/*/seed*") if p.is_dir())
print("test run dirs:", len(alldirs), "with done.json:", len(dirs))
ep, groups, inst, dig_ok, csvmatch = defaultdict(list), Counter(), Counter(), True, True
leftovers = []
for d in alldirs:
    leftovers += [f.name for f in d.iterdir() if f.name not in ("done.json", "best.pt", ".training.lock")]
for p in dirs:
    d = json.loads((p / "done.json").read_text())
    h = d["head"]
    c = sel[h]
    assert (d["lr"], d["weight_decay"], d["epochs"]) == (c["lr"], c["weight_decay"], c["epochs"]), p
    assert len(d["history"]) == d["epochs"] == 20 and [e["epoch"] for e in d["history"]] == list(range(1, 21))
    assert d["phase"] == "wd-test" and d["reconstruction_weight"] == 1e-6 and d["dependency_weight"] == 0
    dig_ok &= s001[d["seed"]] == {d["split_digest"]}
    ep[h].append(d["best_epoch"])
    g = d["wd_groups"]
    groups[(h, json.dumps(g["decay"]), json.dumps(g["no_decay"]), g["wd"])] += 1
    L = [e["train_loss"] for e in d["history"]]
    inst[h] += any(b > 1.05 * a for a, b in zip(L, L[1:]))
    # best_epoch consistent with history: max val top1, ties by lower val CE
    best = max(d["history"], key=lambda e: (e["val_top1"], -e["val_ce"]))
    if best["epoch"] != d["best_epoch"]:
        print("  best-epoch mismatch", p, best["epoch"], d["best_epoch"])
    # csv row consistent with done.json
    for r in rows_new:
        if r["run_dir"] == d["run_dir"]:
            csvmatch &= (int(r["best_epoch"]) == d["best_epoch"] and r["split_digest"] == d["split_digest"]
                         and float(r["val_top1"]) == d["val_top1"])
print("leftover files:", Counter(leftovers))
print("digests equal SEED-001 (single digest per seed):", dig_ok, "| csv rows match done.json:", csvmatch)
print("wd groups:", dict(groups))
print("instability:", dict(inst))
for h, e in ep.items():
    e2 = sorted(e)
    print(f"selected epochs {h}: median {e2[len(e2) // 2]}, >=19: {sum(x >= 19 for x in e)}/21, <=2: "
          f"{sum(x <= 2 for x in e)}/21, all {e2}")

# compare with analysis.json
aj = json.loads((ROOT / "ai_harness/artifacts/WD-001/analysis.json").read_text())
print("\nanalysis.json family:")
for k, v in aj["family"].items():
    print(f"  {k}: mean {100 * v['mean']:+.4f} lo {100 * v['ci_low']:+.4f} hi {100 * v['ci_high']:+.4f} "
          f"p {v['p']:.6g} p_holm {v['p_holm']:.6g}")
