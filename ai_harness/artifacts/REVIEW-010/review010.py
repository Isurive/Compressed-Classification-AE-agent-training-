"""REVIEW-010 independent recompute of RETEST-001 (standard library only; read-only).

Run:  py -I review010.py "<Final Project path>"
Does not import analyze.py. Uses a regularized-incomplete-beta t CDF (not Simpson) and an exact t quantile by bisection.
"""
import csv
import io
import json
import math
import os
import pickletools
import sqlite3
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(sys.argv[1])
RUNS = ROOT / "runs" / "RETEST-001"
ART = ROOT / "ai_harness" / "artifacts" / "RETEST-001"
HEADS = ["sparse-M2048", "sparse-M4096", "dense-M2048", "linear"]
DICT = HEADS[:3]
LEVELS = ["source", "L080", "L090"]
SEEDS = list(range(45, 52))
LOCAL = timezone(timedelta(hours=-7))
MODEL = {"sparse-M2048": ("sparse", "2048", "512"), "sparse-M4096": ("sparse", "4096", "1024"),
         "dense-M2048": ("dense", "2048", "2048"), "linear": ("linear", "0", "0")}
LR = {"sparse-M2048": 1e-4, "sparse-M4096": 1e-4, "dense-M2048": 1e-4, "linear": 3e-4}


# ---------- t distribution via regularized incomplete beta (Numerical Recipes continued fraction) ----------
def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / d if abs(d) > 1e-300 else 1e300
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / d if abs(d) > 1e-300 else 1e300
        c = 1 + aa / c if abs(c) > 1e-300 else 1e300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / d if abs(d) > 1e-300 else 1e300
        c = 1 + aa / c if abs(c) > 1e-300 else 1e300
        de = d * c
        h *= de
        if abs(de - 1) < 1e-15:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b


def p_two(t, df):
    return betai(df / 2, 0.5, df / (df + t * t))


def t_quant(df, conf=0.95):
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if p_two(mid, df) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


TQ6 = t_quant(6)


def st(xs, tq=2.447):
    n = len(xs)
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    se = sd / math.sqrt(n)
    t = m / se
    return {"mean": m, "sd": sd, "lo": m - tq * se, "hi": m + tq * se, "lo_exact": m - TQ6 * se,
            "hi_exact": m + TQ6 * se, "p": p_two(abs(t), n - 1), "pos": sum(x > 0 for x in xs), "n": n}


def fmt(e):
    return (f"{100*e['mean']:+.3f} sd {100*e['sd']:.3f} CI [{100*e['lo']:+.3f}, {100*e['hi']:+.3f}] "
            f"(exact-t [{100*e['lo_exact']:+.3f}, {100*e['hi_exact']:+.3f}]) {e['pos']}/{e['n']} p {e['p']:.5f}")


out = []
P = out.append

# ---------- 1. Timing ----------
P("== 1. timing ==")
for name in ("task.json", "analyze.py", "report.md", "analysis.md", "grid_log.txt"):
    P(f"{name}: mtime {datetime.fromtimestamp(os.path.getmtime(ART / name), LOCAL).isoformat()}")
P(f"src/run_retest.py mtime {datetime.fromtimestamp(os.path.getmtime(ROOT / 'src' / 'run_retest.py'), LOCAL).isoformat()}")
con = sqlite3.connect(f"file:{(ROOT / 'ai_harness' / 'state.sqlite3').as_posix()}?mode=ro", uri=True)
for row in con.execute("select seq, task, action, agent, substr(detail,1,120), at from events where task in "
                       "('RETEST-001','REVIEW-010') order by seq"):
    P(f"ledger event {row}")
for row in con.execute("select id, state, owner, created, updated from tasks where id='RETEST-001'"):
    P(f"ledger task {row}")
spec = json.loads(con.execute("select spec from tasks where id='RETEST-001'").fetchone()[0])
filespec = json.loads((ART / "task.json").read_text(encoding="utf-8"))
P(f"ledger spec == task.json (description, write_paths, acceptance): "
  f"{all(spec.get(k) == filespec.get(k) for k in ('description', 'write_paths', 'acceptance', 'title'))}")

# done.json mtimes (completion order) around analyze.py write
az = os.path.getmtime(ART / "analyze.py")
done_times = sorted((os.path.getmtime(p), p.relative_to(RUNS).as_posix()) for p in RUNS.glob("*/*/seed*/done.json"))
P(f"done.json files: {len(done_times)}; first {datetime.fromtimestamp(done_times[0][0], LOCAL).time()} "
  f"{done_times[0][1]}; last {datetime.fromtimestamp(done_times[-1][0], LOCAL).time()} {done_times[-1][1]}")
P(f"runs finished before analyze.py mtime: {[n for t, n in done_times if t < az]}")

# ---------- 2. results.csv integrity ----------
P("\n== 2. results.csv ==")
rows = list(csv.DictReader(open(RUNS / "results.csv", encoding="utf-8", newline="")))
P(f"rows {len(rows)}; phases {Counter(r['phase'] for r in rows)}; eval_set {Counter(r['eval_set'] for r in rows)}")
keys = Counter((r["run_dir"], r["eval_level"]) for r in rows)
P(f"duplicate (run_dir, eval_level): {[k for k, c in keys.items() if c > 1]}")
by = {}
bad = []
for r in rows:
    parts = r["run_dir"].split("/")
    head, lv, sd = parts[2], parts[3], int(parts[4][4:])
    if (r["model"], r["atoms"], r["k"]) != MODEL[head] or r["train_level"] != lv or int(r["seed"]) != sd:
        bad.append(r["run_dir"])
    if r["eval_level"] not in ("source", lv):
        bad.append(("eval", r["run_dir"], r["eval_level"]))
    if head == "linear" and float(r["weight_decay"]) != 1e-4 or head != "linear" and float(r["weight_decay"]) != 0:
        bad.append(("wd", r["run_dir"]))
    by[(head, lv, sd, r["eval_level"])] = r
P(f"label mismatches: {bad}")
missing = [(h, lv, s, ev) for h in HEADS for lv in LEVELS for s in SEEDS for ev in dict.fromkeys(["source", lv])
           if (h, lv, s, ev) not in by]
P(f"missing expected rows: {missing}; expected count {4*7*(1+2+2)} = 140")

# ---------- 3. done.json checks ----------
P("\n== 3. done.json ==")
digests = defaultdict(set)
sel = defaultdict(list)
instab = defaultdict(list)
issues = []
present = Counter()
for h in HEADS:
    for lv in LEVELS:
        for s in SEEDS:
            d = RUNS / h / lv / f"seed{s}"
            present["last.pt"] += (d / "last.pt").exists()
            present["best.pt"] += (d / "best.pt").exists()
            j = json.loads((d / "done.json").read_text())
            hist = j["history"]
            if not (j["epochs"] == 20 and len(hist) == 20 and [e["epoch"] for e in hist] == list(range(1, 21))):
                issues.append(("epochs", h, lv, s))
            if j["head"] != h or j["train_level"] != lv or j["seed"] != s or j["phase"] != "retest":
                issues.append(("labels", h, lv, s))
            if abs(j["lr"] - LR[h]) > 1e-15:
                issues.append(("lr", h, lv, s, j["lr"]))
            if h in DICT and (j.get("reconstruction_weight"), j.get("dependency_weight"), j.get("coherence_weight")) != (1.0, 0.0, 0.0):
                issues.append(("weights", h, lv, s))
            # Independent best-epoch rule: max (val_top1, -val_ce), first occurrence wins (strict >)
            best, be = None, None
            for e in hist:
                sc = (e["val_top1"], -e["val_ce"])
                if best is None or sc > best:
                    best, be = sc, e["epoch"]
            if be != j["best_epoch"] or abs(hist[be - 1]["val_top1"] - j["val_top1"]) > 1e-12:
                issues.append(("best_epoch", h, lv, s, be, j["best_epoch"]))
            digests[s].add(j["split_digest"])
            sel[h].append(j["best_epoch"])
            if h in DICT:
                L = [e["train_loss"] for e in hist]
                rises = [(i + 2, b / a) for i, (a, b) in enumerate(zip(L, L[1:])) if b > a * 1.05]
                instab[h].append(rises)
                if any(x is None for x in L):
                    issues.append(("noloss", h, lv, s))
            else:
                if any(e["train_loss"] is not None for e in hist):
                    issues.append(("linear-loss", h, lv, s))
            # results rows agree with done.json
            for ev in dict.fromkeys(["source", lv]):
                r = by[(h, lv, s, ev)]
                if (int(r["best_epoch"]) != j["best_epoch"] or abs(float(r["val_top1"]) - j["val_top1"]) > 1e-12
                        or r["split_digest"] != j["split_digest"]):
                    issues.append(("row-vs-done", h, lv, s, ev))
P(f"issues: {issues}")
P(f"files present: {dict(present)} of 84")
s1 = defaultdict(set)
for r in csv.DictReader(open(ROOT / "runs" / "SEED-001" / "results.csv", encoding="utf-8")):
    s1[int(r["seed"])].add(r["split_digest"])
P(f"digests: one per seed {all(len(digests[s]) == 1 for s in SEEDS)}; equal SEED-001 "
  f"{all(digests[s] == s1[s] for s in SEEDS)}; distinct across seeds {len(set().union(*digests.values()))}")
for h in HEADS:
    e = sel[h]
    P(f"selected epochs {h}: median {statistics.median(e)} >=19: {sum(x >= 19 for x in e)} <=2: {sum(x <= 2 for x in e)} "
      f"dist {sorted(Counter(e).items())}")
for h in DICT:
    P(f"instability {h}: {sum(bool(x) for x in instab[h])}/21 runs; max ratio "
      f"{max((max((v for _, v in x), default=0) for x in instab[h]), default=0)}")
    # also max consecutive ratio overall
mx = 0
for h in DICT:
    for lv in LEVELS:
        for s in SEEDS:
            L = [e["train_loss"] for e in json.loads((RUNS / h / lv / f"seed{s}" / "done.json").read_text())["history"]]
            mx = max(mx, max(b / a for a, b in zip(L, L[1:])))
P(f"largest consecutive train-loss ratio across 63 dictionary runs: {mx:.4f}")

# ---------- 4. confirmatory family ----------
P("\n== 4. family ==")
T = {h: {(lv, s): float(by[(h, lv, s, "source")]["top1"]) for lv in LEVELS for s in SEEDS} for h in HEADS}
for h in HEADS:
    P(f"mean {h}: " + "  ".join(f"{lv} {100*statistics.mean(T[h][(lv, s)] for s in SEEDS):.3f}" for lv in LEVELS))


def pooled(a, b, tab=T):
    return [statistics.mean(tab[a][(lv, s)] - tab[b][(lv, s)] for lv in LEVELS) for s in SEEDS]


fam = {"A1": pooled("sparse-M2048", "linear"), "A2": pooled("sparse-M2048", "dense-M2048"),
       "A3": pooled("sparse-M4096", "sparse-M2048"), "A4": pooled("sparse-M4096", "linear")}
for i, lv in enumerate(("L080", "L090")):
    for j, h in enumerate(HEADS):
        fam[f"B{1+4*i+j} {h} {lv}"] = [T[h][("source", s)] - T[h][(lv, s)] for s in SEEDS]
F = {k: st(v) for k, v in fam.items()}
order = sorted(F, key=lambda k: F[k]["p"])
run = 0
for i, k in enumerate(order):
    run = max(run, min(1, (12 - i) * F[k]["p"]))
    F[k]["holm"] = run
P(f"t quantile df6 exact {TQ6:.5f} vs 2.447")
for k, e in F.items():
    det = e["lo"] > 0 or e["hi"] < 0
    mean_ok = abs(e["mean"]) >= 0.01
    word = "finding" if det and mean_ok and e["holm"] < 0.05 else ("small/not robust" if det else "no detected difference")
    P(f"{k}: {fmt(e)} holm {e['holm']:.5f} detected {det} meaningful {det and mean_ok} -> {word}")
    if k == "A2":
        P(f"  A2 per-seed values (pp): {[round(100*x, 3) for x in fam[k]]}")

# per-level secondary
P("\n-- per-level (secondary) --")
for a, b in (("sparse-M2048", "linear"), ("sparse-M4096", "linear"), ("dense-M2048", "linear"),
             ("sparse-M2048", "dense-M2048"), ("sparse-M4096", "sparse-M2048")):
    for lv in LEVELS:
        P(f"{a} - {b} @ {lv}: {fmt(st([T[a][(lv, s)] - T[b][(lv, s)] for s in SEEDS]))}")

# ---------- 5. exploratory ----------
P("\n== 5. exploratory ==")
OLD = {"sparse-M2048": ("SEED-001", "seed-mse", "sparse", "2048"), "dense-M2048": ("SEED-001", "seed-mse", "dense", "2048"),
       "linear": ("SEED-001", "seed-linear", "linear", "0"), "sparse-M4096": ("MSWEEP-001", "msweep-clean", "sparse", "4096")}
old = {}
for h, (task, ph, model, atoms) in OLD.items():
    rs = list(csv.DictReader(open(ROOT / "runs" / task / "results.csv", encoding="utf-8")))
    P(f"{task} phases: {sorted(Counter(r['phase'] for r in rs).items())}")
    sel_rows = [r for r in rs if r["phase"] == ph and r["model"] == model and r["atoms"] == atoms
                and r["eval_level"] == "source" and r["train_level"] in LEVELS and int(r["seed"]) in SEEDS]
    c = Counter((r["train_level"], int(r["seed"])) for r in sel_rows)
    P(f"old {h}: {len(sel_rows)} rows, dup keys {[k for k, v in c.items() if v > 1]}")
    old[h] = {(r["train_level"], int(r["seed"])): float(r["top1"]) for r in sel_rows}
    # sanity: old runs used same split digests
    dd = {int(r["seed"]): r["split_digest"] for r in sel_rows}
    P(f"  old digests equal RETEST digests: {all({dd[s]} == digests[s] for s in SEEDS)}")
for h in HEADS:
    g = [statistics.mean(T[h][(lv, s)] - old[h][(lv, s)] for lv in LEVELS) for s in SEEDS]
    P(f"gain {h} pooled: {fmt(st(g))}; per level " + ", ".join(
        f"{lv} {100*statistics.mean(T[h][(lv, s)] - old[h][(lv, s)] for s in SEEDS):+.2f}" for lv in LEVELS))
for name, a, b in (("A1", "sparse-M2048", "linear"), ("A2", "sparse-M2048", "dense-M2048"),
                   ("A3", "sparse-M4096", "sparse-M2048"), ("A4", "sparse-M4096", "linear")):
    P(f"old gap {name}: {fmt(st(pooled(a, b, old)))}")
P("-- compression-cost differences (new) --")
for lv in ("L080", "L090"):
    for a, b in (("sparse-M2048", "linear"), ("sparse-M2048", "dense-M2048"), ("sparse-M4096", "linear"),
                 ("dense-M2048", "linear"), ("sparse-M4096", "sparse-M2048")):
        v = [(T[a][("source", s)] - T[a][(lv, s)]) - (T[b][("source", s)] - T[b][(lv, s)]) for s in SEEDS]
        P(f"new cost({a}) - cost({b}) @ {lv}: {fmt(st(v))}")
P("-- compression-cost differences (old lr 1e-3, same seeds) --")
for lv in ("L080", "L090"):
    for a, b in (("sparse-M2048", "linear"), ("sparse-M2048", "dense-M2048"), ("sparse-M4096", "linear"),
                 ("dense-M2048", "linear")):
        v = [(old[a][("source", s)] - old[a][(lv, s)]) - (old[b][("source", s)] - old[b][(lv, s)]) for s in SEEDS]
        P(f"old cost({a}) - cost({b}) @ {lv}: {fmt(st(v))}")
P("-- old per-head costs --")
for h in HEADS:
    P(f"old cost {h}: " + ", ".join(f"{lv} {100*statistics.mean(old[h][('source', s)] - old[h][(lv, s)] for s in SEEDS):+.2f}"
                                     for lv in ("L080", "L090")))
# top-5 / train-level
for h in HEADS:
    t5 = [statistics.mean(float(by[(h, lv, s, 'source')]['top5']) for s in SEEDS) for lv in LEVELS]
    tl = [statistics.mean(float(by[(h, lv, s, lv)]['top1']) for s in SEEDS) for lv in ("L080", "L090")]
    drop = [100 * (statistics.mean(T[h][(lv, s)] for s in SEEDS) - x) for lv, x in zip(("L080", "L090"), tl)]
    P(f"{h}: top5 {[round(100*x, 2) for x in t5]}; train-level top1 {[round(100*x, 2) for x in tl]}; "
      f"extra drop vs source eval {[round(x, 2) for x in drop]}")

# ---------- 6. linear lr actually used (pickle opcodes of best.pt 'settings'; no unpickling) ----------
P("\n== 6. linear best.pt settings via pickletools (no object construction) ==")
combos, epoch_ok = set(), True
for lv in LEVELS:
    for s in SEEDS:
        p = RUNS / "linear" / lv / f"seed{s}" / "best.pt"
        with zipfile.ZipFile(p) as z:
            data = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
        ops = list(pickletools.genops(data))  # opcode stream only; nothing is unpickled or executed
        vals = {}
        for i, (op, arg, _) in enumerate(ops):
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE") and arg in ("lr", "epochs", "weight_decay", "batch_size", "epoch"):
                j = i + 1
                while ops[j][0].name in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
                    j += 1
                vals.setdefault(arg, []).append(ops[j][1])
        combos.add(tuple((k, tuple(vals.get(k, []))) for k in ("lr", "epochs", "weight_decay", "batch_size")))
        dj = json.loads((RUNS / "linear" / lv / f"seed{s}" / "done.json").read_text())
        epoch_ok &= vals.get("epoch", [None])[0] == dj["best_epoch"]
P(f"distinct linear best.pt settings over 21 runs: {sorted(combos)}; checkpoint epoch == done.json best_epoch: {epoch_ok}")

print("\n".join(out))
