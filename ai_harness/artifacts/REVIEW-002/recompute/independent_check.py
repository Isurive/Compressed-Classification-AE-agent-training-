"""REVIEW-002 independent checks (reviewer-20261125-a). Read-only on runs/, src/, Datasets/.

    /c/Users/ericw/miniconda3/envs/ai360/python.exe -B ai_harness/artifacts/REVIEW-002/recompute/independent_check.py

Written from scratch (does not reuse the authors' analyze.py). Imports src/image_data.py only to rebuild the
seed splits, which is the code the runs used. Writes independent_check_output.txt next to this file.
"""
import csv
import hashlib
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
OUT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    OUT.append(line)


def rows(task):
    return list(csv.DictReader(open(ROOT / f"runs/{task}/results.csv", encoding="utf-8")))


R = {t: rows(t) for t in ("GRID-001", "GRID-002", "ABL-001")}

# ---------------------------------------------------------------- 1. counts, phases, settings per phase
say("## 1. Row counts and per-phase settings")
for t, rs in R.items():
    c = Counter((r["phase"], r["eval_level"] == "source") for r in rs)
    say(t, len(rs), "rows", dict(c))
    combos = Counter((r["phase"], r["model"], r["atoms"], r["k"], r["fc_init"], r["weight_decay"]) for r in rs)
    for k, v in sorted(combos.items()):
        say("   ", k, v)
    runs = {r["run_dir"] for r in rs}
    say("   distinct run_dirs:", len(runs))

# ---------------------------------------------------------------- 2. split digests per seed across all files
say("\n## 2. Split digests")
dig = defaultdict(set)
for t, rs in R.items():
    for r in rs:
        dig[int(r["seed"])].add(r["split_digest"])
for s, d in sorted(dig.items()):
    say("seed", s, "distinct digests across all three results.csv:", len(d), sorted(d)[0][:16])
from image_data import training_splits  # noqa: E402

POOL = ROOT / "Datasets" / "imagenet-1k" / "val"
splits = {}
for s in (42, 43, 44):
    tr, ti, va, vi, kind, digest = training_splits(str(POOL.resolve()), 0, 0, 0.10, s)
    splits[s] = (tr, list(ti), list(vi))
    keys_tr = {tr.key(i) for i in ti}
    keys_va = {va.key(i) for i in vi}
    say(f"seed {s}: kind={kind}; train {len(ti)} val {len(vi)}; overlap {len(keys_tr & keys_va)}; "
        f"recomputed digest matches csv: {digest in dig[s] and len(dig[s]) == 1}; val source is train source: {va is tr}; "
        f"pool root: {Path(tr.root).name if hasattr(tr, 'root') else '?'}; example key {tr.key(ti[0])}")
    per_class = Counter(tr.labels[i] for i in vi)
    say(f"   val images per class min/max {min(per_class.values())}/{max(per_class.values())}")

# ImageNetV2 vs pool content overlap by sha256
pool_sha = {r["sha256"] for r in csv.DictReader(open(ROOT / "Datasets/_manifests/imagenet-1k-val.csv", encoding="utf-8"))}
v2 = list(csv.DictReader(open(ROOT / "Datasets/_manifests/imagenetv2-matched-frequency.csv", encoding="utf-8")))
say(f"ImageNetV2 files {len(v2)}; pool files {len(pool_sha)}; identical-content files in both: "
    f"{len(pool_sha & {r['sha256'] for r in v2})}")

# ---------------------------------------------------------------- 3. tuning rule recomputed from tune_results.json
say("\n## 3. Freeze rules")
tune = json.loads((ROOT / "runs/GRID-001/tune_results.json").read_text())
for inits, path in ((None, "runs/GRID-001/tuned.json"), ("xavier", "runs/GRID-002/tuned.json")):
    frozen = json.loads((ROOT / path).read_text())
    for m in ("sparse", "dense", "linear"):
        cand = [r for r in tune if r["model"] == m and (inits is None or r["fc_init"] == inits)]
        best = max(cand, key=lambda r: (r["mean_val_top1"], -r["mean_val_ce"]))
        ok = all(best[k] == frozen[m][k] for k in ("atoms", "k", "fc_init", "weight_decay"))
        say(f"{path} {m}: {len(cand)} candidates; rule picks k={best['k']} init={best['fc_init']} "
            f"wd={best['weight_decay']} (val {100 * best['mean_val_top1']:.2f}); matches frozen: {ok}")
for r in sorted(tune, key=lambda r: (r["model"], -r["mean_val_top1"])):
    say(f"   tune {r['model']:6s} k={r['k']:<4} init={r['fc_init']:8s} wd={r['weight_decay']:<7g} "
        f"mean {100 * r['mean_val_top1']:.2f} src {100 * r['val_top1_source']:.2f} L90 {100 * r['val_top1_L090']:.2f}")

# ---------------------------------------------------------------- 4. accuracy tables from scratch
say("\n## 4. ImageNetV2-source top-1 (mean, sd) and paired drops, recomputed")


def cell(rs, phase, model, atoms=None, eval_source=True):
    d = {}
    for r in rs:
        if r["phase"] == phase and r["model"] == model and (atoms is None or int(r["atoms"]) == atoms) \
                and (r["eval_level"] == "source") == eval_source:
            if not eval_source and r["eval_level"] != r["train_level"]:
                continue
            d[(r["train_level"], int(r["seed"]))] = float(r["top1"]) * 100
    return d


def msd(xs):
    return st.mean(xs), st.stdev(xs)


LV = ["source", "L020", "L060", "L080", "L085", "L090", "L095", "L100"]
tabs = {}
for t, phase in (("GRID-001", "grid"), ("GRID-002", "grid-xavier")):
    for m in ("sparse", "dense", "linear"):
        c = cell(R[t], phase, m)
        tabs[(t, m)] = c
        parts = []
        for lv in LV:
            a = msd([c[(lv, s)] for s in (42, 43, 44)])
            if lv == "source":
                parts.append(f"{lv} {a[0]:.2f}±{a[1]:.2f}")
            else:
                d = msd([c[("source", s)] - c[(lv, s)] for s in (42, 43, 44)])
                parts.append(f"{lv} {a[0]:.2f}±{a[1]:.2f} (drop {d[0]:.2f}±{d[1]:.2f})")
        say(f"{t} {m}: " + "; ".join(parts))
for t in ("GRID-001", "GRID-002"):
    for other in ("dense", "linear"):
        g = []
        for lv in LV:
            gap = [tabs[(t, "sparse")][(lv, s)] - tabs[(t, other)][(lv, s)] for s in (42, 43, 44)]
            g.append(f"{lv} {st.mean(gap):+.2f} [{sum(x > 0 for x in gap)}/3]")
        say(f"{t} sparse-{other} gap: " + "; ".join(g))
# M sweep
for M in (512, 1024, 4096):
    c = cell(R["GRID-001"], "msweep", "sparse", M)
    parts = []
    for lv in ("source", "L080", "L090", "L095"):
        a = msd([c[(lv, s)] for s in (42, 43, 44)])
        parts.append(f"{lv} {a[0]:.2f}±{a[1]:.2f}")
    say(f"msweep M={M}: " + "; ".join(parts))
# Xavier minus inherited
for m in ("sparse", "dense", "linear"):
    say(f"xavier-inherited {m}: " + "; ".join(
        f"{lv} {st.mean(tabs[('GRID-002', m)][(lv, s)] - tabs[('GRID-001', m)][(lv, s)] for s in (42, 43, 44)):+.2f}"
        for lv in LV))

# ablation cells
say("\nABL-001 cells (ImageNetV2-source)")
A = {"sparse-full": cell(R["GRID-002"], "grid-xavier", "sparse"), "dense-full": cell(R["GRID-002"], "grid-xavier", "dense"),
     "linear": cell(R["GRID-002"], "grid-xavier", "linear"),
     "sparse-mse": cell(R["ABL-001"], "abl-mse", "sparse"), "dense-mse": cell(R["ABL-001"], "abl-mse", "dense"),
     "sparse-none": cell(R["ABL-001"], "abl-none", "sparse"), "dense-none": cell(R["ABL-001"], "abl-none", "dense")}
AL = ["source", "L060", "L080", "L090", "L095"]
for n, c in A.items():
    say(f"  {n:12s} " + "  ".join(f"{lv} {st.mean(c[(lv, s)] for s in (42, 43, 44)):.2f}" for lv in AL))
for a, b in (("sparse-mse", "sparse-none"), ("dense-mse", "dense-none"), ("sparse-none", "dense-none"),
             ("sparse-mse", "dense-mse"), ("sparse-full", "dense-full"), ("dense-none", "linear"),
             ("sparse-full", "linear"), ("sparse-mse", "linear"), ("dense-mse", "linear"),
             ("sparse-full", "sparse-mse"), ("dense-full", "dense-mse")):
    per = [A[a][(lv, s)] - A[b][(lv, s)] for lv in AL for s in (42, 43, 44)]
    # per-seed average over levels: the 3 split-level replicates
    seedavg = [st.mean(A[a][(lv, s)] - A[b][(lv, s)] for lv in AL) for s in (42, 43, 44)]
    say(f"  {a} - {b}: mean {st.mean(per):+.2f}; positive {sum(x > 0 for x in per)}/15; "
        f"per-seed means over levels {[round(x, 2) for x in seedavg]}")

# ---------------------------------------------------------------- 5. done.json vs results.csv for every run
say("\n## 5. done.json vs results.csv (all runs)")
for t, rs in R.items():
    bad, n = 0, 0
    for rd in sorted({r["run_dir"] for r in rs}):
        done = json.loads((ROOT / rd / "done.json").read_text())
        for r in (r for r in rs if r["run_dir"] == rd):
            n += 1
            checks = [done["split_digest"] == r["split_digest"], int(done["best_epoch"]) == int(r["best_epoch"]),
                      abs(done["val_top1"] - float(r["val_top1"])) < 1e-9, str(done["k"]) == r["k"],
                      str(done["atoms"]) == r["atoms"], done["fc_init"] == r["fc_init"], done["train_level"] == r["train_level"]]
            if not all(checks):
                bad += 1
                say("   MISMATCH", rd, checks)
    say(f"{t}: {n} rows checked against done.json, mismatches {bad}")
abl_w = Counter((json.loads((ROOT / rd / "done.json").read_text())["phase"],
                 json.loads((ROOT / rd / "done.json").read_text())["reconstruction_weight"],
                 json.loads((ROOT / rd / "done.json").read_text())["dependency_weight"],
                 json.loads((ROOT / rd / "done.json").read_text())["coherence_weight"])
                for rd in {r["run_dir"] for r in R["ABL-001"]})
say("ABL-001 done.json loss weights:", dict(abl_w))
be = defaultdict(list)
for t, rs in R.items():
    for r in rs:
        if r["eval_level"] == "source":
            be[(t, r["phase"], r["model"])].append(int(r["best_epoch"]))
for k, v in sorted(be.items()):
    say(f"best epochs {k}: min {min(v)} max {max(v)} counts {dict(sorted(Counter(v).items()))}")

# ---------------------------------------------------------------- 6. log-based loss decomposition (dictionary runs)
say("\n## 6. Loss weights verified from per-epoch training logs: loss - ce - w_mse*mse")
pat_head = re.compile(r"\[(grid-xavier|abl-mse|abl-none|grid)\] (sparse|dense) M=(\d+) k=(\d+) (\S+) seed (\d+)")
for log in ("ai_harness/artifacts/GRID-002/grid_log.txt", "ai_harness/artifacts/ABL-001/grid_log.txt"):
    cur, res = None, defaultdict(list)
    for line in open(ROOT / log, encoding="utf-8", errors="replace"):
        m = pat_head.search(line)
        if m:
            cur = m.group(1)
            continue
        if cur and '"epoch"' in line and '"train"' in line:
            rec = json.loads(line[line.index("{"):])
            tr = rec["train"]
            w = 1e-6 if cur == "abl-none" else 1.0
            res[cur].append((tr["loss"] - tr["ce"] - w * tr["feature_mse"], tr["feature_mse"], tr["ce"]))
    for ph, v in res.items():
        resid = [x[0] for x in v]
        say(f"{log.split('/')[2]} {ph}: {len(v)} epochs; residual min {min(resid):.2e} max {max(resid):.2e}; "
            f"train feature_mse range {min(x[1] for x in v):.3f}-{max(x[1] for x in v):.3f}; "
            f"train CE range {min(x[2] for x in v):.4f}-{max(x[2] for x in v):.3f}")

# ---------------------------------------------------------------- 7. best.pt spot checks
say("\n## 7. best.pt spot checks")
import torch  # noqa: E402

spots = ["runs/GRID-001/grid/sparse/L080/seed43", "runs/GRID-001/grid/linear/L090/seed44",
         "runs/GRID-002/grid/dense/L085/seed42", "runs/GRID-002/grid/linear/L080/seed43",
         "runs/ABL-001/abl-none/dense/L090/seed44", "runs/ABL-001/abl-mse/sparse/L060/seed43"]
for rd in spots:
    b = torch.load(ROOT / rd / "best.pt", map_location="cpu", weights_only=True)
    done = json.loads((ROOT / rd / "done.json").read_text())
    row = next(r for t in R.values() for r in t if r["run_dir"] == rd and r["eval_level"] == "source")
    info = {"format": b["format"], "epoch": b["epoch"], "val_top1": b["validation"]["top1"],
            "digest_ok": b["protocol"]["digest"] == row["split_digest"], "level": b["compression_level"]}
    if "atoms" in b:
        info.update(atoms=b["atoms"], k=b["k"], fc_shape=tuple(b["model"]["fc.weight"].shape),
                    enc_shape=tuple(b["model"]["encoder.weight"].shape), has_settings="settings" in b)
    else:
        info.update(settings=b.get("settings"), head_shape=tuple(b["head"]["weight"].shape))
    consistent = (b["epoch"] == int(row["best_epoch"]) and abs(b["validation"]["top1"] - float(row["val_top1"])) < 1e-9
                  and info["digest_ok"] and b["epoch"] == done["best_epoch"])
    say(rd, json.dumps(info, default=str), "consistent with csv/done:", consistent)
    last = ROOT / rd / "last.pt"
    if last.exists():
        L = torch.load(last, map_location="cpu", weights_only=True)
        s = L["settings"]
        say("   last.pt settings:", {k: s[k] for k in ("k", "atoms", "fc_init", "encoder_init", "reconstruction_weight",
                                                       "dependency_weight", "coherence_weight", "compression_level", "seed")},
            "history best epoch by (top1,-ce):",
            max(L["history"], key=lambda h: (h["validation"]["top1"], -h["validation"]["ce"]))["epoch"])
        del L
    # history.json for linear
    h = ROOT / rd / "history.json"
    if h.exists():
        hist = json.loads(h.read_text())
        best = max(hist, key=lambda e: (e["validation"]["top1"], -e["validation"]["ce"]))
        say("   history.json best epoch:", best["epoch"], "val", best["validation"]["top1"])

# Re-score one linear head on cached ImageNetV2 features (CPU, no training) to check the test pipeline.
say("\n## 8. Re-score linear best.pt on cached ImageNetV2-source features (CPU inference only)")
test = torch.load(ROOT / "runs/GRID-001/features/test_source.pt", map_location="cpu", weights_only=True)
say("test features", tuple(test["features"].shape), "labels", len(test["labels"]), "level", test["level"])
pool = torch.load(ROOT / "runs/GRID-001/features/pool_source.pt", map_location="cpu", weights_only=True)
say("pool features", tuple(pool["features"].shape), "pool keys_digest != test keys_digest:", pool["keys_digest"] != test["keys_digest"])
for rd in ("runs/GRID-001/grid/linear/L090/seed44", "runs/GRID-002/grid/linear/L080/seed43"):
    b = torch.load(ROOT / rd / "best.pt", map_location="cpu", weights_only=True)
    W, bias = b["head"]["weight"], b["head"]["bias"]
    with torch.no_grad():
        pred = (test["features"] @ W.T + bias).argmax(1)
    acc = (pred == test["labels"]).float().mean().item()
    row = next(r for t in R.values() for r in t if r["run_dir"] == rd and r["eval_level"] == "source")
    say(rd, f"rescored top1 {acc:.4f} vs csv {float(row['top1']):.4f}")
ref = json.loads((ROOT / "runs/GRID-001/reference_original_fc.json").read_text())
say("original fc on V2-source:", next(r["top1"] for r in ref if r["eval_level"] == "source"))

# ---------------------------------------------------------------- 9. bytes from the streamed manifest
say("\n## 9. Byte accounting from runs/GRID-001/jpeg_variants/manifest.csv (streamed)")
stem_bytes = defaultdict(dict)
src_bytes_m, raw_m, chan = {}, {}, Counter()
nrows = 0
with open(ROOT / "runs/GRID-001/jpeg_variants/manifest.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        nrows += 1
        stem = Path(r["source_path"]).stem
        q = int(r["quality"])
        stem_bytes[stem][q] = (int(r["jpeg_bytes"]), int(r["header_bytes"]), int(r["payload_bytes"]))
        assert int(r["jpeg_bytes"]) == int(r["header_bytes"]) + int(r["payload_bytes"])
        if q == 76:
            src_bytes_m[stem] = int(r["source_bytes"])
            raw_m[stem] = int(r["raw_bytes"])
            chan[(r["source_mode"], r["channels"])] += 1
            assert int(r["raw_bytes"]) == int(r["width"]) * int(r["height"]) * int(r["channels"])
say(f"manifest rows {nrows}; images {len(stem_bytes)}; qualities per image {Counter(len(v) for v in stem_bytes.values())}")
say("source mode/channels:", dict(chan))
ds = {Path(r["relative_path"]).stem: int(r["bytes"]) for r in csv.DictReader(open(ROOT / "Datasets/_manifests/imagenet-1k-val.csv", encoding="utf-8"))}
say("source bytes: manifest.csv vs Datasets manifest identical for all images:", all(ds[s] == src_bytes_m[s] for s in ds),
    "pool total", f"{sum(ds.values()):,}")
for s in (42, 43, 44):
    tr, ti, vi = splits[s]
    stems = [Path(tr.key(i)).stem for i in ti]
    src = sum(ds[x] for x in stems)
    raw = sum(raw_m[x] for x in stems)
    out = [f"seed {s} train45k: source {src:,} B, raw {raw:,} B"]
    for q in (76, 39, 20, 15, 10, 6, 1):
        b = sum(stem_bytes[x][q][0] for x in stems)
        h = sum(stem_bytes[x][q][1] for x in stems)
        out.append(f"q{q}: {b:,} B R_raw {raw / b:.1f} R_ex {src / b:.2f} save {100 * (1 - b / src):.1f}% hdr {100 * h / b:.1f}%")
    say("\n   ".join(out))
    if s == 42:
        stems_v = [Path(tr.key(i)).stem for i in vi]
        b20 = sum(stem_bytes[x][20][0] for x in stems_v)
        say(f"   seed 42 val5k q20: {b20:,} B; train+val at q20: {b20 + sum(stem_bytes[x][20][0] for x in stems):,} B")
# grayscale raw counted as 1 channel -> what R_raw would be if every image counted as RGB (model input)
tr, ti, _ = splits[42]
stems = [Path(tr.key(i)).stem for i in ti]
px = {}
with open(ROOT / "runs/GRID-001/jpeg_variants/manifest.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["quality"] == "20":
            px[Path(r["source_path"]).stem] = int(r["width"]) * int(r["height"])
raw3 = sum(3 * px[x] for x in stems)
b20 = sum(stem_bytes[x][20][0] for x in stems)
say(f"seed 42 L80 R_raw with every image counted as 3-channel RGB: {raw3 / b20:.2f} (reported 50.3)")

# ---------------------------------------------------------------- 10. integrity
say("\n## 10. Integrity")
for line in (ROOT / "ai_harness/artifacts/GRID-001/src_hashes_before.txt").read_text().splitlines():
    h, p = line.split(" *")
    now = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    say(f"{p}: {'unchanged' if now == h else 'CHANGED'}")
say("__pycache__ in src:", [str(p) for p in (ROOT / "src").rglob("__pycache__")])

(HERE / "independent_check_output.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
