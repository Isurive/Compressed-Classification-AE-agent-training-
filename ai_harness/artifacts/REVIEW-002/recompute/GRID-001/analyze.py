"""GRID-001 analysis: paired accuracy drops and per-seed training-split byte totals.

    python -B ai_harness/artifacts/GRID-001/analyze.py   (from Final Project, ai360 env)

Reads runs/GRID-001/results.csv and runs/GRID-001/jpeg_variants/manifest.csv; writes
analysis.json and analysis.md next to this file. Pre-declared rules (REASON-002 plan section 7):
drop(m, L) = A(m, source) - A(m, L) on ImageNetV2-source, paired by seed; a level retains accuracy
when the mean drop <= 1.0 percentage point.
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # REVIEW-002 copy: five levels up to the project root
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

LEVELS = ["source", "L020", "L060", "L080", "L085", "L090", "L095", "L100"]
TOL = 0.010


def mean_sd(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def accuracy_tables():
    rows = list(csv.DictReader(open(ROOT / "runs/GRID-001/results.csv", encoding="utf-8")))
    acc = defaultdict(dict)  # (phase, model, atoms, k, eval_kind) -> {(level, seed): top1}
    digests = defaultdict(set)
    for r in rows:
        kind = "v2_source" if r["eval_level"] == "source" else "v2_matched"
        acc[(r["phase"], r["model"], int(r["atoms"]), int(r["k"]), kind)][(r["train_level"], int(r["seed"]))] = float(r["top1"])
        digests[int(r["seed"])].add(r["split_digest"])
    # The M sweep reuses grid sparse M=2048 runs.
    out = {"split_digests_per_seed": {s: sorted(d) for s, d in digests.items()}, "groups": []}
    for (phase, model, atoms, k, kind), table in sorted(acc.items()):
        seeds = sorted({s for _, s in table})
        group = {"phase": phase, "model": model, "atoms": atoms, "k": k, "eval": kind, "levels": {}}
        for level in LEVELS:
            if not all((level, s) in table for s in seeds):
                continue
            a = [table[(level, s)] for s in seeds]
            entry = {"top1_mean": mean_sd(a)[0], "top1_sd": mean_sd(a)[1], "n": len(a)}
            if kind == "v2_source" and level != "source" and all(("source", s) in table for s in seeds):
                d = [table[("source", s)] - table[(level, s)] for s in seeds]
                entry["drop_mean"], entry["drop_sd"] = mean_sd(d)
                entry["retains_1pp"] = entry["drop_mean"] <= TOL
            group["levels"][level] = entry
        out["groups"].append(group)
    return out, acc


def interactions(acc):
    """drop(other, L) - drop(sparse, L), paired by seed; positive means sparse dropped less."""
    sparse = acc[("grid", "sparse", 2048, 512, "v2_source")]
    res = {}
    for other_key in [k for k in acc if k[0] == "grid" and k[1] in ("dense", "linear") and k[4] == "v2_source"]:
        other = acc[other_key]
        seeds = sorted({s for _, s in other})
        res[other_key[1]] = {}
        for level in LEVELS[1:]:
            d = [(other[("source", s)] - other[(level, s)]) - (sparse[("source", s)] - sparse[(level, s)]) for s in seeds]
            gap = [other[(level, s)] - sparse[(level, s)] for s in seeds]
            res[other_key[1]][level] = {"interaction_mean": mean_sd(d)[0], "interaction_sd": mean_sd(d)[1],
                                        "absolute_gap_other_minus_sparse": mean_sd(gap)[0]}
    return res


def byte_tables():
    from run_grid import pool_split, SEEDS  # noqa: E402  (pure split computation; no training)
    manifest = ROOT / "runs/GRID-001/jpeg_variants/manifest.csv"
    if not manifest.exists():
        return None
    by_stem = defaultdict(dict)  # stem -> {quality: row}
    for r in csv.DictReader(open(manifest, encoding="utf-8")):
        by_stem[Path(r["source_path"]).stem][int(r["quality"])] = r
    source_bytes = {r["relative_path"].split("/")[-1].rsplit(".", 1)[0]: int(r["bytes"])
                    for r in csv.DictReader(open(ROOT / "Datasets/_manifests/imagenet-1k-val.csv", encoding="utf-8"))}
    qualities = sorted({q for v in by_stem.values() for q in v}, reverse=True)
    pool_total = sum(source_bytes.values())
    tables = {"pool_source_bytes": pool_total, "seeds": {}}
    for seed in SEEDS:
        train, train_ids, _val, val_ids, _split, digest = pool_split(seed)
        groups = {"train45k": [Path(train.key(i)).stem for i in train_ids],
                  "val5k": [Path(train.key(i)).stem for i in val_ids]}
        seed_out = {"split_digest": digest}
        for name, stems in groups.items():
            missing = [s for s in stems if s not in by_stem]
            src = sum(source_bytes[s] for s in stems)
            pixels = sum(int(by_stem[s][qualities[0]]["width"]) * int(by_stem[s][qualities[0]]["height"]) for s in stems)
            raw = sum(int(by_stem[s][qualities[0]]["raw_bytes"]) for s in stems)
            rows = {"source": {"bytes": src, "bpp": 8 * src / pixels, "R_raw": raw / src, "R_existing": 1.0, "saving": 0.0}}
            for q in qualities:
                b = sum(int(by_stem[s][q]["jpeg_bytes"]) for s in stems)
                payload = sum(int(by_stem[s][q]["payload_bytes"]) for s in stems)
                rows[f"q{q}"] = {"bytes": b, "payload_bytes": payload, "header_bytes": b - payload,
                                 "bpp": 8 * b / pixels, "R_raw": raw / b, "R_existing": src / b, "saving": 1 - b / src}
            seed_out[name] = {"images": len(stems), "missing_in_manifest": len(missing), "pixels": pixels,
                              "raw_rgb_bytes": raw, "rows": rows}
        tables["seeds"][seed] = seed_out
    return tables


def main():
    acc_out, acc = accuracy_tables()
    result = {"accuracy": acc_out, "interactions_vs_sparse": interactions(acc), "bytes": byte_tables()}
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# GRID-001 analysis tables (generated by analyze.py)", ""]
    for g in acc_out["groups"]:
        lines.append(f"## {g['phase']} {g['model']} M={g['atoms']} k={g['k']} eval={g['eval']}")
        lines.append("| train level | top-1 mean | sd | drop vs source (paired) | drop sd | within 1.0 pp |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for level, e in g["levels"].items():
            drop = f"{100 * e['drop_mean']:.2f}" if "drop_mean" in e else ""
            dsd = f"{100 * e['drop_sd']:.2f}" if "drop_sd" in e else ""
            ok = ("yes" if e["retains_1pp"] else "no") if "retains_1pp" in e else ""
            lines.append(f"| {level} | {100 * e['top1_mean']:.2f} | {100 * e['top1_sd']:.2f} | {drop} | {dsd} | {ok} |")
        lines.append("")
    lines.append("## Interaction: drop(other) - drop(sparse), pp, paired by seed; and absolute gap other - sparse")
    lines.append("| other | level | interaction mean | sd | absolute gap |")
    lines.append("| --- | --- | --- | --- | --- |")
    for other, by_level in result["interactions_vs_sparse"].items():
        for level, e in by_level.items():
            lines.append(f"| {other} | {level} | {100 * e['interaction_mean']:.2f} | {100 * e['interaction_sd']:.2f} | "
                         f"{100 * e['absolute_gap_other_minus_sparse']:.2f} |")
    lines.append("")
    b = result["bytes"]
    if b:
        for seed, s in b["seeds"].items():
            for name in ("train45k", "val5k"):
                t = s[name]
                lines.append(f"## Bytes, seed {seed}, {name} ({t['images']} images, missing {t['missing_in_manifest']})")
                lines.append("| variant | full-file bytes | header bytes | bpp | R_raw | R_existing | saving vs stored |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- |")
                for v, r in t["rows"].items():
                    lines.append(f"| {v} | {r['bytes']:,} | {r.get('header_bytes', '')} | {r['bpp']:.3f} | {r['R_raw']:.1f} | "
                                 f"{r['R_existing']:.2f} | {100 * r['saving']:.1f}% |")
                lines.append("")
    (HERE / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
