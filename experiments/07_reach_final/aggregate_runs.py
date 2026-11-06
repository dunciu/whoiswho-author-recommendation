"""
Aggregate results/runs/*/metrics.csv into paper-ready tables (mean +- std over seeds).

    python experiments/07_reach_final/aggregate_runs.py [--runs results/runs] [--out results/summary.md]

Produces:
  * Table III: NDCG@K / Recall@K for every expert and baseline at C = 0, 1, 2 (config "full")
  * Table IV: ablations at K = 100 (fusion ablations from "full", feature ablations from other configs)
  * Significance: REACH vs each baseline from each seed's significance.csv
  * Learned lambda and mean REACH gate weights per context level
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--out", default="results/summary.md")
args = ap.parse_args()

metrics = defaultdict(lambda: defaultdict(list))   # (config) -> (method, ctx, K) -> [(ndcg, recall)]
sig = defaultdict(list)                             # (ctx, K, baseline) -> [(diff, lo, hi, p)]
extra = defaultdict(list)                           # config -> [config.json dicts]
stats = None

for d in sorted(glob.glob(os.path.join(args.runs, "*_seed*"))):
    m = re.match(r"(.+)_seed(\d+)$", os.path.basename(d))
    if not m or not os.path.exists(os.path.join(d, "metrics.csv")):
        continue
    cfg = m.group(1)
    with open(os.path.join(d, "metrics.csv")) as f:
        for r in csv.DictReader(f):
            metrics[cfg][(r["method"], int(r["context"]), int(r["K"]))].append((float(r["ndcg"]), float(r["recall"])))
    if cfg == "full" and os.path.exists(os.path.join(d, "significance.csv")):
        with open(os.path.join(d, "significance.csv")) as f:
            for r in csv.DictReader(f):
                sig[(int(r["context"]), int(r["K"]), r["baseline"])].append(
                    tuple(float(r[k]) for k in ("reach_minus_baseline_ndcg", "ci95_low", "ci95_high", "p_value")))
    if os.path.exists(os.path.join(d, "config.json")):
        with open(os.path.join(d, "config.json")) as f:
            extra[cfg].append(json.load(f))
    if stats is None and os.path.exists(os.path.join(d, "dataset_stats.json")):
        with open(os.path.join(d, "dataset_stats.json")) as f:
            stats = json.load(f)

if not metrics:
    raise SystemExit(f"No runs found in {args.runs}")


def cell(vals, i):
    a = np.array([v[i] for v in vals])
    return f"{a.mean():.4f}" + (f" ± {a.std(ddof=1):.4f}" if len(a) > 1 else "")


lines = ["# REACH v2 results", ""]
if stats:
    lines += ["## Dataset (Table I)", "", "| Statistic | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in stats.items()] + [""]

main_cfg = "full" if "full" in metrics else sorted(metrics)[0]
n_seeds = len(extra.get(main_cfg, [])) or "?"
main_methods = ["GNN", "RB", "SEM", "WHO_FIXED", "STATIC_TUNED", "REACH"]
lines += [f"## Main results (Table III), config `{main_cfg}`, {n_seeds} seed(s), mean ± std", ""]
for metric_i, metric in ((0, "NDCG"), (1, "Recall")):
    lines += [f"### {metric}@K", "", "| C | K | " + " | ".join(main_methods) + " |",
              "|---|---|" + "---|" * len(main_methods)]
    for c in (0, 1, 2):
        for K in (5, 10, 20):
            row = [cell(metrics[main_cfg][(mm, c, K)], metric_i) if metrics[main_cfg][(mm, c, K)] else "–"
                   for mm in main_methods]
            lines.append(f"| {c} | {K} | " + " | ".join(row) + " |")
    lines.append("")

lines += ["## Ablations (Table IV), K = 100", "", "| Variant | C=0 NDCG | C=0 Recall | C=2 NDCG | C=2 Recall |", "|---|---|---|---|---|"]
abl = [("REACH", main_cfg, "REACH"), ("w/o GNN expert", main_cfg, "REACH_noGNN"),
       ("w/o SEM expert", main_cfg, "REACH_noSEM"), ("w/o RB expert", main_cfg, "REACH_noRB"),
       ("w/o fusion layer (tuned static)", main_cfg, "STATIC_TUNED"),
       ("w/o VIF", "noVIF", "REACH"), ("w/o TEW (λ = 0)", "noTEW", "REACH"), ("w/o TE (linear year)", "linearTE", "REACH")]
for label, cfg, meth in abl:
    if cfg not in metrics or not metrics[cfg][(meth, 0, 100)]:
        continue
    vals = [cell(metrics[cfg][(meth, c, 100)], i) for c in (0, 2) for i in (0, 1)]
    lines.append(f"| {label} | " + " | ".join(vals) + " |")
lines.append("")

if sig:
    lines += ["## REACH − baseline (NDCG), paired bootstrap per seed", "",
              "| C | K | Baseline | mean diff | 95% CI (per seed) | p (max over seeds) |", "|---|---|---|---|---|---|"]
    for (c, K, b), vs in sorted(sig.items()):
        diffs = np.array([v[0] for v in vs])
        cis = "; ".join(f"[{v[1]:+.4f}, {v[2]:+.4f}]" for v in vs)
        lines.append(f"| {c} | {K} | {b} | {diffs.mean():+.4f} | {cis} | {max(v[3] for v in vs):.4f} |")
    lines.append("")

lines += ["## Learned parameters", ""]
for cfg, cs in sorted(extra.items()):
    lams = [x["learned_lambda"] for x in cs]
    gates = [x["mean_reach_gate_weights_[GNN,SEM,RB]"] for x in cs]
    lines.append(f"- `{cfg}`: λ = {np.mean(lams):.4f}; mean REACH gate [GNN, SEM, RB] per C (seed 1): {gates[0]}")
    lines.append(f"  - tuned static weights (seed 1): {cs[0]['static_tuned_weights_[GNN,SEM,RB]']}")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {args.out}")
