"""End-to-end smoke test: the full REACH v2 pipeline on a small synthetic dataset (CPU, ~1 minute)."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "experiments", "07_reach_final", "reach.py")


def _run(tmp_path, *extra):
    out = tmp_path / "runs"
    cmd = [sys.executable, SCRIPT, "--synthetic", "1500", "--epochs", "3", "--batch-size", "2048",
           "--fusion-papers", "40", "--fusion-epochs", "2", "--max-test-papers", "60", "--bootstrap", "200",
           "--hidden", "32", "--aux-dim", "16", "--seed", "0", "--tag", "ci",
           "--out-dir", str(out), "--cache-dir", str(tmp_path / "cache"), *extra]
    subprocess.run(cmd, check=True, cwd=ROOT, timeout=900)
    return out / "ci_seed0"


def test_full_pipeline_writes_valid_outputs(tmp_path):
    run = _run(tmp_path)
    for f in ("metrics.csv", "significance.csv", "config.json", "dataset_stats.json", "per_query_ndcg.npz"):
        assert (run / f).exists(), f

    rows = list(csv.DictReader(open(run / "metrics.csv")))
    methods = {r["method"] for r in rows}
    assert {"GNN", "RB", "SEM", "WHO_FIXED", "STATIC_TUNED", "REACH",
            "REACH_noGNN", "REACH_noSEM", "REACH_noRB"} <= methods
    for r in rows:
        assert 0.0 <= float(r["ndcg"]) <= 1.0 and 0.0 <= float(r["recall"]) <= 1.0
        assert int(r["n_queries"]) > 0

    cfg = json.load(open(run / "config.json"))
    assert cfg["learned_lambda"] > 0
    assert cfg["mean_reach_gate_weights_[GNN,SEM,RB]"]["0"][2] == 0.0   # no rule expert at cold start

    stats = json.load(open(run / "dataset_stats.json"))
    assert stats["train_papers"] > stats["val_papers"] > 0 and stats["test_papers"] > 0


def test_ablation_flags_run(tmp_path):
    run = _run(tmp_path, "--no-vif", "--no-tew", "--linear-time")
    cfg = json.load(open(run / "config.json"))
    assert cfg["learned_lambda"] == 0.0
