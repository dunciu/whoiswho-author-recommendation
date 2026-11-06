"""
Run REACH v2 over several seeds for the full model and each ablation, then aggregate.

    python experiments/07_reach_final/run_experiments.py                 # 3 seeds x 4 configs
    python experiments/07_reach_final/run_experiments.py --seeds 0 1 2 3 4 --configs full noTEW
    python experiments/07_reach_final/run_experiments.py --extra --max-papers 50000   # pass flags through

Each run writes results/runs/<config>_seed<N>/. Embeddings are cached in cache/, so only the
first run pays the SPECTER2 encoding cost.
"""
import argparse
import subprocess
import sys

CONFIGS = {
    "full": [],
    "noVIF": ["--no-vif"],
    "noTEW": ["--no-tew"],
    "linearTE": ["--linear-time"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra flags passed to reach.py")
args = ap.parse_args()

for cfg in args.configs:
    for seed in args.seeds:
        cmd = [sys.executable, "experiments/07_reach_final/reach.py", "--seed", str(seed),
               "--tag", cfg, *CONFIGS[cfg], *args.extra]
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

subprocess.run([sys.executable, "experiments/07_reach_final/aggregate_runs.py"], check=True)
