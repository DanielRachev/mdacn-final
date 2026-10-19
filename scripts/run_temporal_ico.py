"""Evaluate temporal ICO using the team's existing targeting scores where available."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from mdacn.data import load_interactions
from mdacn.temporal_analysis import filter_core_events
from mdacn.temporal_interventions import evaluate_temporal_ico
from mdacn.temporal_interventions import summarize_temporal_ico


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched temporal ICO evaluation")
    parser.add_argument("--data", type=Path, default=Path("data/raw/soc-RedditHyperlinks-body.tsv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/processed/nodes.json"))
    parser.add_argument("--unsigned", type=Path, default=Path("data/processed/unsigned.npz"))
    parser.add_argument("--signed-leverage", type=Path, help="Teammate's CSV with node,signed_leverage columns")
    parser.add_argument("--output", type=Path, default=Path("reports/temporal_ico"))
    parser.add_argument("--frequency", default="W-SUN")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25])
    parser.add_argument("--initializations", type=int, default=20)
    parser.add_argument("--shuffles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    folder = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    nodes = tuple(json.loads(args.nodes.read_text()))
    adjacency = load_npz(args.unsigned)
    if adjacency.shape != (len(nodes), len(nodes)):
        raise ValueError("Unsigned matrix node order/size mismatch")
    degree_scores = dict(zip(nodes, np.asarray(adjacency.getnnz(axis=1), dtype=float), strict=True))
    signed_scores = None
    if args.signed_leverage is not None:
        table = pd.read_csv(args.signed_leverage)
        if not {"node", "signed_leverage"}.issubset(table.columns) or table["node"].duplicated().any():
            raise ValueError("Signed leverage CSV needs unique node,signed_leverage columns")
        signed_scores = dict(zip(table["node"].astype(str), table["signed_leverage"].astype(float), strict=True))
        if not set(nodes).issubset(signed_scores):
            raise ValueError("Signed leverage CSV is missing core nodes")
    else:
        print("No signed-leverage CSV supplied: evaluating random and degree only")
    events = filter_core_events(load_interactions(args.data), nodes)
    results = evaluate_temporal_ico(events, nodes, degree_scores=degree_scores, signed_leverage_scores=signed_scores, budgets=tuple(args.budgets), fractions=tuple(args.fractions), n_initializations=args.initializations, n_shuffles=args.shuffles, frequency=args.frequency, seed=args.seed)
    results.to_csv(folder / "raw_results.csv", index=False)
    summary = results.groupby(["f", "budget", "strategy", "variant"], as_index=False).agg(mean_g_minus=("g_minus", "mean"), sd_g_minus=("g_minus", "std"), mean_negative_fraction=("negative_fraction", "mean"), mean_switches=("total_switches", "mean"), runs=("g_minus", "count"))
    summary.to_csv(folder / "summary.csv", index=False)
    outcomes, differences, crossings = summarize_temporal_ico(results, n_bootstrap=args.bootstrap, seed=args.seed)
    outcomes.to_csv(folder / "outcomes_and_success.csv", index=False)
    differences.to_csv(folder / "paired_effects.csv", index=False)
    crossings.to_csv(folder / "per_run_phi_c.csv", index=False)
    print("\nPaired effects at f=0.7, Phi=0.05:")
    print(differences.loc[(differences.f == 0.7) & (np.isclose(differences.budget, 0.05))].to_string(index=False))
    print(summary.to_string(index=False))
    print(f"\nSaved to {folder}")


if __name__ == "__main__":
    main()
