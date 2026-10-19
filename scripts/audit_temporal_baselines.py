"""Audit static NCO stopping behavior and timestamp-null structural invariants."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from mdacn.data import load_interactions
from mdacn.dynamics import initialize_opinions, simulate_nco
from mdacn.temporal_analysis import filter_core_events
from mdacn.temporal_nulls import shuffle_timestamps, validate_aggregate_preservation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/soc-RedditHyperlinks-body.tsv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/processed/nodes.json"))
    parser.add_argument("--static", type=Path, default=Path("data/processed/signed.npz"))
    parser.add_argument("--output", type=Path, default=Path("reports/temporal_audit"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[round(x, 1) for x in np.arange(0.1, 1., 0.1)])
    parser.add_argument("--initializations", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    folder = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    nodes = tuple(json.loads(args.nodes.read_text()))
    events = filter_core_events(load_interactions(args.data), nodes)
    null = shuffle_timestamps(events, seed=args.seed)
    invariants = validate_aggregate_preservation(events, null, nodes)
    (folder / "shuffle_invariants.json").write_text(json.dumps(invariants, indent=2))
    signed = load_npz(args.static)
    if signed.shape != (len(nodes), len(nodes)):
        raise ValueError("Static signed matrix does not align with selected core")
    rng = np.random.default_rng(args.seed)
    records = []
    for f in args.fractions:
        for k in range(args.initializations):
            init_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            result = simulate_nco(signed, initialize_opinions(len(nodes), f, seed=init_seed), model="signed", max_steps=args.max_steps)
            records.append({"f": f, "initialization": k, "initial_seed": init_seed, "stop_reason": result.stop_reason, "steps": result.steps, "period": result.cycle_length, "final_negative_fraction": float(np.mean(result.final_state == -1))})
    table = pd.DataFrame(records)
    table.to_csv(folder / "static_stopping.csv", index=False)
    print("Timestamp shuffle preservation:", invariants)
    print(table.groupby(["f", "stop_reason"]).agg(runs=("steps", "size"), mean_steps=("steps", "mean")).to_string())
    print(f"Saved to {folder}")


if __name__ == "__main__":
    main()
