
"""Run paired static-versus-temporal signed NCO experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from mdacn.data import load_interactions
from mdacn.temporal_analysis import run_static_temporal_comparison, summarize_static_temporal
from mdacn.temporal_analysis import paired_bootstrap_summary


METRICS = ("absolute_magnetization", "minority_fraction", "minority_lcc_fraction", "total_switches", "disagreement_with_static")


def plot_comparison(summary: pd.DataFrame, output: Path) -> None:
    """Generate an outcome curve for each metric."""
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        subset = summary.loc[summary["metric"] == metric]

        for variant, group in subset.groupby("variant"):
            group = group.sort_values("f")
            ax.plot(group["f"], group["mean"], marker="o", label=variant)

        ax.set(xlabel="Initial negative-opinion fraction f", ylabel=metric.replace("_", " "))
        ax.set_title(f"Static vs temporal: {metric.replace('_', ' ')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=200)
        plt.close(fig)


def plot_paired_effects(effects: pd.DataFrame, output: Path) -> None:
    """Plot temporal-minus-static differences and bootstrap intervals."""
    for metric in METRICS:
        subset = effects.loc[effects["metric"] == metric].sort_values("f")
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = subset["f"].to_numpy(dtype=float)
        y = subset["difference"].to_numpy(dtype=float)
        low = subset["ci_low"].to_numpy(dtype=float)
        high = subset["ci_high"].to_numpy(dtype=float)

        ax.plot(x, y, marker="o", label="Temporal real - static")
        ax.fill_between(x, low, high, alpha=0.2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel="Initial negative-opinion fraction f", ylabel="Temporal - static")
        ax.set_title(f"Paired effect: {metric.replace('_', ' ')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"effect_{metric}.png", dpi=200)
        plt.close(fig)


def paired_static_effects(results: pd.DataFrame, *, n_bootstrap: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap paired temporal-minus-static effects over initializations."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    rng = np.random.default_rng(seed)
    records = []

    for f, group in results.groupby("f"):
        static = group.loc[group["variant"] == "static"].set_index("initialization")
        temporal = group.loc[group["variant"] == "temporal_real"].set_index("initialization")

        if static.index.has_duplicates or temporal.index.has_duplicates or not static.index.equals(temporal.index):
            raise ValueError("Static and temporal initializations are not uniquely aligned")
        if not static["initial_seed"].equals(temporal["initial_seed"]):
            raise ValueError("Static and temporal initial seeds differ")

        for metric in METRICS:
            differences = (temporal[metric] - static[metric]).to_numpy(dtype=float)
            n = len(differences)
            if n < 2:
                raise ValueError("At least two initializations are needed for uncertainty estimates")

            estimates = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                indices = rng.integers(0, n, n)
                estimates[b] = differences[indices].mean()

            records.append({
                "f": f, "metric": metric,
                "static_mean": float(static[metric].mean()),
                "temporal_mean": float(temporal[metric].mean()),
                "difference": float(differences.mean()),
                "ci_low": float(np.quantile(estimates, 0.025)),
                "ci_high": float(np.quantile(estimates, 0.975)),
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/soc-RedditHyperlinks-body.tsv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/processed/nodes.json"))
    parser.add_argument("--static", type=Path, default=Path("data/processed/signed.npz"))
    parser.add_argument("--output", type=Path, default=Path("reports/static_temporal"))
    parser.add_argument("--frequency", default="W-SUN")
    parser.add_argument("--fractions", type=float, nargs="+", default=[round(x, 1) for x in np.arange(0.1, 1.0, 0.1)])
    parser.add_argument("--initializations", type=int, default=20)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)

    config = {
        "data": str(args.data.resolve()), "nodes": str(args.nodes.resolve()),
        "static": str(args.static.resolve()), "frequency": args.frequency,
        "fractions": args.fractions, "initializations": args.initializations,
        "shuffles": args.shuffles, "bootstrap": args.bootstrap, "seed": args.seed,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))

    nodes = tuple(json.loads(args.nodes.read_text()))
    signed = load_npz(args.static)
    interactions = load_interactions(args.data)
    print(f"Using {len(nodes):,} nodes and static matrix {signed.shape}")

    results = run_static_temporal_comparison(
        interactions, nodes, signed, fractions=tuple(args.fractions),
        frequency=args.frequency, n_initializations=args.initializations,
        n_shuffles=args.shuffles, seed=args.seed,
    )
    results.to_csv(output / "raw_results.csv", index=False)

    summary = summarize_static_temporal(results)
    summary.to_csv(output / "summary.csv", index=False)
    plot_comparison(summary, output)

    effects = paired_static_effects(results, n_bootstrap=args.bootstrap, seed=args.seed)
    effects.to_csv(output / "paired_static_effects.csv", index=False)
    plot_paired_effects(effects, output)

    # Reuse Feature 6's two-level bootstrap for temporal-real versus shuffled temporal.
    bootstrap_input = results.loc[results["variant"].isin(("temporal_real", "temporal_shuffled"))].copy()
    bootstrap_input["variant"] = bootstrap_input["variant"].replace({"temporal_real": "real", "temporal_shuffled": "global"})
    shuffled_effects = paired_bootstrap_summary(bootstrap_input, metrics=METRICS, n_bootstrap=args.bootstrap, seed=args.seed)
    shuffled_effects.to_csv(output / "paired_shuffle_effects.csv", index=False)

    print("\nStatic versus real temporal:")
    print(effects.to_string(index=False))

    print("\nTemporal real versus shuffled:")
    print(shuffled_effects.to_string(index=False))

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
