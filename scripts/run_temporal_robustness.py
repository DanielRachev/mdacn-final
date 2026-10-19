
"""Run temporal opinion robustness and sign-persistence experiments."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mdacn.data import load_interactions
from mdacn.temporal_analysis import run_fraction_sweep, paired_fraction_summary, sign_persistence_null, sign_persistence_summary


def plot_fraction_sweep(results, output):
    metrics = ("absolute_magnetization", "minority_fraction", "largest_opinion_gcc_fraction", "total_switches")

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        grouped = results.groupby(["f", "variant"])[metric].agg(["mean", "std", "count"]).reset_index()

        for variant, group in grouped.groupby("variant"):
            group = group.sort_values("f")
            ax.plot(group["f"], group["mean"], marker="o", label=variant)
            if variant == "real":
                sem = group["std"].fillna(0) / np.sqrt(group["count"])
                ax.fill_between(group["f"].to_numpy(), (group["mean"] - 1.96 * sem).to_numpy(), (group["mean"] + 1.96 * sem).to_numpy(), alpha=0.15)

        ax.set(xlabel="Initial negative-opinion fraction f", ylabel=metric.replace("_", " "), title=f"Temporal NCO: {metric.replace('_', ' ')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/soc-RedditHyperlinks-body.tsv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/processed/nodes.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/temporal_robustness"))
    parser.add_argument("--frequency", default="W-SUN")
    parser.add_argument("--initializations", type=int, default=20)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--sign-shuffles", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs="+", default=[round(x, 2) for x in np.arange(0.1, 1.0, 0.1)])
    parser.add_argument("--modes", nargs="+", choices=("global", "within_source"), default=["global", "within_source"])
    args = parser.parse_args()

    config = {
        "data": str(args.data.resolve()), "nodes": str(args.nodes.resolve()),
        "frequency": args.frequency, "initializations": args.initializations,
        "shuffles": args.shuffles, "sign_shuffles": args.sign_shuffles,
        "seed": args.seed, "fractions": args.fractions, "modes": args.modes,
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = args.output / f"{args.frequency.replace('-', '_')}_{digest}"
    output.mkdir(parents=True, exist_ok=False)

    config["created_utc"] = datetime.now(timezone.utc).isoformat()
    with args.data.open("rb") as file:
        config["data_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2))

    interactions = load_interactions(args.data)
    nodes = tuple(json.loads(args.nodes.read_text()))
    print(f"Loaded {len(interactions):,} original events; using {len(nodes):,} core nodes")

    results = run_fraction_sweep(
        interactions, nodes, fractions=tuple(args.fractions), frequency=args.frequency,
        n_initializations=args.initializations, n_shuffles=args.shuffles,
        seed=args.seed, modes=tuple(args.modes),
    )
    results.to_csv(output / "fraction_sweep.csv", index=False)

    summary = paired_fraction_summary(results)
    summary.to_csv(output / "fraction_sweep_summary.csv", index=False)
    plot_fraction_sweep(results, output)

    transitions = sign_persistence_null(interactions, nodes, frequency=args.frequency, n_shuffles=args.sign_shuffles, seed=args.seed)
    transitions.to_csv(output / "sign_persistence_null.csv", index=False)

    transition_summary = sign_persistence_summary(transitions)
    transition_summary.to_csv(output / "sign_persistence_summary.csv", index=False)

    print("\nReal-versus-null effects:")
    print(summary.to_string(index=False))

    print("\nSign persistence:")
    print(transition_summary.to_string(index=False))
    print(f"\nSaved outputs to: {output}")


if __name__ == "__main__":
    main()
