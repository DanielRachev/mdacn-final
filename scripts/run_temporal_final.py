"""Generate final temporal robustness results and presentation figures."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdacn.data import load_interactions
from mdacn.temporal_analysis import (
    compare_sign_persistence,
    paired_bootstrap_summary,
    run_minority_experiment,
    summarize_sign_persistence,
)


def plot_effects(summary: pd.DataFrame, output: Path) -> None:
    """Plot real-minus-null effects with bootstrap confidence intervals."""
    for metric in summary["metric"].unique():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        subset = summary.loc[summary["metric"] == metric]

        for variant, group in subset.groupby("variant"):
            group = group.sort_values("f")
            x = group["f"].to_numpy(dtype=float)
            y = group["difference"].to_numpy(dtype=float)
            low = group["ci_low"].to_numpy(dtype=float)
            high = group["ci_high"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", label=variant)
            ax.fill_between(x, low, high, alpha=0.15)

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel="Initial negative-opinion fraction f", ylabel="Real minus shuffled")
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"effect_{metric}.png", dpi=200)
        plt.close(fig)


def plot_sign_persistence(summary: pd.DataFrame, output: Path) -> None:
    """Compare observed sign persistence with both randomized baselines."""
    same = summary.loc[summary["previous"] == summary["current"]].copy()
    signs = ("+", "-", "mixed")
    positions = np.arange(len(signs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))

    observed = same.loc[same["variant"] == "global"].set_index("previous").reindex(signs)
    ax.bar(positions - width, observed["observed"], width, label="Real")

    for offset, variant in ((0, "global"), (width, "within_source")):
        data = same.loc[same["variant"] == variant].set_index("previous").reindex(signs)
        ax.bar(positions + offset, data["null_mean"], width, label=variant)

    ax.set_xticks(positions, ("Positive", "Negative", "Mixed"))
    ax.set(xlabel="Previous sign", ylabel="Probability of retaining sign", ylim=(0, 1))
    ax.set_title("Weekly sign persistence: observed versus randomized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "sign_persistence_comparison.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/soc-RedditHyperlinks-body.tsv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/processed/nodes.json"))
    parser.add_argument("--previous", type=Path, required=True, help="Existing fraction_sweep.csv")
    parser.add_argument("--output", type=Path, default=Path("reports/temporal_final"))
    parser.add_argument("--frequency", default="W-SUN")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--initializations", type=int, default=20)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--sign-shuffles", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)

    config = {
        "data": str(args.data.resolve()),
        "nodes": str(args.nodes.resolve()),
        "previous": str(args.previous.resolve()),
        "frequency": args.frequency,
        "fractions": args.fractions,
        "initializations": args.initializations,
        "shuffles": args.shuffles,
        "sign_shuffles": args.sign_shuffles,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))

    interactions = load_interactions(args.data)
    nodes = tuple(json.loads(args.nodes.read_text()))

    print(f"Using {len(nodes):,} nodes; saving to {output}")

    print("\n1. Calculating bootstrap confidence intervals...")
    previous = pd.read_csv(args.previous)
    previous_metrics = (
        "absolute_magnetization",
        "minority_fraction",
        "largest_opinion_gcc_fraction",
        "total_switches",
    )
    effects = paired_bootstrap_summary(
        previous, metrics=previous_metrics, n_bootstrap=args.bootstrap, seed=args.seed
    )
    effects.to_csv(output / "opinion_effects.csv", index=False)
    plot_effects(effects, output)

    print("\n2. Evaluating minority fragmentation...")
    minority = run_minority_experiment(
        interactions,
        nodes,
        fractions=tuple(args.fractions),
        frequency=args.frequency,
        n_initializations=args.initializations,
        n_shuffles=args.shuffles,
        seed=args.seed,
    )
    minority.to_csv(output / "minority_fragmentation.csv", index=False)

    minority_metrics = (
        "minority_lcc_fraction",
        "minority_component_count",
        "minority_isolated_fraction",
    )
    minority_effects = paired_bootstrap_summary(
        minority, metrics=minority_metrics, n_bootstrap=args.bootstrap, seed=args.seed
    )
    minority_effects.to_csv(output / "minority_effects.csv", index=False)
    plot_effects(minority_effects, output)

    print("\n3. Evaluating source-conditioned sign persistence...")
    transitions = compare_sign_persistence(
        interactions,
        nodes,
        frequency=args.frequency,
        n_shuffles=args.sign_shuffles,
        seed=args.seed,
    )
    transitions.to_csv(output / "sign_null_raw.csv", index=False)

    sign_summary = summarize_sign_persistence(transitions)
    sign_summary.to_csv(output / "sign_null_summary.csv", index=False)
    plot_sign_persistence(sign_summary, output)

    print("\nMinority fragmentation effects:")
    print(minority_effects.to_string(index=False))

    print("\nSign persistence results:")
    print(
        sign_summary.loc[sign_summary["previous"] == sign_summary["current"]].to_string(index=False)
    )

    print(f"\nAll results saved to {output}")


if __name__ == "__main__":
    main()
