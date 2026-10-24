"""Run final temporal burstiness, correlation and dyad-sign diagnostics."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdacn.data import load_interactions
from mdacn.temporal_analysis import filter_core_events
from mdacn.temporal_diagnostics import (
    activity_null_experiment,
    correlation_null_experiment,
    dyad_sign_null_experiment,
    node_activity_metrics,
    signed_reciprocity_timestamp_nulls,
    summarize_dyad_sign_null,
)


def null_summary(results: pd.DataFrame, metric: str, *, value_column: str = "mean") -> pd.DataFrame:
    """Summarize a scalar observed statistic against randomized realizations."""
    observed = results.loc[results["variant"] == "real", value_column]
    if len(observed) != 1 or not np.isfinite(observed.iloc[0]):
        return pd.DataFrame()

    records = []
    value = float(observed.iloc[0])

    for variant, group in results.loc[results["variant"] != "real"].groupby("variant"):
        values = group[value_column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue

        lower = (1 + np.count_nonzero(values <= value)) / (len(values) + 1)
        upper = (1 + np.count_nonzero(values >= value)) / (len(values) + 1)

        records.append(
            {
                "metric": metric,
                "variant": variant,
                "observed": value,
                "null_mean": float(values.mean()),
                "difference": value - float(values.mean()),
                "null_low": float(np.quantile(values, 0.025)),
                "null_high": float(np.quantile(values, 0.975)),
                "p_two_sided": min(1.0, 2 * min(lower, upper)),
            }
        )

    return pd.DataFrame(records)


def plot_null_summary(summary: pd.DataFrame, output: Path, title: str) -> None:
    """Plot observed and null means for each metric."""
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(summary))
    labels = (summary["metric"] + " / " + summary["variant"]).tolist()

    ax.bar(positions - 0.18, summary["observed"], width=0.36, label="Real")
    ax.bar(positions + 0.18, summary["null_mean"], width=0.36, label="Null mean")
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set(ylabel="Statistic", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/soc-RedditHyperlinks-body.tsv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/processed/nodes.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/temporal_diagnostics"))
    parser.add_argument("--frequency", default="W-SUN")
    parser.add_argument("--shuffles", type=int, default=30)
    parser.add_argument("--sign-shuffles", type=int, default=100)
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "nodes": str(args.nodes.resolve()),
                "frequency": args.frequency,
                "shuffles": args.shuffles,
                "sign_shuffles": args.sign_shuffles,
                "max_pairs": args.max_pairs,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    interactions = load_interactions(args.data)
    nodes = tuple(json.loads(args.nodes.read_text()))
    events = filter_core_events(interactions, nodes)
    print(f"Using {len(events):,} core interactions and {len(nodes):,} nodes")

    node_activity_metrics(events, nodes, frequency=args.frequency).to_csv(
        output / "node_activity_features.csv", index=False
    )

    print("\n1. Source-level burstiness...")
    activity = activity_null_experiment(
        events, nodes, frequency=args.frequency, n_shuffles=args.shuffles, seed=args.seed
    )
    activity.to_csv(output / "activity_null.csv", index=False)

    activity_summaries = []
    for metric, group in activity.groupby("metric"):
        summary = null_summary(group, metric)
        if not summary.empty:
            activity_summaries.append(summary)

    activity_summary = (
        pd.concat(activity_summaries, ignore_index=True) if activity_summaries else pd.DataFrame()
    )
    activity_summary.to_csv(output / "activity_summary.csv", index=False)
    plot_null_summary(
        activity_summary, output / "activity_comparison.png", "Source-level temporal activity"
    )

    print("\n2. Incident-edge activity correlations...")
    correlations, sampled_pairs = correlation_null_experiment(
        events,
        nodes,
        frequency=args.frequency,
        n_shuffles=args.shuffles,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    correlations.to_csv(output / "incident_correlations.csv", index=False)
    sampled_pairs.to_csv(output / "sampled_incident_pairs.csv", index=False)

    correlation_summary = null_summary(
        correlations, "incident_edge_correlation", value_column="mean_correlation"
    )
    correlation_summary.to_csv(output / "correlation_summary.csv", index=False)
    plot_null_summary(
        correlation_summary,
        output / "correlation_comparison.png",
        "Incident-edge activity correlation",
    )

    print("\n3. Within-dyad sentiment shuffling...")
    transitions = dyad_sign_null_experiment(
        events, nodes, frequency=args.frequency, n_shuffles=args.sign_shuffles, seed=args.seed
    )
    transitions.to_csv(output / "dyad_sign_null.csv", index=False)
    sign_summary = summarize_dyad_sign_null(transitions)
    sign_summary.to_csv(output / "dyad_sign_summary.csv", index=False)

    print("\n4. Seven-day directed negative and non-negative reciprocity...")
    reciprocity = signed_reciprocity_timestamp_nulls(
        events, days=7, n_shuffles=args.shuffles, seed=args.seed
    )
    reciprocity.to_csv(output / "signed_reciprocity_null.csv", index=False)
    sign_summaries = []
    for sign, group in reciprocity.groupby("sign", sort=False):
        summary = null_summary(group, f"seven_day_{sign}_reciprocity", value_column="fraction")
        if not summary.empty:
            summary.insert(1, "sign", sign)
            summary["fold_vs_null"] = np.where(
                summary["null_mean"] > 0, summary["observed"] / summary["null_mean"], np.nan
            )
            sign_summaries.append(summary)
    signed_summary = (
        pd.concat(sign_summaries, ignore_index=True) if sign_summaries else pd.DataFrame()
    )
    signed_summary.to_csv(output / "signed_reciprocity_summary.csv", index=False)
    negative = (
        reciprocity.loc[reciprocity["sign"] == "negative"]
        .rename(columns={"eligible_events": "negative_events"})
        .drop(columns="sign")
    )
    negative.to_csv(output / "negative_reciprocity_null.csv", index=False)
    negative_summary = signed_summary.loc[signed_summary["sign"] == "negative"].drop(
        columns=["sign", "fold_vs_null"]
    )
    negative_summary.to_csv(output / "negative_reciprocity_summary.csv", index=False)
    print(signed_summary.to_string(index=False))
    print("Note: +1 means non-negative (including neutral), not exclusively positive sentiment.")

    print("\nActivity:")
    print(activity_summary.to_string(index=False))
    print("\nIncident-edge correlations:")
    print(correlation_summary.to_string(index=False))
    print("\nDyad-conditioned sign persistence:")
    print(
        sign_summary.loc[sign_summary["previous"] == sign_summary["current"]].to_string(index=False)
    )
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
