"""Run Person-5 temporal characterization and null-model experiments."""

import argparse
import json
from pathlib import Path

from mdacn.data import load_interactions
from mdacn.temporal_analysis import (
    compare_temporal_trajectories,
    filter_core_events,
    plot_resolution_ablation,
    plot_temporal_activity,
    plot_temporal_trajectories,
    resolution_ablation,
    sign_transitions,
    temporal_activity,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/soc-RedditHyperlinks-body.tsv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/processed/nodes.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/temporal"))
    parser.add_argument("--frequency", default="W-SUN")
    parser.add_argument("--negative-fraction", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-mode", choices=("global", "within_source"), default="global")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    interactions = load_interactions(args.data)
    nodes = tuple(json.loads(args.nodes.read_text()))
    events = filter_core_events(interactions, nodes)

    print(f"Loaded {len(events):,} core interactions among {len(nodes):,} nodes")

    activity = temporal_activity(events, nodes, frequency=args.frequency)
    transitions = sign_transitions(events, nodes, frequency=args.frequency)
    activity.to_csv(args.output / "activity.csv", index=False)
    transitions.to_csv(args.output / "sign_transitions.csv", index=False)

    plot_temporal_activity(activity, args.output / "activity.png")

    trajectories = compare_temporal_trajectories(
        events,
        nodes,
        frequency=args.frequency,
        negative_fraction=args.negative_fraction,
        n_repetitions=args.repetitions,
        seed=args.seed,
        shuffle_mode=args.shuffle_mode,
    )
    trajectories.to_csv(args.output / "trajectories.csv", index=False)
    plot_temporal_trajectories(trajectories, args.output / "trajectories.png")

    ablation = resolution_ablation(
        events,
        nodes,
        frequencies=("W-SUN", "2W-SUN", "M"),
        negative_fraction=args.negative_fraction,
        n_initializations=args.repetitions,
        n_shuffles=args.shuffles,
        seed=args.seed,
        shuffle_mode=args.shuffle_mode,
    )
    ablation.to_csv(args.output / "resolution_ablation.csv", index=False)
    plot_resolution_ablation(ablation, args.output / "resolution_ablation.png")

    print("\nTemporal sign transitions:")
    print(transitions.to_string(index=False))

    print("\nResolution sensitivity:")
    print(ablation.to_string(index=False))

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
