
"""Timestamp-shuffling null models and paired temporal NCO experiments."""

from collections import Counter
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from mdacn.temporal_data import build_temporal_snapshots
from mdacn.temporal_dynamics import simulate_temporal_nco


ShuffleMode = Literal["global", "within_source"]


def shuffle_timestamps(interactions: pd.DataFrame, *, mode: ShuffleMode = "global", seed: int = 0) -> pd.DataFrame:
    """Reassign timestamps without changing event endpoints or signs."""
    if mode not in ("global", "within_source"):
        raise ValueError("mode must be 'global' or 'within_source'")

    required = {"source", "target", "sentiment", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(interactions.columns))}")

    result = interactions.copy(deep=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")

    if result["timestamp"].isna().any():
        raise ValueError("Timestamps cannot contain missing values")
    if not result["sentiment"].isin((-1, 1)).all():
        raise ValueError("Sentiment must be +1 or -1")

    rng = np.random.default_rng(seed)
    timestamps = result["timestamp"].to_numpy(copy=True)

    if mode == "global":
        shuffled = rng.permutation(timestamps)
    else:
        shuffled = timestamps.copy()
        groups = result.groupby("source", sort=True, dropna=False).indices
        for indices in groups.values():
            shuffled[indices] = rng.permutation(timestamps[indices])

    result["timestamp"] = shuffled
    return result


def validate_timestamp_shuffle(original: pd.DataFrame, shuffled: pd.DataFrame, *, mode: ShuffleMode = "global") -> None:
    """Raise AssertionError if the null model violates its preservation rules."""
    if mode not in ("global", "within_source"):
        raise ValueError("Invalid shuffle mode")

    if len(original) != len(shuffled):
        raise AssertionError("Event count changed")

    for column in ("source", "target", "sentiment"):
        if not original[column].reset_index(drop=True).equals(shuffled[column].reset_index(drop=True)):
            raise AssertionError(f"{column} changed")

    original_times = pd.to_datetime(original["timestamp"])
    shuffled_times = pd.to_datetime(shuffled["timestamp"])

    if Counter(original_times) != Counter(shuffled_times):
        raise AssertionError("Global timestamp distribution changed")

    if mode == "within_source":
        for source in original["source"].unique():
            before = original_times[original["source"] == source]
            after = shuffled_times[shuffled["source"] == source]
            if Counter(before) != Counter(after):
                raise AssertionError(f"Timestamp distribution changed for source {source}")



def validate_aggregate_preservation(original: pd.DataFrame, shuffled: pd.DataFrame, nodes: tuple[str, ...], *, mode: ShuffleMode = "global") -> dict:
    """Explicitly verify full lifetime unsigned/signed graphs and node-degree vectors."""
    from mdacn.data import aggregate_edges
    from mdacn.temporal_data import build_temporal_snapshots

    validate_timestamp_shuffle(original, shuffled, mode=mode)
    before = aggregate_edges(original)
    after = aggregate_edges(shuffled)
    if set(before.nodes) != set(after.nodes):
        raise AssertionError("Node set changed")
    if dict(before.degree()) != dict(after.degree()):
        raise AssertionError("Lifetime degree sequence changed")
    if set(before.edges()) != set(after.edges()):
        raise AssertionError("Lifetime edge set changed")
    for u, v, attributes in before.edges(data=True):
        if any(attributes[key] != after[u][v][key] for key in ("sign", "n_positive", "n_negative")):
            raise AssertionError("Lifetime signed edge or event totals changed")
    # Explicitly check global per-window counts; weekly node degrees are NOT invariant.
    first = build_temporal_snapshots(original, nodes)
    second = build_temporal_snapshots(shuffled, nodes)
    if first.timestamps != second.timestamps or first.event_counts != second.event_counts:
        raise AssertionError("Global weekly timestamp histogram changed")
    return {"nodes": len(before), "edges": before.number_of_edges(), "degree_preserved": True, "signed_edges_preserved": True, "weekly_global_counts_preserved": True}


def generate_timestamp_nulls(interactions: pd.DataFrame, *, mode: ShuffleMode = "global", n_shuffles: int = 30,
                             seed: int = 0) -> Iterator[pd.DataFrame]:
    """Yield reproducible, independently seeded timestamp permutations."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be positive")
    if mode not in ("global", "within_source"):
        raise ValueError("Invalid shuffle mode")

    rng = np.random.default_rng(seed)
    for _ in range(n_shuffles):
        shuffle_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        yield shuffle_timestamps(interactions, mode=mode, seed=shuffle_seed)


def evaluate_timestamp_nulls(interactions: pd.DataFrame, nodes: tuple[str, ...], *,
                             frequency: str = "W-SUN", tie_policy: str = "zero", model: str = "signed",
                             negative_fraction: float = 0.5, n_initializations: int = 20, n_shuffles: int = 30,
                             seed: int = 0, mode: ShuffleMode = "global") -> pd.DataFrame:
    """Compare real timestamps against nulls with paired opinion initializations."""
    if n_initializations < 1 or n_shuffles < 1:
        raise ValueError("n_initializations and n_shuffles must be positive")
    if not 0 <= negative_fraction <= 1:
        raise ValueError("negative_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    real_snapshots = build_temporal_snapshots(interactions, nodes, frequency=frequency, tie_policy=tie_policy)
    shuffled_snapshots = []

    for shuffled in generate_timestamp_nulls(interactions, mode=mode, n_shuffles=n_shuffles, seed=seed):
        shuffled_snapshots.append(build_temporal_snapshots(shuffled, nodes, frequency=frequency, tie_policy=tie_policy))

    records = []

    for initialization in range(n_initializations):
        initial_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        real = simulate_temporal_nco(real_snapshots, model=model, negative_fraction=negative_fraction, seed=initial_seed)
        initial_state = real.initial_state.copy()

        records.append({
            "initialization": initialization,
            "shuffle": -1,
            "variant": "real",
            "final_positive_fraction": float(real.positive_fraction[-1]),
            "final_negative_fraction": float(real.negative_fraction[-1]),
            "total_switches": int(real.changed_count.sum()),
            "final_week_switches": int(real.changed_count[-1]) if real.steps else 0,
        })

        for shuffle_index, snapshots in enumerate(shuffled_snapshots):
            result = simulate_temporal_nco(snapshots, model=model, initial_state=initial_state)
            records.append({
                "initialization": initialization,
                "shuffle": shuffle_index,
                "variant": "shuffled",
                "final_positive_fraction": float(result.positive_fraction[-1]),
                "final_negative_fraction": float(result.negative_fraction[-1]),
                "total_switches": int(result.changed_count.sum()),
                "final_week_switches": int(result.changed_count[-1]) if result.steps else 0,
            })

    return pd.DataFrame.from_records(records)
