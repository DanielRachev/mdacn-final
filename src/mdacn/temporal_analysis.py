"""Temporal network characterization, NCO comparisons, and statistical analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import t

from mdacn.dynamics import _prepare_matrix, _prepare_state, _step, initialize_opinions
from mdacn.temporal_data import build_temporal_snapshots, temporal_windows
from mdacn.temporal_dynamics import simulate_temporal_nco
from mdacn.temporal_nulls import evaluate_timestamp_nulls, shuffle_timestamps


def filter_core_events(interactions: pd.DataFrame, nodes: tuple[str, ...]) -> pd.DataFrame:
    """Retain valid interactions between distinct nodes in the selected core."""
    required = {"source", "target", "sentiment", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(interactions.columns))}")

    events = interactions.loc[
        interactions["source"].isin(nodes)
        & interactions["target"].isin(nodes)
        & interactions["source"].ne(interactions["target"])
    ].copy()

    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="raise")
    if events["timestamp"].isna().any() or not events["sentiment"].isin((-1, 1)).all():
        raise ValueError("Invalid timestamps or sentiment labels")

    return events


def temporal_activity(
    interactions: pd.DataFrame, nodes: tuple[str, ...], *, frequency: str = "W-SUN"
) -> pd.DataFrame:
    """Summarize event volume and sentiment composition for every time window."""
    events = filter_core_events(interactions, nodes)
    columns = ["timestamp", "total", "positive", "negative", "negative_fraction", "active_nodes"]

    if events.empty:
        return pd.DataFrame(columns=columns)

    events["period"], periods = temporal_windows(events["timestamp"], frequency)
    records = []

    for period in periods:
        window = events.loc[events["period"] == period]
        positive = int((window["sentiment"] == 1).sum())
        negative = int((window["sentiment"] == -1).sum())
        active = len(set(window["source"]).union(window["target"]))

        records.append(
            {
                "timestamp": period.start_time,
                "total": len(window),
                "positive": positive,
                "negative": negative,
                "negative_fraction": negative / len(window) if len(window) else np.nan,
                "active_nodes": active,
            }
        )

    return pd.DataFrame(records)


def sign_transitions(
    interactions: pd.DataFrame, nodes: tuple[str, ...], *, frequency: str = "W-SUN"
) -> pd.DataFrame:
    """Count sign transitions for undirected dyads active in consecutive windows."""
    events = filter_core_events(interactions, nodes)
    columns = ["previous", "current", "count", "probability"]

    if events.empty:
        return pd.DataFrame(columns=columns)

    index = {node: i for i, node in enumerate(nodes)}
    source = events["source"].map(index).to_numpy()
    target = events["target"].map(index).to_numpy()

    events["u"] = np.minimum(source, target)
    events["v"] = np.maximum(source, target)
    events["period"], calendar = temporal_windows(events["timestamp"], frequency)
    events["positive"] = (events["sentiment"] == 1).astype(int)
    events["negative"] = (events["sentiment"] == -1).astype(int)

    grouped = (
        events.groupby(["u", "v", "period"], observed=True)[["positive", "negative"]]
        .sum()
        .reset_index()
    )
    difference = grouped["positive"] - grouped["negative"]
    grouped["sign"] = np.select([difference > 0, difference < 0], ["+", "-"], default="mixed")
    grouped["period_index"] = calendar.get_indexer(grouped["period"])
    grouped = grouped.sort_values(["u", "v", "period_index"])

    pairs = grouped.groupby(["u", "v"], sort=False)
    grouped["previous"] = pairs["sign"].shift()
    grouped["previous_period"] = pairs["period_index"].shift()

    consecutive = grouped.loc[grouped["period_index"] - grouped["previous_period"] == 1]
    counts = consecutive.groupby(["previous", "sign"]).size().reset_index(name="count")
    counts = counts.rename(columns={"sign": "current"})
    counts["probability"] = counts["count"] / counts.groupby("previous")["count"].transform("sum")

    return counts[columns]


def compare_temporal_trajectories(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    *,
    frequency: str = "W-SUN",
    model: str = "signed",
    negative_fraction: float = 0.5,
    n_repetitions: int = 20,
    seed: int = 0,
    shuffle_mode: str = "global",
) -> pd.DataFrame:
    """Compare real and shuffled trajectories using paired opinion initializations."""
    if n_repetitions < 1:
        raise ValueError("n_repetitions must be positive")

    events = filter_core_events(interactions, nodes)
    real_snapshots = build_temporal_snapshots(events, nodes, frequency=frequency)
    rng = np.random.default_rng(seed)
    records = []

    for repetition in range(n_repetitions):
        initial_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        shuffle_seed = int(rng.integers(0, np.iinfo(np.int64).max))

        real = simulate_temporal_nco(
            real_snapshots, model=model, negative_fraction=negative_fraction, seed=initial_seed
        )
        shuffled_events = shuffle_timestamps(events, mode=shuffle_mode, seed=shuffle_seed)
        shuffled_snapshots = build_temporal_snapshots(shuffled_events, nodes, frequency=frequency)
        shuffled = simulate_temporal_nco(
            shuffled_snapshots, model=model, initial_state=real.initial_state
        )

        if real.steps != shuffled.steps:
            raise ValueError("Real and shuffled networks have different simulation horizons")

        for step in range(real.steps + 1):
            records.append(
                {
                    "repetition": repetition,
                    "step": step,
                    "variant": "real",
                    "positive_fraction": real.positive_fraction[step],
                    "minority_fraction": float(
                        min(real.positive_fraction[step], real.negative_fraction[step])
                    ),
                    "initial_minority_fraction": (
                        float(
                            real.negative_fraction[step]
                            if negative_fraction < 0.5
                            else real.positive_fraction[step]
                        )
                        if negative_fraction != 0.5
                        else float("nan")
                    ),
                    "switches": int(real.changed_count[step - 1]) if step else 0,
                }
            )
            records.append(
                {
                    "repetition": repetition,
                    "step": step,
                    "variant": "shuffled",
                    "positive_fraction": shuffled.positive_fraction[step],
                    "minority_fraction": float(
                        min(shuffled.positive_fraction[step], shuffled.negative_fraction[step])
                    ),
                    "initial_minority_fraction": (
                        float(
                            shuffled.negative_fraction[step]
                            if negative_fraction < 0.5
                            else shuffled.positive_fraction[step]
                        )
                        if negative_fraction != 0.5
                        else float("nan")
                    ),
                    "switches": int(shuffled.changed_count[step - 1]) if step else 0,
                }
            )

    return pd.DataFrame(records)


def paired_null_summary(
    results: pd.DataFrame, *, metric: str = "final_positive_fraction", confidence: float = 0.95
) -> dict:
    """Compare real runs with the mean shuffled result for each initialization."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if metric not in results.columns:
        raise ValueError(f"Missing metric: {metric}")

    real = results.loc[results["variant"] == "real"].set_index("initialization")[metric]
    shuffled = (
        results.loc[results["variant"] == "shuffled"].groupby("initialization")[metric].mean()
    )

    if real.index.has_duplicates or not real.index.equals(shuffled.index):
        real, shuffled = real.align(shuffled, join="inner")
        if len(real) == 0 or real.index.has_duplicates:
            raise ValueError("Invalid or unmatched initialization identifiers")

    differences = (real - shuffled).to_numpy(dtype=float)
    n = len(differences)

    if n < 2:
        raise ValueError("At least two paired initializations are required")

    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / np.sqrt(n))
    margin = float(t.ppf((1 + confidence) / 2, df=n - 1) * standard_error)

    return {
        "metric": metric,
        "n_initializations": n,
        "real_mean": float(real.mean()),
        "shuffled_mean": float(shuffled.mean()),
        "difference": mean,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def resolution_ablation(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    *,
    frequencies: tuple[str, ...] = ("W-SUN", "2W-SUN", "M"),
    negative_fraction: float = 0.5,
    n_initializations: int = 20,
    n_shuffles: int = 20,
    seed: int = 0,
    shuffle_mode: str = "global",
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Evaluate real-versus-shuffled dynamics across temporal resolutions."""
    events = filter_core_events(interactions, nodes)
    records = []

    for frequency in frequencies:
        results = evaluate_timestamp_nulls(
            events,
            nodes,
            frequency=frequency,
            negative_fraction=negative_fraction,
            n_initializations=n_initializations,
            n_shuffles=n_shuffles,
            seed=seed,
            mode=shuffle_mode,
        )

        results["f"] = negative_fraction
        summary = paired_bootstrap_summary(
            results,
            metrics=("final_positive_fraction", "total_switches"),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        summary["frequency"] = frequency
        records.extend(summary.to_dict("records"))

    return pd.DataFrame(records)


def plot_temporal_activity(activity: pd.DataFrame, output: str | Path) -> None:
    """Plot weekly positive and negative interaction activity."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(activity["timestamp"], activity["positive"], label="Non-negative events")
    ax.plot(activity["timestamp"], activity["negative"], label="Negative events")
    ax.set(
        xlabel="Time", ylabel="Number of interactions", title="Temporal Reddit interaction activity"
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_temporal_trajectories(trajectories: pd.DataFrame, output: str | Path) -> None:
    """Plot mean opinion trajectories with approximate 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(11, 4))

    for variant, group in trajectories.groupby("variant"):
        stats = group.groupby("step")["positive_fraction"].agg(["mean", "std", "count"])
        sem = stats["std"].fillna(0) / np.sqrt(stats["count"])
        lower = (stats["mean"] - 1.96 * sem).clip(0, 1)
        upper = (stats["mean"] + 1.96 * sem).clip(0, 1)

        ax.plot(stats.index, stats["mean"], label=variant)
        ax.fill_between(stats.index.to_numpy(), lower.to_numpy(), upper.to_numpy(), alpha=0.2)

    ax.set(xlabel="Temporal window / opinion update", ylabel="Fraction holding +1", ylim=(0, 1))
    ax.set_title("Temporal Signed NCO: real versus shuffled timestamps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_minority_trajectories(
    trajectories: pd.DataFrame, output: str | Path, *, metric: str = "minority_fraction"
) -> None:
    """Plot minority survival with between-initialization uncertainty."""
    if metric not in ("minority_fraction", "initial_minority_fraction"):
        raise ValueError("Unknown minority trajectory metric")
    if metric not in trajectories or trajectories[metric].isna().all():
        raise ValueError("Metric is unavailable for this initial opinion fraction")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for variant, group in trajectories.groupby("variant"):
        stats = group.groupby("step")[metric].agg(["mean", "std", "count"])
        sem = stats["std"].fillna(0) / np.sqrt(stats["count"])
        x, mean = stats.index.to_numpy(), stats["mean"].to_numpy()
        delta = 1.96 * sem.to_numpy()
        ax.plot(x, mean, label=variant)
        ax.fill_between(x, np.clip(mean - delta, 0, 1), np.clip(mean + delta, 0, 1), alpha=0.2)
    ylabel = (
        "Current minority fraction"
        if metric == "minority_fraction"
        else "Fraction holding the initially minority opinion"
    )
    ax.set(
        xlabel="Temporal window / opinion update",
        ylabel=ylabel,
        title="Minority trajectory: real vs shuffled timing",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_resolution_ablation(
    ablation: pd.DataFrame, output: str | Path, *, metric: str = "final_positive_fraction"
) -> None:
    """Plot paired real-minus-shuffled differences across time resolutions."""
    data = ablation.loc[ablation["metric"] == metric].copy()
    positions = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(
        positions,
        data["difference"],
        yerr=[
            data["difference"] - data["ci_low"],
            data["ci_high"] - data["difference"],
        ],
        fmt="o",
        capsize=4,
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(positions, data["frequency"])
    ax.set(
        xlabel="Time window", ylabel="Real - shuffled", title=f"Resolution sensitivity: {metric}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


# Consolidated temporal experiment, robustness and comparison functions.


def lifetime_topology(interactions: pd.DataFrame, nodes: tuple[str, ...]) -> csr_matrix:
    """Build the undirected union of all observed edges."""
    events = filter_core_events(interactions, nodes)
    index = {node: i for i, node in enumerate(nodes)}
    u = events["source"].map(index).to_numpy(dtype=np.int64)
    v = events["target"].map(index).to_numpy(dtype=np.int64)
    rows = np.concatenate((u, v))
    cols = np.concatenate((v, u))
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(nodes), len(nodes))
    )
    graph.sum_duplicates()
    graph.data[:] = 1
    return graph


def largest_opinion_component(graph: csr_matrix, state: np.ndarray, opinion: int) -> int:
    """Return the largest connected component containing only the selected opinion."""
    selected = np.flatnonzero(state == opinion)
    if len(selected) == 0:
        return 0

    subgraph = graph[selected][:, selected]
    _, labels = connected_components(subgraph, directed=False)
    return int(np.bincount(labels).max())


def opinion_metrics(graph: csr_matrix, state: np.ndarray) -> dict:
    """Calculate label-invariant outcomes and opinion-specific component sizes."""
    state = np.asarray(state)
    if state.ndim != 1 or len(state) != graph.shape[0] or not np.isin(state, (-1, 1)).all():
        raise ValueError("Invalid opinion state")

    n = len(state)
    positive_fraction = float(np.mean(state == 1)) if n else 0.0
    positive_gcc = largest_opinion_component(graph, state, 1)
    negative_gcc = largest_opinion_component(graph, state, -1)

    return {
        "positive_fraction": positive_fraction,
        "negative_fraction": 1.0 - positive_fraction if n else 0.0,
        "absolute_magnetization": abs(2 * positive_fraction - 1) if n else 0.0,
        "minority_fraction": min(positive_fraction, 1 - positive_fraction) if n else 0.0,
        "positive_gcc_fraction": positive_gcc / n if n else 0.0,
        "negative_gcc_fraction": negative_gcc / n if n else 0.0,
        "largest_opinion_gcc_fraction": max(positive_gcc, negative_gcc) / n if n else 0.0,
    }


def run_fraction_sweep(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    *,
    fractions: tuple[float, ...] = tuple(np.round(np.arange(0.1, 1.0, 0.1), 2)),
    frequency: str = "W-SUN",
    n_initializations: int = 20,
    n_shuffles: int = 20,
    seed: int = 42,
    modes: tuple[str, ...] = ("global", "within_source"),
) -> pd.DataFrame:
    """Run paired real-versus-null experiments across initial opinion fractions."""
    if n_initializations < 1 or n_shuffles < 1:
        raise ValueError("Repetition counts must be positive")
    if not fractions or not all(np.isfinite(f) and 0 <= f <= 1 for f in fractions):
        raise ValueError("fractions must contain values between 0 and 1")
    if len(set(modes)) != len(modes) or not set(modes).issubset({"global", "within_source"}):
        raise ValueError("Invalid or duplicate shuffle modes")

    events = filter_core_events(interactions, nodes)
    graph = lifetime_topology(events, nodes)
    real_snapshots = build_temporal_snapshots(events, nodes, frequency=frequency)
    rng = np.random.default_rng(seed)
    snapshot_sets = [("real", -1, real_snapshots)]

    # Construct shuffled networks once, then reuse them for all f values.
    for mode in modes:
        for shuffle_id in range(n_shuffles):
            shuffle_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            shuffled = shuffle_timestamps(events, mode=mode, seed=shuffle_seed)
            snapshots = build_temporal_snapshots(shuffled, nodes, frequency=frequency)
            snapshot_sets.append((mode, shuffle_id, snapshots))

    records = []

    for f in fractions:
        for initialization in range(n_initializations):
            initial_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            initial_state = initialize_opinions(len(nodes), float(f), seed=initial_seed)

            for variant, shuffle_id, snapshots in snapshot_sets:
                result = simulate_temporal_nco(snapshots, initial_state=initial_state)
                metrics = opinion_metrics(graph, result.final_state)
                records.append(
                    {
                        "f": float(f),
                        "initialization": initialization,
                        "initial_seed": initial_seed,
                        "variant": variant,
                        "shuffle": shuffle_id,
                        "frequency": frequency,
                        "total_switches": int(result.changed_count.sum()),
                        "final_week_switches": int(result.changed_count[-1]) if result.steps else 0,
                        **metrics,
                    }
                )

    return pd.DataFrame.from_records(records)


def paired_fraction_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize real-minus-null effects, pairing by f and initial opinion configuration."""
    metrics = (
        "absolute_magnetization",
        "minority_fraction",
        "largest_opinion_gcc_fraction",
        "total_switches",
    )
    records = []

    for (f, variant), group in results.loc[results["variant"] != "real"].groupby(
        ["f", "variant"], sort=True
    ):
        real = results.loc[(results["f"] == f) & (results["variant"] == "real")].set_index(
            "initialization"
        )
        null = group.groupby("initialization")[list(metrics)].mean()
        shared = real.index.intersection(null.index)

        for metric in metrics:
            differences = (real.loc[shared, metric] - null.loc[shared, metric]).to_numpy(
                dtype=float
            )
            n = len(differences)
            sem = float(differences.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            records.append(
                {
                    "f": f,
                    "variant": variant,
                    "metric": metric,
                    "n_initializations": n,
                    "real_mean": float(real.loc[shared, metric].mean()),
                    "null_mean": float(null.loc[shared, metric].mean()),
                    "difference": float(differences.mean()),
                    "standard_error": sem,
                }
            )

    return pd.DataFrame(records)


def shuffle_sentiments(interactions: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Permute event sentiment labels while keeping endpoints and timestamps fixed."""
    result = interactions.copy(deep=True)
    result["sentiment"] = np.random.default_rng(seed).permutation(result["sentiment"].to_numpy())
    return result


def sign_persistence_null(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    *,
    frequency: str = "W-SUN",
    n_shuffles: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare real weekly dyad-sign transitions with globally shuffled sentiment labels."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be positive")

    events = filter_core_events(interactions, nodes)
    rng = np.random.default_rng(seed)
    records = []

    for realization in range(-1, n_shuffles):
        sample = (
            events
            if realization == -1
            else shuffle_sentiments(events, seed=int(rng.integers(0, np.iinfo(np.int64).max)))
        )
        transitions = sign_transitions(sample, nodes, frequency=frequency)
        counts = {(r.previous, r.current): int(r.count) for r in transitions.itertuples()}

        for previous in ("+", "-", "mixed"):
            denominator = sum(counts.get((previous, current), 0) for current in ("+", "-", "mixed"))
            for current in ("+", "-", "mixed"):
                count = counts.get((previous, current), 0)
                records.append(
                    {
                        "variant": "real" if realization == -1 else "sign_shuffled",
                        "realization": realization,
                        "previous": previous,
                        "current": current,
                        "count": count,
                        "denominator": denominator,
                        "probability": count / denominator if denominator else np.nan,
                    }
                )

    return pd.DataFrame(records)


def sign_persistence_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    """Summarize transition probabilities across sign-shuffled null realizations."""
    real = transitions.loc[transitions["variant"] == "real"]
    null = transitions.loc[transitions["variant"] == "sign_shuffled"]
    summaries = (
        null.groupby(["previous", "current"])["probability"].agg(["mean", "std"]).reset_index()
    )
    summaries = summaries.rename(columns={"mean": "null_mean", "std": "null_std"})
    return (
        real[["previous", "current", "probability", "count"]]
        .merge(summaries, on=["previous", "current"])
        .rename(columns={"probability": "real_probability"})
    )


def minority_connectivity(graph, state: np.ndarray) -> dict:
    """Measure connectivity among nodes holding the less frequent opinion."""
    state = np.asarray(state)
    if (
        state.ndim != 1
        or graph.shape != (len(state), len(state))
        or not np.isin(state, (-1, 1)).all()
    ):
        raise ValueError("Invalid graph or opinion state")

    n = len(state)
    if n == 0:
        return {
            "minority_count": 0,
            "minority_lcc": 0,
            "minority_lcc_fraction": 0.0,
            "minority_component_count": 0,
            "minority_isolated_fraction": 0.0,
        }

    positive = int(np.count_nonzero(state == 1))
    negative = n - positive
    minority = -1 if negative <= positive else 1
    selected = np.flatnonzero(state == minority)

    if len(selected) == 0:
        return {
            "minority_count": 0,
            "minority_lcc": 0,
            "minority_lcc_fraction": 0.0,
            "minority_component_count": 0,
            "minority_isolated_fraction": 0.0,
        }

    subgraph = graph[selected][:, selected].tocsr()
    n_components, labels = connected_components(subgraph, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    isolated = int(np.count_nonzero(sizes == 1))

    return {
        "minority_count": len(selected),
        "minority_lcc": int(sizes.max()),
        "minority_lcc_fraction": float(sizes.max() / len(selected)),
        "minority_component_count": n_components,
        "minority_isolated_fraction": float(isolated / len(selected)),
    }


def run_minority_experiment(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    *,
    fractions: tuple[float, ...] = (0.3, 0.5, 0.7),
    n_initializations: int = 20,
    n_shuffles: int = 20,
    seed: int = 42,
    frequency: str = "W-SUN",
) -> pd.DataFrame:
    """Compare final minority fragmentation on real and randomized temporal networks."""
    if (
        n_initializations < 1
        or n_shuffles < 1
        or not fractions
        or any(not 0 <= f <= 1 for f in fractions)
    ):
        raise ValueError("Invalid experiment configuration")

    events = filter_core_events(interactions, nodes)
    graph = lifetime_topology(events, nodes)
    rng = np.random.default_rng(seed)
    snapshots = [("real", -1, build_temporal_snapshots(events, nodes, frequency=frequency))]

    for mode in ("global", "within_source"):
        for shuffle_id in range(n_shuffles):
            shuffle_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            shuffled = shuffle_timestamps(events, mode=mode, seed=shuffle_seed)
            snapshots.append(
                (mode, shuffle_id, build_temporal_snapshots(shuffled, nodes, frequency=frequency))
            )

    records = []

    for f in fractions:
        for initialization in range(n_initializations):
            initial_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            initial = initialize_opinions(len(nodes), float(f), seed=initial_seed)

            for variant, shuffle_id, network in snapshots:
                result = simulate_temporal_nco(network, initial_state=initial)
                metrics = minority_connectivity(graph, result.final_state)
                records.append(
                    {
                        "f": float(f),
                        "initialization": initialization,
                        "initial_seed": initial_seed,
                        "variant": variant,
                        "shuffle": shuffle_id,
                        **metrics,
                    }
                )

    return pd.DataFrame(records)


def shuffle_source_sentiments(interactions: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Shuffle signs within each source, preserving source-specific sign counts."""
    required = {"source", "target", "sentiment", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError("Missing interaction columns")
    if not interactions["sentiment"].isin((-1, 1)).all():
        raise ValueError("Invalid sentiment labels")

    result = interactions.copy(deep=True)
    rng = np.random.default_rng(seed)
    original = result["sentiment"].to_numpy()
    shuffled = original.copy()

    for indices in result.groupby("source", sort=True, dropna=False).indices.values():
        shuffled[indices] = rng.permutation(original[indices])

    result["sentiment"] = shuffled
    return result


def compare_sign_persistence(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    *,
    n_shuffles: int = 100,
    seed: int = 42,
    frequency: str = "W-SUN",
) -> pd.DataFrame:
    """Compare observed dyad transitions with global and source-conditioned sign nulls."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be positive")

    events = filter_core_events(interactions, nodes)
    rng = np.random.default_rng(seed)
    records = []
    signs = ("+", "-", "mixed")

    for variant in ("real", "global", "within_source"):
        repetitions = 1 if variant == "real" else n_shuffles

        for realization in range(repetitions):
            if variant == "real":
                sample = events
            elif variant == "global":
                sample = events.copy()
                sample["sentiment"] = rng.permutation(events["sentiment"].to_numpy())
            else:
                sample = shuffle_source_sentiments(
                    events, seed=int(rng.integers(0, np.iinfo(np.int64).max))
                )

            transitions = sign_transitions(sample, nodes, frequency=frequency)
            counts = {(r.previous, r.current): int(r.count) for r in transitions.itertuples()}

            for previous in signs:
                denominator = sum(counts.get((previous, current), 0) for current in signs)

                for current in signs:
                    count = counts.get((previous, current), 0)
                    records.append(
                        {
                            "variant": variant,
                            "realization": realization,
                            "previous": previous,
                            "current": current,
                            "count": count,
                            "denominator": denominator,
                            "probability": count / denominator if denominator else np.nan,
                        }
                    )

    return pd.DataFrame(records)


def summarize_sign_persistence(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize transitions, leaving p-values undefined for missing observations."""
    real = results.loc[results["variant"] == "real"].set_index(["previous", "current"])
    records = []

    for variant, group in results.loc[results["variant"] != "real"].groupby("variant"):
        for (previous, current), subset in group.groupby(["previous", "current"]):
            observed = float(real.loc[(previous, current), "probability"])
            null = subset["probability"].dropna().to_numpy(dtype=float)

            if len(null) == 0:
                continue

            p_two_sided = np.nan
            if np.isfinite(observed):
                lower = (1 + np.count_nonzero(null <= observed)) / (len(null) + 1)
                upper = (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
                p_two_sided = min(1.0, 2 * min(lower, upper))

            records.append(
                {
                    "variant": variant,
                    "previous": previous,
                    "current": current,
                    "observed": observed,
                    "null_mean": float(null.mean()),
                    "null_low": float(np.quantile(null, 0.025)),
                    "null_high": float(np.quantile(null, 0.975)),
                    "difference": float(observed - null.mean()),
                    "p_two_sided": p_two_sided,
                }
            )

    return pd.DataFrame(records)


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Resample paired differences over initializations and shared shuffle IDs.

    Rows are initializations and columns are shuffled timelines. A single column
    represents a comparison on the observed timeline only.
    """
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2 or 0 in differences.shape or not np.isfinite(differences).all():
        raise ValueError("Bootstrap requires a complete finite matrix of paired differences")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    n_initializations, n_shuffles = differences.shape
    estimates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        i = rng.integers(0, n_initializations, size=n_initializations)
        j = rng.integers(0, n_shuffles, size=n_shuffles) if n_shuffles > 1 else np.array([0])
        estimates[b] = differences[np.ix_(i, j)].mean()
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def paired_bootstrap_summary(
    results: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    n_bootstrap: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Paired two-level bootstrap over initializations and shared null realizations."""
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")

    rng = np.random.default_rng(seed)
    records = []

    for f, fraction_rows in results.groupby("f", sort=True):
        real_rows = fraction_rows.loc[fraction_rows["variant"] == "real"].set_index(
            "initialization"
        )

        if real_rows.index.has_duplicates:
            raise ValueError("Duplicate real initialization")

        for variant, null_rows in fraction_rows.loc[fraction_rows["variant"] != "real"].groupby(
            "variant"
        ):
            for metric in metrics:
                if metric not in results.columns:
                    raise ValueError(f"Missing metric: {metric}")

                matrix = null_rows.pivot(index="initialization", columns="shuffle", values=metric)
                aligned_real = real_rows[metric].reindex(matrix.index)
                real = aligned_real.to_numpy(dtype=float)
                null = matrix.to_numpy(dtype=float)

                if (
                    len(real) < 2
                    or null.shape[1] < 2
                    or not np.isfinite(real).all()
                    or not np.isfinite(null).all()
                ):
                    raise ValueError(
                        "Bootstrap requires complete paired runs and at least two null realizations"
                    )

                effect = float(real.mean() - null.mean())
                ci_low, ci_high = paired_bootstrap_interval(
                    real[:, None] - null,
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                )
                records.append(
                    {
                        "f": float(f),
                        "variant": variant,
                        "metric": metric,
                        "n_initializations": len(real),
                        "n_shuffles": null.shape[1],
                        "real_mean": float(real.mean()),
                        "null_mean": float(null.mean()),
                        "difference": effect,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    return pd.DataFrame(records)


def simulate_static_fixed_horizon(
    matrix: csr_matrix, initial_state: np.ndarray, *, steps: int
) -> dict:
    """Apply static signed NCO for exactly the temporal experiment's number of steps."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")

    matrix = _prepare_matrix(matrix, "signed")
    state = _prepare_state(initial_state, matrix.shape[0]).copy()
    changed_count = np.zeros(steps, dtype=np.int64)
    first_fixed_point = 0 if steps == 0 else None
    first_repeated_state = None
    seen = {state.tobytes(): 0}

    for step in range(steps):
        next_state = _step(matrix, state)
        changed_count[step] = np.count_nonzero(next_state != state)

        if first_fixed_point is None and changed_count[step] == 0:
            first_fixed_point = step + 1

        key = next_state.tobytes()
        if first_repeated_state is None and key in seen:
            first_repeated_state = step + 1

        if key not in seen:
            seen[key] = step + 1

        state = next_state

    return {
        "final_state": state.copy(),
        "changed_count": changed_count,
        "first_fixed_point": first_fixed_point,
        "first_repeated_state": first_repeated_state,
    }


def final_metrics(graph: csr_matrix, state: np.ndarray, switches: np.ndarray) -> dict:
    """Combine opinion dominance, fragmentation, and switching measurements."""
    opinion = opinion_metrics(graph, state)
    minority = minority_connectivity(graph, state)

    return {
        **opinion,
        **minority,
        "total_switches": int(switches.sum()),
        "switching_updates": int(np.count_nonzero(switches)),
        "final_update_switches": int(switches[-1]) if len(switches) else 0,
    }


def run_static_temporal_comparison(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    static_signed: csr_matrix,
    *,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    frequency: str = "W-SUN",
    n_initializations: int = 20,
    n_shuffles: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare static, observed temporal, and globally shuffled temporal NCO."""
    if n_initializations < 1 or n_shuffles < 1:
        raise ValueError("Repetition counts must be positive")
    if not fractions or any(not np.isfinite(f) or not 0 <= f <= 1 for f in fractions):
        raise ValueError("Invalid initial opinion fractions")
    if static_signed.shape != (len(nodes), len(nodes)):
        raise ValueError("Static matrix shape does not match nodes")

    static_signed = _prepare_matrix(static_signed, "signed")
    events = filter_core_events(interactions, nodes)
    graph = lifetime_topology(events, nodes)
    real_snapshots = build_temporal_snapshots(events, nodes, frequency=frequency)
    horizon = len(real_snapshots.timestamps)
    rng = np.random.default_rng(seed)
    null_snapshots = []

    for shuffle_id in range(n_shuffles):
        shuffle_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        shuffled = shuffle_timestamps(events, mode="global", seed=shuffle_seed)
        snapshots = build_temporal_snapshots(shuffled, nodes, frequency=frequency)

        if len(snapshots.timestamps) != horizon:
            raise ValueError("Shuffling changed the temporal horizon")

        null_snapshots.append((shuffle_id, shuffle_seed, snapshots))

    records = []

    for f in fractions:
        for initialization in range(n_initializations):
            initial_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            initial = initialize_opinions(len(nodes), float(f), seed=initial_seed)

            static = simulate_static_fixed_horizon(static_signed, initial, steps=horizon)
            real = simulate_temporal_nco(real_snapshots, model="signed", initial_state=initial)
            outcomes = [
                ("static", -1, -1, static["final_state"], static["changed_count"]),
                ("temporal_real", -1, -1, real.final_state, real.changed_count),
            ]

            for shuffle_id, shuffle_seed, snapshots in null_snapshots:
                null = simulate_temporal_nco(snapshots, model="signed", initial_state=initial)
                outcomes.append(
                    (
                        "temporal_shuffled",
                        shuffle_id,
                        shuffle_seed,
                        null.final_state,
                        null.changed_count,
                    )
                )

            for variant, shuffle_id, shuffle_seed, final_state, switches in outcomes:
                records.append(
                    {
                        "f": float(f),
                        "initialization": initialization,
                        "initial_seed": initial_seed,
                        "variant": variant,
                        "shuffle": shuffle_id,
                        "shuffle_seed": shuffle_seed,
                        "frequency": frequency,
                        "steps": horizon,
                        "disagreement_with_static": float(
                            np.mean(final_state != static["final_state"])
                        )
                        if len(nodes)
                        else 0.0,
                        "disagreement_with_real": float(np.mean(final_state != real.final_state))
                        if len(nodes)
                        else 0.0,
                        "static_first_fixed_point": static["first_fixed_point"],
                        "static_first_repeated_state": static["first_repeated_state"],
                        **final_metrics(graph, final_state, switches),
                    }
                )

    return pd.DataFrame(records)


def summarize_static_temporal(
    results: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = (
        "absolute_magnetization",
        "minority_fraction",
        "minority_lcc_fraction",
        "total_switches",
        "disagreement_with_static",
    ),
) -> pd.DataFrame:
    """Return mean outcomes for all variants across initial opinion fractions."""
    columns = ["f", "variant", "metric", "mean", "std", "n_initializations", "n_shuffles"]
    records = []

    for (f, variant), group in results.groupby(["f", "variant"], sort=True):
        for metric in metrics:
            if metric not in results.columns:
                raise ValueError(f"Missing metric: {metric}")

            if variant == "temporal_shuffled":
                values = group.groupby("initialization")[metric].mean()
            else:
                values = group.set_index("initialization")[metric]

            records.append(
                {
                    "f": float(f),
                    "variant": variant,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "n_initializations": len(values),
                    "n_shuffles": group["shuffle"].nunique()
                    if variant == "temporal_shuffled"
                    else 0,
                }
            )

    return pd.DataFrame(records, columns=columns)
