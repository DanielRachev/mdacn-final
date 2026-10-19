
"""Temporal burstiness, incident-edge correlations and dyad-conditioned sign nulls."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from mdacn.temporal_analysis import filter_core_events, sign_transitions
from mdacn.temporal_analysis import summarize_sign_persistence
from mdacn.temporal_nulls import shuffle_timestamps


SIGNS = ("+", "-", "mixed")


def _weekly_table(events: pd.DataFrame, nodes: tuple[str, ...], frequency: str) -> tuple[pd.DataFrame, int]:
    """Build source-by-period activity counts, retaining zero-activity windows."""
    if events.empty:
        return pd.DataFrame(index=pd.Index(nodes, name="source")), 0

    periods = events["timestamp"].dt.to_period(frequency)
    calendar = pd.period_range(periods.min(), periods.max(), freq=frequency)
    counts = pd.crosstab(events["source"], periods)
    counts = counts.reindex(index=nodes, columns=calendar, fill_value=0)
    return counts, len(calendar)


def node_activity_metrics(interactions: pd.DataFrame, nodes: tuple[str, ...], *, frequency: str = "W-SUN", min_events: int = 3) -> pd.DataFrame:
    """Calculate source-level event burstiness and weekly activity dispersion."""
    if min_events < 3:
        raise ValueError("min_events must be at least 3")

    events = filter_core_events(interactions, nodes)
    counts, n_windows = _weekly_table(events, nodes, frequency)
    results = []
    grouped_times = {node: group["timestamp"].sort_values().to_numpy(dtype="datetime64[ns]") for node, group in events.groupby("source")}

    for node in nodes:
        times = grouped_times.get(node, np.array([], dtype="datetime64[ns]"))
        activity = counts.loc[node].to_numpy(dtype=float) if n_windows else np.array([])
        total = len(times)
        fano = float(activity.var() / activity.mean()) if total and n_windows else np.nan
        burstiness = np.nan

        if total >= min_events:
            gaps = np.diff(times).astype("timedelta64[ns]").astype(np.float64) / 1e9
            mean_gap = float(gaps.mean())
            sd_gap = float(gaps.std())
            if mean_gap + sd_gap > 0:
                burstiness = (sd_gap - mean_gap) / (sd_gap + mean_gap)

        results.append({
            "node": node, "event_count": total,
            "burstiness": burstiness, "weekly_fano": fano,
            "active_windows": int(np.count_nonzero(activity)),
        })

    return pd.DataFrame(results)


def activity_null_experiment(interactions: pd.DataFrame, nodes: tuple[str, ...], *, frequency: str = "W-SUN", n_shuffles: int = 30, seed: int = 42) -> pd.DataFrame:
    """Compare source-level burstiness with globally timestamp-shuffled events."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be positive")

    events = filter_core_events(interactions, nodes)
    rng = np.random.default_rng(seed)
    records = []

    for realization in range(-1, n_shuffles):
        sample = events if realization == -1 else shuffle_timestamps(events, seed=int(rng.integers(0, np.iinfo(np.int64).max)))
        metrics = node_activity_metrics(sample, nodes, frequency=frequency)

        for metric in ("burstiness", "weekly_fano", "active_windows"):
            eligible = metrics.loc[metrics["event_count"] >= 3, metric].dropna()
            records.append({
                "variant": "real" if realization == -1 else "global",
                "realization": realization, "metric": metric,
                "mean": float(eligible.mean()) if len(eligible) else np.nan,
                "median": float(eligible.median()) if len(eligible) else np.nan,
                "n_nodes": len(eligible),
            })

    return pd.DataFrame(records)


def sample_incident_pairs(interactions: pd.DataFrame, nodes: tuple[str, ...], *, max_pairs: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Sample distinct pairs of undirected edges that share a node."""
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")

    events = filter_core_events(interactions, nodes)
    endpoints = set()

    for source, target in zip(events["source"], events["target"]):
        endpoints.add(tuple(sorted((source, target))))

    incident = {node: [] for node in nodes}
    for edge in sorted(endpoints):
        incident[edge[0]].append(edge)
        incident[edge[1]].append(edge)

    rng = np.random.default_rng(seed)
    candidates = []

    for node, edges in incident.items():
        if len(edges) < 2:
            continue

        # Limit the number of candidates contributed by high-degree nodes.
        local = set()
        target = min(20, len(edges) * (len(edges) - 1) // 2)
        attempts = 0

        while len(local) < target and attempts < max(100, target * 20):
            i, j = rng.choice(len(edges), size=2, replace=False)
            local.add(tuple(sorted((int(i), int(j)))))
            attempts += 1

        for i, j in sorted(local):
            candidates.append((node, edges[i], edges[j]))

    if len(candidates) > max_pairs:
        indices = np.sort(rng.choice(len(candidates), size=max_pairs, replace=False))
        candidates = [candidates[i] for i in indices]

    return pd.DataFrame(candidates, columns=["node", "edge_a", "edge_b"])


def incident_edge_correlations(interactions: pd.DataFrame, nodes: tuple[str, ...], pairs: pd.DataFrame, *, frequency: str = "W-SUN") -> pd.DataFrame:
    """Calculate Pearson correlations of weekly event counts on sampled incident edge pairs."""
    events = filter_core_events(interactions, nodes)

    if pairs.empty:
        return pairs.assign(correlation=pd.Series(dtype=float))

    if events.empty:
        return pairs.assign(correlation=np.nan)

    index = {node: i for i, node in enumerate(nodes)}
    source = events["source"].map(index).to_numpy(dtype=np.int64)
    target = events["target"].map(index).to_numpy(dtype=np.int64)
    u = np.where(source <= target, source, target)
    v = np.where(source <= target, target, source)

    periods = events["timestamp"].dt.to_period(frequency)
    calendar = pd.period_range(periods.min(), periods.max(), freq=frequency)
    period_idx = pd.Categorical(periods, categories=calendar, ordered=True).codes

    edge_keys = list(zip(u, v))
    distinct_edges = sorted(set(edge_keys))
    edge_index = {edge: i for i, edge in enumerate(distinct_edges)}
    edge_rows = np.array([edge_index[edge] for edge in edge_keys], dtype=np.int64)

    activity = csr_matrix(
        (np.ones(len(edge_rows), dtype=np.float64), (edge_rows, period_idx)),
        shape=(len(distinct_edges), len(calendar)),
    )

    n_periods = activity.shape[1]
    totals = np.asarray(activity.sum(axis=1)).ravel()
    squares = np.asarray(activity.multiply(activity).sum(axis=1)).ravel()

    correlations = []

    for row in pairs.itertuples(index=False):
        a = edge_index.get(tuple(sorted((index[row.edge_a[0]], index[row.edge_a[1]]))))
        b = edge_index.get(tuple(sorted((index[row.edge_b[0]], index[row.edge_b[1]]))))

        if a is None or b is None or n_periods < 2:
            correlations.append(np.nan)
            continue

        numerator = float(activity[a].multiply(activity[b]).sum()) - totals[a] * totals[b] / n_periods
        variance_a = squares[a] - totals[a] ** 2 / n_periods
        variance_b = squares[b] - totals[b] ** 2 / n_periods

        correlations.append(numerator / np.sqrt(variance_a * variance_b) if variance_a > 0 and variance_b > 0 else np.nan)

    result = pairs.copy()
    result["correlation"] = correlations
    return result


def correlation_null_experiment(interactions: pd.DataFrame, nodes: tuple[str, ...], *, frequency: str = "W-SUN", n_shuffles: int = 30, max_pairs: int = 3000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare real incident-edge coordination with both timestamp null models."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be positive")

    events = filter_core_events(interactions, nodes)
    pairs = sample_incident_pairs(events, nodes, max_pairs=max_pairs, seed=seed)
    rng = np.random.default_rng(seed)
    records = []

    for variant in ("real", "global", "within_source"):
        repetitions = 1 if variant == "real" else n_shuffles

        for realization in range(repetitions):
            sample = events if variant == "real" else shuffle_timestamps(events, mode=variant, seed=int(rng.integers(0, np.iinfo(np.int64).max)))
            calculated = incident_edge_correlations(sample, nodes, pairs, frequency=frequency)
            valid = calculated["correlation"].dropna()

            records.append({
                "variant": variant, "realization": realization,
                "mean_correlation": float(valid.mean()) if len(valid) else np.nan,
                "median_correlation": float(valid.median()) if len(valid) else np.nan,
                "positive_fraction": float((valid > 0).mean()) if len(valid) else np.nan,
                "n_pairs": len(valid),
            })

    return pd.DataFrame(records), pairs


def shuffle_dyad_sentiments(interactions: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Shuffle sentiment within unordered dyads, preserving each dyad's sign counts."""
    required = {"source", "target", "sentiment", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError("Missing interaction columns")
    if not interactions["sentiment"].isin((-1, 1)).all():
        raise ValueError("Sentiment must be +1 or -1")

    result = interactions.copy(deep=True)
    source = result["source"].to_numpy()
    target = result["target"].to_numpy()
    u = np.minimum(source, target)
    v = np.maximum(source, target)

    groups = pd.DataFrame({"u": u, "v": v}).groupby(["u", "v"], sort=True).indices
    original = result["sentiment"].to_numpy()
    shuffled = original.copy()
    rng = np.random.default_rng(seed)

    for indices in groups.values():
        shuffled[indices] = rng.permutation(original[indices])

    result["sentiment"] = shuffled
    return result


def dyad_sign_null_experiment(interactions: pd.DataFrame, nodes: tuple[str, ...], *, frequency: str = "W-SUN", n_shuffles: int = 100, seed: int = 42) -> pd.DataFrame:
    """Compare observed sign transitions with dyad-conditioned sentiment permutations."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be positive")

    events = filter_core_events(interactions, nodes)
    rng = np.random.default_rng(seed)
    records = []

    for realization in range(-1, n_shuffles):
        sample = events if realization == -1 else shuffle_dyad_sentiments(events, seed=int(rng.integers(0, np.iinfo(np.int64).max)))
        transitions = sign_transitions(sample, nodes, frequency=frequency)
        counts = {(row.previous, row.current): int(row.count) for row in transitions.itertuples()}

        for previous in SIGNS:
            denominator = sum(counts.get((previous, current), 0) for current in SIGNS)

            for current in SIGNS:
                count = counts.get((previous, current), 0)
                records.append({
                    "variant": "real" if realization == -1 else "within_dyad",
                    "realization": realization, "previous": previous, "current": current,
                    "count": count, "denominator": denominator,
                    "probability": count / denominator if denominator else np.nan,
                })

    return pd.DataFrame(records)


def summarize_dyad_sign_null(results: pd.DataFrame) -> pd.DataFrame:
    """Reuse the empirical transition-null summary from Feature 6."""
    return summarize_sign_persistence(results)


# Consolidated from temporal_reciprocity.py; preserve public function interfaces.
def signed_reciprocity(events: pd.DataFrame, *, sign: int, days: int = 7) -> dict:
    """Fraction of directed sign-specific events with a strictly later reverse event.

    Counts each initiating event once even if it has multiple replies. Includes only
    non-self directed events; all such events are denominator-eligible, including those
    near the end of the data window (no right-censoring correction).
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    if days <= 0:
        raise ValueError("days must be positive")
    required = {"source", "target", "sentiment", "timestamp"}
    if not required.issubset(events.columns):
        raise ValueError("Missing directed-event fields")
    selected = events.loc[(events["sentiment"] == sign) & (events["source"] != events["target"]), ["source", "target", "timestamp"]].copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="raise")
    if selected["timestamp"].isna().any():
        raise ValueError("Null event timestamp")
    label = "negative" if sign == -1 else "non_negative"
    if selected.empty:
        return {"sign": label, "eligible_events": 0, "reciprocated_events": 0, "fraction": float("nan"), "days": days}
    pairs = {(u, v): np.sort(group["timestamp"].to_numpy(dtype="datetime64[ns]")) for (u, v), group in selected.groupby(["source", "target"], sort=False)}
    responded = 0
    window_ns = np.timedelta64(days, "D")
    for (u, v), times in pairs.items():
        reverse = pairs.get((v, u))
        if reverse is None:
            continue
        positions = np.searchsorted(reverse, times, side="right")
        eligible = positions < reverse.size
        if eligible.any():
            responded += int(np.count_nonzero(reverse[positions[eligible]] <= times[eligible] + window_ns))
    return {"sign": label, "eligible_events": len(selected), "reciprocated_events": responded, "fraction": responded / len(selected), "days": days}


def negative_reciprocity(events: pd.DataFrame, *, days: int = 7) -> dict:
    """Backward-compatible negative-event reciprocity summary."""
    result = signed_reciprocity(events, sign=-1, days=days)
    return {"negative_events": result["eligible_events"], "reciprocated_events": result["reciprocated_events"], "fraction": result["fraction"], "days": days}


def signed_reciprocity_timestamp_nulls(events: pd.DataFrame, *, days: int = 7, n_shuffles: int = 20, seed: int = 42) -> pd.DataFrame:
    """Use identical timestamp permutations to compare both sentiments fairly."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be positive")
    rng = np.random.default_rng(seed)
    records = []
    for sign in (-1, 1):
        records.append({"variant": "real", "shuffle": -1, **signed_reciprocity(events, sign=sign, days=days)})
    for mode in ("global", "within_source"):
        for k in range(n_shuffles):
            shuffled = shuffle_timestamps(events, mode=mode, seed=int(rng.integers(0, np.iinfo(np.int64).max)))
            for sign in (-1, 1):
                records.append({"variant": mode, "shuffle": k, **signed_reciprocity(shuffled, sign=sign, days=days)})
    return pd.DataFrame(records)


def reciprocity_timestamp_nulls(events: pd.DataFrame, *, days: int = 7, n_shuffles: int = 20, seed: int = 42) -> pd.DataFrame:
    """Backward-compatible negative-only timestamp null experiment."""
    results = signed_reciprocity_timestamp_nulls(events, days=days, n_shuffles=n_shuffles, seed=seed)
    neg = results.loc[results["sign"] == "negative"].copy()
    return neg.rename(columns={"eligible_events": "negative_events"}).drop(columns="sign").reset_index(drop=True)
