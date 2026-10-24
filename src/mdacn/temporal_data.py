"""Construct time-windowed signed and unsigned Reddit networks."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

TiePolicy = Literal["zero", "positive", "negative"]


@dataclass(frozen=True)
class TemporalSnapshots:
    """Aligned temporal graphs using the static pipeline's node ordering."""

    nodes: tuple[str, ...]
    timestamps: tuple[pd.Timestamp, ...]
    unsigned: tuple[csr_matrix, ...]
    signed: tuple[csr_matrix, ...]
    event_counts: tuple[int, ...]


def temporal_windows(timestamps: pd.Series, frequency: str) -> tuple[pd.Series, pd.PeriodIndex]:
    """Assign events to disjoint windows anchored at the first observed base period.

    Pandas labels each week separately even for ``to_period('2W-SUN')``.
    Map those labels back to the calendar, which advances two weeks at a time.
    The timestamp multiset fixes the anchor, so permutations use identical bins.
    """
    periods = timestamps.dt.to_period(frequency)
    if periods.empty:
        return periods, pd.PeriodIndex([], freq=frequency)
    calendar = pd.period_range(periods.min(), periods.max(), freq=frequency)
    positions = np.searchsorted(calendar.asi8, periods.array.asi8, side="right") - 1
    labels = pd.Series(calendar.take(positions), index=timestamps.index, name=timestamps.name)
    return labels, calendar


def build_temporal_snapshots(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    *,
    frequency: str = "W-SUN",
    tie_policy: TiePolicy = "zero",
) -> TemporalSnapshots:
    """Aggregate timestamped interactions into undirected sparse network snapshots."""

    if tie_policy not in ("zero", "positive", "negative"):
        raise ValueError("tie_policy must be 'zero', 'positive', or 'negative'")
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")

    required = {"source", "target", "sentiment", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError(
            f"interactions missing columns: {sorted(required - set(interactions.columns))}"
        )

    try:
        pd.Period("2020-01-01", freq=frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pandas period frequency: {frequency}") from exc

    index = {node: i for i, node in enumerate(nodes)}
    events = interactions.loc[
        interactions["source"].isin(index)
        & interactions["target"].isin(index)
        & interactions["source"].ne(interactions["target"]),
        ["source", "target", "sentiment", "timestamp"],
    ].copy()

    if events.empty:
        return TemporalSnapshots(nodes, (), (), (), ())

    if not events["sentiment"].isin((-1, 1)).all():
        raise ValueError("sentiment must be +1 or -1")

    if not pd.api.types.is_datetime64_any_dtype(events["timestamp"]):
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="raise")
    if events["timestamp"].isna().any():
        raise ValueError("timestamp contains null values")

    periods, full_range = temporal_windows(events["timestamp"], frequency)
    events["period"] = periods

    source_idx = events["source"].map(index).to_numpy(dtype=np.int64)
    target_idx = events["target"].map(index).to_numpy(dtype=np.int64)
    events["u"] = np.minimum(source_idx, target_idx)
    events["v"] = np.maximum(source_idx, target_idx)
    events["positive"] = (events["sentiment"] == 1).astype(np.int64)
    events["negative"] = (events["sentiment"] == -1).astype(np.int64)

    grouped = events.groupby(["period", "u", "v"], sort=True, observed=True)[
        ["positive", "negative"]
    ].sum()
    grouped_periods = dict(tuple(grouped.groupby(level="period", sort=False)))
    event_counts = events.groupby("period", observed=True).size()

    n = len(nodes)
    unsigned, signed, counts = [], [], []

    for period in full_range:
        period_edges = grouped_periods.get(period)

        if period_edges is None:
            unsigned.append(csr_matrix((n, n), dtype=np.float64))
            signed.append(csr_matrix((n, n), dtype=np.float64))
            counts.append(0)
            continue

        endpoints = period_edges.index.droplevel("period")
        u = endpoints.get_level_values("u").to_numpy(dtype=np.int64)
        v = endpoints.get_level_values("v").to_numpy(dtype=np.int64)

        diff = (period_edges["positive"] - period_edges["negative"]).to_numpy()
        signs = np.sign(diff)

        if tie_policy == "positive":
            signs[diff == 0] = 1
        elif tie_policy == "negative":
            signs[diff == 0] = -1

        rows = np.concatenate((u, v))
        cols = np.concatenate((v, u))

        presence = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        weights = csr_matrix(
            (np.concatenate((signs, signs)), (rows, cols)), shape=(n, n), dtype=np.float64
        )
        weights.eliminate_zeros()

        unsigned.append(presence)
        signed.append(weights)
        counts.append(int(event_counts[period]))

    return TemporalSnapshots(
        nodes=nodes,
        timestamps=tuple(period.start_time for period in full_range),
        unsigned=tuple(unsigned),
        signed=tuple(signed),
        event_counts=tuple(counts),
    )
