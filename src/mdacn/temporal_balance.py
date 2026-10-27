"""Temporal P-O-X closure: predict each triangle's last edge sign from structural balance."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from mdacn.artifacts import checksum, write_json
from mdacn.data import load_graph, load_interactions
from mdacn.plots import plot_temporal_balance
from mdacn.signed_structure import (
    SignedEdges,
    node_positions,
    null_comparison,
    safe_ratio,
    signed_edges,
    spawn_seeds,
)
from mdacn.signed_triads import triangle_edges
from mdacn.temporal_analysis import filter_core_events
from mdacn.temporal_nulls import shuffle_timestamps

WEDGES = ("++", "+-", "--")
STATUSES = ("evaluated", "ambiguous_tie", "zero_closing", "zero_wedge")
TemporalNull = Literal["post_label_shuffle", "timestamp_shuffle"]
NULLS: tuple[TemporalNull, ...] = ("post_label_shuffle", "timestamp_shuffle")
_OTHERS = np.array([[1, 2], [0, 2], [0, 1]])


@dataclass(frozen=True)
class EdgeTimeline:
    """Per-edge first appearance and cumulative net sentiment over time.

    Times are integer seconds. keys encode (edge, time) so a single sorted search
    finds the latest cumulative value of any edge at any time.
    """

    first_time: NDArray[np.int64]
    first_sign: NDArray[np.int8]
    keys: NDArray[np.int64]
    cumulative: NDArray[np.int64]
    origin: int
    stride: int


def _seconds(timestamps: pd.Series) -> NDArray[np.int64]:
    """Return timestamps as whole seconds since the epoch."""
    return pd.to_datetime(timestamps).to_numpy().astype("datetime64[s]").astype(np.int64)


def edge_events(events: pd.DataFrame, edges: SignedEdges) -> pd.DataFrame:
    """Map core events to edge indices with integer-second times, in event order."""
    source, target = node_positions(events, edges.nodes)
    n_nodes = len(edges.nodes)
    edge_keys = np.minimum(edges.sources, edges.targets).astype(np.int64) * n_nodes + np.maximum(
        edges.sources, edges.targets
    )
    edge = pd.Index(edge_keys).get_indexer(
        np.minimum(source, target) * n_nodes + np.maximum(source, target)
    )
    if (edge < 0).any():
        raise ValueError("Every event must belong to a core edge")
    table = pd.DataFrame(
        {
            "edge": edge,
            "time": _seconds(events["timestamp"]),
            "sentiment": events["sentiment"].to_numpy(np.int8),
        }
    )
    if "post_id" in events:
        table["post_id"] = events["post_id"].to_numpy()
    return table


def edge_timeline(table: pd.DataFrame, n_edges: int) -> EdgeTimeline:
    """Return the first appearance, first sign, and cumulative net sign per edge.

    The first sign is the majority of the edge's events at its first timestamp;
    equal counts give zero.
    """
    grouped = (
        table.assign(sentiment=table["sentiment"].astype(np.int64))
        .groupby(["edge", "time"], sort=True)["sentiment"]
        .sum()
    )
    edge = grouped.index.get_level_values("edge").to_numpy(np.int64)
    time = grouped.index.get_level_values("time").to_numpy(np.int64)
    net = grouped.to_numpy(np.int64)
    first = np.r_[True, edge[1:] != edge[:-1]]
    if not np.array_equal(edge[first], np.arange(n_edges)):
        raise ValueError("Every edge needs at least one event")
    origin = int(time.min())
    stride = int(time.max()) - origin + 1
    return EdgeTimeline(
        first_time=time[first],
        first_sign=np.sign(net[first]).astype(np.int8),
        keys=edge * stride + (time - origin),
        cumulative=grouped.groupby(level="edge").cumsum().to_numpy(np.int64),
        origin=origin,
        stride=stride,
    )


def signs_as_of(
    timeline: EdgeTimeline, edge: NDArray[np.intp], time: NDArray[np.int64]
) -> NDArray[np.int8]:
    """Return each edge's majority sign over its events up to and including time.

    Every queried time must be at or after the edge's first appearance.
    """
    if (time < timeline.first_time[edge]).any():
        raise ValueError("An edge was queried before its first appearance")
    keys = edge.astype(np.int64) * timeline.stride + (time - timeline.origin)
    position = np.searchsorted(timeline.keys, keys, side="right") - 1
    return np.sign(timeline.cumulative[position]).astype(np.int8)


def closure_predictions(triangles: NDArray[np.intp], timeline: EdgeTimeline) -> pd.DataFrame:
    """Predict each triangle's closing-edge sign as the product of its wedge signs.

    The closing edge appears last. If several edges appear last together and share
    their first sign, any of them can close with the same outcome; otherwise the
    triangle is an ambiguous tie. Wedge signs use events up to the closing time;
    the closing sign is the closing edge's first sign. Triangles with a zero wedge
    or closing sign are excluded from evaluation and counted by status.
    """
    rows = np.arange(len(triangles))
    times = timeline.first_time[triangles]
    closing_time = times.max(axis=1) if len(triangles) else np.empty(0, dtype=np.int64)
    last = times == closing_time[:, None]
    column = last.argmax(axis=1)
    closing = triangles[rows, column]
    wedge = triangles[rows[:, None], _OTHERS[column]]
    closing_sign = timeline.first_sign[closing]
    consistent = ((timeline.first_sign[triangles] == closing_sign[:, None]) | ~last).all(axis=1)
    wedge_signs = signs_as_of(timeline, wedge.ravel(), np.repeat(closing_time, 2)).reshape(-1, 2)
    status = np.select([~consistent, closing_sign == 0, (wedge_signs == 0).any(axis=1)], [1, 2, 3])
    wedge_type = np.where(status == 0, 1 + (wedge_signs == -1).sum(axis=1), 0)
    return pd.DataFrame(
        {
            "closing_time": closing_time,
            "closing_edge": closing,
            "wedge_edge_a": wedge[:, 0],
            "wedge_edge_b": wedge[:, 1],
            "wedge_sign_a": wedge_signs[:, 0],
            "wedge_sign_b": wedge_signs[:, 1],
            "wedge_type": pd.Categorical.from_codes(wedge_type, categories=("", *WEDGES)),
            "closing_sign": closing_sign,
            "predicted_sign": wedge_signs.prod(axis=1).astype(np.int8),
            "tied_last_edges": last.sum(axis=1),
            "status": pd.Categorical.from_codes(status, categories=STATUSES),
        }
    )


def prediction_metrics(closures: pd.DataFrame) -> dict[str, float]:
    """Score balance predictions with negative closing edges as the class of interest.

    The confusion counts use it as "positive": true_positives are negative closings
    predicted negative, false_positives are positive closings predicted negative.
    """
    status = closures["status"].value_counts()
    evaluated = closures.loc[closures["status"] == "evaluated"]
    actual = evaluated["closing_sign"].to_numpy() == -1
    predicted = evaluated["predicted_sign"].to_numpy() == -1
    tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
    fn, tn = int((actual & ~predicted).sum()), int((~actual & ~predicted).sum())
    recall, specificity = safe_ratio(tp, tp + fn), safe_ratio(tn, tn + fp)
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics = {
        "triangles": len(closures),
        **{f"status_{name}": int(status.get(name, 0)) for name in STATUSES},
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "negative_closing_rate": safe_ratio(int(actual.sum()), len(actual)),
        "accuracy": safe_ratio(tp + tn, len(actual)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision_negative": safe_ratio(tp, tp + fp),
        "recall_negative": recall,
        "specificity": specificity,
        "negative_predictive_value": safe_ratio(tn, tn + fn),
        "mcc": (tp * tn - fp * fn) / denominator if denominator else np.nan,
    }
    for wedge in WEDGES:
        group = (evaluated["wedge_type"] == wedge).to_numpy()
        metrics[f"n_{wedge}"] = int(group.sum())
        metrics[f"negative_rate_{wedge}"] = safe_ratio(int(actual[group].sum()), int(group.sum()))
    return metrics


def wedge_table(closures: pd.DataFrame) -> pd.DataFrame:
    """Summarize closings per wedge type with counts, rates, and lift over the base rate."""
    evaluated = closures.loc[closures["status"] == "evaluated"]
    base = (evaluated["closing_sign"] == -1).mean() if len(evaluated) else np.nan
    rows = []
    for wedge in WEDGES:
        group = evaluated.loc[evaluated["wedge_type"] == wedge]
        negative = int((group["closing_sign"] == -1).sum())
        predicted = 1 if wedge != "+-" else -1
        rows.append(
            {
                "wedge_type": wedge,
                "predicted_sign": predicted,
                "triangles": len(group),
                "negative_closings": negative,
                "positive_closings": len(group) - negative,
                "negative_rate": safe_ratio(negative, len(group)),
                "agreement": safe_ratio(
                    int((group["closing_sign"] == predicted).sum()), len(group)
                ),
                "negative_rate_lift": safe_ratio(safe_ratio(negative, len(group)), base),
            }
        )
    return pd.DataFrame(rows)


def period_metrics(closures: pd.DataFrame, frequency: str = "Q") -> pd.DataFrame:
    """Return prediction metrics for triangles grouped by closing period."""
    periods = pd.to_datetime(closures["closing_time"], unit="s").dt.to_period(frequency)
    rows = [
        {"period": str(period), **prediction_metrics(group)}
        for period, group in closures.groupby(periods, sort=True)
    ]
    return pd.DataFrame(rows)


def shuffle_post_labels(events: pd.DataFrame, *, seed: int) -> NDArray[np.int8]:
    """Return sentiments with labels permuted across posts, in row order.

    The label belongs to the source post, so every row of a post receives the same
    permuted label. Timing, endpoints, and the number of negative posts are preserved.
    """
    codes, _ = pd.factorize(events["post_id"])
    sentiment = events["sentiment"].to_numpy(np.int8)
    labels = np.zeros(codes.max() + 1 if len(codes) else 0, dtype=np.int8)
    labels[codes] = sentiment
    if (labels[codes] != sentiment).any():
        raise ValueError("A post has rows with different sentiment labels")
    return np.random.default_rng(seed).permutation(labels)[codes]


def temporal_null_metrics(
    events: pd.DataFrame,
    table: pd.DataFrame,
    triangles: NDArray[np.intp],
    n_edges: int,
    *,
    null: TemporalNull,
    seeds: tuple[int, ...],
    show_progress: bool = False,
) -> pd.DataFrame:
    """Return prediction metrics for each realization of a temporal null model.

    table is edge_events(events, edges), built once: each realization only replaces
    its sentiment or time column, since rows keep the order of events.
    """
    rows = []
    for realization, null_seed in enumerate(
        tqdm(seeds, desc=null, unit="run", disable=not show_progress)
    ):
        if null == "post_label_shuffle":
            shuffled = table.assign(sentiment=shuffle_post_labels(table, seed=null_seed))
        elif null == "timestamp_shuffle":
            times = shuffle_timestamps(events, mode="global", seed=null_seed)["timestamp"]
            shuffled = table.assign(time=_seconds(times))
        else:
            raise ValueError(f"Unknown temporal null: {null}")
        rows.append(
            {"null": null, "realization": realization, "seed": null_seed}
            | prediction_metrics(closure_predictions(triangles, edge_timeline(shuffled, n_edges)))
        )
    return pd.DataFrame(rows)


def temporal_null_summary(empirical: dict[str, float], nulls: pd.DataFrame) -> pd.DataFrame:
    """Compare empirical prediction metrics with each temporal null ensemble."""
    statistics = [
        "negative_closing_rate",
        "balanced_accuracy",
        "mcc",
        "precision_negative",
        "recall_negative",
        "specificity",
        "accuracy",
        *(f"negative_rate_{wedge}" for wedge in WEDGES),
        "status_evaluated",
        "status_ambiguous_tie",
    ]
    rows = []
    for null, group in nulls.groupby("null", sort=False):
        for statistic in statistics:
            values = group[statistic].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                rows.append(
                    {"null": null, "statistic": statistic}
                    | null_comparison(float(empirical[statistic]), values)
                )
    return pd.DataFrame(rows)


def run_temporal_balance_analysis(
    input_path: Path,
    graph_path: Path,
    *,
    results_dir: Path = Path("results/temporal_balance"),
    realizations: int = 200,
    seed: int = 0,
    frequency: str = "Q",
) -> pd.DataFrame:
    """Save P-O-X closure predictions, metrics, temporal null comparisons, and figures.

    The analysis is undirected, so direction and status theory are not modelled. Temporal
    order supports prediction, not causation, and the probability that an open wedge
    closes is not estimated. Degree-preserving rewiring does not apply: rewired edges
    have no event times.
    """
    if isinstance(realizations, bool) or not isinstance(realizations, int) or realizations < 0:
        raise ValueError("realizations must be a non-negative integer")
    graph = load_graph(graph_path)
    edges = signed_edges(graph)
    triangles = triangle_edges(edges)
    events = filter_core_events(load_interactions(input_path, include_post_id=True), edges.nodes)
    print(
        f"Temporal balance: {len(events):,} core events, {len(triangles):,} triangles",
        flush=True,
    )
    table = edge_events(events, edges)
    timeline = edge_timeline(table, len(edges.signs))
    closures = closure_predictions(triangles, timeline)
    metrics = prediction_metrics(closures)
    wedges = wedge_table(closures)
    periods = period_metrics(closures, frequency)

    null_seeds = {
        null: spawn_seeds([seed, 2 + index], realizations) for index, null in enumerate(NULLS)
    }
    nulls = pd.concat(
        [
            temporal_null_metrics(
                events,
                table,
                triangles,
                len(edges.signs),
                null=null,
                seeds=seeds,
                show_progress=True,
            )
            for null, seeds in null_seeds.items()
            if seeds
        ]
        or [pd.DataFrame(columns=["null"])],
        ignore_index=True,
    )
    summary = temporal_null_summary(metrics, nulls)

    results_dir.mkdir(parents=True, exist_ok=True)
    names = np.array(edges.nodes)
    low = names[np.minimum(edges.sources, edges.targets)]
    high = names[np.maximum(edges.sources, edges.targets)]
    readable = closures.assign(
        closing_time=pd.to_datetime(closures["closing_time"], unit="s"),
        closing_u=low[closures["closing_edge"]],
        closing_v=high[closures["closing_edge"]],
        third_node=[
            next(iter({a, b, c, d} - {u, v}))
            for a, b, c, d, u, v in zip(
                low[closures["wedge_edge_a"]],
                high[closures["wedge_edge_a"]],
                low[closures["wedge_edge_b"]],
                high[closures["wedge_edge_b"]],
                low[closures["closing_edge"]],
                high[closures["closing_edge"]],
                strict=True,
            )
        ],
    ).drop(columns=["closing_edge", "wedge_edge_a", "wedge_edge_b"])
    readable.to_csv(results_dir / "closures.csv.gz", index=False)
    pd.DataFrame([metrics]).to_csv(results_dir / "prediction_metrics.csv", index=False)
    wedges.to_csv(results_dir / "wedge_closures.csv", index=False)
    periods.to_csv(results_dir / "period_metrics.csv", index=False)
    nulls.to_csv(results_dir / "null_realizations.csv", index=False)
    summary.to_csv(results_dir / "null_summary.csv", index=False)

    metadata = {
        "input_path": str(input_path.resolve()),
        "input_sha256": checksum(input_path),
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": checksum(graph_path),
        "core_events": len(events),
        "triangles": len(triangles),
        "realizations": realizations,
        "seed": seed,
        "null_seeds": {null: list(seeds) for null, seeds in null_seeds.items()},
        "period_frequency": frequency,
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy", "pandas")},
    }
    write_json(results_dir / "metadata.json", metadata)
    if metrics["status_evaluated"]:
        for extension in ("png", "pdf"):
            plot_temporal_balance(
                metrics, wedges, nulls, periods, results_dir / f"temporal_balance.{extension}"
            )
    else:
        print("No evaluated triangles; skipping the temporal-balance figure", flush=True)
    print(wedges.to_string(index=False), flush=True)
    print(
        summary.loc[:, ["null", "statistic", "empirical", "mean", "q025", "q975"]].to_string(
            index=False
        ),
        flush=True,
    )
    print(f"Saved temporal-balance results to {results_dir}", flush=True)
    return closures
