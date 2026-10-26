"""Temporal P-O-X closure ordering, ties, as-of signs, metrics, and nulls."""

import json

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mdacn.data import aggregate_edges, export_graph
from mdacn.signed_structure import signed_edges
from mdacn.signed_triads import triangle_edges
from mdacn.temporal_balance import (
    closure_predictions,
    edge_events,
    edge_timeline,
    prediction_metrics,
    run_temporal_balance_analysis,
    shuffle_post_labels,
    signs_as_of,
    wedge_table,
)


def _events(rows: list[tuple]) -> pd.DataFrame:
    """Rows of (source, target, sentiment, day, post_id)."""
    frame = pd.DataFrame(rows, columns=["source", "target", "sentiment", "day", "post_id"])
    frame["timestamp"] = pd.Timestamp("2015-01-01") + pd.to_timedelta(frame["day"], unit="D")
    frame["sentiment"] = frame["sentiment"].astype(np.int8)
    return frame.drop(columns="day")


def _closures(events: pd.DataFrame) -> pd.DataFrame:
    graph = aggregate_edges(events)
    edges = signed_edges(graph)
    timeline = edge_timeline(edge_events(events, edges), len(edges.signs))
    closures = closure_predictions(triangle_edges(edges), timeline)
    names = np.array(edges.nodes)
    low = names[np.minimum(edges.sources, edges.targets)]
    high = names[np.maximum(edges.sources, edges.targets)]
    closures["closing"] = [f"{low[edge]}{high[edge]}" for edge in closures["closing_edge"]]
    return closures


def test_last_edge_closes_with_signs_known_at_closing_time():
    closures = _closures(
        _events(
            [
                ("a", "b", 1, 1, "p1"),
                ("b", "a", -1, 2, "p2"),
                ("a", "b", -1, 2, "p3"),
                ("a", "c", 1, 3, "p4"),
                ("b", "c", 1, 4, "p5"),
                ("a", "b", 1, 5, "p6"),
                ("a", "b", 1, 6, "p7"),
            ]
        )
    )
    row = closures.iloc[0]
    assert row["closing"] == "bc"
    assert row["status"] == "evaluated"
    assert {row["wedge_sign_a"], row["wedge_sign_b"]} == {-1, 1}
    assert row["wedge_type"] == "+-"
    assert row["predicted_sign"] == -1
    assert row["closing_sign"] == 1


def test_simultaneous_last_edges():
    same = _closures(
        _events(
            [
                ("a", "b", -1, 1, "p1"),
                ("c", "a", 1, 2, "p2"),
                ("c", "b", 1, 2, "p2"),
            ]
        )
    ).iloc[0]
    assert same["status"] == "evaluated"
    assert same["tied_last_edges"] == 2
    assert same["wedge_type"] == "+-"
    assert same["predicted_sign"] == -1 and same["closing_sign"] == 1

    mixed = _closures(
        _events([("a", "b", 1, 1, "p1"), ("c", "a", 1, 2, "p2"), ("c", "b", -1, 2, "p3")])
    ).iloc[0]
    assert mixed["status"] == "ambiguous_tie"
    assert mixed["wedge_type"] == ""

    all_three = _closures(
        _events([("a", "b", -1, 1, "p1"), ("a", "c", -1, 1, "p1"), ("b", "c", -1, 1, "p2")])
    ).iloc[0]
    assert all_three["status"] == "evaluated"
    assert all_three["wedge_type"] == "--"


def test_zero_signs_are_excluded():
    zero_closing = _closures(
        _events(
            [
                ("a", "b", 1, 1, "p1"),
                ("a", "c", 1, 2, "p2"),
                ("b", "c", 1, 3, "p3"),
                ("c", "b", -1, 3, "p4"),
            ]
        )
    ).iloc[0]
    assert zero_closing["status"] == "zero_closing"
    zero_wedge = _closures(
        _events(
            [
                ("a", "b", 1, 1, "p1"),
                ("b", "a", -1, 1, "p2"),
                ("a", "c", 1, 2, "p3"),
                ("b", "c", 1, 3, "p4"),
            ]
        )
    ).iloc[0]
    assert zero_wedge["status"] == "zero_wedge"


def test_signs_as_of_rejects_queries_before_first_event():
    events = _events([("a", "b", 1, 2, "p1"), ("a", "b", -1, 3, "p2"), ("a", "b", -1, 4, "p3")])
    edges = signed_edges(aggregate_edges(events))
    timeline = edge_timeline(edge_events(events, edges), 1)
    day = 24 * 3600
    start = int(timeline.first_time[0])
    assert signs_as_of(
        timeline, np.array([0, 0, 0]), np.array([start, start + day, start + 2 * day])
    ).tolist() == [1, 0, -1]
    with pytest.raises(ValueError, match="before"):
        signs_as_of(timeline, np.array([0]), np.array([start - 1]))


def test_every_edge_needs_events():
    graph = nx.Graph()
    graph.add_edge("a", "b", sign=1)
    graph.add_edge("b", "c", sign=1)
    edges = signed_edges(graph)
    with pytest.raises(ValueError, match="at least one event"):
        edge_timeline(edge_events(_events([("a", "b", 1, 1, "p")]), edges), 2)


def test_metrics_confusion_counts():
    closures = pd.DataFrame(
        {
            "status": ["evaluated"] * 5 + ["zero_wedge"],
            "wedge_type": ["+-", "+-", "++", "++", "--", ""],
            "predicted_sign": [-1, -1, 1, 1, 1, 1],
            "closing_sign": [-1, 1, -1, 1, 1, 1],
        }
    )
    metrics = prediction_metrics(closures)
    assert metrics["status_evaluated"] == 5 and metrics["status_zero_wedge"] == 1
    assert metrics["recall_negative"] == pytest.approx(0.5)
    assert metrics["precision_negative"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert metrics["negative_rate_+-"] == pytest.approx(0.5)
    table = wedge_table(closures).set_index("wedge_type")
    assert table.loc["+-", "agreement"] == pytest.approx(0.5)
    assert table.loc["++", "negative_rate_lift"] == pytest.approx(0.5 / 0.4)


def test_always_positive_prediction_scores_chance():
    closures = pd.DataFrame(
        {
            "status": ["evaluated"] * 4,
            "wedge_type": ["++"] * 4,
            "predicted_sign": [1] * 4,
            "closing_sign": [1, 1, 1, -1],
        }
    )
    metrics = prediction_metrics(closures)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert np.isnan(metrics["mcc"])


def test_post_label_shuffle_keeps_posts_consistent():
    events = _events(
        [
            ("a", "b", 1, 1, "p1"),
            ("a", "c", 1, 1, "p1"),
            ("b", "c", -1, 2, "p2"),
            ("c", "a", 1, 3, "p3"),
        ]
    )
    shuffled = events.assign(sentiment=shuffle_post_labels(events, seed=1))
    assert shuffled.groupby("post_id")["sentiment"].nunique().max() == 1
    per_post = shuffled.drop_duplicates("post_id")["sentiment"]
    assert sorted(per_post) == [-1, 1, 1]
    mixed = events.assign(sentiment=np.array([1, -1, -1, 1], dtype=np.int8))
    with pytest.raises(ValueError, match="different sentiment"):
        shuffle_post_labels(mixed, seed=0)


def test_pipeline_saves_outputs(tmp_path):
    rng = np.random.default_rng(0)
    graph = nx.gnm_random_graph(12, 40, seed=1)
    rows = []
    for index, (u, v) in enumerate(graph.edges()):
        for repeat in range(2):
            rows.append(
                {
                    "SOURCE_SUBREDDIT": f"s{u}",
                    "TARGET_SUBREDDIT": f"s{v}",
                    "POST_ID": f"p{index}_{repeat}",
                    "TIMESTAMP": (
                        pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 200)))
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "LINK_SENTIMENT": int(rng.choice([1, 1, 1, -1])),
                }
            )
    input_path = tmp_path / "body.tsv"
    pd.DataFrame(rows).to_csv(input_path, sep="\t", index=False)
    from mdacn.data import load_interactions

    graph_path = tmp_path / "core.graphml"
    export_graph(aggregate_edges(load_interactions(input_path)), graph_path)
    results_dir = tmp_path / "pox"
    closures = run_temporal_balance_analysis(
        input_path, graph_path, results_dir=results_dir, realizations=3, seed=1
    )
    assert len(closures) == sum(nx.triangles(graph).values()) // 3
    for name in (
        "closures.csv.gz",
        "prediction_metrics.csv",
        "wedge_closures.csv",
        "period_metrics.csv",
        "null_realizations.csv",
        "null_summary.csv",
        "metadata.json",
        "temporal_balance.png",
        "temporal_balance.pdf",
    ):
        assert (results_dir / name).is_file()
    nulls = pd.read_csv(results_dir / "null_realizations.csv")
    assert nulls["null"].value_counts().to_dict() == {
        "post_label_shuffle": 3,
        "timestamp_shuffle": 3,
    }
    saved = pd.read_csv(results_dir / "closures.csv.gz")
    assert len(saved) == len(closures)
    assert {"closing_u", "closing_v", "third_node"} <= set(saved.columns)
    metadata = json.loads((results_dir / "metadata.json").read_text())
    assert len(metadata["null_seeds"]["timestamp_shuffle"]) == 3
