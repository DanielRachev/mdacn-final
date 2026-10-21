"""Tests for temporal activity, incident correlations and dyad sign nulls."""

import pandas as pd
import pytest

from mdacn.temporal_diagnostics import (
    activity_null_experiment,
    correlation_null_experiment,
    dyad_sign_null_experiment,
    incident_edge_correlations,
    node_activity_metrics,
    sample_incident_pairs,
    shuffle_dyad_sentiments,
    summarize_dyad_sign_null,
)
from mdacn.temporal_nulls import shuffle_timestamps

NODES = ("A", "B", "C", "D")


def make_events():
    rows = [
        ("A", "B", 1, "2026-01-05"),
        ("A", "C", -1, "2026-01-06"),
        ("A", "B", -1, "2026-01-07"),
        ("B", "C", 1, "2026-01-08"),
        ("A", "B", 1, "2026-01-12"),
        ("A", "C", -1, "2026-01-13"),
        ("B", "D", 1, "2026-01-14"),
        ("A", "C", 1, "2026-01-19"),
        ("A", "B", -1, "2026-01-20"),
        ("C", "D", -1, "2026-01-21"),
        ("A", "D", 1, "2026-01-26"),
        ("A", "C", -1, "2026-01-27"),
    ]
    df = pd.DataFrame(rows, columns=["source", "target", "sentiment", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_source_burstiness_is_invariant_under_within_source_shuffle():
    events = make_events()
    shuffled = shuffle_timestamps(events, mode="within_source", seed=42)

    first = node_activity_metrics(events, NODES).set_index("node")
    second = node_activity_metrics(shuffled, NODES).set_index("node")
    pd.testing.assert_frame_equal(first, second)


def test_activity_null_runs():
    result = activity_null_experiment(make_events(), NODES, n_shuffles=3, seed=42)

    assert len(result) == 4 * 3
    assert set(result["metric"]) == {"burstiness", "weekly_fano", "active_windows"}
    assert set(result["variant"]) == {"real", "global"}


def test_incident_pairs_share_a_node():
    pairs = sample_incident_pairs(make_events(), NODES, max_pairs=20, seed=42)

    assert len(pairs) > 0
    assert len(pairs) <= 20

    for row in pairs.itertuples(index=False):
        assert row.edge_a != row.edge_b
        assert row.node in row.edge_a
        assert row.node in row.edge_b


def test_incident_correlations_are_bounded():
    events = make_events()
    pairs = sample_incident_pairs(events, NODES, max_pairs=20, seed=42)
    result = incident_edge_correlations(events, NODES, pairs)

    assert len(result) == len(pairs)
    assert result["correlation"].dropna().between(-1.000001, 1.000001).all()


def test_correlation_null_runs():
    results, pairs = correlation_null_experiment(
        make_events(), NODES, n_shuffles=2, max_pairs=20, seed=42
    )

    assert len(results) == 5
    assert set(results["variant"]) == {"real", "global", "within_source"}
    assert len(pairs) > 0


def test_dyad_shuffle_preserves_each_pair_sign_counts():
    events = make_events()
    original = events.copy(deep=True)
    shuffled = shuffle_dyad_sentiments(events, seed=42)

    for df in (events, shuffled):
        df["dyad"] = [tuple(sorted((a, b))) for a, b in zip(df["source"], df["target"])]

    before = events.groupby(["dyad", "sentiment"]).size()
    after = shuffled.groupby(["dyad", "sentiment"]).size()

    pd.testing.assert_series_equal(before, after)
    pd.testing.assert_frame_equal(events.drop(columns="dyad"), original)


def test_dyad_sign_null_runs():
    result = dyad_sign_null_experiment(make_events(), NODES, n_shuffles=3, seed=42)
    summary = summarize_dyad_sign_null(result)

    assert len(result) == 4 * 9
    assert set(result["variant"]) == {"real", "within_dyad"}
    assert set(summary["variant"]).issubset({"within_dyad"})
    assert summary["null_mean"].dropna().between(0, 1).all()


def test_invalid_counts():
    with pytest.raises(ValueError, match="positive"):
        activity_null_experiment(make_events(), NODES, n_shuffles=0)

    with pytest.raises(ValueError, match="positive"):
        dyad_sign_null_experiment(make_events(), NODES, n_shuffles=0)
