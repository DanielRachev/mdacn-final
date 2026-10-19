
"""Tests for temporal characterization and experimental analysis."""

import numpy as np
import pandas as pd
import pytest

from mdacn.temporal_analysis import (
    temporal_activity, sign_transitions, compare_temporal_trajectories,
    paired_null_summary, resolution_ablation,
)


NODES = ("A", "B", "C", "D")


def make_events():
    rows = [
        ("A", "B", 1, "2026-01-05"),
        ("A", "B", 1, "2026-01-06"),
        ("A", "C", -1, "2026-01-06"),
        ("A", "B", -1, "2026-01-12"),
        ("A", "C", -1, "2026-01-13"),
        ("B", "D", 1, "2026-01-13"),
        ("A", "B", 1, "2026-01-26"),
        ("C", "D", -1, "2026-01-27"),
    ]
    df = pd.DataFrame(rows, columns=["source", "target", "sentiment", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_temporal_activity_includes_inactive_weeks():
    activity = temporal_activity(make_events(), NODES)

    assert len(activity) == 4
    assert activity["total"].tolist() == [3, 3, 0, 2]
    assert activity["positive"].tolist() == [2, 1, 0, 1]
    assert activity["negative"].tolist() == [1, 2, 0, 1]
    assert activity["active_nodes"].tolist() == [3, 4, 0, 4]
    assert np.isnan(activity["negative_fraction"].iloc[2])


def test_sign_transitions_only_count_consecutive_active_windows():
    transitions = sign_transitions(make_events(), NODES)

    observed = {(row.previous, row.current): row.count for row in transitions.itertuples()}

    assert observed[("+", "-")] == 1
    assert observed[("-", "-")] == 1
    assert len(observed) == 2
    assert np.allclose(transitions.groupby("previous")["probability"].sum(), 1)


def test_temporal_trajectories_are_paired_and_reproducible():
    first = compare_temporal_trajectories(make_events(), NODES, n_repetitions=3, seed=42)
    second = compare_temporal_trajectories(make_events(), NODES, n_repetitions=3, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3 * 2 * 5
    assert first["positive_fraction"].between(0, 1).all()

    initial = first.loc[first["step"] == 0].pivot(index="repetition", columns="variant", values="positive_fraction")
    np.testing.assert_allclose(initial["real"], initial["shuffled"])


def test_paired_null_summary():
    results = pd.DataFrame([
        (0, "real", 0.7), (0, "shuffled", 0.5), (0, "shuffled", 0.6),
        (1, "real", 0.8), (1, "shuffled", 0.6), (1, "shuffled", 0.7),
        (2, "real", 0.6), (2, "shuffled", 0.5), (2, "shuffled", 0.4),
    ], columns=["initialization", "variant", "final_positive_fraction"])

    summary = paired_null_summary(results)

    assert summary["n_initializations"] == 3
    assert summary["difference"] == pytest.approx(0.15)
    assert summary["ci_low"] <= summary["difference"] <= summary["ci_high"]


def test_resolution_ablation_runs():
    results = resolution_ablation(
        make_events(), NODES, frequencies=("W-SUN", "M"),
        n_initializations=3, n_shuffles=2, seed=42,
    )

    assert len(results) == 4
    assert set(results["frequency"]) == {"W-SUN", "M"}
    assert set(results["metric"]) == {"final_positive_fraction", "total_switches"}
    assert np.isfinite(results["difference"]).all()


def test_invalid_repetitions():
    with pytest.raises(ValueError, match="positive"):
        compare_temporal_trajectories(make_events(), NODES, n_repetitions=0)
