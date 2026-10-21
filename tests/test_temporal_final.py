"""Tests for final temporal robustness experiments."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from mdacn.temporal_analysis import (
    compare_sign_persistence,
    minority_connectivity,
    paired_bootstrap_summary,
    run_minority_experiment,
    shuffle_source_sentiments,
    summarize_sign_persistence,
)

NODES = ("A", "B", "C", "D")


def make_events():
    rows = [
        ("A", "B", 1, "2026-01-05"),
        ("A", "C", -1, "2026-01-06"),
        ("A", "D", 1, "2026-01-07"),
        ("B", "C", -1, "2026-01-07"),
        ("A", "B", -1, "2026-01-12"),
        ("A", "C", 1, "2026-01-13"),
        ("A", "D", -1, "2026-01-14"),
        ("B", "C", -1, "2026-01-15"),
        ("C", "D", 1, "2026-01-20"),
    ]
    df = pd.DataFrame(rows, columns=["source", "target", "sentiment", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_minority_connectivity_connected():
    graph = csr_matrix(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    result = minority_connectivity(graph, np.array([-1, -1, 1, 1]))

    assert result["minority_count"] == 2
    assert result["minority_lcc"] == 2
    assert result["minority_lcc_fraction"] == 1.0
    assert result["minority_component_count"] == 1


def test_minority_connectivity_fragmented():
    graph = csr_matrix(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    )
    result = minority_connectivity(graph, np.array([-1, 1, -1, 1, 1]))

    assert result["minority_count"] == 2
    assert result["minority_lcc"] == 1
    assert result["minority_lcc_fraction"] == 0.5
    assert result["minority_component_count"] == 2
    assert result["minority_isolated_fraction"] == 1.0


def test_source_shuffle_preserves_source_sign_counts():
    original = make_events()
    shuffled = shuffle_source_sentiments(original, seed=42)

    pd.testing.assert_series_equal(original["source"], shuffled["source"])
    pd.testing.assert_series_equal(original["target"], shuffled["target"])
    pd.testing.assert_series_equal(original["timestamp"], shuffled["timestamp"])

    for source in original["source"].unique():
        before = original.loc[original["source"] == source, "sentiment"].value_counts()
        after = shuffled.loc[shuffled["source"] == source, "sentiment"].value_counts()
        pd.testing.assert_series_equal(before.sort_index(), after.sort_index())


def test_sign_persistence_comparison():
    results = compare_sign_persistence(make_events(), NODES, n_shuffles=3, seed=42)
    summary = summarize_sign_persistence(results)

    assert len(results) == (1 + 3 + 3) * 9
    assert len(summary) == 2 * 6
    assert set(summary["variant"]) == {"global", "within_source"}
    assert set(summary["previous"]) == {"+", "-"}
    assert summary["null_mean"].dropna().between(0, 1).all()


def test_minority_experiment_is_reproducible():
    first = run_minority_experiment(
        make_events(), NODES, fractions=(0.5,), n_initializations=2, n_shuffles=2, seed=42
    )
    second = run_minority_experiment(
        make_events(), NODES, fractions=(0.5,), n_initializations=2, n_shuffles=2, seed=42
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2 * (1 + 2 * 2)
    assert first["minority_lcc_fraction"].between(0, 1).all()


def test_paired_bootstrap():
    rows = []

    for initialization in range(4):
        rows.append(
            {
                "f": 0.5,
                "initialization": initialization,
                "variant": "real",
                "shuffle": -1,
                "score": 0.8,
            }
        )

        for shuffle in range(3):
            rows.append(
                {
                    "f": 0.5,
                    "initialization": initialization,
                    "variant": "global",
                    "shuffle": shuffle,
                    "score": 0.6,
                }
            )

    results = pd.DataFrame(rows)
    first = paired_bootstrap_summary(results, metrics=("score",), n_bootstrap=100, seed=42)
    second = paired_bootstrap_summary(results, metrics=("score",), n_bootstrap=100, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["difference"].iloc[0] == pytest.approx(0.2)
    assert first["ci_low"].iloc[0] == pytest.approx(0.2)
    assert first["ci_high"].iloc[0] == pytest.approx(0.2)


def test_invalid_bootstrap_inputs():
    with pytest.raises(ValueError, match="Invalid bootstrap"):
        paired_bootstrap_summary(pd.DataFrame(), metrics=("score",), n_bootstrap=0)
