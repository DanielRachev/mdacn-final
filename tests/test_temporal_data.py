
"""Tests for temporal network snapshot construction."""

import numpy as np
import pandas as pd
import pytest

from mdacn.temporal_data import build_temporal_snapshots


def events(rows):
    return (pd.DataFrame(rows, columns=["source", "target", "sentiment", "timestamp"])
            .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"])))


def test_repeated_directed_events_collapse_and_tie_remains_unsigned():
    data = events([
        ("A", "B", 1, "2026-01-05"),
        ("B", "A", -1, "2026-01-06"),
        ("A", "C", -1, "2026-01-07"),
        ("C", "C", 1, "2026-01-07"),
        ("A", "outside", 1, "2026-01-07"),
    ])

    snapshots = build_temporal_snapshots(data, ("A", "B", "C"))

    assert len(snapshots.timestamps) == 1
    assert snapshots.event_counts == (3,)
    np.testing.assert_array_equal(snapshots.unsigned[0].toarray(), [[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(snapshots.signed[0].toarray(), [[0, 0, -1], [0, 0, 0], [-1, 0, 0]])


def test_time_windows_include_empty_weeks_and_keep_node_order():
    data = events([
        ("B", "A", 1, "2026-01-05"),
        ("B", "A", -1, "2026-01-19"),
    ])

    result = build_temporal_snapshots(data, ("B", "A", "isolated"))

    assert len(result.timestamps) == 3
    assert result.event_counts == (1, 0, 1)
    assert result.unsigned[1].nnz == 0
    assert result.signed[0][0, 1] == 1
    assert result.signed[2][0, 1] == -1
    assert all(matrix.shape == (3, 3) for matrix in result.signed)


def test_monthly_ablation_and_tie_policy():
    data = events([
        ("A", "B", 1, "2026-01-05"),
        ("A", "B", -1, "2026-01-26"),
    ])

    result = build_temporal_snapshots(data, ("A", "B"), frequency="M", tie_policy="negative")

    assert len(result.signed) == 1
    assert result.signed[0][0, 1] == -1


def test_empty_and_invalid_inputs():
    empty = events([])
    result = build_temporal_snapshots(empty, ("A", "B"))

    assert result.signed == ()

    with pytest.raises(ValueError, match="unique"):
        build_temporal_snapshots(empty, ("A", "A"))

    with pytest.raises(ValueError, match="sentiment"):
        build_temporal_snapshots(events([("A", "B", 0, "2026-01-05")]), ("A", "B"))
