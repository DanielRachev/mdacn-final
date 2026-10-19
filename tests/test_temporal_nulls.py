
"""Tests for temporal timestamp-shuffling null models."""

import numpy as np
import pandas as pd
import pytest

from mdacn.temporal_data import build_temporal_snapshots
from mdacn.temporal_nulls import (shuffle_timestamps, validate_timestamp_shuffle,
                                  generate_timestamp_nulls, evaluate_timestamp_nulls)


def make_events():
    return pd.DataFrame([
        ("A", "B", 1, "2026-01-05"),
        ("A", "B", -1, "2026-01-06"),
        ("A", "C", 1, "2026-01-13"),
        ("A", "D", -1, "2026-01-20"),
        ("B", "C", 1, "2026-01-07"),
        ("B", "D", -1, "2026-01-14"),
        ("C", "D", 1, "2026-01-21"),
        ("D", "A", -1, "2026-01-22"),
    ], columns=["source", "target", "sentiment", "timestamp"]).assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))


def test_global_shuffle_preserves_events_and_timestamps():
    original = make_events()
    shuffled = shuffle_timestamps(original, seed=42)

    validate_timestamp_shuffle(original, shuffled)
    assert not original["timestamp"].equals(shuffled["timestamp"])
    pd.testing.assert_frame_equal(original, make_events())


def test_global_shuffle_preserves_aggregate_signed_network():
    original = make_events()
    shuffled = shuffle_timestamps(original, seed=42)
    nodes = ("A", "B", "C", "D")

    real = build_temporal_snapshots(original, nodes)
    null = build_temporal_snapshots(shuffled, nodes)

    real_unsigned = sum(real.unsigned)
    null_unsigned = sum(null.unsigned)

    # Weekly edge presence may change even though lifetime events are preserved.
    assert real_unsigned.shape == null_unsigned.shape
    validate_timestamp_shuffle(original, shuffled)

    real_lifetime = original.groupby(["source", "target", "sentiment"]).size()
    null_lifetime = shuffled.groupby(["source", "target", "sentiment"]).size()
    pd.testing.assert_series_equal(real_lifetime, null_lifetime)


def test_within_source_shuffle_preserves_source_activity():
    original = make_events()
    shuffled = shuffle_timestamps(original, mode="within_source", seed=123)

    validate_timestamp_shuffle(original, shuffled, mode="within_source")

    for source in original["source"].unique():
        before = original.loc[original["source"] == source, "timestamp"].value_counts()
        after = shuffled.loc[shuffled["source"] == source, "timestamp"].value_counts()
        pd.testing.assert_series_equal(before.sort_index(), after.sort_index())


def test_shuffle_is_reproducible_and_does_not_modify_input():
    original = make_events()
    untouched = original.copy(deep=True)

    first = shuffle_timestamps(original, seed=42)
    second = shuffle_timestamps(original, seed=42)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(original, untouched)


def test_multiple_shuffles():
    original = make_events()
    shuffles = list(generate_timestamp_nulls(original, n_shuffles=5, seed=42))

    assert len(shuffles) == 5
    for shuffled in shuffles:
        validate_timestamp_shuffle(original, shuffled)

    assert any(not shuffles[0]["timestamp"].equals(other["timestamp"]) for other in shuffles[1:])


def test_paired_monte_carlo_experiments():
    original = make_events()
    results = evaluate_timestamp_nulls(original, ("A", "B", "C", "D"), n_initializations=3, n_shuffles=4, seed=42)

    assert len(results) == 3 * (1 + 4)
    assert (results["variant"] == "real").sum() == 3
    assert (results["variant"] == "shuffled").sum() == 12
    assert results["final_positive_fraction"].between(0, 1).all()
    assert results["final_negative_fraction"].between(0, 1).all()
    np.testing.assert_allclose(results["final_positive_fraction"] + results["final_negative_fraction"], 1)


def test_invalid_configuration():
    original = make_events()

    with pytest.raises(ValueError, match="mode"):
        shuffle_timestamps(original, mode="invalid")

    with pytest.raises(ValueError, match="positive"):
        list(generate_timestamp_nulls(original, n_shuffles=0))

    with pytest.raises(ValueError, match="positive"):
        evaluate_timestamp_nulls(original, ("A", "B", "C", "D"), n_initializations=0)
