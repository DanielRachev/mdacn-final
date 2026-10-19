
"""Tests for temporal unsigned and signed NCO simulations."""

import numpy as np
import pandas as pd
import pytest

from mdacn.dynamics import simulate_nco
from mdacn.temporal_data import build_temporal_snapshots
from mdacn.temporal_dynamics import simulate_temporal_nco


def make_snapshots(rows, nodes=("A", "B", "C")):
    df = pd.DataFrame(rows, columns=["source", "target", "sentiment", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return build_temporal_snapshots(df, nodes)


def test_temporal_matches_static_nco_for_one_step():
    snapshots = make_snapshots([
        ("A", "B", -1, "2026-01-05"),
        ("A", "C", 1, "2026-01-05"),
    ])
    initial = np.array([1, 1, -1], dtype=np.int8)

    for model in ("signed", "unsigned"):
        temporal = simulate_temporal_nco(snapshots, model=model, initial_state=initial)
        matrix = snapshots.signed[0] if model == "signed" else snapshots.unsigned[0]
        static = simulate_nco(matrix, initial, model=model, max_steps=1)

        np.testing.assert_array_equal(temporal.final_state, static.final_state)


def test_signed_and_unsigned_can_produce_different_results():
    snapshots = make_snapshots([
        ("A", "B", -1, "2026-01-05"),
        ("A", "C", 1, "2026-01-05"),
    ])
    initial = np.array([1, 1, -1], dtype=np.int8)

    signed = simulate_temporal_nco(snapshots, model="signed", initial_state=initial)
    unsigned = simulate_temporal_nco(snapshots, model="unsigned", initial_state=initial)

    assert signed.final_state[0] == -1
    assert unsigned.final_state[0] == 1


def test_simulation_continues_after_temporary_stability():
    snapshots = make_snapshots([
        ("A", "B", 1, "2026-01-05"),
        ("A", "B", -1, "2026-01-12"),
        ("A", "C", 1, "2026-01-12"),
    ])
    initial = np.array([1, 1, -1], dtype=np.int8)

    result = simulate_temporal_nco(snapshots, initial_state=initial, record_states=True)

    assert result.steps == 2
    assert result.changed_count[0] == 0
    assert result.changed_count[1] > 0
    assert result.states.shape == (3, 3)
    np.testing.assert_array_equal(result.states[0], initial)
    np.testing.assert_array_equal(result.states[-1], result.final_state)


def test_random_initialization_is_reproducible():
    snapshots = make_snapshots([
        ("A", "B", 1, "2026-01-05"),
    ])

    first = simulate_temporal_nco(snapshots, negative_fraction=0.4, seed=42)
    second = simulate_temporal_nco(snapshots, negative_fraction=0.4, seed=42)

    np.testing.assert_array_equal(first.initial_state, second.initial_state)
    np.testing.assert_array_equal(first.final_state, second.final_state)


def test_fraction_history_and_changes():
    snapshots = make_snapshots([
        ("A", "B", -1, "2026-01-05"),
        ("A", "C", 1, "2026-01-05"),
    ])
    initial = np.array([1, 1, -1], dtype=np.int8)

    result = simulate_temporal_nco(snapshots, initial_state=initial)

    assert len(result.positive_fraction) == result.steps + 1
    assert len(result.negative_fraction) == result.steps + 1
    assert len(result.changed_count) == result.steps
    assert result.states is None
    np.testing.assert_allclose(result.positive_fraction + result.negative_fraction, 1.0)
    assert result.changed_count[0] == np.count_nonzero(initial != result.final_state)


def test_initial_state_is_not_modified():
    snapshots = make_snapshots([
        ("A", "B", -1, "2026-01-05"),
        ("A", "C", 1, "2026-01-05"),
    ])
    initial = np.array([1, 1, -1], dtype=np.int8)
    original = initial.copy()

    simulate_temporal_nco(snapshots, initial_state=initial)

    np.testing.assert_array_equal(initial, original)


def test_invalid_initialization_and_model():
    snapshots = make_snapshots([("A", "B", 1, "2026-01-05")])

    with pytest.raises(ValueError, match="Provide"):
        simulate_temporal_nco(snapshots)

    with pytest.raises(ValueError, match="Provide either"):
        simulate_temporal_nco(snapshots, initial_state=np.ones(3), negative_fraction=0.5, seed=42)

    with pytest.raises(ValueError, match="model"):
        simulate_temporal_nco(snapshots, model="invalid", negative_fraction=0.5, seed=42)


def test_empty_snapshots_return_initial_state():
    snapshots = make_snapshots([], nodes=("A", "B"))
    initial = np.array([1, -1], dtype=np.int8)

    result = simulate_temporal_nco(snapshots, initial_state=initial, record_states=True)

    assert result.steps == 0
    assert result.changed_count.size == 0
    assert result.states.shape == (1, 2)
    np.testing.assert_array_equal(result.final_state, initial)
