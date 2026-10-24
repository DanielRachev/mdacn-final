"""Tests for static-versus-temporal NCO comparisons."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from mdacn.dynamics import simulate_nco
from mdacn.temporal_analysis import (
    run_static_temporal_comparison,
    simulate_static_fixed_horizon,
    summarize_static_temporal,
)

NODES = ("A", "B", "C", "D")


def make_events():
    rows = [
        ("A", "B", 1, "2026-01-05"),
        ("B", "C", -1, "2026-01-06"),
        ("A", "C", 1, "2026-01-12"),
        ("C", "D", -1, "2026-01-13"),
        ("A", "D", 1, "2026-01-19"),
        ("B", "D", 1, "2026-01-20"),
    ]
    df = pd.DataFrame(rows, columns=["source", "target", "sentiment", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def make_static_matrix():
    return csr_matrix(
        [
            [0, 1, 1, 1],
            [1, 0, -1, 1],
            [1, -1, 0, -1],
            [1, 1, -1, 0],
        ]
    )


def test_static_single_step_matches_existing_nco():
    matrix = make_static_matrix()
    initial = np.array([1, 1, -1, -1], dtype=np.int8)

    ours = simulate_static_fixed_horizon(matrix, initial, steps=1)
    existing = simulate_nco(matrix, initial, model="signed", max_steps=1)

    np.testing.assert_array_equal(ours["final_state"], existing.final_state)


def test_static_fixed_horizon_and_stability():
    matrix = csr_matrix((3, 3))
    initial = np.array([1, -1, 1], dtype=np.int8)
    result = simulate_static_fixed_horizon(matrix, initial, steps=5)

    np.testing.assert_array_equal(result["final_state"], initial)
    np.testing.assert_array_equal(result["changed_count"], [0, 0, 0, 0, 0])
    assert result["first_fixed_point"] == 1
    assert result["first_repeated_state"] == 1


def test_static_does_not_modify_initial_state():
    initial = np.array([1, 1, -1, -1], dtype=np.int8)
    saved = initial.copy()
    simulate_static_fixed_horizon(make_static_matrix(), initial, steps=4)
    np.testing.assert_array_equal(initial, saved)


def test_paired_experiment_shapes_and_reproducibility():
    first = run_static_temporal_comparison(
        make_events(),
        NODES,
        make_static_matrix(),
        fractions=(0.3, 0.5),
        n_initializations=2,
        n_shuffles=2,
        seed=42,
    )
    second = run_static_temporal_comparison(
        make_events(),
        NODES,
        make_static_matrix(),
        fractions=(0.3, 0.5),
        n_initializations=2,
        n_shuffles=2,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2 * 2 * (2 + 2)
    assert set(first["variant"]) == {"static", "temporal_real", "temporal_shuffled"}
    assert (first.groupby(["f", "initialization"])["initial_seed"].nunique() == 1).all()
    assert (first.groupby(["f", "initialization"])["steps"].nunique() == 1).all()
    assert first["disagreement_with_static"].between(0, 1).all()
    assert first["minority_lcc_fraction"].between(0, 1).all()


def test_static_disagreement_is_zero():
    results = run_static_temporal_comparison(
        make_events(),
        NODES,
        make_static_matrix(),
        fractions=(0.5,),
        n_initializations=2,
        n_shuffles=2,
    )
    static = results.loc[results["variant"] == "static"]

    assert (static["disagreement_with_static"] == 0).all()


def test_summary():
    results = run_static_temporal_comparison(
        make_events(),
        NODES,
        make_static_matrix(),
        fractions=(0.5,),
        n_initializations=3,
        n_shuffles=2,
    )
    summary = summarize_static_temporal(results)

    assert len(summary) == 3 * 5
    assert summary["mean"].notna().all()


def test_invalid_configuration():
    with pytest.raises(ValueError, match="nonnegative"):
        simulate_static_fixed_horizon(make_static_matrix(), np.ones(4), steps=-1)

    with pytest.raises(ValueError, match="shape"):
        run_static_temporal_comparison(
            make_events(), NODES, csr_matrix((3, 3)), n_initializations=1, n_shuffles=1
        )


def test_identical_graph_update_has_zero_disagreement():
    matrix = csr_matrix([[0, 1], [1, 0]])
    initial = np.array([1, -1], dtype=np.int8)
    static = simulate_static_fixed_horizon(matrix, initial, steps=3)

    from mdacn.temporal_data import TemporalSnapshots
    from mdacn.temporal_dynamics import simulate_temporal_nco

    snapshots = TemporalSnapshots(
        nodes=("A", "B"),
        timestamps=tuple(pd.Timestamp("2026-01-05") + pd.Timedelta(weeks=i) for i in range(3)),
        unsigned=(matrix, matrix, matrix),
        signed=(matrix, matrix, matrix),
        event_counts=(1, 1, 1),
    )
    temporal = simulate_temporal_nco(snapshots, initial_state=initial)

    np.testing.assert_array_equal(static["final_state"], temporal.final_state)
    np.testing.assert_array_equal(static["changed_count"], temporal.changed_count)
