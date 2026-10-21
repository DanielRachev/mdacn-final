"""Tests for temporal robustness and label-invariant metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from mdacn.temporal_analysis import (
    largest_opinion_component,
    lifetime_topology,
    opinion_metrics,
    paired_fraction_summary,
    run_fraction_sweep,
    shuffle_sentiments,
    sign_persistence_null,
    sign_persistence_summary,
)

NODES = ("A", "B", "C", "D")


def make_events():
    rows = [
        ("A", "B", 1, "2026-01-05"),
        ("B", "C", -1, "2026-01-06"),
        ("A", "B", -1, "2026-01-12"),
        ("B", "C", 1, "2026-01-13"),
        ("C", "D", 1, "2026-01-19"),
        ("A", "D", -1, "2026-01-26"),
    ]
    df = pd.DataFrame(rows, columns=["source", "target", "sentiment", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_label_invariant_metrics():
    graph = csr_matrix(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    state = np.array([1, 1, -1, -1], dtype=np.int8)
    metrics = opinion_metrics(graph, state)
    flipped = opinion_metrics(graph, -state)

    assert metrics["absolute_magnetization"] == 0
    assert metrics["minority_fraction"] == 0.5
    assert metrics["largest_opinion_gcc_fraction"] == 0.5
    assert metrics["absolute_magnetization"] == flipped["absolute_magnetization"]
    assert metrics["largest_opinion_gcc_fraction"] == flipped["largest_opinion_gcc_fraction"]
    assert largest_opinion_component(graph, state, 1) == 2


def test_lifetime_topology_is_undirected():
    graph = lifetime_topology(make_events(), NODES)
    assert graph.shape == (4, 4)
    assert (graph != graph.T).nnz == 0
    assert graph[0, 1] == 1


def test_fraction_sweep_is_reproducible_and_paired():
    first = run_fraction_sweep(
        make_events(), NODES, fractions=(0.3, 0.5), n_initializations=2, n_shuffles=2, seed=42
    )
    second = run_fraction_sweep(
        make_events(), NODES, fractions=(0.3, 0.5), n_initializations=2, n_shuffles=2, seed=42
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2 * 2 * (1 + 2 * 2)
    assert first["absolute_magnetization"].between(0, 1).all()
    assert first["largest_opinion_gcc_fraction"].between(0, 1).all()

    paired = first.groupby(["f", "initialization"])["initial_seed"].nunique()
    assert (paired == 1).all()


def test_paired_fraction_summary():
    results = run_fraction_sweep(
        make_events(), NODES, fractions=(0.3,), n_initializations=3, n_shuffles=2, seed=42
    )
    summary = paired_fraction_summary(results)

    assert len(summary) == 2 * 4
    assert (summary["n_initializations"] == 3).all()
    assert np.isfinite(summary["difference"]).all()


def test_sign_shuffling_preserves_counts_and_events():
    original = make_events()
    shuffled = shuffle_sentiments(original, seed=42)

    assert sorted(original["sentiment"]) == sorted(shuffled["sentiment"])
    pd.testing.assert_frame_equal(
        original.drop(columns="sentiment"), shuffled.drop(columns="sentiment")
    )


def test_sign_persistence_null():
    transitions = sign_persistence_null(make_events(), NODES, n_shuffles=3, seed=42)
    summary = sign_persistence_summary(transitions)

    assert set(transitions["variant"]) == {"real", "sign_shuffled"}
    assert len(transitions) == 4 * 9
    assert len(summary) == 9


def test_invalid_fraction_sweep():
    with pytest.raises(ValueError, match="fractions"):
        run_fraction_sweep(
            make_events(), NODES, fractions=(-0.1,), n_initializations=1, n_shuffles=1
        )
