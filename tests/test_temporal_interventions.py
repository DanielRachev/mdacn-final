import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from mdacn.temporal_data import TemporalSnapshots
from mdacn.temporal_interventions import (
    evaluate_temporal_ico,
    simulate_temporal_ico,
    summarize_temporal_ico,
)
from mdacn.temporal_nulls import shuffle_timestamps, validate_aggregate_preservation


def test_pinned_stays_positive_and_unpinned_follows_signed_rule():
    matrix = csr_matrix(np.array([[0, -1], [-1, 0]]))
    snapshots = TemporalSnapshots(
        ("a", "b"), (pd.Timestamp("2020-01-01"),), (abs(matrix),), (matrix,), (1,)
    )
    ordinary = simulate_temporal_ico(snapshots, np.array([-1, -1], dtype=np.int8))
    pinned = simulate_temporal_ico(snapshots, np.array([-1, -1], dtype=np.int8), contrarians=("a",))
    assert ordinary["final_state"].tolist() == [-1, -1]
    assert pinned["final_state"].tolist() == [1, -1]
    with pytest.raises(ValueError):
        simulate_temporal_ico(snapshots, np.array([-1, -1]), contrarians=("not-node",))


def test_timestamp_invariants_with_multiple_repeated_events():
    events = pd.DataFrame(
        {
            "source": ["a", "a", "b", "b", "a"],
            "target": ["b", "b", "a", "c", "c"],
            "sentiment": [1, -1, 1, -1, -1],
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-08", "2020-01-01", "2020-01-15", "2020-01-08"]
            ),
        }
    )
    for mode in ("global", "within_source"):
        shuffled = shuffle_timestamps(events, mode=mode, seed=5)
        outcome = validate_aggregate_preservation(events, shuffled, ("a", "b", "c"), mode=mode)
        assert outcome["degree_preserved"] and outcome["signed_edges_preserved"]


def test_temporal_ico_matched_baseline_and_budget():
    events = pd.DataFrame(
        {
            "source": ["a", "b", "b"],
            "target": ["b", "c", "a"],
            "sentiment": [1, -1, 1],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
        }
    )
    outcome = evaluate_temporal_ico(
        events,
        ("a", "b", "c"),
        degree_scores={"a": 1, "b": 2, "c": 1},
        budgets=(0.0, 1.0),
        fractions=(0.5,),
        n_initializations=1,
        n_shuffles=1,
        seed=42,
    )
    assert set(outcome["strategy"]) == {"random", "degree"}
    assert set(outcome["variant"]) == {"real", "global_shuffle"}
    assert (outcome.loc[outcome.budget == 1.0, "g_minus"] == 0).all()
    assert (outcome.loc[outcome.budget == 0.0, "pinned_count"] == 0).all()


def test_ico_bootstrap_includes_shared_timeline_variation_and_pairs_strategies():
    records = []
    for initialization in range(4):
        for strategy, real, shuffled in (
            ("degree", 0.5, (0.1, 0.9)),
            ("random", 0.75, (0.2, 1.0)),
        ):
            for variant, shuffle, value in (
                ("real", -1, real),
                ("global_shuffle", 0, shuffled[0]),
                ("global_shuffle", 1, shuffled[1]),
            ):
                records.append(
                    dict(
                        f=0.7,
                        budget=0.05,
                        strategy=strategy,
                        variant=variant,
                        initialization=initialization,
                        shuffle=shuffle,
                        g_minus=value,
                    )
                )

    _, paired, _ = summarize_temporal_ico(pd.DataFrame(records), n_bootstrap=1000, seed=42)
    paired = paired.set_index("comparison")
    timing = paired.loc["degree:real-minus-shuffled"]
    assert timing["mean_difference"] == pytest.approx(0)
    assert timing["ci_low"] == pytest.approx(-0.4)
    assert timing["ci_high"] == pytest.approx(0.4)
    strategies = paired.loc["global_shuffle:degree-minus-random"]
    assert strategies["ci_low"] == pytest.approx(-0.1)
    assert strategies["ci_high"] == pytest.approx(-0.1)
