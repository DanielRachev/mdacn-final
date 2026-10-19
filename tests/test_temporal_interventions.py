import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from mdacn.temporal_data import TemporalSnapshots
from mdacn.temporal_interventions import simulate_temporal_ico, evaluate_temporal_ico
from mdacn.temporal_nulls import shuffle_timestamps, validate_aggregate_preservation


def test_pinned_stays_positive_and_unpinned_follows_signed_rule():
    matrix = csr_matrix(np.array([[0, -1], [-1, 0]]))
    snapshots = TemporalSnapshots(("a", "b"), (pd.Timestamp("2020-01-01"),), (abs(matrix),), (matrix,), (1,))
    ordinary = simulate_temporal_ico(snapshots, np.array([-1, -1], dtype=np.int8))
    pinned = simulate_temporal_ico(snapshots, np.array([-1, -1], dtype=np.int8), contrarians=("a",))
    assert ordinary["final_state"].tolist() == [-1, -1]
    assert pinned["final_state"].tolist() == [1, -1]
    with pytest.raises(ValueError):
        simulate_temporal_ico(snapshots, np.array([-1, -1]), contrarians=("not-node",))


def test_timestamp_invariants_with_multiple_repeated_events():
    events = pd.DataFrame({"source": ["a", "a", "b", "b", "a"], "target": ["b", "b", "a", "c", "c"], "sentiment": [1, -1, 1, -1, -1], "timestamp": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-01", "2020-01-15", "2020-01-08"])})
    for mode in ("global", "within_source"):
        shuffled = shuffle_timestamps(events, mode=mode, seed=5)
        outcome = validate_aggregate_preservation(events, shuffled, ("a", "b", "c"), mode=mode)
        assert outcome["degree_preserved"] and outcome["signed_edges_preserved"]


def test_temporal_ico_matched_baseline_and_budget():
    events = pd.DataFrame({"source": ["a", "b", "b"], "target": ["b", "c", "a"], "sentiment": [1, -1, 1], "timestamp": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"])})
    outcome = evaluate_temporal_ico(events, ("a", "b", "c"), degree_scores={"a": 1, "b": 2, "c": 1}, budgets=(0., 1.), fractions=(0.5,), n_initializations=1, n_shuffles=1, seed=42)
    assert set(outcome["strategy"]) == {"random", "degree"}
    assert set(outcome["variant"]) == {"real", "global_shuffle"}
    assert (outcome.loc[outcome.budget == 1., "g_minus"] == 0).all()
    assert (outcome.loc[outcome.budget == 0., "pinned_count"] == 0).all()
