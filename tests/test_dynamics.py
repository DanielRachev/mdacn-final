"""Behavioral checks for Daniel's synchronous NCO engine."""

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mdacn.dynamics import (
    detect_attractor,
    initialize_opinions,
    simulate_nco,
)


def adjacency(graph):
    return csr_matrix(nx.to_scipy_sparse_array(graph, dtype=np.int8))


def test_initialization_is_seeded_exact_and_does_not_change_global_rng():
    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    first = initialize_opinions(10, 0.25, seed=7)
    assert np.random.random() == expected_next
    np.testing.assert_array_equal(first, initialize_opinions(10, 0.25, seed=7))
    assert first.dtype == np.int8
    assert np.count_nonzero(first == -1) == 3
    assert np.all(initialize_opinions(8, 0, seed=1) == 1)
    assert np.all(initialize_opinions(8, 1, seed=1) == -1)
    assert initialize_opinions(0, 0.5, seed=1).shape == (0,)


def test_unsigned_majority_and_self_vote():
    # Leaves tie because self counts; the center adopts its two neighbors' opinion.
    matrix = adjacency(nx.path_graph(3))
    result = simulate_nco(matrix, np.array([1, -1, 1]), model="unsigned", max_steps=1)
    np.testing.assert_array_equal(result.final_state, [1, 1, 1])


@pytest.mark.parametrize("model", ["unsigned", "signed"])
def test_tied_vote_preserves_state(model):
    state = np.array([1, -1], dtype=np.int8)
    result = simulate_nco(adjacency(nx.path_graph(2)), state, model=model, max_steps=1)
    np.testing.assert_array_equal(result.final_state, state)


def test_updates_are_synchronous_and_preserve_input():
    matrix = adjacency(nx.cycle_graph(4))
    original_matrix = matrix.copy()
    state = np.array([1, -1, 1, -1], dtype=np.int8)
    result = simulate_nco(matrix, state, model="unsigned", max_steps=1)
    np.testing.assert_array_equal(result.final_state, [-1, 1, -1, 1])
    np.testing.assert_array_equal(state, [1, -1, 1, -1])
    assert not np.shares_memory(result.final_state, state)
    assert (matrix != original_matrix).nnz == 0


def test_negative_edge_favors_disagreement():
    matrix = -adjacency(nx.path_graph(3))
    result = simulate_nco(matrix, np.ones(3, dtype=np.int8), model="signed", max_steps=1)
    np.testing.assert_array_equal(result.final_state, [1, -1, 1])
    result = simulate_nco(
        -adjacency(nx.path_graph(2)), np.array([1, -1]), model="signed", max_steps=1
    )
    np.testing.assert_array_equal(result.final_state, [1, -1])


@pytest.mark.parametrize("model", ["unsigned", "signed"])
def test_isolated_node_preserves_state(model):
    graph = nx.path_graph(3)
    graph.add_node(3)
    matrix = adjacency(graph) * (-1 if model == "signed" else 1)
    result = simulate_nco(matrix, np.array([1, 1, 1, -1]), model=model, max_steps=10)
    assert result.final_state[-1] == -1
    state = np.array([-1, 1], dtype=np.int8)
    result = simulate_nco(csr_matrix((2, 2)), state, model=model, max_steps=10)
    np.testing.assert_array_equal(result.final_state, state)
    assert (result.stop_reason, result.steps) == ("steady_state", 1)


def test_steady_state_limit_cycle_and_update_limit():
    matrix = adjacency(nx.cycle_graph(4))
    steady = simulate_nco(matrix, np.ones(4, dtype=np.int8), model="unsigned", max_steps=10)
    assert (steady.stop_reason, steady.steps, steady.cycle_length) == ("steady_state", 1, None)
    state = np.array([1, -1, 1, -1], dtype=np.int8)
    cycle = simulate_nco(matrix, state, model="unsigned", max_steps=2)
    assert (cycle.stop_reason, cycle.steps, cycle.cycle_length) == ("limit_cycle", 2, 2)
    np.testing.assert_array_equal(cycle.final_state, state)
    limited = simulate_nco(matrix, state, model="unsigned", max_steps=1)
    assert (limited.stop_reason, limited.steps, limited.cycle_length) == ("max_steps", 1, None)
    np.testing.assert_array_equal(limited.final_state, -state)
    zero = simulate_nco(matrix, state, model="unsigned", max_steps=0)
    assert (zero.stop_reason, zero.steps) == ("max_steps", 0)
    np.testing.assert_array_equal(zero.final_state, state)
    assert not np.shares_memory(zero.final_state, state)
    signed_cycle = simulate_nco(
        -adjacency(nx.complete_graph(3)), np.ones(3, dtype=np.int8), model="signed", max_steps=3
    )
    assert (signed_cycle.stop_reason, signed_cycle.cycle_length) == ("limit_cycle", 2)


def test_history_attractor_detection():
    a, b, c = np.array([1, 1]), np.array([-1, 1]), np.array([1, -1])
    assert detect_attractor([]) is None
    assert detect_attractor([a]) is None
    assert detect_attractor([a, b, c]) is None
    assert detect_attractor([a, b, b.copy()]) == ("steady_state", None)
    assert detect_attractor([a, b, a.copy()]) == ("limit_cycle", 2)
    assert detect_attractor([a, b, c, a.copy()]) == ("limit_cycle", 3)


@pytest.mark.parametrize("model,sign", [("unsigned", 1), ("signed", -1)])
def test_hub_votes_do_not_overflow_int8(model, sign):
    matrix = sign * adjacency(nx.star_graph(256))
    state = np.ones(257, dtype=np.int8)
    state[0] = -1
    result = simulate_nco(matrix, state, model=model, max_steps=1)
    assert result.final_state[0] == sign


@pytest.mark.parametrize("seed", range(10))
def test_sparse_simulation_matches_scalar_reference(seed):
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.choice([-1, 0, 1], size=(8, 8)), k=1)
    weights = upper + upper.T
    initial = rng.choice([-1, 1], size=8).astype(np.int8)
    for model, matrix in (("signed", weights), ("unsigned", np.abs(weights))):
        state = initial.copy()
        history = [state.copy()]
        reason, period = "max_steps", None
        for step in range(1, 21):
            next_state = []
            for i in range(8):
                vote = int(state[i]) + sum(int(matrix[i, j]) * int(state[j]) for j in range(8))
                next_state.append(1 if vote > 0 else -1 if vote < 0 else state[i])
            state = np.array(next_state, dtype=np.int8)
            history.append(state.copy())
            attractor = detect_attractor(history)
            if attractor:
                reason, period = attractor
                break
        result = simulate_nco(csr_matrix(matrix), initial, model=model, max_steps=20)
        np.testing.assert_array_equal(result.final_state, state)
        assert (result.steps, result.stop_reason, result.cycle_length) == (step, reason, period)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError, match="negative_fraction"):
        initialize_opinions(4, fraction, seed=0)


@pytest.mark.parametrize(
    "matrix,state,model,max_steps",
    [
        (csr_matrix((2, 3)), [1, 1], "unsigned", 10),
        (csr_matrix([[0, 1], [0, 0]]), [1, 1], "unsigned", 10),
        (csr_matrix(np.eye(2)), [1, 1], "unsigned", 10),
        (csr_matrix([[0, 2], [2, 0]]), [1, 1], "signed", 10),
        (csr_matrix([[0, -1], [-1, 0]]), [1, 1], "unsigned", 10),
        (csr_matrix([[0, np.nan], [np.nan, 0]]), [1, 1], "signed", 10),
        (csr_matrix((2, 2)), [1, 0], "unsigned", 10),
        (csr_matrix((2, 2)), [1], "unsigned", 10),
        (csr_matrix((2, 2)), [[1, -1]], "unsigned", 10),
        (csr_matrix((2, 2)), [1, -1], "temporal", 10),
        (csr_matrix((2, 2)), [1, -1], "unsigned", -1),
        (csr_matrix((2, 2)), [1, -1], "unsigned", 1.5),
    ],
)
def test_invalid_simulation_inputs_rejected(matrix, state, model, max_steps):
    with pytest.raises(ValueError):
        simulate_nco(matrix, np.array(state), model=model, max_steps=max_steps)
