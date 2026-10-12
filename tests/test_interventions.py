import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from mdacn.contracts import NetworkMatrices
from mdacn.data import build_matrices
from mdacn.experiments import run_intervention_sweep
from mdacn.interventions import (
    opposing_cluster_size,
    select_contrarians,
    simulate_ico,
)


def test_select_contrarians_handles_random_and_ranked_strategies() -> None:
    nodes = ("a", "b", "c", "d")

    random_choice = select_contrarians(nodes, 0.5, strategy="random", seed=7)
    assert len(random_choice) == 2
    assert set(random_choice).issubset(set(nodes))
    assert len(set(random_choice)) == len(random_choice)

    degree_scores = {"a": 0.8, "b": 0.9, "c": 0.7, "d": 0.9}
    degree_choice = select_contrarians(nodes, 0.5, strategy="degree", seed=7, scores=degree_scores)
    assert degree_choice == ("b", "d")

    leverage_scores = {"a": 0.2, "b": 0.8, "c": 0.7, "d": 0.4}
    leverage_choice = select_contrarians(
        nodes, 0.75, strategy="leverage", seed=7, scores=leverage_scores
    )
    assert leverage_choice == ("b", "c", "d")


def test_simulate_ico_keeps_contrarians_fixed_at_positive_one() -> None:
    nodes = ("a", "b", "c")
    adjacency = csr_matrix(
        np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )
    )
    initial_state = np.array([1, -1, 1], dtype=np.int8)

    result = simulate_ico(adjacency, initial_state, nodes=nodes, contrarians=("b",), max_steps=10)

    assert result.stop_reason in {"steady_state", "max_steps"}
    assert result.final_state[1] == 1
    assert result.final_state.tolist() == [1, 1, 1]


def test_opposing_cluster_size_tracks_the_largest_negative_component() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("d", "e")])
    state = np.array([-1, -1, -1, 1, -1], dtype=np.int8)

    assert opposing_cluster_size(graph, ("a", "b", "c", "d", "e"), state) == 3


def test_run_intervention_sweep_records_cluster_decay() -> None:
    graph = nx.path_graph(4)
    graph = nx.relabel_nodes(graph, {0: "a", 1: "b", 2: "c", 3: "d"})
    nodes = ("a", "b", "c", "d")
    unsigned = csr_matrix(nx.to_scipy_sparse_array(graph, dtype=float, weight=None))
    network = NetworkMatrices(nodes=nodes, unsigned=unsigned, signed=unsigned.copy())

    results = run_intervention_sweep(
        graph,
        network,
        budgets=(0.25,),
        initial_negative_fraction=0.5,
        realizations=1,
        seed=0,
        max_steps=20,
    )

    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) >= {"strategy", "budget", "cluster_size", "stop_reason", "steps"}
    assert len(results) == 3
    assert set(results["strategy"]) == {"random", "degree", "leverage"}


def test_intervention_starts_from_shared_settled_baseline(monkeypatch):
    import mdacn.experiments as experiments

    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    initial = np.array([-1, -1, 1, -1], dtype=np.int8)
    monkeypatch.setattr(experiments, "initialize_opinions", lambda *args, **kwargs: initial.copy())
    starts = []

    def capture(adjacency, state, **kwargs):
        starts.append(state.copy())
        return simulate_ico(adjacency, state, **kwargs)

    monkeypatch.setattr(experiments, "simulate_ico", capture)
    results = run_intervention_sweep(
        graph,
        network,
        budgets=(0, 0.25),
        initial_negative_fraction=0.75,
        realizations=1,
        max_steps=20,
    )

    assert len(starts) == 6
    assert all(state.tolist() == [-1, -1, -1, -1] for state in starts)
    assert (results["baseline_negative_fraction"] == 1).all()
    assert (results["baseline_steps"] == 2).all()
    assert (results.loc[results["budget"] == 0, "cluster_size"] == 4).all()
    targeted = results.loc[(results["budget"] == 0.25) & (results["strategy"] == "degree")]
    assert targeted["cluster_size"].tolist() == [2]
    assert initial.tolist() == [-1, -1, 1, -1]


@pytest.mark.parametrize(
    ("max_steps", "expected_status"), [(0, "max_steps"), (1, "max_steps"), (20, "limit_cycle")]
)
def test_intervention_rejects_nonsteady_baseline(monkeypatch, max_steps, expected_status):
    import mdacn.experiments as experiments

    graph = nx.complete_bipartite_graph(3, 3)
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    initial = np.array([-1, -1, -1, 1, 1, 1], dtype=np.int8)
    monkeypatch.setattr(experiments, "initialize_opinions", lambda *args, **kwargs: initial.copy())

    def unexpected_intervention(*args, **kwargs):
        pytest.fail("Intervention must not start from a nonsteady baseline")

    monkeypatch.setattr(experiments, "simulate_ico", unexpected_intervention)
    with pytest.raises(
        RuntimeError, match=f"Baseline did not reach a steady state: {expected_status}"
    ):
        run_intervention_sweep(
            graph,
            network,
            budgets=(0.25,),
            initial_negative_fraction=0.5,
            realizations=1,
            max_steps=max_steps,
        )
