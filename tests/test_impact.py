import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mdacn.data import build_matrices
from mdacn.dynamics import simulate_nco
from mdacn.impact import iter_node_impacts
from mdacn.interventions import simulate_ico


def test_signed_ico_repels_neighbors_and_keeps_pin():
    graph = nx.complete_graph(["a", "b", "c"])
    nx.set_edge_attributes(graph, -1, "sign")
    network = build_matrices(graph)
    initial = np.ones(3, dtype=np.int8)
    signed = simulate_ico(
        network.signed,
        initial,
        nodes=network.nodes,
        contrarians=("a",),
        model="signed",
        max_steps=10,
    )
    unsigned = simulate_ico(
        network.unsigned, initial, nodes=network.nodes, contrarians=("a",), max_steps=10
    )
    assert signed.final_state.tolist() == [1, -1, -1]
    assert signed.stop_reason == "steady_state"
    assert unsigned.final_state.tolist() == [1, 1, 1]
    assert initial.tolist() == [1, 1, 1]


@pytest.mark.parametrize("model", ["unsigned", "signed"])
def test_unpinned_ico_matches_nco_cycles_and_limits(model):
    graph = nx.complete_bipartite_graph(3, 3)
    nx.set_edge_attributes(graph, -1 if model == "signed" else 1, "sign")
    network = build_matrices(graph)
    initial = np.array([1, 1, 1, -1, -1, -1], dtype=np.int8)
    matrix = getattr(network, model)
    for limit in (0, 1, 20):
        expected = simulate_nco(matrix, initial, model=model, max_steps=limit)
        actual = simulate_ico(
            matrix, initial, nodes=network.nodes, contrarians=(), model=model, max_steps=limit
        )
        np.testing.assert_array_equal(actual.final_state, expected.final_state)
        assert (actual.steps, actual.stop_reason, actual.cycle_length) == (
            expected.steps,
            expected.stop_reason,
            expected.cycle_length,
        )


def test_single_node_impacts_pair_settled_baseline_and_use_gcc(monkeypatch):
    import mdacn.impact as impact

    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    initial = np.array([-1, -1, 1, -1], dtype=np.int8)
    monkeypatch.setattr(impact, "initialize_opinions", lambda *a, **k: initial.copy())
    starts = []

    def capture(matrix, state, **kwargs):
        starts.append(state.copy())
        return simulate_ico(matrix, state, **kwargs)

    monkeypatch.setattr(impact, "simulate_ico", capture)
    runs = pd.DataFrame(
        iter_node_impacts(graph, build_matrices(graph), realizations=2, max_steps=20)
    )
    assert len(runs) == 16
    assert set(runs.node) == set(graph)
    assert runs.groupby(["model", "realization"]).node.nunique().eq(len(graph)).all()
    assert all(state.tolist() == [-1, -1, -1, -1] for state in starts)
    assert runs.groupby("realization").initial_seed.nunique().eq(1).all()
    middle = runs.loc[runs.node == "b"]
    assert middle.baseline_g_minus.eq(1).all()
    assert middle.intervention_g_minus.eq(0.5).all()
    assert middle.intervention_s1.eq(0.75).all()
    assert middle.impact.eq(0.5).all()
    assert runs.eligible.all()
    assert initial.tolist() == [-1, -1, 1, -1]


def test_nonsteady_baselines_are_retained_but_not_intervened(monkeypatch):
    import mdacn.impact as impact

    graph = nx.complete_bipartite_graph(3, 3)
    nx.set_edge_attributes(graph, 1, "sign")
    initial = np.array([-1, -1, -1, 1, 1, 1], dtype=np.int8)
    monkeypatch.setattr(impact, "initialize_opinions", lambda *a, **k: initial.copy())
    monkeypatch.setattr(impact, "simulate_ico", lambda *a, **k: pytest.fail("Baseline cycles"))
    runs = pd.DataFrame(
        iter_node_impacts(graph, build_matrices(graph), models=("unsigned",), realizations=1)
    )
    assert runs.baseline_stop_reason.eq("limit_cycle").all()
    assert runs.intervention_stop_reason.eq("not_run").all()
    assert runs.impact.isna().all()
    assert not runs.eligible.any()


def test_all_nodes_include_positive_nodes_and_share_one_baseline(monkeypatch):
    import mdacn.impact as impact

    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    initial = np.array([-1, -1, 1, 1], dtype=np.int8)
    monkeypatch.setattr(impact, "initialize_opinions", lambda *a, **k: initial.copy())
    baselines = []

    def capture(*args, **kwargs):
        baselines.append(kwargs["model"])
        return simulate_nco(*args, **kwargs)

    monkeypatch.setattr(impact, "simulate_nco", capture)
    runs = pd.DataFrame(iter_node_impacts(graph, build_matrices(graph), realizations=2))
    assert baselines == ["unsigned", "signed"] * 2
    assert len(runs) == 16
    positive = runs[runs.baseline_node_opinion == 1]
    assert set(positive.node) == {"c", "d"}
    assert positive.impact.eq(0).all()
    assert positive.eligible.all()


def test_empty_core_is_rejected():
    graph = nx.Graph()
    with pytest.raises(ValueError, match="Core must contain"):
        list(iter_node_impacts(graph, build_matrices(graph)))
