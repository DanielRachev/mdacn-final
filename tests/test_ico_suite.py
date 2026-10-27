import networkx as nx
import numpy as np
import pandas as pd
import pytest

import mdacn.experiments as experiments
from mdacn.backfire import backfire_metrics
from mdacn.data import build_matrices
from mdacn.experiments import ALL_STRATEGIES, run_intervention_sweep
from mdacn.ico_pipeline import compute_phi_c
from mdacn.interventions import RANKED_STRATEGIES, select_contrarians
from mdacn.plots import plot_phi_c_comparison


@pytest.mark.parametrize("strategy", RANKED_STRATEGIES)
def test_select_contrarians_ranks_every_score_strategy(strategy) -> None:
    nodes = ("a", "b", "c", "d")
    scores = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7}

    selection = select_contrarians(nodes, 0.5, strategy=strategy, seed=0, scores=scores)

    assert selection == ("b", "d")


def test_select_contrarians_rejects_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        select_contrarians(("a", "b"), 0.5, strategy="popularity", seed=0)


def _signed_pair():
    graph = nx.Graph()
    graph.add_edge("a", "b", sign=-1)
    graph.add_edge("b", "c", sign=1)
    return build_matrices(graph)


def test_backfire_counts_negative_links_and_repelled_neighbours() -> None:
    network = _signed_pair()
    baseline = np.array([1, 1, 1], dtype=np.int8)
    final = np.array([1, -1, 1], dtype=np.int8)

    metrics = backfire_metrics(
        network,
        contrarians=("a",),
        baseline_state=baseline,
        final_state=final,
        model="signed",
    )

    assert metrics["negative_influence_links"] == 1
    assert metrics["repelled_neighbours"] == 1
    assert metrics["repelled_share"] == 1.0


def test_backfire_ignores_unsigned_runs_and_neighbours_that_did_not_flip() -> None:
    network = _signed_pair()
    baseline = np.array([1, 1, 1], dtype=np.int8)

    unsigned = backfire_metrics(
        network, contrarians=("a",), baseline_state=baseline, final_state=baseline, model="unsigned"
    )
    steady = backfire_metrics(
        network, contrarians=("a",), baseline_state=baseline, final_state=baseline, model="signed"
    )

    assert unsigned == {
        "negative_influence_links": 0,
        "repelled_neighbours": 0,
        "repelled_share": 0,
    }
    assert steady["negative_influence_links"] == 1
    assert steady["repelled_neighbours"] == 0


def test_backfire_requires_core_contrarians() -> None:
    network = _signed_pair()
    state = np.array([1, 1, 1], dtype=np.int8)

    with pytest.raises(ValueError, match="Contrarians must be core nodes"):
        backfire_metrics(
            network,
            contrarians=("missing",),
            baseline_state=state,
            final_state=state,
            model="signed",
        )


def test_all_seven_strategies_run_on_the_signed_model() -> None:
    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    nodes = network.nodes
    scores = pd.DataFrame(
        {
            "node": list(nodes),
            "degree": [1, 2, 2, 1],
            "leverage": [-1 / 3, 1 / 6, 1 / 6, -1 / 3],
            "betweenness": [0, 2 / 3, 2 / 3, 0],
            "eigenvector": [0.5, 0.5, 0.5, 0.5],
        }
    )
    strategy_scores = {
        "signed_leverage": {node: 0.0 for node in nodes},
        "temporal_bridge": {node: 0.1 for node in nodes},
    }

    results = run_intervention_sweep(
        graph,
        network,
        scores=scores,
        strategies=ALL_STRATEGIES,
        strategy_scores=strategy_scores,
        model="signed",
        budgets=(0.0, 0.25),
        initial_negative_fraction=0.5,
        realizations=1,
        max_steps=20,
    )

    assert len(results) == len(ALL_STRATEGIES) * 2
    assert set(results["strategy"]) == set(ALL_STRATEGIES)
    assert set(results["model"]) == {"signed"}
    assert {"baseline_g_minus", "delta_g_minus", "backfired", "repelled_share"} <= set(results)


def test_sweep_requires_scores_for_requested_strategies() -> None:
    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    nodes = network.nodes
    scores = pd.DataFrame(
        {
            "node": list(nodes),
            "degree": [1, 2, 2, 1],
            "leverage": [-1 / 3, 1 / 6, 1 / 6, -1 / 3],
        }
    )

    with pytest.raises(ValueError, match="Missing scores for \\['temporal_bridge'\\]"):
        run_intervention_sweep(
            graph,
            network,
            scores=scores,
            strategies=("random", "temporal_bridge"),
            budgets=(0.25,),
            initial_negative_fraction=0.5,
            realizations=1,
        )


def test_signed_sweep_settles_and_intervenes_with_the_signed_model(monkeypatch) -> None:
    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    nodes = network.nodes
    scores = pd.DataFrame(
        {
            "node": list(nodes),
            "degree": [1, 2, 2, 1],
            "leverage": [-1 / 3, 1 / 6, 1 / 6, -1 / 3],
        }
    )
    calls = []

    def capture(name):
        original = getattr(experiments, f"simulate_{name}")

        def wrapper(matrix, state, **kwargs):
            calls.append((name, kwargs.get("model"), matrix is network.signed))
            return original(matrix, state, **kwargs)

        return wrapper

    monkeypatch.setattr(experiments, "simulate_nco", capture("nco"))
    monkeypatch.setattr(experiments, "simulate_ico", capture("ico"))
    run_intervention_sweep(
        graph,
        network,
        scores=scores,
        model="signed",
        strategies=("random", "degree"),
        budgets=(0.25,),
        initial_negative_fraction=0.5,
        realizations=1,
        max_steps=20,
    )

    assert [call[0] for call in calls] == ["nco", "ico", "ico"]
    assert all(model == "signed" and signed for _, model, signed in calls)


def test_sweep_rejects_an_unknown_model() -> None:
    graph = nx.path_graph(["a", "b"])
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    scores = pd.DataFrame({"node": ["a", "b"], "degree": [1, 1], "leverage": [0.0, 0.0]})

    with pytest.raises(ValueError, match="model must be 'unsigned' or 'signed'"):
        run_intervention_sweep(
            graph,
            network,
            scores=scores,
            model="magnetic",
            budgets=(0,),
            initial_negative_fraction=0.5,
            realizations=1,
        )


def _phi_frame(g_minus, budgets, strategy="degree", model="unsigned"):
    return pd.DataFrame(
        {
            "model": [model] * len(budgets),
            "f": [0.5] * len(budgets),
            "strategy": [strategy] * len(budgets),
            "realization": [0] * len(budgets),
            "budget": list(budgets),
            "g_minus": list(g_minus),
        }
    )


def test_phi_c_reports_the_first_threshold_crossing() -> None:
    results = _phi_frame([0.9, 0.7, 0.4, 0.2], [0.0, 0.05, 0.1, 0.25])

    phi_c = compute_phi_c(results, threshold=0.5)

    row = phi_c.iloc[0]
    assert row.mean_phi_c == pytest.approx(0.1)
    assert row.reached_fraction == 1.0
    assert row.status == "reached"


def test_phi_c_distinguishes_zero_from_not_reached() -> None:
    below = compute_phi_c(_phi_frame([0.3, 0.2], [0.0, 0.05]), threshold=0.5)
    never = compute_phi_c(_phi_frame([0.9, 0.8], [0.0, 0.05]), threshold=0.5)

    assert below.iloc[0].mean_phi_c == 0.0
    assert below.iloc[0].status == "reference below threshold at Phi = 0"
    assert np.isnan(never.iloc[0].mean_phi_c)
    assert never.iloc[0].reached_runs == 0
    assert never.iloc[0].status == "not reached in the tested range"


def test_phi_c_requires_budget_and_component_columns() -> None:
    with pytest.raises(ValueError, match="requires g_minus"):
        compute_phi_c(pd.DataFrame({"budget": [0.0]}))


def test_phi_c_comparison_figure_is_saved(tmp_path) -> None:
    import matplotlib.pyplot as plt

    phi_c = pd.DataFrame(
        {
            "model": ["unsigned", "unsigned", "signed", "signed"],
            "f": [0.5, 0.5, 0.5, 0.5],
            "strategy": ["degree", "signed_leverage", "degree", "signed_leverage"],
            "mean_phi_c": [0.1, 0.05, np.nan, 0.2],
        }
    )
    path = tmp_path / "phi_c.png"

    plot_phi_c_comparison(phi_c, path)

    assert path.stat().st_size > 1000
    assert not plt.get_fignums()
