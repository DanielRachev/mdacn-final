import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mdacn.data import build_matrices, edge_order_metadata
from mdacn.experiments import run_intervention_sweep, shuffle_edge_signs
from mdacn.ico_pipeline import compute_phi_c
from mdacn.impact import iter_node_impacts
from mdacn.interventions import select_contrarians
from mdacn.rewiring import rewire
from mdacn.signed_structure import signed_edges


def test_budget_grid_changes_do_not_change_paired_trials():
    graph = nx.complete_graph(list("abcdefg"))
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    scores = pd.DataFrame({"node": network.nodes, "degree": 6, "leverage": 0.0})
    options = dict(scores=scores, initial_negative_fraction=0.5, realizations=3, max_steps=20)
    first = run_intervention_sweep(graph, network, budgets=(0, 0.3, 0.6), **options)
    changed = run_intervention_sweep(graph, network, budgets=(0.6, 0.1, 0.3, 0), **options)
    keys = ["realization", "strategy", "budget"]
    common = changed[changed.budget.isin(first.budget.unique())]
    pd.testing.assert_frame_equal(
        first.sort_values(keys).reset_index(drop=True),
        common.sort_values(keys).reset_index(drop=True),
    )
    assert first.groupby("realization").initial_seed.nunique().eq(1).all()
    assert first.groupby("realization").baseline_g_minus.nunique().eq(1).all()


def test_random_targets_are_nested_across_budgets():
    nodes = tuple(f"n{i}" for i in range(30))
    selected = [
        set(select_contrarians(nodes, b, strategy="random", seed=8))
        for b in [0, 0.1, 0.3, 0.5, 0.9, 1]
    ]
    assert all(a <= b for a, b in zip(selected, selected[1:]))


def threshold_rows():
    rows = []
    for realization, baseline, final in [(0, 0.2, 0.2), (1, 0.9, 0.8), (2, 0.9, 0.2)]:
        for budget, value in [(0, baseline), (0.25, final)]:
            rows.append(
                dict(
                    model="signed",
                    f=0.5,
                    strategy="random",
                    realization=realization,
                    initial_seed=realization,
                    baseline_g_minus=baseline,
                    budget=budget,
                    g_minus=value,
                    stop_reason="steady_state",
                )
            )
    return pd.DataFrame(rows)


def test_threshold_summary_separates_success_baseline_and_censoring():
    row = compute_phi_c(threshold_rows()).iloc[0]
    assert (row.runs, row.reached_runs, row.zero_budget_runs, row.censored_runs) == (3, 2, 1, 1)
    assert row.mean_phi_c == pytest.approx(0.125)
    assert row.mean_intervention_phi_c == 0.25
    assert row.intervention_reached_fraction == 0.5
    assert row.reached_fraction == pytest.approx(2 / 3)


def test_independent_budget_seeds_cannot_be_reported_as_paired_thresholds():
    rows = threshold_rows()
    rows.loc[rows.budget.eq(0.25), "initial_seed"] += 10
    with pytest.raises(ValueError, match="one initial seed across budgets"):
        compute_phi_c(rows)


def test_transient_cycle_state_is_not_a_steady_threshold_crossing():
    rows = threshold_rows().query("realization==2").copy()
    rows.loc[rows.budget.eq(0.25), "stop_reason"] = "limit_cycle"
    row = compute_phi_c(rows).iloc[0]
    assert row.reached_runs == 0
    assert row.unresolved_runs == 1
    assert row.censored_runs == 0


def test_seeded_nulls_ignore_edge_and_node_insertion_order():
    records = [
        ("a", "b", 1),
        ("b", "c", -1),
        ("c", "d", 1),
        ("d", "e", 0),
        ("e", "f", 1),
        ("f", "a", -1),
        ("b", "e", 1),
        ("a", "d", 1),
    ]
    first, second = nx.Graph(), nx.Graph()
    for graph, edges in [(first, records), (second, list(reversed(records)))]:
        for u, v, sign in edges:
            graph.add_edge(u, v, sign=sign, n_positive=int(sign >= 0), n_negative=int(sign <= 0))
        nx.set_node_attributes(graph, {n: i // 2 for i, n in enumerate(sorted(graph))}, "community")
    assert edge_order_metadata(first) == edge_order_metadata(second)
    a, b = signed_edges(first), signed_edges(second)
    for field in ["sources", "targets", "signs"]:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    shuffled = [shuffle_edge_signs(graph, seed=7) for graph in [first, second]]
    np.testing.assert_array_equal(
        build_matrices(shuffled[0]).signed.toarray(), build_matrices(shuffled[1]).signed.toarray()
    )
    a, b = [rewire(graph, attempts=300, seed=7) for graph in [first, second]]
    assert a.stats == b.stats
    np.testing.assert_array_equal(
        build_matrices(a.graph).signed.toarray(), build_matrices(b.graph).signed.toarray()
    )


def test_spillover_excludes_direct_fragmentation_and_detects_signed_backfire(monkeypatch):
    import mdacn.impact as impact

    graph = nx.path_graph(list("abcd"))
    nx.set_edge_attributes(graph, 1, "sign")
    monkeypatch.setattr(impact, "initialize_opinions", lambda *a, **k: -np.ones(4, dtype=np.int8))
    rows = pd.DataFrame(
        iter_node_impacts(graph, build_matrices(graph), models=("unsigned",), realizations=1)
    )
    row = rows[rows.node.eq("b")].iloc[0]
    assert row.impact == row.direct_impact == 0.5
    assert row.spillover_impact == 0
    assert row.unpinned_changes == 0

    graph = nx.Graph()
    graph.add_edge("a", "b", sign=-1)
    graph.add_edge("b", "c", sign=1)
    monkeypatch.setattr(impact, "initialize_opinions", lambda *a, **k: np.array([-1, 1, -1]))
    rows = pd.DataFrame(
        iter_node_impacts(graph, build_matrices(graph), models=("signed",), realizations=1)
    )
    row = rows[rows.node.eq("a")].iloc[0]
    assert row.direct_impact == 0
    assert row.spillover_impact == pytest.approx(-1 / 3)
    assert row.unpinned_plus_to_minus == row.unpinned_changes == 1
    assert row.impact == pytest.approx(row.direct_impact + row.spillover_impact)
