"""Negative incident ratio, sign-shuffle reference, binning, and community rates."""

import json

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mdacn.data import export_graph
from mdacn.experiments import shuffle_edge_signs
from mdacn.signed_structure import (
    community_sign_rates,
    concentration_summary,
    negative_incident_ratios,
    ratio_bin_edges,
    ratio_distribution,
    ratio_histogram,
    run_incident_sign_analysis,
    shuffled_negative_ratios,
)


def _signed_graph() -> nx.Graph:
    """Two triangles joined by a negative bridge, with one zero-sign edge."""
    graph = nx.Graph()
    signs = {
        ("a", "b"): 1,
        ("a", "c"): -1,
        ("b", "c"): 0,
        ("c", "d"): -1,
        ("d", "e"): 1,
        ("d", "f"): 1,
        ("e", "f"): 1,
    }
    for (u, v), sign in signs.items():
        graph.add_edge(u, v, sign=sign)
    for node in graph:
        graph.nodes[node]["community"] = 0 if node in "abc" else 1
    return graph


def test_negative_incident_ratio_counts():
    table = negative_incident_ratios(_signed_graph()).set_index("node")
    assert table.index.tolist() == sorted("abcdef")
    assert table.loc[
        "c", ["degree", "positive_degree", "negative_degree", "zero_degree"]
    ].tolist() == [
        3,
        0,
        2,
        1,
    ]
    assert table.loc["c", "r_neg"] == pytest.approx(2 / 3)
    assert table.loc["e", "r_neg"] == 0
    assert (
        table["degree"] == table[["positive_degree", "negative_degree", "zero_degree"]].sum(axis=1)
    ).all()
    assert table["community"].to_dict() == {node: int(node not in "abc") for node in "abcdef"}
    assert table["p_upper"].between(0, 1).all()
    assert table.loc["e", "p_upper"] == 1


def test_isolated_nodes_are_rejected():
    graph = _signed_graph()
    graph.add_node("z")
    with pytest.raises(ValueError, match="isolated"):
        negative_incident_ratios(graph)


def test_shuffles_replay_through_shuffle_edge_signs():
    graph = _signed_graph()
    ratios, seeds = shuffled_negative_ratios(graph, realizations=5, seed=3)
    again, same_seeds = shuffled_negative_ratios(graph, realizations=5, seed=3)
    np.testing.assert_array_equal(ratios, again)
    assert seeds == same_seeds
    for row, shuffle_seed in zip(ratios, seeds, strict=True):
        replay = negative_incident_ratios(shuffle_edge_signs(graph, seed=shuffle_seed))
        np.testing.assert_allclose(row, replay["r_neg"])
    degree = negative_incident_ratios(graph)["degree"].to_numpy()
    assert np.allclose((ratios * degree).sum(axis=1), 4)


def test_right_closed_bins_keep_exact_fractions_together():
    edges = ratio_bin_edges(np.array([]), scheme="width", width=0.05)
    values = np.array([0, 1 / 10, 2 / 20, 3 / 30, 0.1000001, 1e-4, 1.0])
    counts = ratio_histogram(values, edges)[0]
    assert counts[0] == 1
    assert counts[2] == 3
    assert counts[3] == 1
    assert counts[1] == 1
    assert counts[-1] == 1
    assert counts.sum() == values.size


def test_quantile_bins_merge_ties_and_cover_unit_interval():
    reference = np.array([[0, 0.1, 0.1, 0.1, 0.1, 0.5]])
    edges = ratio_bin_edges(reference, scheme="quantile", n_bins=4)
    assert edges[0] == 0 and edges[-1] == 1
    assert np.all(np.diff(edges) > 0)
    with pytest.raises(ValueError, match="width"):
        ratio_bin_edges(reference, scheme="width", width=0.3)


def test_distribution_probabilities_and_density():
    empirical = np.array([0, 0, 0.1, 0.3])
    reference = np.array([[0, 0.1, 0.1, 0.3], [0.1, 0.1, 0.1, 0.1]])
    edges = ratio_bin_edges(reference, width=0.25)
    table = ratio_distribution(empirical, reference, edges)
    assert table["bin"].iloc[0] == "zero"
    assert table["probability"].sum() == pytest.approx(1)
    assert table["reference_mean"].sum() == pytest.approx(1)
    assert table["count"].tolist() == [2, 1, 1, 0, 0]
    assert np.isnan(table["density"].iloc[0])
    positive = table.iloc[1:]
    assert (positive["density"] * (positive["right"] - positive["left"])).sum() == pytest.approx(
        0.5
    )


def test_concentration_summary_detects_concentrated_signs():
    graph = nx.star_graph(20)
    for u, v in graph.edges():
        graph[u][v]["sign"] = -1 if v <= 4 else 1
    table = negative_incident_ratios(graph)
    reference, _ = shuffled_negative_ratios(graph, realizations=50, seed=0)
    summary = concentration_summary(table, reference).set_index("statistic")
    assert summary.loc["zero_fraction", "empirical"] == pytest.approx(16 / 21)
    assert summary["p_upper"].between(0, 1).all()


def test_community_rates_count_every_edge_once():
    rates = community_sign_rates(_signed_graph())
    total = rates.set_index(["community", "scope"])
    assert total.loc[("all", "within"), "edges"] + total.loc[("all", "between"), "edges"] == 7
    assert total.loc[("all", "between"), "negative_edges"] == 1
    assert total.loc[("0", "within"), ["edges", "negative_edges", "zero_edges"]].tolist() == [
        3,
        1,
        1,
    ]
    assert total.loc[("1", "boundary"), "edges"] == 1
    assert total.loc[("1", "within"), "negative_rate"] == 0
    assert (rates["reference_q025"] <= rates["reference_q975"]).all()


def test_community_rates_need_labels():
    graph = _signed_graph()
    del graph.nodes["a"]["community"]
    with pytest.raises(ValueError, match="community"):
        community_sign_rates(graph)


def test_pipeline_saves_outputs(tmp_path):
    graph_path = tmp_path / "core.graphml"
    export_graph(_signed_graph(), graph_path)
    results_dir = tmp_path / "incident"
    table = run_incident_sign_analysis(
        graph_path, results_dir=results_dir, realizations=10, seed=1, bin_width=0.25
    )
    for name in (
        "node_incident_signs.csv",
        "r_neg_distribution.csv",
        "r_neg_summary.csv",
        "community_sign_rates.csv",
        "metadata.json",
        "nodes.json",
        "r_neg_distribution.png",
        "r_neg_distribution.pdf",
    ):
        assert (results_dir / name).is_file()
    saved = pd.read_csv(results_dir / "node_incident_signs.csv")
    assert saved["node"].tolist() == table["node"].tolist()
    metadata = json.loads((results_dir / "metadata.json").read_text())
    assert len(metadata["shuffle_seeds"]) == 10
    assert metadata["sign_counts"] == {"1": 4, "-1": 2, "0": 1}
    assert json.loads((results_dir / "nodes.json").read_text()) == sorted("abcdef")
