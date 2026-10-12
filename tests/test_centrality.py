import networkx as nx
import pytest

from mdacn.centrality import (
    betweenness_centrality,
    degree_centrality,
    eigenvector_centrality,
    leverage_centrality,
    rank_nodes,
)


def test_degree_and_rank_nodes_are_deterministic():
    graph = nx.path_graph(4)
    scores = degree_centrality(graph)
    assert scores == pytest.approx({0: 1 / 3, 1: 2 / 3, 2: 2 / 3, 3: 1 / 3})
    assert rank_nodes(scores) == [1, 2, 0, 3]


def test_betweenness_and_eigenvector_match_reference_values():
    graph = nx.path_graph(4)
    betweenness = betweenness_centrality(graph)
    assert betweenness == pytest.approx({0: 0.0, 1: 0.6666666667, 2: 0.6666666667, 3: 0.0})

    eigenvector = eigenvector_centrality(graph)
    assert set(eigenvector) == set(graph.nodes)
    assert max(eigenvector.values()) == pytest.approx(max(eigenvector.values()))
    assert eigenvector[1] > eigenvector[0]


def test_leverage_matches_formula_on_small_graph():
    graph = nx.Graph()
    graph.add_edges_from([("hub", "a"), ("hub", "b"), ("hub", "c")])
    scores = leverage_centrality(graph)
    expected = {
        "hub": (1 / 3) * 3 * ((3 - 1) / (3 + 1)),
        "a": (1 / 1) * ((1 - 3) / (1 + 3)),
        "b": (1 / 1) * ((1 - 3) / (1 + 3)),
        "c": (1 / 1) * ((1 - 3) / (1 + 3)),
    }
    assert scores == pytest.approx(expected)
