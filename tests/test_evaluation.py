"""Opinion GCC measurements on the fixed unsigned topology."""

import networkx as nx
import numpy as np
import pytest

from mdacn.evaluation import compute_opinion_gcc


def test_opinion_gcc_keeps_full_denominator_and_node_alignment():
    graph = nx.path_graph(["e", "d", "c", "b", "a"])
    nx.set_edge_attributes(graph, 0, "sign")
    # Negative nodes a, b, d form components of sizes two and one.
    state = np.array([-1, -1, 1, -1, 1], dtype=np.int8)
    expected = {"g_minus": 2 / 5, "g_plus": 1 / 5}
    assert compute_opinion_gcc(graph, state) == expected
    assert compute_opinion_gcc(graph, state[::-1], nodes=tuple(graph)) == expected
    assert compute_opinion_gcc(graph, np.ones(5, dtype=np.int8)) == {"g_minus": 0, "g_plus": 1}
    assert compute_opinion_gcc(graph, -np.ones(5, dtype=np.int8)) == {"g_minus": 1, "g_plus": 0}
    assert compute_opinion_gcc(nx.Graph(), np.array([], dtype=np.int8)) == {
        "g_minus": 0,
        "g_plus": 0,
    }
    with pytest.raises(ValueError, match="node order"):
        compute_opinion_gcc(graph, state, nodes=("a", "b", "c", "d", "missing"))
