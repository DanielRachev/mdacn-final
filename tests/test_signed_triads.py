"""Signed triangle enumeration, balance counts, and null-model inputs."""

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import save_npz

from mdacn.data import build_matrices, export_graph
from mdacn.signed_structure import signed_edges
from mdacn.signed_triads import (
    erdos_renyi_signed,
    rewired_networks,
    run_signed_triad_analysis,
    signed_degrees,
    triad_counts,
    triad_counts_from_matrices,
    triad_null_summary,
    triangle_edges,
)


def _random_signed_graph(seed: int = 0) -> nx.Graph:
    rng = np.random.default_rng(seed)
    graph = nx.gnm_random_graph(25, 110, seed=seed)
    graph = nx.relabel_nodes(graph, {node: f"n{node:02d}" for node in graph})
    for u, v in graph.edges():
        graph[u][v]["sign"] = int(rng.choice([1, 1, 1, -1, 0]))
    return graph


def test_triangle_edges_enumerate_each_triangle_once():
    graph = _random_signed_graph()
    edges = signed_edges(graph)
    triangles = triangle_edges(edges)
    assert len(triangles) == sum(nx.triangles(graph).values()) // 3
    endpoints = np.stack(
        [np.minimum(edges.sources, edges.targets), np.maximum(edges.sources, edges.targets)], 1
    )
    nodes = {frozenset(endpoints[triangle].ravel()) for triangle in triangles}
    assert len(nodes) == len(triangles)
    assert all(len(triangle) == 3 for triangle in nodes)


def test_triad_types_follow_negative_edge_count():
    graph = nx.Graph()
    signs = {("a", "b"): 1, ("a", "c"): -1, ("b", "c"): -1, ("c", "d"): 0, ("b", "d"): 1}
    for (u, v), sign in signs.items():
        graph.add_edge(u, v, sign=sign)
    edges = signed_edges(graph)
    counts = triad_counts(triangle_edges(edges), edges.signs)
    assert counts["triangles"] == 2
    assert counts["zero_sign_triangles"] == 1
    assert counts["+--"] == 1
    assert counts["balanced_fraction"] == 1


def test_matrix_counts_match_enumeration():
    graph = _random_signed_graph(3)
    edges = signed_edges(graph)
    network = build_matrices(graph)
    assert triad_counts(triangle_edges(edges), edges.signs) == triad_counts_from_matrices(
        network.unsigned, network.signed
    )


def test_erdos_renyi_keeps_size_and_sign_multiset():
    signs = np.array([1] * 40 + [-1] * 8 + [0] * 2, dtype=np.int8)
    unsigned, signed = erdos_renyi_signed(20, signs, seed=4)
    assert unsigned.nnz == 2 * len(signs)
    assert (unsigned != unsigned.T).nnz == 0
    assert Counter(signed.data) == {1.0: 80, -1.0: 16}
    again = erdos_renyi_signed(20, signs, seed=4)
    assert (again[1] != signed).nnz == 0


def test_rewired_networks_check_node_order(tmp_path):
    network = build_matrices(_random_signed_graph())
    folder = tmp_path / "replicate_0"
    folder.mkdir()
    save_npz(folder / "unsigned.npz", network.unsigned)
    save_npz(folder / "signed.npz", network.signed)
    loaded = rewired_networks(tmp_path, len(network.nodes))
    assert [name for name, *_ in loaded] == ["replicate_0"]
    np.testing.assert_array_equal(
        signed_degrees(*loaded[0][1:]), signed_degrees(network.unsigned, network.signed)
    )
    with pytest.raises(ValueError, match="node order"):
        rewired_networks(tmp_path, len(network.nodes) + 1)


def test_null_summary_p_values():
    table = pd.DataFrame(
        {
            "model": ["empirical", "null", "null", "null"],
            "balanced_fraction": [0.9, 0.5, 0.6, 0.95],
            **{
                column: [1, 1, 1, 1]
                for column in (
                    "fraction_+++",
                    "fraction_++-",
                    "fraction_+--",
                    "fraction_---",
                    "triangles",
                    "zero_sign_triangles",
                    "nonzero_triangles",
                    "+++",
                    "++-",
                    "+--",
                    "---",
                )
            },
        }
    )
    summary = triad_null_summary(table).set_index("statistic")
    assert summary.loc["balanced_fraction", "p_upper"] == pytest.approx(2 / 4)
    assert summary.loc["balanced_fraction", "p_lower"] == pytest.approx(3 / 4)
    assert np.isnan(summary.loc["triangles", "z"])


def test_pipeline_saves_all_models(tmp_path):
    graph = _random_signed_graph(5)
    graph_path = tmp_path / "core.graphml"
    export_graph(graph, graph_path)
    network = build_matrices(graph)
    folder = tmp_path / "rewired" / "replicate_0"
    folder.mkdir(parents=True)
    save_npz(folder / "unsigned.npz", network.unsigned)
    save_npz(folder / "signed.npz", network.signed)
    broken = network.signed.tolil()
    u, v = np.argwhere(network.signed.toarray() > 0)[0]
    broken[u, v] = broken[v, u] = -1
    folder = tmp_path / "rewired" / "replicate_1"
    folder.mkdir()
    save_npz(folder / "unsigned.npz", network.unsigned)
    save_npz(folder / "signed.npz", broken.tocsr())
    results_dir = tmp_path / "triads"
    table = run_signed_triad_analysis(
        graph_path,
        results_dir=results_dir,
        realizations=4,
        er_realizations=2,
        rewired_dir=tmp_path / "rewired",
        seed=2,
    )
    assert table["model"].value_counts().to_dict() == {
        "empirical": 1,
        "sign_shuffle": 4,
        "degree_preserving_rewiring": 1,
        "erdos_renyi": 2,
    }
    shuffled = table.loc[table["model"] == "sign_shuffle"]
    assert (shuffled["triangles"] == table.loc[0, "triangles"]).all()
    for name in (
        "triad_realizations.csv",
        "triad_summary.csv",
        "triad_sign_patterns.csv",
        "metadata.json",
        "signed_triads.png",
        "signed_triads.pdf",
    ):
        assert (results_dir / name).is_file()
    metadata = json.loads((results_dir / "metadata.json").read_text())
    assert metadata["rewiring"]["replicates"] == [
        {"name": "replicate_0", "signed_degrees_preserved": True, "included": True},
        {"name": "replicate_1", "signed_degrees_preserved": False, "included": False},
    ]
    patterns = pd.read_csv(results_dir / "triad_sign_patterns.csv")
    assert patterns["triangles"].sum() == table.loc[0, "triangles"]

    skipped = run_signed_triad_analysis(
        graph_path,
        results_dir=tmp_path / "skip",
        realizations=1,
        er_realizations=0,
        rewired_dir=tmp_path / "missing",
    )
    assert set(skipped["model"]) == {"empirical", "sign_shuffle"}
