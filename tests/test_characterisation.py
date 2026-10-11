"""Checks for Jacek's data and characterisation handoff."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, load_npz

from mdacn.characterisation import (
    community_metrics,
    degree_assortativity,
    degree_distribution,
    detect_communities,
    fit_degree_baselines,
    fitted_degree_probabilities,
    spectral_metrics,
)
from mdacn.data import (
    aggregate_edges,
    build_matrices,
    export_graph,
    extract_core,
    load_graph,
    load_interactions,
)


@pytest.fixture
def body_tsv(tmp_path: Path) -> Path:
    path = tmp_path / "body.tsv"
    events = [
        ("b", "a", 1),
        ("b", "a", 1),
        ("a", "b", -1),
        ("a", "b", -1),
        ("a", "c", -1),
        ("c", "d", 1),
        ("c", "d", 1),
        ("alone", "alone", 1),
    ]
    header = "SOURCE_SUBREDDIT\tTARGET_SUBREDDIT\tPOST_ID\tTIMESTAMP\tLINK_SENTIMENT\tPROPERTIES\n"
    rows = "".join(
        f"{source}\t{target}\tp{i}\t2020-01-01 00:00:00\t{sentiment}\t0.1,0.2\n"
        for i, (source, target, sentiment) in enumerate(events)
    )
    path.write_text(header + rows)
    return path


def test_parsing_and_signed_aggregation(body_tsv: Path) -> None:
    interactions = load_interactions(body_tsv)
    graph = aggregate_edges(interactions)
    assert len(interactions) == 8
    assert list(interactions.columns) == ["source", "target", "sentiment", "timestamp"]
    assert pd.api.types.is_datetime64_any_dtype(interactions["timestamp"])
    assert graph["a"]["b"] == {"n_positive": 2, "n_negative": 2, "sign": 1}
    assert graph["a"]["c"]["sign"] == -1
    assert nx.number_of_selfloops(graph) == 0
    assert graph.nodes["alone"]["n_interactions"] == 0


def test_invalid_input_reports_tsv_line(body_tsv: Path) -> None:
    body_tsv.write_text(body_tsv.read_text().replace("\t-1\t", "\t0\t", 1))
    with pytest.raises(ValueError, match=r"TSV lines \[4\]"):
        load_interactions(body_tsv)


def test_core_selection_and_graph_matrix_roundtrip(body_tsv: Path, tmp_path: Path) -> None:
    graph = aggregate_edges(load_interactions(body_tsv))
    assert not extract_core(graph, method="k_core", k=2)
    core = extract_core(graph, method="activity_lcc", min_interactions=4)
    assert set(core) == {"a", "b"}
    assert core["a"]["b"] == graph["a"]["b"]
    assert set(extract_core(graph, method="k_core", k=1)) == {"a", "b", "c", "d"}

    path = tmp_path / "nested" / "graph.graphml"
    export_graph(graph, path)
    restored = load_graph(path)
    assert dict(restored.nodes(data=True)) == dict(graph.nodes(data=True))
    assert dict(restored.edges) == dict(graph.edges)
    network = build_matrices(restored)
    assert network.nodes == ("a", "alone", "b", "c", "d")
    index = {node: position for position, node in enumerate(network.nodes)}
    assert network.unsigned[index["a"], index["c"]] == 1
    assert network.signed[index["a"], index["c"]] == -1
    np.testing.assert_array_equal(network.unsigned.toarray(), np.abs(network.signed.toarray()))

    hub = nx.relabel_nodes(nx.star_graph(140), {0: "hub", **{i: f"n{i}" for i in range(1, 141)}})
    nx.set_edge_attributes(hub, 1, "sign")
    hub_network = build_matrices(hub)
    votes = hub_network.unsigned @ np.ones(141, dtype=np.int8)
    assert votes[hub_network.nodes.index("hub")] == 140


@pytest.mark.parametrize(
    ("graph", "radius", "connectivity"),
    [
        (nx.path_graph(3), np.sqrt(2), 1),
        (nx.star_graph(39), np.sqrt(39), 1),
        (nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)), 1, 0),
        (nx.empty_graph(1), 0, 0),
    ],
)
def test_known_spectral_values(graph: nx.Graph, radius: float, connectivity: float) -> None:
    adjacency = csr_matrix(nx.to_scipy_sparse_array(graph, dtype=float, weight=None))
    metrics = spectral_metrics(adjacency)
    assert metrics["spectral_radius"] == pytest.approx(radius, abs=1e-7)
    assert metrics["algebraic_connectivity"] == pytest.approx(connectivity, abs=1e-7)


def test_degree_fits_and_unsigned_community_metrics() -> None:
    graph = nx.star_graph(200)
    distribution = degree_distribution(graph)
    assert distribution["probability"].sum() == pytest.approx(1)
    fits = fit_degree_baselines(graph)
    assert fits["status"].tolist() == ["ok", "ok"]
    assert fits.iloc[0]["aic"] < fits.iloc[1]["aic"]
    for fit in fits.itertuples(index=False):
        probabilities = fitted_degree_probabilities(
            np.arange(1, 10_001), model=fit.model, parameter=fit.parameter_value, k_min=fit.k_min
        )
        assert probabilities.sum() == pytest.approx(1, abs=1e-7)

    triangle = nx.relabel_nodes(nx.complete_graph(3), {0: "a", 1: "b", 2: "c"})
    nx.set_edge_attributes(triangle, -1, "sign")
    communities = detect_communities(triangle, seed=0)
    assert set(communities) == set(triangle)
    assert len(set(communities.values())) == 1
    metrics = community_metrics(triangle, communities)
    assert metrics == {"modularity": 0.0, "average_clustering": 1.0}
    assert np.isnan(degree_assortativity(triangle))
    assert fit_degree_baselines(triangle)["status"].tolist() == ["constant_degrees"] * 2


def test_pipeline_exports_loadable_artifacts(body_tsv: Path, tmp_path: Path) -> None:
    from mdacn.pipeline import characterise

    processed = tmp_path / "processed"
    results = tmp_path / "results"
    summary = characterise(body_tsv, k=1, processed_dir=processed, results_dir=results)
    graph = nx.read_graphml(processed / "core.graphml")
    aggregate = nx.read_graphml(processed / "aggregate.graphml")
    network = build_matrices(graph)
    nodes = json.loads((processed / "nodes.json").read_text())
    assert set(aggregate) == {"a", "b", "c", "d", "alone"}
    assert nodes == list(network.nodes) == ["a", "b", "c", "d"]
    assert (load_npz(processed / "signed.npz") != network.signed).nnz == 0
    assert (load_npz(processed / "unsigned.npz") != network.unsigned).nnz == 0
    assert summary["core"]["nodes"] == 4
    assert summary["metrics"]["degree_assortativity"] == pytest.approx(-0.5)
    assert json.loads((results / "metrics.json").read_text()) == summary
    assert graph.graph["input_sha256"] == summary["input"]["sha256"]
    for extension in ("png", "pdf"):
        assert (results / f"degree_distribution.{extension}").stat().st_size > 0
