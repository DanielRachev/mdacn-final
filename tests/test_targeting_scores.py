from dataclasses import replace

import networkx as nx
import pandas as pd
import pytest

from mdacn import __main__ as cli
from mdacn import workflow
from mdacn.centrality import leverage_centrality
from mdacn.data import build_matrices, export_graph, save_matrices
from mdacn.targeting import (
    bridge_fraction,
    signed_leverage_centrality,
    temporal_activity_bridge_centrality,
)
from mdacn.targeting_pipeline import save_targeting_scores


def signed_path(signs):
    graph = nx.Graph()
    graph.add_edges_from(zip(["a", "b", "c"], ["b", "c", "d"], strict=True))
    nx.set_edge_attributes(graph, dict(zip(graph.edges(), signs, strict=True)), "sign")
    return graph


def test_signed_leverage_matches_the_proposed_formula() -> None:
    scores = signed_leverage_centrality(build_matrices(signed_path((1, -1, 1))))

    assert scores["a"] == pytest.approx(0.0)
    assert scores["b"] == pytest.approx(1 / 6)
    assert scores["c"] == pytest.approx(1 / 6)
    assert scores["d"] == pytest.approx(0.0)
    assert leverage_centrality(signed_path((1, -1, 1)))["b"] == pytest.approx(1 / 6)


def test_signed_leverage_penalizes_a_negative_smaller_degree_neighbour() -> None:
    # b has degree 2; a has degree 1, so a contributes leverage 1/3, signed -1.
    graph = signed_path((-1, 1, 1))
    scores = signed_leverage_centrality(build_matrices(graph))

    assert scores["b"] == pytest.approx(-1 / 6)


def test_signed_leverage_keeps_zero_sign_edges_in_the_denominator() -> None:
    # b has degree 2: a positive leaf a and a zero-sign edge to the hub h.
    graph = nx.Graph()
    graph.add_edge("a", "b", sign=1)
    graph.add_edge("b", "h", sign=0)
    for leaf in ("x", "y", "z"):
        graph.add_edge("h", leaf, sign=1)

    scores = signed_leverage_centrality(build_matrices(graph))

    # Dropping the zero-sign edge would leave b with degree 1 and score 0.
    assert scores["b"] == pytest.approx(1 / 6)
    assert scores["a"] == pytest.approx(0.0)


def test_signed_leverage_scores_isolated_and_empty_graphs() -> None:
    graph = signed_path((1, 1, 1))
    graph.add_node("isolated")
    assert signed_leverage_centrality(build_matrices(graph))["isolated"] == 0.0

    empty = build_matrices(nx.Graph())
    assert signed_leverage_centrality(empty) == {}


def test_signed_leverage_rejects_signed_edges_outside_the_topology() -> None:
    network = build_matrices(signed_path((1, 1, 1)))
    extra = network.signed.tolil()
    extra[0, 3] = 1.0
    extra[3, 0] = 1.0
    broken = replace(network, signed=extra.tocsr())

    with pytest.raises(ValueError, match="Signed edges must belong to the unsigned topology"):
        signed_leverage_centrality(broken)


def test_bridge_fraction_counts_cross_community_neighbours() -> None:
    graph = signed_path((1, 1, 1))
    communities = {"a": 0, "b": 0, "c": 1, "d": 1}

    scores = bridge_fraction(graph, communities)

    assert scores["a"] == pytest.approx(0.0)
    assert scores["b"] == pytest.approx(0.5)
    assert scores["c"] == pytest.approx(0.5)
    assert scores["d"] == pytest.approx(0.0)


def test_bridge_fraction_requires_complete_labels() -> None:
    graph = signed_path((1, 1, 1))
    with pytest.raises(ValueError, match="Missing community labels"):
        bridge_fraction(graph, {"a": 0, "b": 0, "c": 1})


def test_temporal_activity_bridge_is_a_geometric_mean() -> None:
    graph = signed_path((1, 1, 1))
    communities = {"a": 0, "b": 0, "c": 1, "d": 1}
    activity = {"a": 1, "b": 4, "c": 4, "d": 1}

    scores = temporal_activity_bridge_centrality(graph, communities, activity)

    assert scores["a"] == pytest.approx(0.0)
    assert scores["b"] == pytest.approx(0.5**0.5)
    assert scores["c"] == pytest.approx(0.5**0.5)
    assert scores["d"] == pytest.approx(0.0)


def test_temporal_activity_bridge_defaults_to_the_core_interaction_counts() -> None:
    graph = signed_path((1, 1, 1))
    nx.set_node_attributes(graph, {"a": 1, "b": 4, "c": 4, "d": 1}, "n_interactions")
    communities = {"a": 0, "b": 0, "c": 1, "d": 1}

    scores = temporal_activity_bridge_centrality(graph, communities)

    assert scores["b"] == pytest.approx(0.5**0.5)


def test_temporal_activity_bridge_requires_activity_values() -> None:
    graph = signed_path((1, 1, 1))
    communities = {"a": 0, "b": 0, "c": 1, "d": 1}

    with pytest.raises(ValueError, match="no n_interactions"):
        temporal_activity_bridge_centrality(graph, communities)
    with pytest.raises(ValueError, match="Missing activity values"):
        temporal_activity_bridge_centrality(graph, communities, {"a": 1, "b": 2, "c": 3})
    with pytest.raises(ValueError, match="finite and non-negative"):
        temporal_activity_bridge_centrality(graph, communities, {"a": 1, "b": 2, "c": 3, "d": -1})


def _saved_core(tmp_path, graph, communities):
    processed = tmp_path / "processed"
    processed.mkdir()
    core = processed / "core.graphml"
    export_graph(graph, core)
    save_matrices(build_matrices(graph), core)
    pd.DataFrame(
        {"node": sorted(graph), "community": [communities[node] for node in sorted(graph)]}
    ).to_csv(processed / "communities.csv", index=False)
    return core


def test_save_targeting_scores_exports_aligned_node_rows(tmp_path) -> None:
    graph = signed_path((1, -1, 1))
    nx.set_node_attributes(graph, {"a": 1, "b": 4, "c": 4, "d": 1}, "n_interactions")
    core = _saved_core(tmp_path, graph, {"a": 0, "b": 0, "c": 1, "d": 1})

    save_targeting_scores(core)

    processed = core.parent
    signed_leverage = pd.read_csv(processed / "signed_leverage.csv")
    bridge = pd.read_csv(processed / "temporal_bridge_centrality.csv")
    assert signed_leverage.columns.tolist() == ["node", "signed_leverage"]
    assert bridge.columns.tolist() == ["node", "temporal_bridge"]
    assert signed_leverage.node.tolist() == ["a", "b", "c", "d"]
    assert bridge.node.tolist() == ["a", "b", "c", "d"]
    assert signed_leverage.signed_leverage.tolist() == pytest.approx([0, 1 / 6, 1 / 6, 0])
    assert bridge.temporal_bridge.tolist() == pytest.approx([0, 0.5**0.5, 0.5**0.5, 0])
    assert (processed / "targeting_scores.json").is_file()


def test_save_targeting_scores_accepts_a_measured_activity_export(tmp_path) -> None:
    graph = signed_path((1, -1, 1))
    core = _saved_core(tmp_path, graph, {"a": 0, "b": 0, "c": 1, "d": 1})
    activity = core.parent / "temporal_activity.csv"
    pd.DataFrame({"node": ["a", "b", "c", "d"], "temporal_activity": [8, 2, 4, 1]}).to_csv(
        activity, index=False
    )

    save_targeting_scores(core, activity_path=activity)

    bridge = pd.read_csv(core.parent / "temporal_bridge_centrality.csv")
    assert bridge.temporal_bridge.tolist() == pytest.approx(
        [0, (0.5 * 0.25) ** 0.5, (0.5 * 0.5) ** 0.5, 0]
    )


def test_save_targeting_scores_requires_community_labels(tmp_path) -> None:
    graph = signed_path((1, -1, 1))
    core = _saved_core(tmp_path, graph, {"a": 0, "b": 0, "c": 1, "d": 1})
    (core.parent / "communities.csv").unlink()

    with pytest.raises(ValueError, match="Run characterise first"):
        save_targeting_scores(core)


def test_targeting_scores_command_uses_project_paths(monkeypatch) -> None:
    monkeypatch.setitem(cli.config.PATHS, "processed_dir", "custom/processed")
    calls = []
    monkeypatch.setattr(workflow, "_require", lambda *args: None)
    monkeypatch.setattr(
        workflow,
        "save_targeting_scores",
        lambda graph_path, **kwargs: calls.append((graph_path, kwargs)),
    )
    monkeypatch.setattr("sys.argv", ["mdacn", "targeting-scores"])
    cli.main()

    graph, options = calls[0]
    assert graph == cli.Path("custom/processed/core.graphml")
    assert options == {"activity_path": None}
