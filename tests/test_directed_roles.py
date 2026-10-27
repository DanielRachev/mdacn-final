"""Directed roles, reciprocity, sign fractions, and community bridges."""

import json

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mdacn.data import aggregate_edges, export_graph
from mdacn.directed_roles import (
    community_bridge_rates,
    directed_node_roles,
    directed_reciprocity,
    run_directed_role_analysis,
    sign_fractions,
    signed_bridges,
)


def _events() -> pd.DataFrame:
    rows = [
        ("a", "b", -1, "p1"),
        ("a", "c", -1, "p1"),
        ("a", "b", -1, "p2"),
        ("b", "a", -1, "p3"),
        ("b", "c", 1, "p4"),
        ("c", "b", 1, "p5"),
        ("c", "d", 1, "p6"),
        ("d", "c", -1, "p7"),
    ]
    frame = pd.DataFrame(rows, columns=["source", "target", "sentiment", "post_id"])
    frame["sentiment"] = frame["sentiment"].astype(np.int8)
    frame["timestamp"] = pd.Timestamp("2015-01-01")
    return frame


def _graph(events: pd.DataFrame) -> nx.Graph:
    graph = aggregate_edges(events)
    for node in graph:
        graph.nodes[node]["community"] = 0 if node in "ab" else 1
    return graph


def test_directed_node_roles():
    roles = directed_node_roles(_events(), ("a", "b", "c", "d")).set_index("node")
    a, b, c = roles.loc["a"], roles.loc["b"], roles.loc["c"]
    assert (a["out_degree"], a["in_degree"]) == (2, 1)
    assert (a["negative_out_strength"], a["negative_out_posts"]) == (3, 2)
    assert a["negative_in_strength"] == 1
    assert a["negative_role_balance"] == pytest.approx(0.5)
    assert a["reciprocated_out_fraction"] == pytest.approx(0.5)
    assert a["negative_reciprocated_out_fraction"] == pytest.approx(0.5)
    assert b["negative_role_balance"] == pytest.approx((1 - 2) / 3)
    assert c["negative_out_strength"] == 0 and c["negative_in_strength"] == 2
    assert c["negative_role_balance"] == -1
    assert np.isnan(c["negative_reciprocated_out_fraction"])
    with pytest.raises(ValueError, match="listed nodes"):
        directed_node_roles(_events(), ("a", "b", "c"))


def test_directed_reciprocity_by_sign():
    table = directed_reciprocity(_events(), ("a", "b", "c", "d")).set_index("pair_sign")
    assert table.loc["any", "pairs"] == 7
    assert table.loc["any", "reverse_any"] == 6
    assert table.loc["negative", "pairs"] == 4
    assert table.loc["negative", "reverse_negative"] == 2
    assert table.loc["negative", "reverse_positive"] == 1
    assert table["density_reference"].iloc[0] == pytest.approx(7 / 12)


def test_sign_fractions_by_unit():
    events = _events()
    table = sign_fractions(events, events.iloc[:4]).set_index(["scope", "unit"])
    assert table.loc[("body", "event"), "negative"] == 5
    assert table.loc[("body", "source_post"), ["total", "negative"]].tolist() == [7, 4]
    assert table.loc[("body", "undirected_edge"), ["total", "negative", "zero"]].tolist() == [
        4,
        2,
        1,
    ]
    assert table.loc[("core", "event"), "negative_fraction"] == 1


def test_signed_bridges_and_community_rates():
    events = _events()
    graph = _graph(events)
    bridges = signed_bridges(graph).set_index("node")
    assert bridges.loc["a", ["between_degree", "negative_between"]].tolist() == [1, 1]
    assert bridges.loc["a", "bridge_ratio"] == pytest.approx(0.5)
    assert bridges.loc["a", "signed_bridge_ratio"] == pytest.approx(-0.5)
    assert bridges.loc["b", "signed_bridge_ratio"] == pytest.approx(0.5)
    assert np.isnan(bridges.loc["d", "negative_bridge_share"])
    rates = community_bridge_rates(graph, events).set_index("scope")
    assert rates.loc["within", ["edges", "negative_edges", "zero_edges"]].tolist() == [2, 1, 1]
    assert rates.loc[
        "between", ["edges", "negative_edges", "events", "negative_events"]
    ].tolist() == [
        2,
        1,
        3,
        1,
    ]
    del graph.nodes["a"]["community"]
    with pytest.raises(ValueError, match="community"):
        signed_bridges(graph)


def test_pipeline_saves_outputs(tmp_path):
    events = _events()
    tsv = events.rename(
        columns={
            "source": "SOURCE_SUBREDDIT",
            "target": "TARGET_SUBREDDIT",
            "sentiment": "LINK_SENTIMENT",
            "post_id": "POST_ID",
            "timestamp": "TIMESTAMP",
        }
    )
    input_path = tmp_path / "body.tsv"
    tsv.to_csv(input_path, sep="\t", index=False, date_format="%Y-%m-%d %H:%M:%S")
    graph_path = tmp_path / "core.graphml"
    export_graph(_graph(events), graph_path)
    results_dir = tmp_path / "roles"
    feature_path = tmp_path / "processed" / "incident_signs.csv"
    table = run_directed_role_analysis(
        input_path, graph_path, results_dir=results_dir, feature_path=feature_path
    )
    features = pd.read_csv(feature_path)
    assert features.columns.tolist() == ["node", "negative_incident_ratio", "signed_bridge_ratio"]
    assert features["node"].tolist() == ["a", "b", "c", "d"]
    assert np.isfinite(features[["negative_incident_ratio", "signed_bridge_ratio"]]).all(axis=None)
    assert table["node"].tolist() == ["a", "b", "c", "d"]
    assert {"r_neg", "signed_bridge_ratio", "negative_role_balance", "community"} <= set(table)
    for name in (
        "node_roles.csv",
        "reciprocity.csv",
        "sign_fractions.csv",
        "community_bridges.csv",
        "metadata.json",
        "nodes.json",
        "directed_roles.png",
        "directed_roles.pdf",
    ):
        assert (results_dir / name).is_file()
    assert json.loads((results_dir / "nodes.json").read_text()) == ["a", "b", "c", "d"]
