"""Directed interaction roles, reciprocity, and signed community bridges on the core.

Labels mark source posts that a classifier scored as negative toward the target.
Sender and receiver roles describe who posted and who was linked.
"""

from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mdacn.artifacts import checksum, write_json
from mdacn.data import community_labels, edge_sign_counts, load_graph, load_interactions
from mdacn.plots import plot_directed_roles
from mdacn.signed_structure import (
    community_sign_rates,
    incident_counts,
    negative_incident_ratios,
    node_positions,
    safe_ratio,
    signed_edges,
)
from mdacn.temporal_analysis import filter_core_events


def directed_node_roles(events: pd.DataFrame, nodes: tuple[str, ...]) -> pd.DataFrame:
    """Return directed degrees, signed strengths, and reciprocity per node.

    Strength counts events (hyperlink rows); degree counts distinct directed
    neighbours. negative_out_posts counts distinct source posts, because one post
    linking several targets yields several rows with one shared label.
    negative_role_balance = (negative out - negative in) / (negative out + negative in):
    +1 only sends negative links, -1 only receives them.
    Reciprocated fractions refer to the node's out-neighbours: any link back, or
    for negative out-neighbours, a negative link back.
    """
    n_nodes = len(nodes)
    source, target = node_positions(events, nodes)
    negative = events["sentiment"].to_numpy() == -1

    def count(values: NDArray, mask: NDArray | None = None) -> NDArray[np.int64]:
        return np.bincount(values if mask is None else values[mask], minlength=n_nodes)

    pairs = np.unique(source * n_nodes + target)
    negative_pairs = np.unique((source * n_nodes + target)[negative])
    reverse = (pairs % n_nodes) * n_nodes + pairs // n_nodes
    negative_reverse = (negative_pairs % n_nodes) * n_nodes + negative_pairs // n_nodes
    out_degree = count(pairs // n_nodes)
    negative_out_degree = count(negative_pairs // n_nodes)
    table = pd.DataFrame(
        {
            "node": nodes,
            "out_degree": out_degree,
            "in_degree": count(pairs % n_nodes),
            "positive_out_strength": count(source, ~negative),
            "negative_out_strength": count(source, negative),
            "positive_in_strength": count(target, ~negative),
            "negative_in_strength": count(target, negative),
            "negative_out_degree": negative_out_degree,
            "negative_in_degree": count(negative_pairs % n_nodes),
            "reciprocated_out_fraction": safe_ratio(
                count(pairs // n_nodes, np.isin(reverse, pairs)), out_degree
            ),
            "negative_reciprocated_out_fraction": safe_ratio(
                count(negative_pairs // n_nodes, np.isin(negative_reverse, negative_pairs)),
                negative_out_degree,
            ),
        }
    )
    if "post_id" in events:
        negative_posts = events.loc[negative, ["source", "post_id"]].drop_duplicates()
        table["negative_out_posts"] = (
            negative_posts["source"].value_counts().reindex(nodes, fill_value=0).to_numpy()
        )
    out_strength = table["positive_out_strength"] + table["negative_out_strength"]
    in_strength = table["positive_in_strength"] + table["negative_in_strength"]
    negative_total = table["negative_out_strength"] + table["negative_in_strength"]
    table["negative_out_fraction"] = safe_ratio(table["negative_out_strength"], out_strength)
    table["negative_in_fraction"] = safe_ratio(table["negative_in_strength"], in_strength)
    table["negative_role_balance"] = safe_ratio(
        table["negative_out_strength"] - table["negative_in_strength"], negative_total
    )
    return table


def directed_reciprocity(events: pd.DataFrame, nodes: tuple[str, ...]) -> pd.DataFrame:
    """Return reciprocity of ordered pairs by the sign of their events.

    A pair (u, v) has a sign if it has at least one event of that sign; one pair can
    count as both positive and negative. reverse_* columns count pairs whose reverse
    (v, u) has any, a positive, or a negative event. density_reference is the
    reciprocity expected if links were placed independently at the observed density.
    """
    n_nodes = len(nodes)
    source, target = node_positions(events, nodes)
    keys = source * n_nodes + target
    negative = events["sentiment"].to_numpy() == -1
    sets = {
        "any": np.unique(keys),
        "positive": np.unique(keys[~negative]),
        "negative": np.unique(keys[negative]),
    }
    rows = []
    for name, pairs in sets.items():
        reverse = (pairs % n_nodes) * n_nodes + pairs // n_nodes
        row = {"pair_sign": name, "pairs": len(pairs)}
        for other in ("any", "positive", "negative"):
            matched = int(np.isin(reverse, sets[other]).sum())
            row[f"reverse_{other}"] = matched
            row[f"reverse_{other}_fraction"] = matched / len(pairs) if len(pairs) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    possible = n_nodes * (n_nodes - 1)
    table["density_reference"] = len(sets["any"]) / possible if possible else np.nan
    return table


def signed_bridges(graph: nx.Graph) -> pd.DataFrame:
    """Return per-node community bridge counts and ratios in nodes.json order.

    bridge_ratio = between-community edges / degree.
    negative_bridge_share = negative between-community edges / between-community edges.
    signed_bridge_ratio = (positive - negative between-community edges) / degree: near +1
    for a node whose ties mostly cross communities positively, negative for hostile
    bridges, near 0 for nodes inside their own community or with balanced bridges.
    Zero-sign edges count in the degree and between counts but in neither sign.
    """
    communities = community_labels(graph)
    edges = signed_edges(graph)
    labels = np.array([communities[node] for node in edges.nodes])
    between = labels[edges.sources] != labels[edges.targets]
    degree = incident_counts(edges)
    between_degree = incident_counts(edges, between)
    positive = incident_counts(edges, between & (edges.signs == 1))
    negative = incident_counts(edges, between & (edges.signs == -1))
    return pd.DataFrame(
        {
            "node": edges.nodes,
            "community": labels,
            "between_degree": between_degree,
            "positive_between": positive,
            "negative_between": negative,
            "zero_between": incident_counts(edges, between & (edges.signs == 0)),
            "bridge_ratio": safe_ratio(between_degree, degree),
            "negative_bridge_share": safe_ratio(negative, between_degree),
            "signed_bridge_ratio": safe_ratio(positive - negative, degree),
        }
    )


def sign_fractions(interactions: pd.DataFrame, core_events: pd.DataFrame) -> pd.DataFrame:
    """Return negative fractions by unit for the body file and the core.

    Units: events (rows), source posts, directed pairs and undirected edges by
    majority sign (zero for ties). Majority collapse and repeated rows per post are
    why event, post, and edge fractions differ.
    """
    rows = []
    for scope, frame in (("body", interactions), ("core", core_events)):
        counts = [("event", len(frame), int((frame["sentiment"] == -1).sum()), 0)]
        if "post_id" in frame:
            posts = frame.groupby("post_id")["sentiment"].first()
            counts.append(("source_post", len(posts), int((posts == -1).sum()), 0))
        directed = np.sign(
            frame.loc[frame["source"] != frame["target"]]
            .groupby(["source", "target"])["sentiment"]
            .sum()
        )
        undirected = edge_sign_counts(frame)["sign"]
        for unit, signs in (("directed_pair", directed), ("undirected_edge", undirected)):
            counts.append((unit, len(signs), int((signs < 0).sum()), int((signs == 0).sum())))
        rows.extend(
            {
                "scope": scope,
                "unit": unit,
                "total": total,
                "negative": negative,
                "zero": zero,
                "negative_fraction": safe_ratio(negative, total),
            }
            for unit, total, negative, zero in counts
        )
    return pd.DataFrame(rows)


def community_bridge_rates(graph: nx.Graph, core_events: pd.DataFrame) -> pd.DataFrame:
    """Return negative rates within and between fixed communities for edges and events.

    Edge counts are the all-community rows of community_sign_rates; events add the
    directed rows behind those edges.
    """
    communities = community_labels(graph)
    edge_rates = community_sign_rates(graph).set_index(["community", "scope"])
    event_between = (
        core_events["source"].map(communities).to_numpy()
        != core_events["target"].map(communities).to_numpy()
    )
    event_negative = core_events["sentiment"].to_numpy() == -1
    rows = []
    for scope, between in (("within", False), ("between", True)):
        edge_row = edge_rates.loc[("all", scope)]
        event_mask = event_between == between
        rows.append(
            {
                "scope": scope,
                "edges": int(edge_row["edges"]),
                "negative_edges": int(edge_row["negative_edges"]),
                "zero_edges": int(edge_row["zero_edges"]),
                "events": int(event_mask.sum()),
                "negative_events": int((event_mask & event_negative).sum()),
            }
        )
    table = pd.DataFrame(rows)
    table["negative_edge_rate"] = safe_ratio(table["negative_edges"], table["edges"])
    table["negative_event_rate"] = safe_ratio(table["negative_events"], table["events"])
    return table


def run_directed_role_analysis(
    input_path: Path,
    graph_path: Path,
    *,
    results_dir: Path = Path("results/directed_roles"),
    feature_path: Path | None = Path("data/processed/incident_signs.csv"),
) -> pd.DataFrame:
    """Save per-node directed roles and bridges, reciprocity, sign fractions, and figures.

    feature_path receives the node-feature export read by the shared features stage:
    node, negative_incident_ratio, and signed_bridge_ratio for every core node.

    Roles describe posting and being linked (LINK_SENTIMENT of the source post toward the
    target), not real aggressors or factual correctness.
    """
    graph = load_graph(graph_path)
    nodes = tuple(sorted(graph))
    interactions = load_interactions(input_path, include_post_id=True)
    events = filter_core_events(interactions, nodes)
    print(f"Directed roles: {len(events):,} core events among {len(nodes):,} nodes", flush=True)

    incident = negative_incident_ratios(graph).drop(columns=["community"], errors="ignore")
    table = (
        signed_bridges(graph)
        .merge(incident, on="node", validate="one_to_one")
        .merge(directed_node_roles(events, nodes), on="node", validate="one_to_one")
    )
    if table["node"].tolist() != list(nodes):
        raise RuntimeError("Node table lost the nodes.json order")
    reciprocity = directed_reciprocity(events, nodes)
    fractions = sign_fractions(interactions, events)
    bridges = community_bridge_rates(graph, events)

    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "node_roles.csv", index=False)
    if feature_path is not None:
        features = table[["node", "r_neg", "signed_bridge_ratio"]].rename(
            columns={"r_neg": "negative_incident_ratio"}
        )
        if not np.isfinite(features[["negative_incident_ratio", "signed_bridge_ratio"]]).all(
            axis=None
        ):
            raise RuntimeError("Feature export needs finite scores for every core node")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(feature_path, index=False)
    reciprocity.to_csv(results_dir / "reciprocity.csv", index=False)
    fractions.to_csv(results_dir / "sign_fractions.csv", index=False)
    bridges.to_csv(results_dir / "community_bridges.csv", index=False)
    metadata = {
        "input_path": str(input_path.resolve()),
        "input_sha256": checksum(input_path),
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": checksum(graph_path),
        "core_events": len(events),
        "feature_export": str(feature_path.resolve()) if feature_path is not None else None,
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "pandas")},
    }
    write_json(results_dir / "metadata.json", metadata)
    write_json(results_dir / "nodes.json", list(nodes))
    for extension in ("png", "pdf"):
        plot_directed_roles(
            table, reciprocity, bridges, results_dir / f"directed_roles.{extension}"
        )
    print(reciprocity.to_string(index=False), flush=True)
    print(fractions.to_string(index=False), flush=True)
    print(bridges.to_string(index=False), flush=True)
    print(f"Saved directed-role results to {results_dir}", flush=True)
    return table
