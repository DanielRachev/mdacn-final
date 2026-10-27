"""Export the proposed targeting scores."""

from pathlib import Path

import numpy as np
import pandas as pd

from mdacn.artifacts import checksum, write_json
from mdacn.config import processed_paths
from mdacn.data import load_graph, load_matrices
from mdacn.targeting import signed_leverage_centrality, temporal_activity_bridge_centrality


def load_communities(path: Path, nodes: tuple[str, ...]) -> dict[str, int]:
    """Read baseline community labels and require exactly one per core node."""
    if not path.is_file():
        raise ValueError(f"Missing community labels: {path}. Run characterise first.")
    table = pd.read_csv(path, dtype={"node": str}, keep_default_na=False)
    if not {"node", "community"} <= set(table):
        raise ValueError(f"{path}: require node and community columns")
    if table.node.duplicated().any() or set(table.node) != set(nodes):
        raise ValueError(f"{path}: community labels must cover the core nodes exactly once")
    return dict(zip(table.node, table.community.astype(int), strict=True))


def load_activity(path: Path, nodes: tuple[str, ...], column: str = "temporal_activity"):
    """Read a per-node activity export and require exactly one value per core node."""
    if not path.is_file():
        raise ValueError(f"Missing activity table: {path}")
    table = pd.read_csv(path, dtype={"node": str}, keep_default_na=False)
    if not {"node", column} <= set(table):
        raise ValueError(f"{path}: require node and {column} columns")
    if table.node.duplicated().any() or set(table.node) != set(nodes):
        raise ValueError(f"{path}: activity must cover the core nodes exactly once")
    values = pd.to_numeric(table[column], errors="raise")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{path}: activity values must be finite")
    return dict(zip(table.node, values.astype(float), strict=True))


def save_targeting_scores(
    graph_path: Path,
    *,
    activity_path: Path | None = None,
    activity_column: str = "temporal_activity",
) -> None:
    """Save signed-leverage and temporal-activity bridge scores for every core node.

    Signed leverage reads the saved signed matrix. The bridge score reads the
    community labels from characterise and the core graph's interaction counts,
    unless ``activity_path`` supplies a per-node export.
    """
    paths = processed_paths(graph_path.parent)
    if not graph_path.is_file():
        raise ValueError(f"Missing core graph: {graph_path}. Run characterise first.")
    network = load_matrices(graph_path)
    graph = load_graph(graph_path)
    if tuple(sorted(graph)) != network.nodes:
        raise ValueError("Saved matrices do not match the core; run characterise again")

    communities = load_communities(paths["communities"], network.nodes)
    activity = (
        load_activity(activity_path, network.nodes, activity_column)
        if activity_path is not None
        else None
    )
    signed_leverage = signed_leverage_centrality(network)
    bridge = temporal_activity_bridge_centrality(graph, communities, activity)

    pd.DataFrame(
        {"node": network.nodes, "signed_leverage": [signed_leverage[n] for n in network.nodes]}
    ).to_csv(paths["signed_leverage"], index=False)
    pd.DataFrame(
        {"node": network.nodes, "temporal_bridge": [bridge[n] for n in network.nodes]}
    ).to_csv(paths["temporal_bridge"], index=False)
    write_json(
        paths["targeting_metadata"],
        {
            "graph_sha256": checksum(graph_path),
            "communities_sha256": checksum(paths["communities"]),
            "signed_leverage_sha256": checksum(paths["signed_leverage"]),
            "temporal_bridge_sha256": checksum(paths["temporal_bridge"]),
            "signed_leverage": (
                "(1 / d_i) * sum_j W_ij * max(0, (d_i - d_j) / (d_i + d_j)); "
                "d_i is the unsigned core degree"
            ),
            "temporal_bridge": (
                "sqrt(bridge_fraction * activity / max(activity)); "
                "bridge_fraction is the share of neighbours outside the node's community"
            ),
            "activity_source": ("none" if activity_path is None else str(activity_path.resolve())),
            "activity_column": None if activity_path is None else activity_column,
        },
    )
    print(f"Saved targeting scores: {paths['signed_leverage']} and {paths['temporal_bridge']}")
