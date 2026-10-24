"""Export the existing centrality implementations for later stages."""

import json
from pathlib import Path

import pandas as pd

from mdacn.artifacts import checksum, write_json
from mdacn.centrality import (
    betweenness_centrality,
    degree_centrality,
    eigenvector_centrality,
    leverage_centrality,
)
from mdacn.config import processed_paths
from mdacn.data import load_graph


def save_centrality_scores(graph_path: Path) -> None:
    paths = processed_paths(graph_path.parent)
    graph = load_graph(graph_path)
    nodes = json.loads(paths["nodes"].read_text(encoding="utf-8"))
    if nodes != sorted(graph):
        raise ValueError("Saved nodes do not match the core; run characterise again")
    scores = (
        pd.DataFrame(
            {
                "degree": dict(graph.degree()),
                "degree_centrality": degree_centrality(graph),
                "betweenness": betweenness_centrality(graph),
                "eigenvector": eigenvector_centrality(graph),
                "leverage": leverage_centrality(graph),
            }
        )
        .reindex(nodes)
        .rename_axis("node")
        .reset_index()
    )
    scores.to_csv(paths["centrality"], index=False)
    write_json(
        paths["centrality_metadata"],
        {
            "graph_sha256": checksum(graph_path),
            "centrality_sha256": checksum(paths["centrality"]),
        },
    )
    print(f"Saved centrality scores: {paths['centrality']}")


def load_centrality_scores(graph_path: Path) -> pd.DataFrame:
    paths = processed_paths(graph_path.parent)
    for name in ("centrality", "centrality_metadata"):
        if not paths[name].is_file():
            raise ValueError(f"Missing centrality input: {paths[name]}. Run centrality first.")
    metadata = json.loads(paths["centrality_metadata"].read_text(encoding="utf-8"))
    if metadata["graph_sha256"] != checksum(graph_path):
        raise ValueError("Core changed since centrality export; run centrality again")
    if metadata["centrality_sha256"] != checksum(paths["centrality"]):
        raise ValueError("Centrality table changed; run centrality again")
    return pd.read_csv(paths["centrality"], dtype={"node": str}, keep_default_na=False)
