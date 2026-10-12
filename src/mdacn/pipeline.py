"""Run preprocessing and characterisation pipeline"""

from pathlib import Path
from typing import Literal

import networkx as nx


def graph_summary(graph: nx.Graph) -> dict[str, int | float | None]:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_negative = sum(edge["sign"] == -1 for _, _, edge in graph.edges(data=True))

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "connected_components": nx.number_connected_components(graph),
        "average_degree": (2 * n_edges / n_nodes) if n_nodes else 0.0,
        "density": nx.density(graph),
        "negative_edge_fraction": (n_negative / n_edges) if n_edges else None,
    }


def characterise(
    input_path: Path,
    *,
    core_method: Literal["k_core", "activity_lcc"] = "k_core",
    k: int = 10,
    min_interactions: int = 100,
    seed: int = 42,
    processed_dir: Path = Path("data/processed"),
    results_dir: Path = Path("results"),
) -> dict[str, object]:
    """Save the aggregate/core graphs, aligned core matrices, metrics, and plots

    The pipeline uses the unsigned core topology for characterisation, keeping
    signs available for the opinion dynamics and intervention work.
    """
    raise NotImplementedError("characterise pipeline not ready yet")
