"""Run preprocessing and characterisation pipeline"""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import networkx as nx
import pandas as pd
from scipy.sparse import save_npz

from mdacn.characterisation import (
    community_metrics,
    degree_assortativity,
    degree_distribution,
    detect_communities,
    fit_degree_baselines,
    spectral_metrics,
)
from mdacn.data import (
    DATASET_URL,
    aggregate_edges,
    build_matrices,
    export_graph,
    extract_core,
    largest_component,
    load_interactions,
)
from mdacn.plots import plot_degree_distribution


def _graph_summary(graph: nx.Graph) -> dict[str, int | float | None]:
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
    print(f"Reading {input_path}", flush=True)
    interactions = load_interactions(input_path)
    if interactions.empty:
        raise ValueError("The input contains no interactions")
    with input_path.open("rb") as raw:
        checksum = hashlib.file_digest(raw, "sha256").hexdigest()
    self_links = int((interactions["source"] == interactions["target"]).sum())

    graph = aggregate_edges(interactions)
    graph.graph.update(
        dataset_url=DATASET_URL,
        source_path=str(input_path.resolve()),
        input_sha256=checksum,
        self_interactions_removed=self_links,
    )
    core = largest_component(
        extract_core(graph, method=core_method, k=k, min_interactions=min_interactions)
    )
    if not core:
        raise ValueError(
            "The extracted core is empty; lower k or min_interactions and ensure the graph is valid"
        )
    threshold = {"k": k} if core_method == "k_core" else {"min_interactions": min_interactions}
    core.graph.update(core_method=core_method, **threshold)

    print(f"Aggregate: {len(graph):,} nodes, {graph.number_of_edges():,} edges")
    print(f"Core: {len(core):,} nodes, {core.number_of_edges():,} edges")

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    export_graph(graph, processed_dir / "aggregate.graphml")
    network = build_matrices(core)
    save_npz(processed_dir / "unsigned.npz", network.unsigned)
    save_npz(processed_dir / "signed.npz", network.signed)
    (processed_dir / "nodes.json").write_text(json.dumps(network.nodes, indent=2) + "\n")

    print("Computing communities, degree fits, and spectral metrics...")
    communities = detect_communities(core, seed=seed)
    nx.set_node_attributes(core, communities, "community")
    export_graph(core, processed_dir / "core.graphml")

    pd.DataFrame(
        {"node": network.nodes, "community": [communities[node] for node in network.nodes]}
    ).to_csv(processed_dir / "communities.csv", index=False)

    distribution = degree_distribution(core)
    fits = fit_degree_baselines(core)
    distribution.to_csv(results_dir / "degree_distribution.csv", index=False)
    fits.to_csv(results_dir / "degree_fits.csv", index=False)

    metrics = {
        "degree_assortativity": degree_assortativity(core),
        "community_count": len(set(communities.values())),
        **community_metrics(core, communities),
        **spectral_metrics(network.unsigned),
    }

    summary: dict[str, object] = {
        "input": {
            "path": str(input_path.resolve()),
            "dataset_url": DATASET_URL,
            "sha256": checksum,
            "interactions": len(interactions),
            "self_interactions_removed": self_links,
            "non_self_interactions": len(interactions) - self_links,
            "negative_interactions_fraction": float((interactions["sentiment"] == -1).mean()),
            "first_timestamp": interactions["timestamp"].min().isoformat(),
            "last_timestamp": interactions["timestamp"].max().isoformat(),
        },
        "preprocessing": {
            "core_method": core_method,
            **threshold,
            "largest_connected_component": True,
            "seed": seed,
        },
        "aggregate": _graph_summary(graph),
        "core": _graph_summary(core),
        "metrics": {
            name: value if math.isfinite(value) else None for name, value in metrics.items()
        },
        "degree_fit_method": "Discrete MLE on k>= smallest positive core degree; no bootstrap",
    }

    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_degree_distribution(distribution, fits, results_dir / "degree_distribution.pdf")
    print(f"Saved graphs and matrices to {processed_dir}, and metrics and plots to {results_dir}")
    return summary
