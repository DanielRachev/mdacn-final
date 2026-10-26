"""A small structural comparison of the existing core extraction methods."""

import hashlib
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from mdacn.characterisation import degree_distribution
from mdacn.comparison_plots import (
    plot_extraction_degrees,
    plot_extraction_retention,
    plot_summary_table,
)
from mdacn.data import (
    DATASET_URL,
    aggregate_edges,
    extract_core,
    largest_component,
    load_interactions,
)
from mdacn.pipeline import characterise_core, graph_summary


def _ordered_core(core: nx.Graph) -> nx.Graph:
    """Give seeded Louvain a stable node and neighbor iteration order."""
    ordered = nx.Graph()
    ordered.graph.update(core.graph)
    ordered.add_nodes_from((node, core.nodes[node]) for node in sorted(core))
    edges = sorted(tuple(sorted(edge)) for edge in core.edges())
    ordered.add_edges_from((u, v, core[u][v]) for u, v in edges)
    return ordered


def compare_extractions(
    input_path: Path,
    *,
    ks: tuple[int, ...] = (8, 10, 12),
    min_interactions: int = 100,
    seed: int = 0,
    processed_dir: Path = Path("data/processed/extraction_comparison"),
    results_dir: Path = Path("results/extraction_comparison"),
) -> dict[str, object]:
    """Export four independent cores and summarize their structural differences.

    Event retention counts non-self events with both endpoints in the core,
    divided by all non-self events in the input. No dynamics are run here.
    """
    if not ks or len(set(ks)) != len(ks) or any(k < 0 for k in ks):
        raise ValueError("Supply distinct, non-negative k values")
    if min_interactions < 0:
        raise ValueError("min_interactions must be non-negative")

    print(f"Reading {input_path}", flush=True)
    interactions = load_interactions(input_path)
    if interactions.empty:
        raise ValueError("The input contains no interactions")
    with input_path.open("rb") as raw:
        checksum = hashlib.file_digest(raw, "sha256").hexdigest()
    graph = aggregate_edges(interactions)
    graph.graph.update(input_sha256=checksum, dataset_url=DATASET_URL)
    aggregate = graph_summary(graph)
    configurations = [(f"k_core_{k}", {"method": "k_core", "k": k}) for k in ks] + [
        (
            f"activity_lcc_{min_interactions}",
            {"method": "activity_lcc", "min_interactions": min_interactions},
        )
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    degree_tables = [degree_distribution(graph).assign(configuration="aggregate")]
    rows = []
    for label, settings in configurations:
        print(f"Characterising {label}...", flush=True)
        core = _ordered_core(largest_component(extract_core(graph, **settings)))
        core.graph.update(
            core_method=settings["method"],
            seed=seed,
            **{key: value for key, value in settings.items() if key != "method"},
        )
        output = processed_dir / label
        metrics = characterise_core(core, seed=seed, processed_dir=output)
        counts = graph_summary(core)
        row = {
            "configuration": label,
            "status": "ok" if core else "empty",
            **counts,
            **metrics,
            "node_retention": len(core) / len(graph),
            "edge_retention": (
                core.number_of_edges() / aggregate["edges"] if aggregate["edges"] else None
            ),
            "event_retention": (
                counts["non_self_interactions"] / aggregate["non_self_interactions"]
                if aggregate["non_self_interactions"]
                else None
            ),
        }
        rows.append(row)
        degree_table = degree_distribution(core)
        degree_tables.append(degree_table.assign(configuration=label))
        configuration_results = results_dir / label
        configuration_results.mkdir(parents=True, exist_ok=True)
        degree_table.to_csv(configuration_results / "degree_distribution.csv", index=False)
        metadata = {
            "input_sha256": checksum,
            "settings": {
                **settings,
                "seed": seed,
                "largest_connected_component": True,
                "community_iteration_order": "alphabetical nodes and lexicographic edges",
            },
            "nodes_sha256": hashlib.sha256((output / "nodes.json").read_bytes()).hexdigest(),
            "processed_dir": str(output.resolve()),
            **row,
        }
        (configuration_results / "metrics.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        print(f"  {len(core):,} nodes; {core.number_of_edges():,} edges", flush=True)

    summary = {
        "input": {
            "path": str(input_path.resolve()),
            "dataset_url": DATASET_URL,
            "sha256": checksum,
            "interactions": len(interactions),
            "self_interactions_removed": int((interactions.source == interactions.target).sum()),
            "first_timestamp": interactions.timestamp.min().isoformat(),
            "last_timestamp": interactions.timestamp.max().isoformat(),
        },
        "seed": seed,
        "aggregate": aggregate,
        "configurations": rows,
    }
    (results_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(results_dir / "comparison.csv", index=False)
    distributions = pd.concat(degree_tables, ignore_index=True)
    distributions.to_csv(results_dir / "degree_distributions.csv", index=False)
    for extension in ("png", "pdf"):
        plot_summary_table(comparison, results_dir / f"topology_table.{extension}")
        plot_extraction_degrees(distributions, results_dir / f"degree_comparison.{extension}")
        plot_extraction_retention(comparison, results_dir / f"retention_and_signs.{extension}")
    print(f"Saved extraction comparison to {results_dir}", flush=True)
    return summary
