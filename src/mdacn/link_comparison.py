"""Coverage and overlap audit of SNAP body and title hyperlinks."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

from mdacn.data import (
    DATASET_URLS,
    aggregate_edges,
    extract_core,
    largest_component,
    load_interactions,
)
from mdacn.pipeline import graph_summary


def _overlap(body: set, title: set) -> dict[str, int | float | None]:
    """Report both directional coverages, rather than an ambiguous overlap rate."""
    shared = len(body & title)
    union = len(body | title)
    return {
        "body_count": len(body),
        "title_count": len(title),
        "shared": shared,
        "body_only": len(body - title),
        "title_only": len(title - body),
        "union": union,
        "body_covered_by_title": shared / len(body) if body else None,
        "title_covered_by_body": shared / len(title) if title else None,
        "jaccard": shared / union if union else None,
    }


def _edges(graph: nx.Graph) -> set[tuple[str, str]]:
    return {tuple(sorted((u, v))) for u, v in graph.edges()}


def compare_link_sources(
    body_path: Path,
    title_path: Path,
    *,
    k: int = 10,
    results_dir: Path = Path("results/link_sources"),
) -> dict[str, object]:
    """Compare coverage, overlap, and signs without running opinion dynamics.

    Graphs omit self-links and merge both directions, as in the static pipeline.
    Post-link overlap uses all raw (source, target, POST_ID) keys; shared event
    occurrences use the minimum multiplicity per key, not a deduplicated total.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    frames = {}
    graphs = {}
    records = {}
    sources = {}
    for name, path in (("body", body_path), ("title", title_path)):
        print(f"Reading {name} links: {path}", flush=True)
        events = load_interactions(path, include_post_id=True)
        if events.empty:
            raise ValueError(f"The {name} input contains no interactions")
        graph = aggregate_edges(events)
        with path.open("rb") as raw:
            checksum = hashlib.file_digest(raw, "sha256").hexdigest()
        post_links = Counter(
            events[["source", "target", "post_id"]].itertuples(index=False, name=None)
        )
        frames[name] = events
        graphs[name] = graph
        records[name] = post_links
        sources[name] = {
            "path": str(path.resolve()),
            "dataset_url": DATASET_URLS[name],
            "sha256": checksum,
            "interactions": len(events),
            "self_interactions": int((events.source == events.target).sum()),
            "first_timestamp": events.timestamp.min().isoformat(),
            "last_timestamp": events.timestamp.max().isoformat(),
            "negative_interaction_fraction": float((events.sentiment == -1).mean()),
            "positive_interaction_fraction": float((events.sentiment == 1).mean()),
            "distinct_post_links": len(post_links),
            "repeated_post_link_rows": len(events) - len(post_links),
            **graph_summary(graph),
        }

    body_graph, title_graph = graphs["body"], graphs["title"]
    body_edges, title_edges = _edges(body_graph), _edges(title_graph)
    shared_edges = body_edges & title_edges
    agreeing_signs = sum(
        body_graph[u][v]["sign"] == title_graph[u][v]["sign"] for u, v in shared_edges
    )
    directed_pairs = {
        name: set(
            events.loc[events.source != events.target, ["source", "target"]].itertuples(
                index=False, name=None
            )
        )
        for name, events in frames.items()
    }
    cores = {
        name: largest_component(extract_core(graph, method="k_core", k=k))
        for name, graph in graphs.items()
    }
    body_core = cores["body"]
    title_on_body_core = title_graph.subgraph(body_core).copy()
    # Retain absent nodes as isolates so coverage uses the original body-core N.
    title_on_body_core.add_nodes_from(body_core)
    covered_nodes = len(set(body_core) & set(title_graph))
    active_nodes = sum(degree > 0 for _, degree in title_on_body_core.degree())
    body_core_edges = _edges(body_core)
    covered_edges = len(body_core_edges & title_edges)
    matched_occurrences = sum((records["body"] & records["title"]).values())

    overlap = {
        "nodes": _overlap(set(body_graph), set(title_graph)),
        "directed_pairs": _overlap(directed_pairs["body"], directed_pairs["title"]),
        "undirected_edges": _overlap(body_edges, title_edges),
        "post_link_keys": _overlap(set(records["body"]), set(records["title"])),
        "core_nodes": _overlap(set(body_core), set(cores["title"])),
    }
    summary = {
        "sources": sources,
        "overlap": overlap,
        "post_link_occurrences": {
            "key": ["source", "target", "POST_ID"],
            "matched_occurrences": matched_occurrences,
            "unmatched_body_occurrences": len(frames["body"]) - matched_occurrences,
            "unmatched_title_occurrences": len(frames["title"]) - matched_occurrences,
            "body_covered_by_title": matched_occurrences / len(frames["body"]),
            "title_covered_by_body": matched_occurrences / len(frames["title"]),
        },
        "shared_edge_signs": {
            "shared_edges": len(shared_edges),
            "agreeing": agreeing_signs,
            "disagreeing": len(shared_edges) - agreeing_signs,
            "agreement_fraction": agreeing_signs / len(shared_edges) if shared_edges else None,
        },
        "cores": {name: graph_summary(core) for name, core in cores.items()},
        "core_k": k,
        "title_coverage_of_body_core": {
            "original_nodes": len(body_core),
            "nodes_present_in_title": covered_nodes,
            "node_coverage": covered_nodes / len(body_core) if body_core else None,
            "nodes_with_title_neighbors_in_body_core": active_nodes,
            "active_node_fraction": active_nodes / len(body_core) if body_core else None,
            "original_edges": len(body_core_edges),
            "edges_present_in_title": covered_edges,
            "edge_coverage": covered_edges / len(body_core_edges) if body_core_edges else None,
            "title_graph_on_fixed_body_nodes": graph_summary(title_on_body_core),
            "largest_component_nodes": len(largest_component(title_on_body_core)),
        },
        "methods": {
            "graph": "Undirected majority sign; self-links removed; zero-sign edges retained",
            "post_links": "All raw directed keys; shared occurrences use minimum multiplicity",
            "coverage": "Presence and overlap only; no dynamics or combined-dataset rerun",
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame.from_dict(sources, orient="index").rename_axis("source").to_csv(
        results_dir / "source_summary.csv"
    )
    pd.DataFrame.from_dict(overlap, orient="index").rename_axis("quantity").to_csv(
        results_dir / "overlap.csv"
    )
    members = sorted(set(body_core) | set(cores["title"]))
    pd.DataFrame(
        {
            "node": members,
            "body_core": [node in body_core for node in members],
            "title_core": [node in cores["title"] for node in members],
        }
    ).to_csv(results_dir / "core_membership.csv", index=False)
    print(f"Saved link-source comparison to {results_dir}", flush=True)
    return summary
