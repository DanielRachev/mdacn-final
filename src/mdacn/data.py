"""Reddit input, static core extraction, and shared graph export."""

import hashlib
import json
from pathlib import Path
from shutil import copyfileobj
from typing import Literal
from urllib.request import urlopen

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from mdacn.artifacts import checksum, write_json
from mdacn.config import processed_paths
from mdacn.contracts import NetworkMatrices

DATASET_URLS = {
    source: f"https://snap.stanford.edu/data/soc-redditHyperlinks-{source}.tsv"
    for source in ("body", "title")
}
DATASET_URL = DATASET_URLS["body"]


def canonical_edges(graph: nx.Graph) -> list[tuple]:
    """Return undirected edge records ordered by their sorted endpoints."""
    return sorted(
        ((min(u, v), max(u, v), data) for u, v, data in graph.edges(data=True)),
        key=lambda edge: (edge[0], edge[1]),
    )


def edge_order_metadata(graph: nx.Graph) -> dict:
    """Identify the signed edge sequence consumed by seeded randomization."""
    records = [(str(u), str(v), int(data["sign"])) for u, v, data in canonical_edges(graph)]
    payload = json.dumps(records, separators=(",", ":")).encode()
    return {
        "edge_order": "canonical-endpoints-v1",
        "signed_edge_sequence_sha256": hashlib.sha256(payload).hexdigest(),
    }


def download_dataset(path: Path, *, dataset: Literal["body", "title"] = "body") -> Path:
    """Download a SNAP link dataset, keeping an existing file."""
    if dataset not in DATASET_URLS:
        raise ValueError(f"Unknown link dataset: {dataset}")
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")

    try:
        with urlopen(DATASET_URLS[dataset], timeout=60) as response, temp.open("wb") as output:
            copyfileobj(response, output)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def load_interactions(
    path: Path, *, include_post_id: bool = False, post_ids: bool = False
) -> pd.DataFrame:
    """Owner: Jacek. Read a body/title TSV into the interaction table contract.

    Return source, target, sentiment, and timestamp columns. Preserve directed
    events and repeated interactions. Optionally retain post_id for overlap audits.
    """
    include_post_id = include_post_id or post_ids
    columns = {
        "SOURCE_SUBREDDIT": "source",
        "TARGET_SUBREDDIT": "target",
        "LINK_SENTIMENT": "sentiment",
        "TIMESTAMP": "timestamp",
    }
    if include_post_id:
        columns["POST_ID"] = "post_id"

    interactions = pd.read_csv(
        path, sep="\t", usecols=list(columns), dtype=str, keep_default_na=False
    ).rename(columns=columns)
    interactions = interactions[list(columns.values())]
    sentiments = pd.to_numeric(interactions["sentiment"], errors="coerce")
    timestamps = pd.to_datetime(
        interactions["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    invalid = (
        interactions["source"].str.strip().eq("")
        | interactions["target"].str.strip().eq("")
        | ~sentiments.isin([-1, 1])
        | timestamps.isna()
    )
    if include_post_id:
        invalid |= interactions["post_id"].str.strip().eq("")

    if invalid.any():
        lines = (interactions.index[invalid] + 2).tolist()
        raise ValueError(f"Invalid interaction records at TSV lines {lines[:5]}")

    interactions["sentiment"] = sentiments.astype(np.int8)
    interactions["timestamp"] = timestamps
    return interactions


def edge_sign_counts(interactions: pd.DataFrame) -> pd.DataFrame:
    """Return u < v, n_positive, n_negative, and majority sign per unordered pair.

    Self-links are dropped and both directions are merged. The sign is the sign of
    n_positive - n_negative; equal counts give 0.
    """
    events = interactions.loc[interactions["source"] != interactions["target"]]
    endpoints = np.sort(events[["source", "target"]].to_numpy(dtype=str), axis=1)
    counts = (
        pd.DataFrame(
            {
                "u": endpoints[:, 0],
                "v": endpoints[:, 1],
                "n_positive": (events["sentiment"].to_numpy() == 1).astype(int),
                "n_negative": (events["sentiment"].to_numpy() == -1).astype(int),
            }
        )
        .groupby(["u", "v"], sort=True)
        .sum()
        .reset_index()
    )
    counts["sign"] = np.sign(counts["n_positive"] - counts["n_negative"]).astype(int)
    return counts


def aggregate_edges(interactions: pd.DataFrame) -> nx.Graph:
    """Owner: Jacek. Return a simple undirected graph of non-self interactions.

    Each edge records n_positive, n_negative, and sign across both directions.
    Its sign is the sign of n_positive - n_negative. Equal counts give sign 0;
    the edge remains in the unsigned topology but has no signed influence.
    """
    graph = nx.Graph()
    graph.add_nodes_from(sorted(set(interactions["source"]) | set(interactions["target"])))

    for u, v, n_positive, n_negative, sign in edge_sign_counts(interactions).itertuples(
        index=False, name=None
    ):
        graph.add_edge(
            u,
            v,
            n_positive=int(n_positive),
            n_negative=int(n_negative),
            sign=int(sign),
        )

    for node in graph:
        graph.nodes[node]["n_interactions"] = sum(
            edge["n_positive"] + edge["n_negative"] for edge in graph[node].values()
        )
    return graph


def extract_core(
    graph: nx.Graph,
    *,
    method: Literal["k_core", "activity_lcc"],
    k: int | None = None,
    min_interactions: int | None = None,
) -> nx.Graph:
    """Owner: Jacek. Return an active core with original edge attributes.

    k_core selects the requested k-core. activity_lcc selects the largest
    connected component after a minimum incident-interaction threshold.
    Only the threshold for the selected method is required.
    """
    if method == "k_core":
        if k is None or k < 0:
            raise ValueError("k_core requires a non-negative k")
        return nx.k_core(graph, k=k)
    if method == "activity_lcc":
        if min_interactions is None or min_interactions < 0:
            raise ValueError("activity_lcc requires a non-negative min_interactions")
        nodes = [
            node
            for node in graph
            if sum(edge["n_positive"] + edge["n_negative"] for edge in graph[node].values())
            >= min_interactions
        ]
        return largest_component(graph.subgraph(nodes))
    raise ValueError(f"Unknown core method: {method}")


def largest_component(graph: nx.Graph) -> nx.Graph:
    """Copy the largest connected component, resolving ties by node name."""
    if not graph:
        return graph.copy()
    nodes = min(nx.connected_components(graph), key=lambda x: (-len(x), min(x)))
    return graph.subgraph(nodes).copy()


def community_labels(graph: nx.Graph, attribute: str = "community") -> dict:
    """Return the fixed community label of every node; every node must have one."""
    missing = [node for node in graph if attribute not in graph.nodes[node]]
    if missing:
        raise ValueError(f"Nodes missing '{attribute}' labels: {missing[:5]}")
    return {node: graph.nodes[node][attribute] for node in graph}


def build_matrices(graph: nx.Graph) -> NetworkMatrices:
    """Owner: Jacek. Return binary adjacency and signed adjacency in one node order."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Matrices require a simple undirected graph without self loops")
    if any(edge.get("sign") not in (-1, 0, 1) for _, _, edge in graph.edges(data=True)):
        raise ValueError("Every edge must have a sign of -1, 0, or +1")

    nodes = tuple(sorted(graph))
    if not nodes:
        return NetworkMatrices(nodes, csr_matrix((0, 0)), csr_matrix((0, 0)))

    unsigned = csr_matrix(
        nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None, dtype=float, format="csr")
    )
    signed = csr_matrix(
        nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="sign", dtype=float, format="csr")
    )
    signed.eliminate_zeros()
    return NetworkMatrices(nodes, unsigned, signed)


def export_graph(graph: nx.Graph, path: Path) -> None:
    """Owner: Jacek. Save a GraphML graph with subreddit names and edge attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def save_matrices(network: NetworkMatrices, graph_path: Path) -> None:
    """Save aligned matrices and their association with the exported graph."""
    paths = processed_paths(graph_path.parent)
    save_npz(paths["unsigned"], network.unsigned)
    save_npz(paths["signed"], network.signed)
    write_json(paths["nodes"], network.nodes)
    write_json(
        paths["network_metadata"],
        {
            "graph_sha256": checksum(graph_path),
            "files": {name: checksum(paths[name]) for name in ("unsigned", "signed", "nodes")},
        },
    )


def load_matrices(graph_path: Path) -> NetworkMatrices:
    """Load previously exported matrices; missing or stale inputs are errors."""
    paths = processed_paths(graph_path.parent)
    for name in ("unsigned", "signed", "nodes", "network_metadata"):
        if not paths[name].is_file():
            raise ValueError(f"Missing matrix input: {paths[name]}. Run characterise first.")
    metadata = json.loads(paths["network_metadata"].read_text(encoding="utf-8"))
    if metadata["graph_sha256"] != checksum(graph_path):
        raise ValueError("Core changed since matrices were exported; run characterise again")
    for name in ("unsigned", "signed", "nodes"):
        if metadata["files"][name] != checksum(paths[name]):
            raise ValueError(f"Saved {name} changed; run characterise again")
    nodes = tuple(json.loads(paths["nodes"].read_text(encoding="utf-8")))
    unsigned, signed = (load_npz(paths[name]).tocsr() for name in ("unsigned", "signed"))
    if len(set(nodes)) != len(nodes) or tuple(sorted(nodes)) != nodes:
        raise ValueError("Saved node order must be sorted and unique")
    if unsigned.shape != (len(nodes), len(nodes)) or signed.shape != unsigned.shape:
        raise ValueError("Saved matrices do not match the node order")
    return NetworkMatrices(nodes, unsigned, signed)


def load_graph(path: Path) -> nx.Graph:
    """Owner: Jacek. Restore the GraphML graph for Daniel and Polly."""
    return nx.read_graphml(path)
