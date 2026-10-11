"""Reddit input, static core extraction, and shared graph export."""

from pathlib import Path
from shutil import copyfileobj
from typing import Literal
from urllib.request import urlopen

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from mdacn.contracts import NetworkMatrices

DATASET_URL = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"


def download_dataset(path: Path) -> Path:
    """Download the body TSV, doing nothing if already exists"""
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")

    try:
        with urlopen(DATASET_URL, timeout=60) as response, temp.open("wb") as output:
            copyfileobj(response, output)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def load_interactions(path: Path) -> pd.DataFrame:
    """Owner: Jacek. Read the body TSV into the interaction table contract.

    Return source, target, sentiment, and timestamp columns. Preserve directed
    events and repeated interactions.
    """
    columns = {
        "SOURCE_SUBREDDIT": "source",
        "TARGET_SUBREDDIT": "target",
        "LINK_SENTIMENT": "sentiment",
        "TIMESTAMP": "timestamp",
    }

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

    if invalid.any():
        lines = (interactions.index[invalid] + 2).tolist()
        raise ValueError(f"Invalid interaction records at TSV lines {lines[:5]}")

    interactions["sentiment"] = sentiments.astype(np.int8)
    interactions["timestamp"] = timestamps
    return interactions


def aggregate_edges(interactions: pd.DataFrame) -> nx.Graph:
    """Owner: Jacek. Return a simple undirected graph of non-self interactions.

    Each edge records n_positive, n_negative, and sign across both directions.
    Its sign is +1 when n_positive >= n_negative, and -1 otherwise.
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
    )

    graph = nx.Graph()
    graph.add_nodes_from(sorted(set(interactions["source"]) | set(interactions["target"])))

    for u, v, n_positive, n_negative in counts.reset_index().itertuples(index=False, name=None):
        graph.add_edge(
            u,
            v,
            n_positive=int(n_positive),
            n_negative=int(n_negative),
            sign=1 if n_positive >= n_negative else -1,
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
            node for node in graph
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


def build_matrices(graph: nx.Graph) -> NetworkMatrices:
    """Owner: Jacek. Return binary adjacency and signed adjacency in one node order."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Matrices require a simple undirected graph without self loops")
    if any(edge.get("sign") not in (-1, 1) for _, _, edge in graph.edges(data=True)):
        raise ValueError("Every edge must have a sign of -1 or +1 (for now?)")

    nodes = tuple(sorted(graph))
    if not nodes:
        return NetworkMatrices(nodes, csr_matrix((0, 0)), csr_matrix((0, 0)))

    unsigned = csr_matrix(
        nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None, dtype=float, format="csr")
    )
    signed = csr_matrix(
        nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="sign", dtype=float, format="csr")
    )
    return NetworkMatrices(nodes, unsigned, signed)


def export_graph(graph: nx.Graph, path: Path) -> None:
    """Owner: Jacek. Save a GraphML graph with subreddit names and edge attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def load_graph(path: Path) -> nx.Graph:
    """Owner: Jacek. Restore the GraphML graph for Daniel and Polly."""
    return nx.read_graphml(path)
