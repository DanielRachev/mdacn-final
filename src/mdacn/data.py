"""Reddit input, static core extraction, and shared graph export."""

from pathlib import Path
from shutil import copyfileobj
from typing import Literal
from urllib.request import urlopen

import networkx as nx
import numpy as np
import pandas as pd

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
        raise ValueError(f"Something is wrong with the input data at lines {lines[:5]}")

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
            sign=1 if n_positive > n_negative else -1,
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
    raise NotImplementedError("Jacek: extract_core")


def build_matrices(graph: nx.Graph) -> NetworkMatrices:
    """Owner: Jacek. Return binary adjacency and signed adjacency in one node order."""
    raise NotImplementedError("Jacek: build_matrices")


def export_graph(graph: nx.Graph, path: Path) -> None:
    """Owner: Jacek. Save a GraphML graph with subreddit names and edge attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def load_graph(path: Path) -> nx.Graph:
    """Owner: Jacek. Restore the GraphML graph for Daniel and Polly."""
    return nx.read_graphml(path)
