"""Reddit input, static core extraction, and shared graph export."""

from pathlib import Path
from typing import Literal

import networkx as nx
import pandas as pd

from mdacn.contracts import NetworkMatrices


def load_interactions(path: Path) -> pd.DataFrame:
    """Owner: Jacek. Read the body TSV into the interaction table contract.

    Return source, target, sentiment, and timestamp columns. Preserve directed
    events and repeated interactions.
    """
    raise NotImplementedError("Jacek: load_interactions")


def aggregate_edges(interactions: pd.DataFrame) -> nx.Graph:
    """Owner: Jacek. Return a simple undirected graph of non-self interactions.

    Each edge records n_positive, n_negative, and sign across both directions.
    Its sign is +1 when n_positive >= n_negative, and -1 otherwise.
    """
    raise NotImplementedError("Jacek: aggregate_edges")


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
    raise NotImplementedError("Jacek: export_graph")


def load_graph(path: Path) -> nx.Graph:
    """Owner: Jacek. Restore the GraphML graph for Daniel and Polly."""
    raise NotImplementedError("Jacek: load_graph")
