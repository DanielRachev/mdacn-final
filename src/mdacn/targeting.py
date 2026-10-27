"""Proposed targeting scores for contrarian selection."""

from collections.abc import Mapping

import networkx as nx
import numpy as np

from mdacn.contracts import NetworkMatrices


def _degrees(network: NetworkMatrices) -> np.ndarray:
    return np.asarray(network.unsigned.tocsr().sum(axis=1), dtype=float).ravel()


def signed_leverage_centrality(network: NetworkMatrices) -> dict[str, float]:
    """Owner: Georgi. Return signed leverage scores per subreddit.

    A neighbour contributes its leverage only when its degree is smaller, and a
    negative edge lowers the score. d_i is the unsigned core degree, so zero-sign
    edges count in the denominator. Isolated nodes score 0.0.
    """
    n_nodes = len(network.nodes)
    if network.unsigned.shape != (n_nodes, n_nodes) or network.signed.shape != (n_nodes, n_nodes):
        raise ValueError("Matrices must match the node order length")
    unsigned = network.unsigned.tocsr()
    signed = network.signed.tocsr()
    if (signed - signed.multiply(unsigned)).nnz:
        raise ValueError("Signed edges must belong to the unsigned topology")

    degrees = _degrees(network)
    scores: dict[str, float] = {}
    for index, node in enumerate(network.nodes):
        degree = degrees[index]
        if degree == 0:
            scores[node] = 0.0
            continue
        start, end = signed.indptr[index], signed.indptr[index + 1]
        total = 0.0
        for neighbour, weight in zip(
            signed.indices[start:end], signed.data[start:end], strict=True
        ):
            neighbour_degree = degrees[neighbour]
            leverage = (degree - neighbour_degree) / (degree + neighbour_degree)
            if leverage > 0:
                total += float(weight) * float(leverage)
        scores[node] = total / float(degree)
    return scores


def bridge_fraction(graph: nx.Graph, communities: Mapping[str, int]) -> dict[str, float]:
    """Return the share of each node's neighbours outside its community.

    Uses the unsigned topology. Isolated nodes score 0.0.
    """
    missing = [node for node in graph if node not in communities]
    if missing:
        raise ValueError(f"Missing community labels for nodes: {missing}")
    scores: dict[str, float] = {}
    for node in graph:
        neighbours = list(graph.neighbors(node))
        if not neighbours:
            scores[node] = 0.0
            continue
        own = communities[node]
        scores[node] = sum(communities[neighbour] != own for neighbour in neighbours) / len(
            neighbours
        )
    return scores


def _resolve_activity(graph: nx.Graph, activity: Mapping[str, float] | None) -> dict[str, float]:
    if activity is None:
        counts = {node: data.get("n_interactions") for node, data in graph.nodes(data=True)}
        if any(value is None for value in counts.values()):
            raise ValueError(
                "Core graph has no n_interactions attribute; supply a per-node activity table"
            )
        activity = counts
    missing = [node for node in graph if node not in activity]
    if missing:
        raise ValueError(f"Missing activity values for nodes: {missing}")
    values = {node: float(activity[node]) for node in graph}
    if not all(np.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("Activity values must be finite and non-negative")
    return values


def temporal_activity_bridge_centrality(
    graph: nx.Graph,
    communities: Mapping[str, int],
    activity: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Owner: Georgi. Return temporal-activity bridge scores per subreddit.

    The score is the geometric mean of the bridge fraction and the node's
    activity relative to the largest activity. Activity defaults to the core
    graph's n_interactions count. Nodes without cross-community edges or without
    activity score 0.0.
    """
    bridge = bridge_fraction(graph, communities)
    counts = _resolve_activity(graph, activity)
    largest = max(counts.values(), default=0.0)
    scores: dict[str, float] = {}
    for node in graph:
        normalized = counts[node] / largest if largest > 0 else 0.0
        scores[node] = float(np.sqrt(bridge[node] * normalized))
    return scores
