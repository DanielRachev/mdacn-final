"""Centrality scores on the unsigned, unweighted core topology."""

import networkx as nx


def degree_centrality(graph: nx.Graph) -> dict[str, float]:
    """Owner: Polly. Return degree scores d_i, keyed by subreddit."""
    return {node: float(score) for node, score in nx.degree_centrality(graph).items()}


def betweenness_centrality(graph: nx.Graph) -> dict[str, float]:
    """Owner: Polly. Return shortest-path betweenness scores, keyed by subreddit."""
    return {node: float(score) for node, score in nx.betweenness_centrality(graph).items()}


def eigenvector_centrality(graph: nx.Graph) -> dict[str, float]:
    """Owner: Polly. Return eigenvector centrality scores, keyed by subreddit."""
    if graph.number_of_nodes() == 0:
        return {}
    return {node: float(score) for node, score in nx.eigenvector_centrality(graph).items()}


def leverage_centrality(graph: nx.Graph) -> dict[str, float]:
    """Owner: Polly. Return node leverage centrality l_ni per subreddit.

    The metric measures a node's degree advantage relative to its neighbors.
    """
    degrees = dict(graph.degree())
    scores: dict[str, float] = {}
    for node in graph:
        degree = degrees[node]
        if degree == 0:
            scores[node] = 0.0
            continue
        total = sum(
            (degree - degrees[neighbor]) / (degree + degrees[neighbor])
            for neighbor in graph.neighbors(node)
        )
        scores[node] = total / degree
    return scores


def rank_nodes(scores: dict[str, float]) -> list[str]:
    """Owner: Polly. Return subreddit names ranked from highest to lowest score.

    Equal scores have a deterministic order.
    """
    return [node for node, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
