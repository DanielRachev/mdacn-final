"""Centrality scores on the unsigned, unweighted core topology."""

import networkx as nx


def degree_centrality(graph: nx.Graph) -> dict[str, float]:
    """Owner: Polly. Return degree scores d_i, keyed by subreddit."""
    raise NotImplementedError("Polly: degree_centrality")


def betweenness_centrality(graph: nx.Graph) -> dict[str, float]:
    """Owner: Polly. Return shortest-path betweenness scores, keyed by subreddit."""
    raise NotImplementedError("Polly: betweenness_centrality")


def eigenvector_centrality(graph: nx.Graph) -> dict[str, float]:
    """Owner: Polly. Return eigenvector centrality scores, keyed by subreddit."""
    raise NotImplementedError("Polly: eigenvector_centrality")


def leverage_centrality(graph: nx.Graph) -> dict[str, float]:
    """Owner: Polly. Return node leverage centrality l_ni per subreddit.

    The metric measures a node's degree advantage relative to its neighbors.
    """
    raise NotImplementedError("Polly: leverage_centrality")


def rank_nodes(scores: dict[str, float]) -> list[str]:
    """Owner: Polly. Return subreddit names ranked from highest to lowest score.

    Equal scores have a deterministic order.
    """
    raise NotImplementedError("Polly: rank_nodes")
