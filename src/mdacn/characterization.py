"""Static metrics on the unsigned, unweighted core topology."""

import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix


def degree_distribution(graph: nx.Graph) -> pd.DataFrame:
    """Owner: Jacek. Return degree, count, and probability columns for P(k)."""
    raise NotImplementedError("Jacek: degree_distribution")


def fit_degree_baselines(graph: nx.Graph) -> pd.DataFrame:
    """Owner: Jacek. Return power-law and Poisson fit summaries.

    Record model, parameters, fitting support, sample size, and fit evidence.
    These fits assess heavy tails; they do not presume scale-free structure.
    """
    raise NotImplementedError("Jacek: fit_degree_baselines")


def degree_assortativity(graph: nx.Graph) -> float:
    """Owner: Jacek. Return the degree correlation coefficient rho_D."""
    raise NotImplementedError("Jacek: degree_assortativity")


def detect_communities(graph: nx.Graph, *, seed: int) -> dict[str, int]:
    """Owner: Jacek. Return one community label per subreddit."""
    raise NotImplementedError("Jacek: detect_communities")


def community_metrics(graph: nx.Graph, communities: dict[str, int]) -> dict[str, float]:
    """Owner: Jacek. Return modularity and average_clustering for the given partition."""
    raise NotImplementedError("Jacek: community_metrics")


def spectral_metrics(adjacency: csr_matrix) -> dict[str, float]:
    """Owner: Jacek. Return spectral_radius and algebraic_connectivity.

    Input is unsigned binary adjacency. Algebraic connectivity is the
    second-smallest eigenvalue of the combinatorial Laplacian.
    """
    raise NotImplementedError("Jacek: spectral_metrics")
