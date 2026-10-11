"""Static metrics on the unsigned, unweighted core topology."""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.special import zeta
from scipy.stats import poisson


def degree_distribution(graph: nx.Graph) -> pd.DataFrame:
    """Owner: Jacek. Return degree, count, and probability columns for P(k)."""
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=np.int64)
    values, counts = np.unique(degrees, return_counts=True)
    probabilities = counts / degrees.size if degrees.size else np.array([], dtype=float)
    return pd.DataFrame({"degree": values, "count": counts, "probability": probabilities})


def fitted_degree_probabilities(
        degrees: NDArray[np.int64], *, model: str, parameter: float, k_min: int
) -> NDArray[np.float64]:
    """Evaluate a fitted discrete PMF on its support"""
    probabilities = np.zeros(degrees.shape, dtype=float)
    support = degrees >= k_min
    values = degrees[support].astyoe(float)
    if model == "power_law":
        probabilities[support] = np.power(values, -parameter) / zeta(parameter, k_min)
    elif model == "poisson":
        probabilities[support] = np.exp(
            poisson.logpmf(values, parameter) - poisson.logsf(k_min - 1, parameter)
        )
    else:
        raise ValueError(f"Unknown degree modelling model: {model}")
    return probabilities

def _ks_distance(
        degrees: NDArray[np.float64], *, model: str, parameter: float, k_min: int
) -> float:
    """Compare discrete CDF before and after observed degree values"""
    values, counts = np.unique(degrees, return_counts=True)
    empirical_after = np.cumsum(counts) / degrees.size
    empirical_before = empirical_after - counts/degrees.size

    if model == "power_law":
        normaliser = zeta(parameter, k_min)
        model_before = 1 - zeta(parameter, values) / normaliser
        model_after = 1 - zeta(parameter, values + 1) / normaliser
    elif model == "poisson":
        log_normaliser = poisson.logsf(k_min - 1, parameter)
        model_before = 1 - np.exp(poisson.logsf(values - 1, parameter) - log_normaliser)
        model_after = 1 - np.exp(poisson.logsf(values, parameter) - log_normaliser)
    else:
        raise ValueError(f"Unsupported degree model {model}")
    return max(
        np.max(np.abs(empirical_before - model_before)),
        np.max(np.abs(empirical_after - model_after))
    )


def fit_degree_baselines(graph: nx.Graph) -> pd.DataFrame:
    """Owner: Jacek. Return power-law and Poisson fit summaries.

    Record model, parameters, fitting support, sample size, and fit evidence.
    These fits assess heavy tails; they do not presume scale-free structure.
    """
    raise NotImplementedError("Jacek: fit_degree_baselines")


def degree_assortativity(graph: nx.Graph) -> float:
    """Owner: Jacek. Return the degree correlation coefficient rho_D."""
    if len({degree for _, degree in graph.degree() if degree > 0}) < 2:
        return float("nan")
    return float(nx.degree_assortativity_coefficient(graph, weight=None))


def detect_communities(graph: nx.Graph, *, seed: int) -> dict[str, int]:
    """Owner: Jacek. Return one community label per subreddit."""
    groups = nx.community.louvain_communities(graph, weight=None, seed=seed)
    return {
        node: label
        for label, group in
        enumerate(sorted(groups, key=min)) for node in sorted(group)
    }


def community_metrics(graph: nx.Graph, communities: dict[str, int]) -> dict[str, float]:
    """Owner: Jacek. Return modularity and average_clustering for the given partition."""
    groups: dict[int, set[str]] = {}

    for node, label in communities.items():
        groups.setdefault(label, set()).add(node)

    modularity = (
        nx.community.modularity(graph, groups.values(), weight=None)
        if graph.number_of_edges()
        else float("nan")
    )
    clustering = nx.average_clustering(graph, weight=None) if graph else float("nan")
    return {"modularity": float(modularity), "average_clustering": float(clustering)}


def spectral_metrics(adjacency: csr_matrix) -> dict[str, float]:
    """Owner: Jacek. Return spectral_radius and algebraic_connectivity.

    Input is unsigned binary adjacency. Algebraic connectivity is the
    second-smallest eigenvalue of the combinatorial Laplacian.
    """
    n_nodes = adjacency.shape[0]

    adjacency = adjacency.astype(float)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(degrees) - adjacency
    n_components = connected_components(adjacency, directed=False, return_labels=False)
    if n_nodes <= 32:
        radius = float(np.linalg.eigvalsh(adjacency.toarray())[-1])
        connectivity = (
            float(np.linalg.eigvalsh(laplacian.toarray())[-1])
            if n_nodes > 1 and n_components == 1
            else 0.0
        )
    else:
        rng = np.random.default_rng(42)
        radius = float(
            eigsh(
                adjacency,
                k=1,
                which="LA",
                v0=rng.normal(size=n_nodes),
                tol=1e-8,
                return_eigenvectors=False,
            )[0]
        )
        connectivity = 0.0
        if n_components == 1:
            eigenvalues = eigsh(
                laplacian,
                k=2,
                which="SM",
                v0=rng.normal(size=n_nodes),
                tol=1e-8,
                return_eigenvectors=False
            )
            connectivity = float(np.sort(eigenvalues)[1])
    return {"spectral_radius": max(0.0, radius), "algebraic_connectivity": max(0.0, connectivity)}