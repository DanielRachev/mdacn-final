"""Incident sign heterogeneity on the static signed core, and shared signed helpers."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import false_discovery_control, hypergeom

from mdacn.artifacts import checksum, write_json
from mdacn.data import community_labels, load_graph
from mdacn.plots import plot_negative_ratio_distribution

BinScheme = Literal["width", "quantile"]
_EDGE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SignedEdges:
    """Edge endpoints and signs in graph.edges order; nodes in nodes.json order.

    Keeping graph.edges order lets shuffle seeds replay through shuffle_edge_signs.
    """

    nodes: tuple[str, ...]
    sources: NDArray[np.intp]
    targets: NDArray[np.intp]
    signs: NDArray[np.int8]


def safe_ratio(numerator: ArrayLike, denominator: ArrayLike) -> NDArray[np.float64] | float:
    """Divide element-wise; a zero denominator gives NaN. Scalars return a float."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    shape = np.broadcast_shapes(numerator.shape, denominator.shape)
    result = np.divide(numerator, denominator, out=np.full(shape, np.nan), where=denominator != 0)
    return float(result) if result.ndim == 0 else result


def spawn_seeds(entropy: int | Sequence[int], count: int) -> tuple[int, ...]:
    """Return count independent integer seeds spawned from one SeedSequence."""
    return tuple(
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(entropy).spawn(count)
    )


def shuffled_signs(signs: NDArray[np.int8], seeds: Sequence[int]) -> Iterator[NDArray[np.int8]]:
    """Yield one exact sign permutation per seed, as shuffle_edge_signs draws it."""
    for seed in seeds:
        yield np.random.default_rng(seed).permutation(signs)


def null_comparison(observed: float, values: ArrayLike) -> dict[str, float | int]:
    """Compare one empirical statistic with a null ensemble.

    p_upper and p_lower are one-sided Monte Carlo p-values with the +1 correction;
    z uses the ensemble standard deviation and is NaN without spread.
    """
    values = np.asarray(values, dtype=float)
    spread = values.std(ddof=1) if values.size > 1 else np.nan
    return {
        "empirical": float(observed),
        "realizations": int(values.size),
        "mean": float(values.mean()),
        "std": float(spread),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "z": float((observed - values.mean()) / spread) if spread > 0 else np.nan,
        "p_upper": float((1 + np.sum(values >= observed)) / (1 + values.size)),
        "p_lower": float((1 + np.sum(values <= observed)) / (1 + values.size)),
    }


def node_positions(events: pd.DataFrame, nodes: Sequence[str]) -> tuple[NDArray, NDArray]:
    """Return source and target positions in nodes for every event."""
    index = {node: i for i, node in enumerate(nodes)}
    source, target = events["source"].map(index), events["target"].map(index)
    if source.isna().any() or target.isna().any():
        raise ValueError("Every event must connect two listed nodes")
    return source.to_numpy(np.int64), target.to_numpy(np.int64)


def signed_edges(graph: nx.Graph) -> SignedEdges:
    """Return aligned edge arrays for a simple undirected signed graph."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Incident signs require a simple undirected graph without self loops")
    nodes = tuple(sorted(graph))
    index = {node: i for i, node in enumerate(nodes)}
    edges = list(graph.edges(data="sign"))
    if any(sign not in (-1, 0, 1) for _, _, sign in edges):
        raise ValueError("Every edge must have a sign of -1, 0, or +1")
    return SignedEdges(
        nodes=nodes,
        sources=np.array([index[u] for u, _, _ in edges], dtype=np.intp),
        targets=np.array([index[v] for _, v, _ in edges], dtype=np.intp),
        signs=np.array([sign for _, _, sign in edges], dtype=np.int8),
    )


def incident_counts(edges: SignedEdges, mask: NDArray[np.bool_] | None = None) -> NDArray[np.int64]:
    """Count incident edges per node, optionally only those selected by mask."""
    sources = edges.sources if mask is None else edges.sources[mask]
    targets = edges.targets if mask is None else edges.targets[mask]
    n_nodes = len(edges.nodes)
    return np.bincount(sources, minlength=n_nodes) + np.bincount(targets, minlength=n_nodes)


def _degrees(edges: SignedEdges) -> NDArray[np.int64]:
    degree = incident_counts(edges)
    if (degree == 0).any():
        raise ValueError("r_neg is undefined for isolated nodes")
    return degree


def negative_incident_ratios(graph: nx.Graph) -> pd.DataFrame:
    """Return per-node signed degrees and r_neg = d_neg / d in nodes.json order."""
    edges = signed_edges(graph)
    degree = _degrees(edges)
    negative = incident_counts(edges, edges.signs == -1)
    n_edges, n_negative = len(edges.signs), int((edges.signs == -1).sum())
    p_upper = hypergeom.sf(negative - 1, n_edges, n_negative, degree)
    table = pd.DataFrame(
        {
            "node": edges.nodes,
            "degree": degree,
            "positive_degree": incident_counts(edges, edges.signs == 1),
            "negative_degree": negative,
            "zero_degree": incident_counts(edges, edges.signs == 0),
            "r_neg": negative / degree,
            "expected_negative_degree": degree * n_negative / n_edges,
            "p_upper": p_upper,
            "q_upper": false_discovery_control(p_upper) if len(p_upper) else p_upper,
        }
    )
    communities = nx.get_node_attributes(graph, "community")
    if communities:
        table.insert(1, "community", [communities[node] for node in edges.nodes])
    return table


def shuffled_negative_ratios(
    graph: nx.Graph, *, realizations: int = 1000, seed: int = 0
) -> tuple[NDArray[np.float64], tuple[int, ...]]:
    """Return r_neg for each sign-shuffled realization, shaped (realizations, nodes).

    Shuffle i equals shuffle_edge_signs(graph, seed=shuffle_seeds[i]).
    """
    if isinstance(realizations, bool) or not isinstance(realizations, int) or realizations < 1:
        raise ValueError("realizations must be a positive integer")
    edges = signed_edges(graph)
    degree = _degrees(edges)
    shuffle_seeds = spawn_seeds(seed, realizations)
    ratios = np.stack(
        [
            incident_counts(edges, signs == -1) / degree
            for signs in shuffled_signs(edges.signs, shuffle_seeds)
        ]
    )
    return ratios, shuffle_seeds


def ratio_bin_edges(
    reference: NDArray[np.float64],
    *,
    scheme: BinScheme = "width",
    width: float = 0.025,
    n_bins: int = 20,
) -> NDArray[np.float64]:
    """Return bin edges on (0, 1] for positive r_neg; r_neg = 0 stays separate.

    width uses equal-width bins. quantile uses equal-count bins of the positive
    reference (sign-shuffled) values, so each bin has a similar expected count.
    Tied quantiles merge, so fewer than n_bins bins can result.
    """
    if scheme == "width":
        n_width = round(1 / width)
        if width <= 0 or not np.isclose(n_width * width, 1):
            raise ValueError("width must divide 1 into a whole number of bins")
        return np.round(np.linspace(0, 1, n_width + 1), 12)
    if scheme == "quantile":
        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        positive = reference[reference > 0]
        if not positive.size:
            raise ValueError("The reference has no positive r_neg values")
        inner = np.quantile(positive, np.linspace(0, 1, n_bins + 1)[1:-1])
        return np.unique(np.concatenate(([0.0], inner, [1.0])))
    raise ValueError(f"Unknown bin scheme: {scheme}")


def ratio_histogram(
    values: NDArray[np.float64], bin_edges: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Count r_neg = 0 first, then right-closed bins (left, right], one row per sample.

    Right-closed bins with a small tolerance keep exact fractions such as 1/10
    in the same bin whatever floating-point error the bin edges carry.
    """
    values = np.atleast_2d(values)
    bins = np.searchsorted(bin_edges, values - _EDGE_TOLERANCE, side="left")
    bins = np.clip(bins, 1, len(bin_edges) - 1)
    bins[values == 0] = 0
    return np.stack([np.bincount(row, minlength=len(bin_edges)) for row in bins])


def ratio_distribution(
    empirical: NDArray[np.float64],
    reference: NDArray[np.float64],
    bin_edges: NDArray[np.float64],
) -> pd.DataFrame:
    """Compare the empirical P(r_neg) with the sign-shuffled reference per bin.

    The first row is the point mass r_neg = 0. Density divides probability by bin
    width, so the positive part integrates to 1 - P(r_neg = 0).
    """
    n_nodes = empirical.size
    observed = ratio_histogram(empirical, bin_edges)[0] / n_nodes
    shuffled = ratio_histogram(reference, bin_edges) / n_nodes
    left = np.concatenate(([0.0], bin_edges[:-1]))
    right = np.concatenate(([0.0], bin_edges[1:]))
    width = right - left
    table = pd.DataFrame(
        {
            "bin": ["zero", *(f"({a:.4g}, {b:.4g}]" for a, b in zip(left[1:], right[1:]))],
            "left": left,
            "right": right,
            "count": np.rint(observed * n_nodes).astype(int),
            "probability": observed,
            "reference_mean": shuffled.mean(axis=0),
            "reference_q025": np.quantile(shuffled, 0.025, axis=0),
            "reference_q975": np.quantile(shuffled, 0.975, axis=0),
        }
    )
    for column in ("probability", "reference_mean", "reference_q025", "reference_q975"):
        name = column.replace("probability", "density").replace("reference_", "reference_density_")
        table[name] = safe_ratio(table[column], width)
    return table


def _top_share(negative_degree: NDArray, share: float) -> NDArray[np.float64]:
    """Share of negative edge ends held by the top fraction of nodes, per row."""
    negative_degree = np.atleast_2d(negative_degree)
    top = max(1, int(np.ceil(share * negative_degree.shape[1])))
    ordered = -np.sort(-negative_degree, axis=1)
    return safe_ratio(ordered[:, :top].sum(axis=1), negative_degree.sum(axis=1))


def concentration_summary(
    table: pd.DataFrame,
    reference: NDArray[np.float64],
    *,
    tail_threshold: float = 0.2,
    top_share: float = 0.05,
) -> pd.DataFrame:
    """Compare summary statistics of r_neg with the sign-shuffled reference."""
    empirical = table["r_neg"].to_numpy()
    degree = table["degree"].to_numpy()
    statistics = {
        "zero_fraction": lambda r: np.mean(np.atleast_2d(r) == 0, axis=1),
        f"fraction_above_{tail_threshold:g}": lambda r: np.mean(
            np.atleast_2d(r) > tail_threshold, axis=1
        ),
        "variance": lambda r: np.var(np.atleast_2d(r), axis=1),
        f"top_{top_share:g}_share_of_negative_ends": lambda r: _top_share(
            np.rint(np.atleast_2d(r) * degree), top_share
        ),
    }
    return pd.DataFrame(
        {"statistic": name} | null_comparison(statistic(empirical)[0], statistic(reference))
        for name, statistic in statistics.items()
    )


def community_sign_rates(graph: nx.Graph) -> pd.DataFrame:
    """Return negative-edge rates within and on the boundary of each community."""
    communities = community_labels(graph)
    edges = signed_edges(graph)
    labels = np.array([communities[node] for node in edges.nodes])
    source, target = labels[edges.sources], labels[edges.targets]
    negative = edges.signs == -1
    n_edges, n_negative = len(edges.signs), int(negative.sum())
    r_neg = negative_incident_ratios(graph)["r_neg"].to_numpy()

    def row(name: str, nodes: int, mask: NDArray[np.bool_], scope: str) -> dict:
        count, k = int(mask.sum()), int((negative & mask).sum())
        distribution = hypergeom(n_edges, n_negative, count)
        return {
            "community": name,
            "scope": scope,
            "nodes": nodes,
            "edges": count,
            "negative_edges": k,
            "zero_edges": int(((edges.signs == 0) & mask).sum()),
            "negative_rate": safe_ratio(k, count),
            "expected_negative_edges": count * n_negative / n_edges,
            "reference_q025": distribution.ppf(0.025) if count else np.nan,
            "reference_q975": distribution.ppf(0.975) if count else np.nan,
            "p_upper": float(distribution.sf(k - 1)) if count else np.nan,
            "p_lower": float(distribution.cdf(k)) if count else np.nan,
        }

    rows = []
    for label in sorted(set(labels)):
        members = labels == label
        within = (source == label) & (target == label)
        boundary = (source == label) != (target == label)
        for scope, mask in (("within", within), ("boundary", boundary)):
            rows.append(row(str(label), int(members.sum()), mask, scope))
            rows[-1]["mean_member_r_neg"] = float(r_neg[members].mean())
    rows.append(row("all", len(labels), source == target, "within"))
    rows.append(row("all", len(labels), source != target, "between"))
    return pd.DataFrame(rows)


def run_incident_sign_analysis(
    graph_path: Path,
    *,
    results_dir: Path = Path("results/incident_signs"),
    realizations: int = 1000,
    seed: int = 0,
    bins: BinScheme = "width",
    bin_width: float = 0.025,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Save per-node r_neg, P(r_neg) against sign shuffles, community rates, and figures."""
    graph = load_graph(graph_path)
    print(f"Incident signs: {len(graph):,} nodes, {graph.number_of_edges():,} edges", flush=True)
    table = negative_incident_ratios(graph)
    reference, shuffle_seeds = shuffled_negative_ratios(graph, realizations=realizations, seed=seed)
    bin_edges = ratio_bin_edges(reference, scheme=bins, width=bin_width, n_bins=n_bins)
    distribution = ratio_distribution(table["r_neg"].to_numpy(), reference, bin_edges)
    summary = concentration_summary(table, reference)
    communities = community_sign_rates(graph) if "community" in table else None

    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "node_incident_signs.csv", index=False)
    distribution.to_csv(results_dir / "r_neg_distribution.csv", index=False)
    summary.to_csv(results_dir / "r_neg_summary.csv", index=False)
    if communities is not None:
        communities.to_csv(results_dir / "community_sign_rates.csv", index=False)
    signs = [sign for _, _, sign in graph.edges(data="sign")]
    write_json(
        results_dir / "metadata.json",
        {
            "graph_path": str(graph_path.resolve()),
            "graph_sha256": checksum(graph_path),
            "graph_metadata": graph.graph,
            "n_nodes": len(table),
            "n_edges": len(signs),
            "sign_counts": {str(value): signs.count(value) for value in (1, -1, 0)},
            "null_model": "sign_shuffling",
            "realizations": realizations,
            "seed": seed,
            "shuffle_seeds": list(shuffle_seeds),
            "bins": {"scheme": bins, "edges": bin_edges.tolist()},
            "versions": {
                name: version(name) for name in ("mdacn-final", "numpy", "scipy", "pandas")
            },
        },
    )
    write_json(results_dir / "nodes.json", table["node"].tolist())
    for extension in ("png", "pdf"):
        plot_negative_ratio_distribution(
            table["r_neg"].to_numpy(),
            reference,
            distribution,
            results_dir / f"r_neg_distribution.{extension}",
        )
    print(summary.to_string(index=False), flush=True)
    print(f"Saved incident-sign results to {results_dir}", flush=True)
    return table
