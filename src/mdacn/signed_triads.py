"""Static signed triads and structural balance on the core."""

from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, load_npz

from mdacn.artifacts import checksum, write_json
from mdacn.data import build_matrices, load_graph
from mdacn.plots import plot_signed_triads
from mdacn.signed_structure import (
    SignedEdges,
    null_comparison,
    safe_ratio,
    shuffled_signs,
    signed_edges,
    spawn_seeds,
)

TRIADS = ("+++", "++-", "+--", "---")
BALANCED = ("+++", "+--")


def triangle_edges(edges: SignedEdges) -> NDArray[np.intp]:
    """Return every triangle once as three edge indices into edges.signs.

    For nodes u < v < w the columns are the edges (u, v), (u, w), and (v, w).
    """
    n_nodes = len(edges.nodes)
    low = np.minimum(edges.sources, edges.targets)
    high = np.maximum(edges.sources, edges.targets)
    upper = csr_matrix((np.arange(len(low), dtype=np.int64), (low, high)), shape=(n_nodes, n_nodes))
    upper.sort_indices()
    indptr, indices, ids = upper.indptr, upper.indices, upper.data
    triangles = []
    for edge, (u, v) in enumerate(zip(low, high)):
        u_slice, v_slice = slice(indptr[u], indptr[u + 1]), slice(indptr[v], indptr[v + 1])
        _, u_pos, v_pos = np.intersect1d(
            indices[u_slice], indices[v_slice], assume_unique=True, return_indices=True
        )
        if u_pos.size:
            triangles.append(
                np.column_stack(
                    (np.full(u_pos.size, edge), ids[u_slice][u_pos], ids[v_slice][v_pos])
                )
            )
    if not triangles:
        return np.empty((0, 3), dtype=np.intp)
    return np.concatenate(triangles).astype(np.intp)


def _with_fractions(counts: dict[str, int]) -> dict[str, float | int]:
    nonzero = sum(counts[triad] for triad in TRIADS)
    balanced = sum(counts[triad] for triad in BALANCED)
    row: dict[str, float | int] = {
        **counts,
        "nonzero_triangles": nonzero,
        "balanced": balanced,
        "unbalanced": nonzero - balanced,
        "balanced_fraction": safe_ratio(balanced, nonzero),
    }
    for triad in TRIADS:
        row[f"fraction_{triad}"] = safe_ratio(counts[triad], nonzero)
    return row


def triad_counts(triangles: NDArray[np.intp], signs: NDArray[np.int8]) -> dict:
    """Count the four signed triads; triangles with any zero-sign edge stay separate."""
    triangle_signs = signs[triangles]
    zero = (triangle_signs == 0).any(axis=1)
    negatives = np.bincount((triangle_signs[~zero] == -1).sum(axis=1), minlength=4)
    counts = {"triangles": len(triangles), "zero_sign_triangles": int(zero.sum())}
    counts |= {triad: int(count) for triad, count in zip(TRIADS, negatives, strict=True)}
    return _with_fractions(counts)


def triad_counts_from_matrices(unsigned: csr_matrix, signed: csr_matrix) -> dict:
    """Count signed triads from aligned matrices with closed-walk traces.

    A triangle with one negative edge closes two walks of the pattern +, +, -,
    so trace(P P N) / 2 counts (+, +, -); trace(N N P) / 2 counts (+, -, -).
    """
    adjacency = (csr_matrix(unsigned) != 0).astype(np.int64)
    positive = (csr_matrix(signed) > 0).astype(np.int64)
    negative = (csr_matrix(signed) < 0).astype(np.int64)

    def closed(first: csr_matrix, second: csr_matrix, last: csr_matrix) -> int:
        return int((first @ second).multiply(last).sum())

    triangles = closed(adjacency, adjacency, adjacency) // 6
    counts = {
        "+++": closed(positive, positive, positive) // 6,
        "++-": closed(positive, positive, negative) // 2,
        "+--": closed(negative, negative, positive) // 2,
        "---": closed(negative, negative, negative) // 6,
    }
    nonzero = sum(counts.values())
    return _with_fractions(
        {"triangles": triangles, "zero_sign_triangles": triangles - nonzero, **counts}
    )


def erdos_renyi_signed(
    n_nodes: int, signs: NDArray[np.int8], *, seed: int
) -> tuple[csr_matrix, csr_matrix]:
    """Return G(N, M) with M = len(signs) and the same sign multiset placed at random."""
    rng = np.random.default_rng(seed)
    graph = nx.gnm_random_graph(n_nodes, len(signs), seed=int(rng.integers(2**32)))
    rows, cols = np.array(graph.edges(), dtype=np.intp).reshape(-1, 2).T
    shuffled = rng.permutation(signs).astype(float)
    shape = (n_nodes, n_nodes)
    both = (np.concatenate((rows, cols)), np.concatenate((cols, rows)))
    unsigned = csr_matrix((np.ones(2 * len(rows)), both), shape=shape)
    signed = csr_matrix((np.tile(shuffled, 2), both), shape=shape)
    signed.eliminate_zeros()
    return unsigned, signed


def rewired_networks(directory: Path, n_nodes: int) -> list[tuple[str, csr_matrix, csr_matrix]]:
    """Load the replicate matrices exported by the degree-preserving rewiring command."""
    folders = sorted(path for path in directory.iterdir() if (path / "signed.npz").is_file())
    networks = []
    for folder in folders:
        unsigned, signed = load_npz(folder / "unsigned.npz"), load_npz(folder / "signed.npz")
        if unsigned.shape != (n_nodes, n_nodes) or signed.shape != (n_nodes, n_nodes):
            raise ValueError(f"{folder} does not match the core node order")
        networks.append((folder.name, csr_matrix(unsigned), csr_matrix(signed)))
    return networks


def signed_degrees(unsigned: csr_matrix, signed: csr_matrix) -> NDArray[np.int64]:
    """Return (degree, positive, negative) per node, shaped (nodes, 3)."""
    signed = csr_matrix(signed)
    return np.column_stack(
        [
            np.asarray((unsigned != 0).sum(axis=1)).ravel(),
            np.asarray((signed > 0).sum(axis=1)).ravel(),
            np.asarray((signed < 0).sum(axis=1)).ravel(),
        ]
    )


def triad_null_summary(realizations: pd.DataFrame) -> pd.DataFrame:
    """Compare the empirical triad statistics with each null model ensemble."""
    empirical = realizations.loc[realizations["model"] == "empirical"].iloc[0]
    statistics = [
        "balanced_fraction",
        *(f"fraction_{triad}" for triad in TRIADS),
        "triangles",
        "zero_sign_triangles",
        "nonzero_triangles",
        *TRIADS,
    ]
    nulls = realizations.loc[realizations["model"] != "empirical"]
    return pd.DataFrame(
        {"model": model, "statistic": statistic}
        | null_comparison(float(empirical[statistic]), group[statistic])
        for model, group in nulls.groupby("model", sort=False)
        for statistic in statistics
    )


def empirical_sign_patterns(triangles: NDArray[np.intp], signs: NDArray[np.int8]) -> pd.DataFrame:
    """Count every unordered sign pattern, including those with zero-sign edges."""
    symbols = np.array(["-", "0", "+"])
    ordered = -np.sort(-signs[triangles], axis=1)
    patterns = pd.Series(["".join(row) for row in symbols[ordered + 1]], name="pattern")
    table = patterns.value_counts().rename_axis("pattern").reset_index(name="triangles")
    table["contains_zero"] = table["pattern"].str.contains("0")
    table["fraction"] = table["triangles"] / len(triangles)
    return table


def run_signed_triad_analysis(
    graph_path: Path,
    *,
    results_dir: Path = Path("results/signed_triads"),
    realizations: int = 1000,
    er_realizations: int = 20,
    rewired_dir: Path | None = Path("results/rewiring/graphs/reference"),
    seed: int = 0,
) -> pd.DataFrame:
    """Save signed triad counts for the core and its null models, with figures.

    Sign shuffling uses the same seeds as the incident-sign analysis. Rewired
    replicates come from the rewiring command's exported matrices; a missing
    directory skips that model and records it in the metadata.
    """
    for name, count in (("realizations", realizations), ("er_realizations", er_realizations)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    graph = load_graph(graph_path)
    edges = signed_edges(graph)
    n_nodes = len(edges.nodes)
    print(f"Signed triads: {n_nodes:,} nodes, {len(edges.signs):,} edges", flush=True)
    triangles = triangle_edges(edges)
    print(f"Triangles: {len(triangles):,}", flush=True)

    rows = [
        {"model": "empirical", "realization": 0, "source": "core"}
        | triad_counts(triangles, edges.signs)
    ]
    shuffle_seeds = spawn_seeds(seed, realizations)
    for realization, (shuffle_seed, signs) in enumerate(
        zip(shuffle_seeds, shuffled_signs(edges.signs, shuffle_seeds), strict=True)
    ):
        rows.append(
            {"model": "sign_shuffle", "realization": realization, "source": str(shuffle_seed)}
            | triad_counts(triangles, signs)
        )

    network = build_matrices(graph)
    empirical_degrees = signed_degrees(network.unsigned, network.signed)
    rewiring: dict[str, object] = {"directory": None, "replicates": []}
    if rewired_dir is not None and rewired_dir.is_dir():
        rewiring["directory"] = str(rewired_dir.resolve())
        for realization, (name, unsigned, signed) in enumerate(
            rewired_networks(rewired_dir, n_nodes)
        ):
            preserved = bool(np.array_equal(signed_degrees(unsigned, signed), empirical_degrees))
            rewiring["replicates"].append(
                {"name": name, "signed_degrees_preserved": preserved, "included": preserved}
            )
            if not preserved:
                # Null Model B must keep every node's signed degrees; anything else is not B.
                print(f"Warning: excluding {name}; its signed degrees differ", flush=True)
                continue
            rows.append(
                {"model": "degree_preserving_rewiring", "realization": realization, "source": name}
                | triad_counts_from_matrices(unsigned, signed)
            )
    else:
        print(f"No rewired replicates at {rewired_dir}; skipping that model", flush=True)

    er_seeds = spawn_seeds([seed, 1], er_realizations)
    for realization, er_seed in enumerate(er_seeds):
        rows.append(
            {"model": "erdos_renyi", "realization": realization, "source": str(er_seed)}
            | triad_counts_from_matrices(*erdos_renyi_signed(n_nodes, edges.signs, seed=er_seed))
        )

    table = pd.DataFrame(rows)
    summary = triad_null_summary(table)
    patterns = empirical_sign_patterns(triangles, edges.signs)

    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "triad_realizations.csv", index=False)
    summary.to_csv(results_dir / "triad_summary.csv", index=False)
    patterns.to_csv(results_dir / "triad_sign_patterns.csv", index=False)
    metadata = {
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": checksum(graph_path),
        "graph_metadata": graph.graph,
        "n_nodes": n_nodes,
        "n_edges": len(edges.signs),
        "triads": {"balanced": list(BALANCED), "unbalanced": ["++-", "---"]},
        "realizations": realizations,
        "er_realizations": er_realizations,
        "seed": seed,
        "shuffle_seeds": list(shuffle_seeds),
        "er_seeds": list(er_seeds),
        "rewiring": rewiring,
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy", "pandas")},
    }
    write_json(results_dir / "metadata.json", metadata)
    for extension in ("png", "pdf"):
        plot_signed_triads(table, results_dir / f"signed_triads.{extension}")
    shown = summary.loc[summary["statistic"].isin(["balanced_fraction", "+--", "---"])]
    print(
        shown[["model", "statistic", "empirical", "mean", "q025", "q975", "z"]].to_string(
            index=False
        ),
        flush=True,
    )
    print(f"Saved signed-triad results to {results_dir}", flush=True)
    return table
