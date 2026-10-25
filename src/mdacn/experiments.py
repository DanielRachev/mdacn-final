"""Experiment interfaces."""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from mdacn.config import SIMULATION
from mdacn.contracts import NetworkMatrices
from mdacn.dynamics import initialize_opinions, simulate_nco
from mdacn.evaluation import compute_opinion_components, compute_opinion_gcc
from mdacn.interventions import select_contrarians, simulate_ico

DEFAULT_FRACTIONS = tuple(SIMULATION["fractions"])


def shuffle_edge_signs(graph: nx.Graph, *, seed: int = 0) -> nx.Graph:
    """Return a copy with edge signs permuted across the same topology.

    The graph structure, node set, and sign multiset are preserved exactly.
    """
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Sign shuffling requires a simple undirected graph without self loops")

    edges = list(graph.edges(data=True))
    signs = [edge_data.get("sign") for _, _, edge_data in edges]
    if any(sign not in (-1, 0, 1) for sign in signs):
        raise ValueError("Every edge must have a sign of -1, 0, or +1")

    shuffled = np.random.default_rng(seed).permutation(signs)
    shuffled_graph = graph.copy()
    for (u, v, _), sign in zip(edges, shuffled, strict=True):
        shuffled_graph[u][v]["sign"] = int(sign)
    return shuffled_graph


def run_fraction_sweep(
    network: NetworkMatrices,
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """Owner: Daniel. Compare unsigned and signed NCO over initial negative fractions.

    Return one row per model, fraction, and realisation, with reproducibility
    fields, raw opinion fractions, first/second component fractions, and stop status.
    Initial states match across models within each fraction and realisation.
    """
    if isinstance(realizations, bool) or not isinstance(realizations, int) or realizations < 1:
        raise ValueError("realizations must be a positive integer")
    fractions = tuple(fractions)
    if not fractions or not all(np.isfinite(f) and 0 <= f <= 1 for f in fractions):
        raise ValueError("fractions must be a nonempty sequence of finite values in [0, 1]")
    if len(set(fractions)) != len(fractions):
        raise ValueError("fractions must be unique")
    n_nodes = len(network.nodes)
    if not n_nodes or len(set(network.nodes)) != n_nodes:
        raise ValueError("network must have nonempty, unique node names")
    if any(m.shape != (n_nodes, n_nodes) for m in (network.unsigned, network.signed)):
        raise ValueError("network matrices must match the node order length")
    if (network.signed - network.signed.multiply(network.unsigned)).nnz:
        raise ValueError("signed edges must belong to the unsigned topology")

    graph = nx.from_scipy_sparse_array(network.unsigned)
    graph = nx.relabel_nodes(graph, dict(enumerate(network.nodes)))
    seeds = np.random.SeedSequence(seed).spawn(len(fractions) * realizations)
    rows = []
    for fraction_index, fraction in enumerate(fractions):
        for realization in range(realizations):
            initial_seed = int(
                seeds[fraction_index * realizations + realization].generate_state(1)[0]
            )
            initial = initialize_opinions(n_nodes, fraction, seed=initial_seed)
            for model, matrix in (("unsigned", network.unsigned), ("signed", network.signed)):
                result = simulate_nco(matrix, initial, model=model, max_steps=max_steps)
                rows.append(
                    {
                        "model": model,
                        "f": fraction,
                        "realization": realization,
                        "seed": seed,
                        "initial_seed": initial_seed,
                        "n_nodes": n_nodes,
                        "initial_negative_fraction": float(np.mean(initial == -1)),
                        "final_negative_fraction": float(np.mean(result.final_state == -1)),
                        **compute_opinion_components(
                            graph, result.final_state, nodes=network.nodes
                        ),
                        "steps": result.steps,
                        "stop_reason": result.stop_reason,
                        "cycle_length": result.cycle_length,
                    }
                )
    results = pd.DataFrame(rows)
    results["cycle_length"] = results["cycle_length"].astype("Int64")
    return results


def run_intervention_sweep(
    graph: nx.Graph,
    network: NetworkMatrices,
    *,
    budgets: tuple[float, ...],
    initial_negative_fraction: float,
    scores: pd.DataFrame,
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """Owner: Polly. Compare random, degree, and leverage targeting on unsigned NCO.

    Return one row per strategy, budget, and realisation, including opposing
    cluster size, both opinion GCC fractions, and stop status. Seeds and baseline
    states support comparison
    across strategies. Settle each baseline with unsigned NCO before intervention.
    Raise RuntimeError if a baseline cycles or reaches the update limit; neither
    is a steady state. max_steps applies separately to both simulation stages.
    """
    if isinstance(realizations, bool) or not isinstance(realizations, int) or realizations < 1:
        raise ValueError("realizations must be a positive integer")
    budgets = tuple(budgets)
    if not budgets or not all(np.isfinite(b) and 0 <= b <= 1 for b in budgets):
        raise ValueError("budgets must be a nonempty sequence of finite values in [0, 1]")
    if not np.isfinite(initial_negative_fraction) or not 0 <= initial_negative_fraction <= 1:
        raise ValueError("initial_negative_fraction must be finite and in [0, 1]")
    if len(network.nodes) != len(set(network.nodes)):
        raise ValueError("network nodes must be unique")

    if (
        "node" not in scores
        or scores.node.duplicated().any()
        or set(scores.node) != set(network.nodes)
    ):
        raise ValueError("Centrality scores must cover the network nodes exactly once")
    if not {"degree", "leverage"} <= set(scores):
        raise ValueError("Missing degree or leverage scores; run centrality first")
    aligned = scores.set_index("node").reindex(network.nodes)
    if not np.isfinite(aligned[["degree", "leverage"]].to_numpy(float)).all():
        raise ValueError("Centrality scores must be finite")
    degree_scores = aligned.degree.to_dict()
    leverage_scores = aligned.leverage.to_dict()
    strategies = ("random", "degree", "leverage")
    seeds = np.random.SeedSequence(seed).spawn(len(budgets) * realizations)

    rows = []
    for budget_index, budget in enumerate(budgets):
        for realization in range(realizations):
            initial_seed = int(
                seeds[budget_index * realizations + realization].generate_state(1)[0]
            )
            initial = initialize_opinions(
                len(network.nodes), initial_negative_fraction, seed=initial_seed
            )
            baseline = simulate_nco(
                network.unsigned, initial, model="unsigned", max_steps=max_steps
            )
            if baseline.stop_reason != "steady_state":
                raise RuntimeError(
                    f"Baseline did not reach a steady state: {baseline.stop_reason} "
                    f"(budget={budget}, realization={realization}, initial_seed={initial_seed})"
                )
            for strategy in strategies:
                score_map = {"degree": degree_scores, "leverage": leverage_scores}.get(strategy)
                contrarians = select_contrarians(
                    network.nodes,
                    budget,
                    strategy=strategy,
                    seed=initial_seed + realization + len(strategy),
                    scores=score_map,
                )
                result = simulate_ico(
                    network.unsigned,
                    baseline.final_state,
                    nodes=network.nodes,
                    contrarians=contrarians,
                    max_steps=max_steps,
                )
                gcc = compute_opinion_gcc(graph, result.final_state, nodes=network.nodes)
                rows.append(
                    {
                        "strategy": strategy,
                        "budget": budget,
                        "realization": realization,
                        "seed": seed,
                        "initial_seed": initial_seed,
                        "n_nodes": len(network.nodes),
                        "baseline_steps": baseline.steps,
                        "baseline_negative_fraction": float(np.mean(baseline.final_state == -1)),
                        "initial_negative_fraction": float(np.mean(initial == -1)),
                        "cluster_size": round(gcc["g_minus"] * len(network.nodes)),
                        **gcc,
                        "steps": result.steps,
                        "stop_reason": result.stop_reason,
                        "cycle_length": result.cycle_length,
                    }
                )
    results = pd.DataFrame(rows)
    results["cycle_length"] = results["cycle_length"].astype("Int64")
    return results


def run_sign_shuffle_fraction_sweep(
    graph: nx.Graph,
    network: NetworkMatrices,
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compare the empirical signed network against a sign-shuffled null model.

    Return one row per model, fraction, and realization. The same initial opinion
    state is replayed on both the empirical and sign-shuffled signed networks.
    Component sizes use the original unsigned topology, including zero-sign edges.
    """
    if isinstance(realizations, bool) or not isinstance(realizations, int) or realizations < 1:
        raise ValueError("realizations must be a positive integer")
    fractions = tuple(fractions)
    if not fractions or not all(np.isfinite(f) and 0 <= f <= 1 for f in fractions):
        raise ValueError("fractions must be a nonempty sequence of finite values in [0, 1]")
    if len(set(fractions)) != len(fractions):
        raise ValueError("fractions must be unique")

    empirical = network
    if empirical.nodes != tuple(sorted(graph)):
        raise ValueError("Saved matrices must use the core node order")
    n_nodes = len(empirical.nodes)
    if not n_nodes:
        raise ValueError("graph must have at least one node")
    node_index = {node: index for index, node in enumerate(empirical.nodes)}
    # Preserve edge order so shuffle seeds still replay through shuffle_edge_signs.
    edges = list(graph.edges(data="sign"))
    sources = np.array([node_index[u] for u, _, _ in edges], dtype=np.intp)
    targets = np.array([node_index[v] for _, v, _ in edges], dtype=np.intp)
    edge_rows = np.concatenate((sources, targets))
    edge_columns = np.concatenate((targets, sources))
    edge_signs = np.array([sign for _, _, sign in edges], dtype=np.int8)
    seeds = np.random.SeedSequence(seed).spawn(len(fractions) * realizations)
    rows = []

    for run_index in tqdm(
        range(len(seeds)), desc="Sign shuffle", unit="pair", disable=not show_progress
    ):
        fraction_index, realization = divmod(run_index, realizations)
        fraction = fractions[fraction_index]
        iteration_seed = int(seeds[run_index].generate_state(1)[0])
        rng = np.random.default_rng(iteration_seed)
        initial_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        shuffle_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        initial = initialize_opinions(n_nodes, fraction, seed=initial_seed)
        shuffled_signs = np.random.default_rng(shuffle_seed).permutation(edge_signs)
        shuffled = csr_matrix(
            (np.tile(shuffled_signs, 2), (edge_rows, edge_columns)),
            shape=empirical.signed.shape,
            dtype=float,
        )

        for model, matrix in (("empirical", empirical.signed), ("sign_shuffled", shuffled)):
            result = simulate_nco(matrix, initial, model="signed", max_steps=max_steps)
            rows.append(
                {
                    "model": model,
                    "f": fraction,
                    "realization": realization,
                    "seed": seed,
                    "initial_seed": initial_seed,
                    "shuffle_seed": shuffle_seed,
                    "n_nodes": n_nodes,
                    "initial_negative_fraction": float(np.mean(initial == -1)),
                    "final_negative_fraction": float(np.mean(result.final_state == -1)),
                    **compute_opinion_components(graph, result.final_state, nodes=empirical.nodes),
                    "steps": result.steps,
                    "stop_reason": result.stop_reason,
                    "cycle_length": result.cycle_length,
                }
            )

    results = pd.DataFrame(rows)
    results["cycle_length"] = results["cycle_length"].astype("Int64")
    return results
