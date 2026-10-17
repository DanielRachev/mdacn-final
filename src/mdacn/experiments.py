"""Experiment interfaces."""

import networkx as nx
import numpy as np
import pandas as pd

from mdacn.contracts import NetworkMatrices
from mdacn.data import build_matrices
from mdacn.dynamics import initialize_opinions, simulate_nco
from mdacn.interventions import (
    _degree_scores,
    _leverage_scores,
    opposing_cluster_size,
    select_contrarians,
    simulate_ico,
)


def shuffle_edge_signs(graph: nx.Graph, *, seed: int = 0) -> nx.Graph:
    """Return a copy with edge signs permuted across the same topology.

    The graph structure, node set, and sign multiset are preserved exactly.
    """
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Sign shuffling requires a simple undirected graph without self loops")

    edges = list(graph.edges(data=True))
    signs = np.array([edge_data.get("sign") for _, _, edge_data in edges], dtype=np.int8)
    if not np.isin(signs, (-1, 0, 1)).all():
        raise ValueError("Every edge must have a sign of -1, 0, or +1")

    shuffled = np.random.default_rng(seed).permutation(signs)
    shuffled_graph = graph.copy()
    for (u, v, _), sign in zip(edges, shuffled, strict=True):
        shuffled_graph[u][v]["sign"] = int(sign)
    return shuffled_graph


def run_fraction_sweep(
    network: NetworkMatrices,
    *,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """Owner: Daniel. Compare unsigned and signed NCO over initial negative fractions.

    Return one row per model, fraction, and realization, with reproducibility
    fields, opinion fractions, and stop status.
    Initial states match across models within each fraction and realization.
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
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """Owner: Polly. Compare random, degree, and leverage targeting on unsigned NCO.

    Return one row per strategy, budget, and realization, including opposing
    cluster size and stop status. Seeds and baseline states support comparison
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

    degree_scores = _degree_scores(graph)
    leverage_scores = _leverage_scores(graph)
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
                        "cluster_size": int(
                            opposing_cluster_size(graph, network.nodes, result.final_state)
                        ),
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
    *,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """Compare the empirical signed network against a sign-shuffled null model.

    Return one row per model, fraction, and realization. The same initial opinion
    state is replayed on both the empirical and sign-shuffled signed networks.
    """
    if isinstance(realizations, bool) or not isinstance(realizations, int) or realizations < 1:
        raise ValueError("realizations must be a positive integer")
    fractions = tuple(fractions)
    if not fractions or not all(np.isfinite(f) and 0 <= f <= 1 for f in fractions):
        raise ValueError("fractions must be a nonempty sequence of finite values in [0, 1]")
    if len(set(fractions)) != len(fractions):
        raise ValueError("fractions must be unique")

    empirical = build_matrices(graph)
    n_nodes = len(empirical.nodes)
    seeds = np.random.SeedSequence(seed).spawn(len(fractions) * realizations)
    rows = []

    for fraction_index, fraction in enumerate(fractions):
        for realization in range(realizations):
            iteration_seed = int(
                seeds[fraction_index * realizations + realization].generate_state(1)[0]
            )
            rng = np.random.default_rng(iteration_seed)
            initial_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            shuffle_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            initial = initialize_opinions(n_nodes, fraction, seed=initial_seed)
            shuffled = build_matrices(shuffle_edge_signs(graph, seed=shuffle_seed))

            for model, matrix in (("empirical", empirical.signed), ("sign_shuffled", shuffled.signed)):
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
                        "steps": result.steps,
                        "stop_reason": result.stop_reason,
                        "cycle_length": result.cycle_length,
                    }
                )

    results = pd.DataFrame(rows)
    results["cycle_length"] = results["cycle_length"].astype("Int64")
    return results
