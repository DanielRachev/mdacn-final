"""Experiment interfaces."""

import networkx as nx
import numpy as np
import pandas as pd

from mdacn.contracts import NetworkMatrices
from mdacn.dynamics import initialize_opinions, simulate_nco
from mdacn.evaluation import compute_opinion_gcc
from mdacn.interventions import (
    _degree_scores,
    _leverage_scores,
    select_contrarians,
    simulate_ico,
)


def run_fraction_sweep(
    network: NetworkMatrices,
    *,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """Owner: Daniel. Compare unsigned and signed NCO over initial negative fractions.

    Return one row per model, fraction, and realisation, with reproducibility
    fields, raw opinion fractions, opinion GCC fractions, and stop status.
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
                        **compute_opinion_gcc(graph, result.final_state, nodes=network.nodes),
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
