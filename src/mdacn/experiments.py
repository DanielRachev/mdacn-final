"""Experiment interfaces."""

import networkx as nx
import numpy as np
import pandas as pd

from mdacn.contracts import NetworkMatrices
from mdacn.dynamics import initialize_opinions, simulate_nco


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
    across strategies.
    """
    raise NotImplementedError("Polly: run_intervention_sweep")
