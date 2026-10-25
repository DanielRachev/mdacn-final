"""Single-node interventions for every node of the fixed aggregate core."""

from collections.abc import Iterator

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from mdacn.contracts import NetworkMatrices
from mdacn.dynamics import initialize_opinions, simulate_nco
from mdacn.interventions import opposing_cluster_size, simulate_ico


def validate_design(
    fractions: tuple[float, ...], models: tuple[str, ...], realizations: int, max_steps: int
) -> None:
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("fractions must be nonempty and unique")
    if not all(np.isfinite(f) and 0 <= f <= 1 for f in fractions):
        raise ValueError("fractions must be finite values in [0, 1]")
    if not models or len(set(models)) != len(models) or not set(models) <= {"unsigned", "signed"}:
        raise ValueError("models must be unique unsigned/signed values")
    if realizations < 1 or max_steps < 1:
        raise ValueError("realizations and max_steps must be positive")


def iter_node_impacts(
    graph: nx.Graph,
    network: NetworkMatrices,
    *,
    fractions: tuple[float, ...] = (0.5,),
    models: tuple[str, ...] = ("unsigned", "signed"),
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
    show_progress: bool = False,
) -> Iterator[dict]:
    """Yield paired single-node outcomes, including all unsuccessful stops.

    Every candidate uses the same settled baseline for a model/f/realization.
    Pin each core node to +1 separately; divide opposing GCC sizes by the full core N.
    The no-intervention state remains unchanged because that baseline is fixed.
    Nonsteady baselines are recorded but never used as intervention starts.
    Nonsteady interventions retain terminal impact, excluded from primary analysis.
    """
    validate_design(fractions, models, realizations, max_steps)
    if network.nodes != tuple(sorted(graph)):
        raise ValueError("Saved matrices must use the core node order")
    n = len(network.nodes)
    if not n:
        raise ValueError("Core must contain at least one node")
    positions = {node: index for index, node in enumerate(network.nodes)}
    seeds = np.random.SeedSequence(seed).spawn(len(fractions) * realizations)
    with tqdm(
        total=len(fractions) * realizations * len(models) * n,
        desc="Single-node interventions",
        disable=not show_progress,
    ) as progress:
        for f_index, fraction in enumerate(fractions):
            for realization in range(realizations):
                initial_seed = int(seeds[f_index * realizations + realization].generate_state(1)[0])
                initial = initialize_opinions(n, fraction, seed=initial_seed)
                for model in models:
                    matrix = getattr(network, model)
                    baseline = simulate_nco(matrix, initial, model=model, max_steps=max_steps)
                    baseline_g = (
                        opposing_cluster_size(graph, network.nodes, baseline.final_state) / n
                    )
                    for node in network.nodes:
                        row = {
                            "node": node,
                            "model": model,
                            "f": fraction,
                            "realization": realization,
                            "initial_seed": initial_seed,
                            "n_nodes": n,
                            "initial_negative_fraction": float(np.mean(initial == -1)),
                            "baseline_stop_reason": baseline.stop_reason,
                            "baseline_steps": baseline.steps,
                            "baseline_cycle_length": baseline.cycle_length,
                            "baseline_g_minus": baseline_g,
                            "baseline_s1": float(np.mean(baseline.final_state == -1)),
                            "baseline_node_opinion": int(baseline.final_state[positions[node]]),
                            "intervention_stop_reason": "not_run",
                            "intervention_steps": None,
                            "intervention_cycle_length": None,
                            "intervention_g_minus": None,
                            "intervention_s1": None,
                            "impact": None,
                            "eligible": False,
                        }
                        if baseline.stop_reason == "steady_state":
                            result = simulate_ico(
                                matrix,
                                baseline.final_state,
                                nodes=network.nodes,
                                contrarians=(node,),
                                model=model,
                                max_steps=max_steps,
                            )
                            g = opposing_cluster_size(graph, network.nodes, result.final_state) / n
                            row.update(
                                intervention_stop_reason=result.stop_reason,
                                intervention_steps=result.steps,
                                intervention_cycle_length=result.cycle_length,
                                intervention_g_minus=g,
                                intervention_s1=float(np.mean(result.final_state == -1)),
                                impact=baseline_g - g,
                                eligible=result.stop_reason == "steady_state",
                            )
                        yield row
                        progress.update(1)
