"""Sampled single-node intervention study on the fixed aggregate core."""

from collections.abc import Iterator

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mdacn.contracts import NetworkMatrices
from mdacn.dynamics import initialize_opinions, simulate_nco
from mdacn.interventions import opposing_cluster_size, simulate_ico


def stratified_sample(
    features: pd.DataFrame, *, sample_size: int = 30, bins: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample across degree and incident-sign ranges before seeing outcomes.

    Quantile cut points do not split equal values. Zero negative-incident ratios
    form their own group. Allocate one node per occupied stratum, then cycle
    through strata in seeded random order until the requested size is reached.
    This is a coverage sample, not a representative probability sample.
    """
    if bins < 1 or not 1 <= sample_size <= len(features):
        raise ValueError("Require bins >= 1 and 1 <= sample_size <= core node count")
    frame = features.sort_values("node").reset_index(drop=True).copy()

    def groups(values: pd.Series) -> np.ndarray:
        cuts = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        return np.searchsorted(cuts, values, side="left")

    frame["degree_bin"] = groups(frame.degree)
    frame["negative_ratio_bin"] = 0
    positive = frame.negative_incident_ratio > 0
    if positive.any():
        frame.loc[positive, "negative_ratio_bin"] = 1 + groups(
            frame.loc[positive, "negative_incident_ratio"]
        )
    keys = ["degree_bin", "negative_ratio_bin"]
    strata = list(frame.groupby(keys, sort=True).groups.items())
    if sample_size < len(strata):
        raise ValueError(f"sample_size must cover all {len(strata)} occupied strata")
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(indices)) for _, indices in strata]
    selected = []
    while len(selected) < sample_size:
        for index in rng.permutation(len(pools)):
            if pools[index]:
                selected.append(pools[index].pop())
                if len(selected) == sample_size:
                    break
    frame["selected"] = frame.index.isin(selected)
    coverage = frame.groupby(keys, as_index=False).agg(
        core_nodes=("node", "size"),
        sampled_nodes=("selected", "sum"),
        degree_min=("degree", "min"),
        degree_max=("degree", "max"),
        negative_ratio_min=("negative_incident_ratio", "min"),
        negative_ratio_max=("negative_incident_ratio", "max"),
    )
    return frame.loc[frame.selected].drop(columns="selected").reset_index(drop=True), coverage


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
    sample_nodes: tuple[str, ...],
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
    Pin any sampled node to +1; divide opposing GCC sizes by the full core N.
    The no-intervention state remains unchanged because that baseline is fixed.
    Nonsteady baselines are recorded but never used as intervention starts.
    Nonsteady interventions retain terminal impact, excluded from primary analysis.
    """
    validate_design(fractions, models, realizations, max_steps)
    if network.nodes != tuple(sorted(graph)):
        raise ValueError("Saved matrices must use the core node order")
    if not sample_nodes or len(set(sample_nodes)) != len(sample_nodes):
        raise ValueError("sample_nodes must be nonempty and unique")
    if not set(sample_nodes) <= set(network.nodes):
        raise ValueError("Sample nodes must belong to the core")
    n = len(network.nodes)
    positions = {node: index for index, node in enumerate(network.nodes)}
    seeds = np.random.SeedSequence(seed).spawn(len(fractions) * realizations)
    with tqdm(
        total=len(fractions) * realizations * len(models) * len(sample_nodes),
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
                    for node in sample_nodes:
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
