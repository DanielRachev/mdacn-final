"""Matched temporal signed ICO experiments using the team's targeting and NCO rules."""

import numpy as np
import pandas as pd

from mdacn.dynamics import _prepare_matrix, _prepare_state, _step, initialize_opinions
from mdacn.interventions import select_contrarians
from mdacn.temporal_analysis import (
    largest_opinion_component,
    lifetime_topology,
    paired_bootstrap_interval,
)
from mdacn.temporal_data import TemporalSnapshots, build_temporal_snapshots
from mdacn.temporal_nulls import shuffle_timestamps


def simulate_temporal_ico(
    snapshots: TemporalSnapshots,
    initial_state: np.ndarray,
    *,
    contrarians: tuple[str, ...] = (),
    model: str = "signed",
) -> dict:
    """One synchronous update per window; selected nodes are pinned to +1 from time zero."""
    if len(set(snapshots.nodes)) != len(snapshots.nodes) or not set(contrarians).issubset(
        snapshots.nodes
    ):
        raise ValueError("Invalid node order or contrarian names")
    state = _prepare_state(initial_state, len(snapshots.nodes))
    contrarian_set = set(contrarians)
    fixed = np.asarray(
        [i for i, node in enumerate(snapshots.nodes) if node in contrarian_set], dtype=int
    )
    state[fixed] = 1
    changed = []
    for matrix in (
        snapshots.signed if model == "signed" else snapshots.unsigned if model == "unsigned" else ()
    ):
        updated = _step(_prepare_matrix(matrix, model), state)
        updated[fixed] = 1
        changed.append(int(np.count_nonzero(updated != state)))
        state = updated
    if model not in ("signed", "unsigned"):
        raise ValueError("model must be signed or unsigned")
    return {
        "final_state": state,
        "changed_count": np.asarray(changed, dtype=int),
        "pinned_count": len(fixed),
    }


def evaluate_temporal_ico(
    interactions: pd.DataFrame,
    nodes: tuple[str, ...],
    *,
    degree_scores: dict[str, float],
    signed_leverage_scores: dict[str, float] | None = None,
    budgets: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25),
    fractions: tuple[float, ...] = (0.3, 0.5, 0.7),
    n_initializations: int = 20,
    n_shuffles: int = 0,
    frequency: str = "W-SUN",
    seed: int = 42,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Paired real and optional shuffled temporal ICO; output negative GCC / fixed core N."""
    if n_initializations < 1 or n_shuffles < 0 or not budgets or not fractions:
        raise ValueError("Invalid repetitions or empty parameter grid")
    if any(not 0 <= x <= 1 for x in (*budgets, *fractions)):
        raise ValueError("Budgets and fractions must lie in [0, 1]")
    graph = lifetime_topology(interactions, nodes)
    real = build_temporal_snapshots(interactions, nodes, frequency=frequency)
    snapshots = [("real", -1, real)]
    rng = np.random.default_rng(seed)
    for k in range(n_shuffles):
        null_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        shuffled = shuffle_timestamps(interactions, seed=null_seed)
        null = build_temporal_snapshots(shuffled, nodes, frequency=frequency)
        if null.timestamps != real.timestamps:
            raise ValueError("Timestamp shuffle changed observation horizon")
        snapshots.append(("global_shuffle", k, null))

    strategies = [("random", None), ("degree", degree_scores)]
    if signed_leverage_scores is not None:
        strategies.append(("signed_leverage", signed_leverage_scores))
    records = []
    for f in fractions:
        for initialization in range(n_initializations):
            init_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            initial = initialize_opinions(len(nodes), float(f), seed=init_seed)
            for budget in budgets:
                for strategy, scores in strategies:
                    selection_seed = (
                        int(rng.integers(0, np.iinfo(np.int64).max))
                        if strategy == "random"
                        else seed
                    )
                    # Reuse the team's ranked selector for signed leverage as well as degree.
                    selector = "random" if strategy == "random" else "degree"
                    pinned = select_contrarians(
                        nodes, float(budget), strategy=selector, seed=selection_seed, scores=scores
                    )
                    for variant, shuffle, sequence in snapshots:
                        result = simulate_temporal_ico(sequence, initial, contrarians=pinned)
                        state = result["final_state"]
                        records.append(
                            {
                                "f": float(f),
                                "initialization": initialization,
                                "initial_seed": init_seed,
                                "budget": float(budget),
                                "strategy": strategy,
                                "variant": variant,
                                "shuffle": shuffle,
                                "target_seed": selection_seed,
                                "pinned_count": len(pinned),
                                "n_nodes": len(nodes),
                                "negative_fraction": float(np.mean(state == -1))
                                if len(nodes)
                                else 0.0,
                                "g_minus": largest_opinion_component(graph, state, -1) / len(nodes)
                                if len(nodes)
                                else 0.0,
                                "total_switches": int(result["changed_count"].sum()),
                                "steps": len(sequence.timestamps),
                            }
                        )
            if show_progress:
                print(
                    f"ICO f={f:g}: initialization {initialization + 1}/{n_initializations}",
                    flush=True,
                )
    return pd.DataFrame.from_records(records)


# Consolidated from temporal_ico_statistics.py; preserve public function interfaces.
def summarize_temporal_ico(
    raw: pd.DataFrame, *, threshold: float = 0.5, n_bootstrap: int = 2000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return outcome summaries, paired differences, and per-run first threshold crossings.

    Bootstrap both initializations and the shuffled timelines shared across them.
    Strategies remain paired on the same initialization and shuffle ID.
    """
    required = {"f", "budget", "strategy", "variant", "initialization", "shuffle", "g_minus"}
    if not required.issubset(raw):
        raise ValueError(f"Missing columns: {sorted(required - set(raw))}")
    if n_bootstrap < 1 or not 0 < threshold < 1:
        raise ValueError("Invalid bootstrap repetitions or threshold")
    rng = np.random.default_rng(seed)
    base = raw.groupby(
        ["f", "budget", "strategy", "variant", "initialization"], as_index=False
    ).agg(
        g_minus=("g_minus", "mean"),
        success_rate=("g_minus", lambda s: float((s < threshold).mean())),
    )
    summaries = base.groupby(["f", "budget", "strategy", "variant"], as_index=False).agg(
        mean_g_minus=("g_minus", "mean"),
        sd_across_initializations=("g_minus", "std"),
        success_probability=("success_rate", "mean"),
        n_initializations=("initialization", "nunique"),
    )
    comparisons = []
    for f, group in raw.groupby("f"):
        for budget, part in group.groupby("budget"):
            for strategy in part.strategy.unique():
                real = part[(part.strategy == strategy) & (part.variant == "real")].set_index(
                    "initialization"
                )["g_minus"]
                null = part[(part.strategy == strategy) & (part.variant == "global_shuffle")].pivot(
                    index="initialization", columns="shuffle", values="g_minus"
                )
                if not null.empty:
                    difference = real.reindex(null.index).to_numpy()[:, None] - null.to_numpy()
                    comparisons.append(
                        _paired_stats(
                            difference,
                            f,
                            budget,
                            f"{strategy}:real-minus-shuffled",
                            rng,
                            n_bootstrap,
                        )
                    )
            for variant in part.variant.unique():
                strategies = part[part.variant == variant].pivot(
                    index=["initialization", "shuffle"], columns="strategy", values="g_minus"
                )
                if {"degree", "random"}.issubset(strategies.columns):
                    difference = (
                        (strategies["degree"] - strategies["random"]).unstack("shuffle").to_numpy()
                    )
                    comparisons.append(
                        _paired_stats(
                            difference,
                            f,
                            budget,
                            f"{variant}:degree-minus-random",
                            rng,
                            n_bootstrap,
                        )
                    )
    # Find each run's first threshold crossing, rather than a crossing of the mean curve.
    crossings = []
    keys = ["f", "strategy", "variant", "initialization", "shuffle"]
    for key, grp in raw.groupby(keys, sort=False):
        met = grp.loc[grp.g_minus < threshold, "budget"]
        crossings.append(
            dict(
                zip(keys, key),
                phi_c=float(met.min()) if len(met) else float("nan"),
                threshold_reached=bool(len(met)),
            )
        )
    return summaries, pd.DataFrame(comparisons), pd.DataFrame(crossings)


def _paired_stats(
    difference: np.ndarray,
    f: float,
    budget: float,
    comparison: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict:
    ci_low, ci_high = paired_bootstrap_interval(difference, rng=rng, n_bootstrap=n_bootstrap)
    return {
        "f": f,
        "budget": budget,
        "comparison": comparison,
        "mean_difference": float(difference.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "paired_initializations": len(difference),
    }
