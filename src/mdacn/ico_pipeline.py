"""Save and report the seven-strategy targeting benchmark."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mdacn import config
from mdacn.artifacts import checksum, write_json
from mdacn.centrality_pipeline import load_centrality_scores
from mdacn.config import processed_paths
from mdacn.data import load_graph, load_matrices
from mdacn.experiments import ALL_STRATEGIES, run_intervention_sweep
from mdacn.plots import plot_intervention_comparison, plot_phi_c_comparison
from mdacn.thresholds import critical_budget_runs, summarize_thresholds

# The opposing GCC counts as crossed below this share of the core.
PHI_C_THRESHOLD = 0.5
BENCHMARK_MODELS = ("unsigned", "signed")
# Scores exported by targeting-scores: strategy -> (path key, column).
EXTRA_SCORES = {
    "signed_leverage": ("signed_leverage", "signed_leverage"),
    "temporal_bridge": ("temporal_bridge", "temporal_bridge"),
}


def _load_score_column(path: Path, column: str, nodes: tuple[str, ...]) -> dict[str, float]:
    """Read one exported targeting score and require exactly one value per core node."""
    if not path.is_file():
        raise ValueError(f"Missing {column} scores: {path}. Run targeting-scores first.")
    table = pd.read_csv(path, dtype={"node": str}, keep_default_na=False)
    if not {"node", column} <= set(table):
        raise ValueError(f"{path}: require node and {column} columns")
    if table.node.duplicated().any() or set(table.node) != set(nodes):
        raise ValueError(f"{path}: scores must cover the core nodes exactly once")
    values = pd.to_numeric(table[column], errors="raise")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{path}: scores must be finite")
    return dict(zip(table.node, values.astype(float), strict=True))


def run_ico_experiment(
    graph_path: Path,
    *,
    results_dir: Path,
    fractions: tuple[float, ...],
    budgets: tuple[float, ...],
    models: tuple[str, ...] | None = None,
    strategies: tuple[str, ...] | None = None,
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
    phi_c_threshold: float | None = None,
    save_reports: bool = True,
) -> pd.DataFrame:
    """Run the full targeting benchmark on the saved core.

    Models and strategies default to `ICO` in config.py. Centrality scores supply
    degree, betweenness, eigenvector, and leverage; signed leverage and the
    temporal-activity bridge come from targeting-scores.
    """
    graph = load_graph(graph_path)
    network = load_matrices(graph_path)
    scores = load_centrality_scores(graph_path)
    models = tuple(models if models is not None else config.ICO.get("models", BENCHMARK_MODELS))
    strategies = tuple(
        strategies if strategies is not None else config.ICO.get("strategies", ALL_STRATEGIES)
    )
    threshold = float(
        config.ICO.get("phi_c_threshold", PHI_C_THRESHOLD)
        if phi_c_threshold is None
        else phi_c_threshold
    )
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("fractions must be nonempty and unique")
    if not models:
        raise ValueError("models must be nonempty")
    if not strategies:
        raise ValueError("strategies must be nonempty")

    paths = processed_paths(graph_path.parent)
    strategy_scores = {
        strategy: _load_score_column(paths[key], column, network.nodes)
        for strategy, (key, column) in EXTRA_SCORES.items()
        if strategy in strategies
    }
    tables = []
    for fraction in fractions:
        for model in models:
            tables.append(
                run_intervention_sweep(
                    graph,
                    network,
                    scores=scores,
                    strategies=strategies,
                    strategy_scores=strategy_scores,
                    model=model,
                    budgets=budgets,
                    initial_negative_fraction=fraction,
                    realizations=realizations,
                    seed=seed,
                    max_steps=max_steps,
                ).assign(f=fraction)
            )
    results = pd.concat(tables, ignore_index=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "intervention_sweep.csv", index=False)
    write_json(results_dir / "nodes.json", network.nodes)
    write_json(
        results_dir / "metadata.json",
        {
            "graph_path": str(graph_path.resolve()),
            "graph_sha256": checksum(graph_path),
            "centrality_sha256": checksum(paths["centrality"]),
            "score_sha256": {
                strategy: checksum(paths[key]) for strategy, (key, _) in EXTRA_SCORES.items()
            },
            "fractions": fractions,
            "budgets": budgets,
            "models": models,
            "strategies": strategies,
            "phi_c_threshold": threshold,
            "realizations": realizations,
            "seed": seed,
            "max_steps": max_steps,
            "seed_protocol": "paired-baseline-v2",
            "random_target_protocol": "nested-permutation-v1",
        },
    )
    (results_dir / "intervention_summary.csv").unlink(missing_ok=True)
    (results_dir / "phi_c.csv").unlink(missing_ok=True)
    for path in results_dir.glob("intervention_comparison_*.png"):
        path.unlink()
    if save_reports:
        save_ico_reports(results_dir)
    print(f"Saved ICO results: {results_dir}")
    return results


def compute_phi_c(results: pd.DataFrame, *, threshold: float = PHI_C_THRESHOLD) -> pd.DataFrame:
    """Summarize paired first crossings, baseline successes, and censored runs."""
    runs = critical_budget_runs(results, threshold=threshold)
    keys = [key for key in ("model", "f", "strategy") if key in runs]
    return summarize_thresholds(runs, keys)


def save_ico_reports(results_dir: Path) -> None:
    """Save summaries, Phi_c, and comparison figures from stored ICO results."""
    results = pd.read_csv(results_dir / "intervention_sweep.csv")
    metadata = json.loads((results_dir / "metadata.json").read_text(encoding="utf-8"))
    threshold = float(metadata.get("phi_c_threshold", PHI_C_THRESHOLD))
    if "model" not in results:
        results["model"] = "unsigned"

    measurements = {
        "runs": ("realization", "size"),
        "mean_cluster_fraction": ("cluster_fraction", "mean"),
        "std_cluster_fraction": ("cluster_fraction", "std"),
        "mean_steps": ("steps", "mean"),
    }
    if {"delta_g_minus", "backfired", "repelled_share"} <= set(results):
        measurements["mean_delta_g_minus"] = ("delta_g_minus", "mean")
        measurements["backfire_run_fraction"] = ("backfired", "mean")
        measurements["mean_repelled_share"] = ("repelled_share", "mean")
    summary = (
        results.assign(cluster_fraction=results.cluster_size / results.n_nodes)
        .groupby(["model", "f", "strategy", "budget", "stop_reason"], as_index=False)
        .agg(**measurements)
    )
    summary["run_fraction"] = summary.runs / metadata["realizations"]
    summary.to_csv(results_dir / "intervention_summary.csv", index=False)

    phi_c = compute_phi_c(results, threshold=threshold)
    phi_c.to_csv(results_dir / "phi_c.csv", index=False)
    critical_budget_runs(results, threshold=threshold).to_csv(
        results_dir / "phi_c_runs.csv", index=False
    )
    for (model, fraction), group in results.groupby(["model", "f"], sort=True):
        plot_intervention_comparison(
            group, results_dir / f"intervention_comparison_{model}_f{fraction:g}.png"
        )
    plot_phi_c_comparison(phi_c, results_dir / "phi_c_comparison.png")
