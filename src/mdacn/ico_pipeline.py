"""Save and report the existing random/degree/leverage ICO sweep."""

import json
from pathlib import Path

import pandas as pd

from mdacn.artifacts import checksum, write_json
from mdacn.centrality_pipeline import load_centrality_scores
from mdacn.config import processed_paths
from mdacn.data import load_graph, load_matrices
from mdacn.experiments import run_intervention_sweep
from mdacn.plots import plot_intervention_comparison


def run_ico_experiment(
    graph_path: Path,
    *,
    results_dir: Path,
    fractions: tuple[float, ...],
    budgets: tuple[float, ...],
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
    save_reports: bool = True,
) -> pd.DataFrame:
    graph = load_graph(graph_path)
    network = load_matrices(graph_path)
    scores = load_centrality_scores(graph_path)
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("fractions must be nonempty and unique")
    tables = []
    for fraction in fractions:
        table = run_intervention_sweep(
            graph,
            network,
            scores=scores,
            budgets=budgets,
            initial_negative_fraction=fraction,
            realizations=realizations,
            seed=seed,
            max_steps=max_steps,
        )
        tables.append(table.assign(f=fraction))
    results = pd.concat(tables, ignore_index=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "intervention_sweep.csv", index=False)
    write_json(results_dir / "nodes.json", network.nodes)
    write_json(
        results_dir / "metadata.json",
        {
            "graph_path": str(graph_path.resolve()),
            "graph_sha256": checksum(graph_path),
            "centrality_sha256": checksum(processed_paths(graph_path.parent)["centrality"]),
            "fractions": fractions,
            "budgets": budgets,
            "realizations": realizations,
            "seed": seed,
            "max_steps": max_steps,
        },
    )
    (results_dir / "intervention_summary.csv").unlink(missing_ok=True)
    for path in results_dir.glob("intervention_comparison_f*.png"):
        path.unlink()
    if save_reports:
        save_ico_reports(results_dir)
    print(f"Saved ICO results: {results_dir}")
    return results


def save_ico_reports(results_dir: Path) -> None:
    results = pd.read_csv(results_dir / "intervention_sweep.csv")
    metadata = json.loads((results_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = (
        results.assign(cluster_fraction=results.cluster_size / results.n_nodes)
        .groupby(["f", "strategy", "budget", "stop_reason"], as_index=False)
        .agg(
            runs=("realization", "size"),
            mean_cluster_fraction=("cluster_fraction", "mean"),
            std_cluster_fraction=("cluster_fraction", "std"),
            mean_steps=("steps", "mean"),
        )
    )
    summary["run_fraction"] = summary.runs / metadata["realizations"]
    summary.to_csv(results_dir / "intervention_summary.csv", index=False)
    for fraction, group in results.groupby("f", sort=True):
        plot_intervention_comparison(
            group, results_dir / f"intervention_comparison_f{fraction:g}.png"
        )
