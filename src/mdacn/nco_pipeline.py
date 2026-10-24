"""Reproducible NCO experiments on an exported Reddit core."""

import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from mdacn.artifacts import checksum, write_json
from mdacn.data import load_graph, load_matrices
from mdacn.experiments import run_fraction_sweep, run_sign_shuffle_fraction_sweep
from mdacn.plots import plot_phase_transition, plot_sign_shuffle_phase_transition


def run_nco_experiment(
    graph_path: Path,
    *,
    results_dir: Path = Path("results/nco"),
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
    save_reports: bool = True,
) -> pd.DataFrame:
    """Run on preprocessed graph signs and save outcomes, provenance, and figures."""
    graph = load_graph(graph_path)
    network = load_matrices(graph_path)
    print(f"NCO core: {len(network.nodes):,} nodes, {graph.number_of_edges():,} edges", flush=True)
    results = run_fraction_sweep(
        network, fractions=fractions, realizations=realizations, seed=seed, max_steps=max_steps
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "fraction_sweep.csv", index=False)
    _clear_reports(results_dir, "fraction_summary", "phase_transition")
    graph_checksum = checksum(graph_path)
    metadata = {
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": graph_checksum,
        "graph_metadata": graph.graph,
        "n_nodes": len(network.nodes),
        "n_edges": graph.number_of_edges(),
        "negative_edges": sum(e["sign"] == -1 for _, _, e in graph.edges(data=True)),
        "zero_edges": sum(e["sign"] == 0 for _, _, e in graph.edges(data=True)),
        "fractions": list(fractions),
        "realizations": realizations,
        "seed": seed,
        "max_steps": max_steps,
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy", "pandas")},
    }
    write_json(results_dir / "metadata.json", metadata)
    write_json(results_dir / "nodes.json", network.nodes)
    if save_reports:
        save_nco_reports(results_dir)
    print(results.groupby(["model", "stop_reason"]).size().to_string(), flush=True)
    print(f"Saved NCO results to {results_dir}", flush=True)
    return results


def run_sign_shuffle_experiment(
    graph_path: Path,
    *,
    results_dir: Path = Path("results/sign_shuffle"),
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
    save_reports: bool = True,
) -> pd.DataFrame:
    """Run the empirical-vs-sign-shuffled null model comparison and save outputs."""
    graph = load_graph(graph_path)
    network = load_matrices(graph_path)
    print(
        f"Sign-shuffle core: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges",
        flush=True,
    )
    results = run_sign_shuffle_fraction_sweep(
        graph,
        network,
        fractions=fractions,
        realizations=realizations,
        seed=seed,
        max_steps=max_steps,
        show_progress=True,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "sign_shuffle_sweep.csv", index=False)
    _clear_reports(results_dir, "sign_shuffle_summary", "sign_shuffle_phase_transition")
    graph_checksum = checksum(graph_path)
    metadata = {
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": graph_checksum,
        "graph_metadata": graph.graph,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "negative_edges": sum(e["sign"] == -1 for _, _, e in graph.edges(data=True)),
        "zero_edges": sum(e["sign"] == 0 for _, _, e in graph.edges(data=True)),
        "fractions": list(fractions),
        "realizations": realizations,
        "seed": seed,
        "max_steps": max_steps,
        "null_model": "sign_shuffling",
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy", "pandas")},
    }
    write_json(results_dir / "metadata.json", metadata)
    write_json(results_dir / "nodes.json", network.nodes)
    if save_reports:
        save_nco_reports(results_dir, sign_shuffle=True)
    print(results.groupby(["model", "stop_reason"]).size().to_string(), flush=True)
    print(f"Saved sign-shuffle results to {results_dir}", flush=True)
    return results


def _clear_reports(results_dir: Path, summary: str, figure: str) -> None:
    for name in (f"{summary}.csv", f"{figure}.png", f"{figure}.pdf"):
        (results_dir / name).unlink(missing_ok=True)


def save_nco_reports(results_dir: Path, *, sign_shuffle: bool = False) -> None:
    """Save summary tables and figures derived from stored NCO runs."""
    sweep = "sign_shuffle_sweep" if sign_shuffle else "fraction_sweep"
    summary_name = "sign_shuffle_summary" if sign_shuffle else "fraction_summary"
    figure = "sign_shuffle_phase_transition" if sign_shuffle else "phase_transition"
    results = pd.read_csv(results_dir / f"{sweep}.csv")
    metadata = json.loads((results_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = results.groupby(["model", "f", "stop_reason"], as_index=False).agg(
        runs=("realization", "size"),
        mean_initial_negative_fraction=("initial_negative_fraction", "mean"),
        mean_final_negative_fraction=("final_negative_fraction", "mean"),
        std_final_negative_fraction=("final_negative_fraction", "std"),
        mean_steps=("steps", "mean"),
    )
    summary["run_fraction"] = summary["runs"] / metadata["realizations"]
    summary.to_csv(results_dir / f"{summary_name}.csv", index=False)
    plot = plot_sign_shuffle_phase_transition if sign_shuffle else plot_phase_transition
    for extension in ("png", "pdf"):
        plot(results, results_dir / f"{figure}.{extension}")
