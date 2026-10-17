"""Reproducible NCO experiments on an exported Reddit core."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from mdacn.data import build_matrices, load_graph
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
) -> pd.DataFrame:
    """Run on preprocessed graph signs and save outcomes, provenance, and figures."""
    graph = load_graph(graph_path)
    network = build_matrices(graph)
    print(f"NCO core: {len(network.nodes):,} nodes, {graph.number_of_edges():,} edges", flush=True)
    results = run_fraction_sweep(
        network, fractions=fractions, realizations=realizations, seed=seed, max_steps=max_steps
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "fraction_sweep.csv", index=False)
    summary = results.groupby(["model", "f", "stop_reason"], as_index=False).agg(
        runs=("realization", "size"),
        mean_initial_negative_fraction=("initial_negative_fraction", "mean"),
        mean_final_negative_fraction=("final_negative_fraction", "mean"),
        std_final_negative_fraction=("final_negative_fraction", "std"),
        mean_steps=("steps", "mean"),
    )
    summary["run_fraction"] = summary["runs"] / realizations
    summary.to_csv(results_dir / "fraction_summary.csv", index=False)
    with graph_path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": checksum,
        "graph_metadata": graph.graph,
        "n_nodes": len(network.nodes),
        "n_edges": graph.number_of_edges(),
        "negative_edges": sum(e["sign"] == -1 for _, _, e in graph.edges(data=True)),
        "zero_edges": sum(e["sign"] == 0 for _, _, e in graph.edges(data=True)),
        "fractions": list(fractions),
        "realizations": realizations,
        "seed": seed,
        "max_steps": max_steps,
        "initialization": "Uniform sampling without replacement; nearest count, halves up",
        "opinion_ties": "Keep current state",
        "interaction_ties": "W=0; retain unsigned edge",
        "s1_definition": "Steady negative node fraction, not largest cluster fraction",
        "nonsteady_outcomes": "Terminal phase/state only; not an equilibrium estimate",
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy", "pandas")},
    }
    (results_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (results_dir / "nodes.json").write_text(json.dumps(network.nodes) + "\n", encoding="utf-8")
    for extension in ("png", "pdf"):
        plot_phase_transition(results, results_dir / f"phase_transition.{extension}")
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
) -> pd.DataFrame:
    """Run the empirical-vs-sign-shuffled null model comparison and save outputs."""
    graph = load_graph(graph_path)
    print(
        f"Sign-shuffle core: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges",
        flush=True,
    )
    results = run_sign_shuffle_fraction_sweep(
        graph,
        fractions=fractions,
        realizations=realizations,
        seed=seed,
        max_steps=max_steps,
        show_progress=True,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "sign_shuffle_sweep.csv", index=False)
    summary = results.groupby(["model", "f", "stop_reason"], as_index=False).agg(
        runs=("realization", "size"),
        mean_initial_negative_fraction=("initial_negative_fraction", "mean"),
        mean_final_negative_fraction=("final_negative_fraction", "mean"),
        std_final_negative_fraction=("final_negative_fraction", "std"),
        mean_steps=("steps", "mean"),
    )
    summary["run_fraction"] = summary["runs"] / realizations
    summary.to_csv(results_dir / "sign_shuffle_summary.csv", index=False)
    with graph_path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": checksum,
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
        "initialization": "Uniform sampling without replacement; nearest count, halves up",
        "opinion_ties": "Keep current state",
        "interaction_ties": "W=0; retain unsigned edge",
        "s1_definition": "Steady negative node fraction, not largest cluster fraction",
        "nonsteady_outcomes": "Terminal phase/state only; not an equilibrium estimate",
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy", "pandas")},
    }
    (results_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (results_dir / "nodes.json").write_text(
        json.dumps(tuple(sorted(graph))) + "\n", encoding="utf-8"
    )
    for extension in ("png", "pdf"):
        plot_sign_shuffle_phase_transition(
            results, results_dir / f"sign_shuffle_phase_transition.{extension}"
        )
    print(results.groupby(["model", "stop_reason"]).size().to_string(), flush=True)
    print(f"Saved sign-shuffle results to {results_dir}", flush=True)
    return results
