"""Reproducible NCO experiments on an exported Reddit core."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from mdacn.data import build_matrices, load_graph
from mdacn.experiments import DEFAULT_FRACTIONS, run_fraction_sweep
from mdacn.plots import plot_opinion_gcc, plot_phase_transition


def run_nco_experiment(
    graph_path: Path,
    *,
    results_dir: Path = Path("results/nco"),
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
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
        mean_g_minus=("g_minus", "mean"),
        std_g_minus=("g_minus", "std"),
        mean_g_plus=("g_plus", "mean"),
        std_g_plus=("g_plus", "std"),
        mean_g_minus_second=("g_minus_second", "mean"),
        std_g_minus_second=("g_minus_second", "std"),
        mean_g_plus_second=("g_plus_second", "mean"),
        std_g_plus_second=("g_plus_second", "std"),
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
        "g_minus_definition": "Largest -1 component on the fixed unsigned core / full core N",
        "g_plus_definition": "Largest +1 component on the fixed unsigned core / full core N",
        "g_minus_second_definition": "Second-largest -1 component / full core N",
        "g_plus_second_definition": "Second-largest +1 component / full core N",
        "seed_assignment": "SeedSequence(seed).spawn by recorded fraction order, then realization",
        "component_figures": "Individual runs and means; horizontal jitter is for visibility only",
        "nonsteady_outcomes": "Terminal phase/state only; not an equilibrium estimate",
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy", "pandas")},
    }
    (results_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (results_dir / "nodes.json").write_text(json.dumps(network.nodes) + "\n", encoding="utf-8")
    for extension in ("png", "pdf"):
        plot_phase_transition(results, results_dir / f"phase_transition.{extension}")
        plot_opinion_gcc(results, results_dir / f"opinion_gcc_phase_transition.{extension}")
        plot_opinion_gcc(
            results,
            results_dir / f"positive_opinion_gcc_phase_transition.{extension}",
            opinion="plus",
        )
    print(results.groupby(["model", "stop_reason"]).size().to_string(), flush=True)
    print(f"Saved NCO results to {results_dir}", flush=True)
    return results
