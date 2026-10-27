"""Reusable preparation, simulation, and analysis stages for node-impact studies."""

import csv
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from mdacn.artifacts import checksum, write_json
from mdacn.data import load_graph, load_matrices
from mdacn.features import PREDICTORS, validate_features
from mdacn.impact import iter_node_impacts, validate_design
from mdacn.impact_analysis import calculate_impact_statistics
from mdacn.impact_plots import save_impact_plots


def _provenance() -> dict:
    package = Path(__file__).parent
    return {
        "versions": {
            name: version(name) for name in ("mdacn-final", "networkx", "numpy", "scipy", "pandas")
        },
        "source_sha256": {path.name: checksum(path) for path in sorted(package.glob("*.py"))},
    }


def _read_features(paths: tuple[Path, ...]) -> tuple[pd.DataFrame | None, list[dict]]:
    merged = None
    sources = []
    for path in paths:
        table = pd.read_csv(path, dtype={"node": str}, keep_default_na=False, na_values=[""])
        if "node" not in table or table.node.isna().any() or table.node.duplicated().any():
            raise ValueError(f"{path}: require unique, nonempty node column")
        if merged is None:
            merged = table
        else:
            overlap = (set(table) & set(merged)) - {"node"}
            if overlap:
                raise ValueError(f"Overlapping feature columns: {sorted(overlap)}")
            merged = merged.merge(table, on="node", how="outer", validate="one_to_one")
        sources.append({"path": str(path.resolve()), "sha256": checksum(path)})
    return merged, sources


def prepare_impact_study(
    graph_path: Path,
    *,
    results_dir: Path,
    features_path: Path,
    feature_metadata_path: Path,
    fractions: tuple[float, ...] = (0.5,),
    models: tuple[str, ...] = ("unsigned", "signed"),
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> dict:
    """Freeze all core-node features, input hashes, and simulation settings."""
    validate_design(fractions, models, realizations, max_steps)
    np.random.SeedSequence(seed)
    if results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError("Use a new or empty results directory to preserve the declared study")
    graph = load_graph(graph_path)
    metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    if metadata["graph_sha256"] != checksum(graph_path):
        raise ValueError("Core changed since features were computed; run features again")
    if metadata["features_sha256"] != checksum(features_path):
        raise ValueError("Feature table changed; run features again")
    features, sources = _read_features((features_path,))
    if set(features.node) != set(graph):
        raise ValueError("Feature table does not cover the core; run features again")
    features = validate_features(features, graph)
    results_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(results_dir / "features.csv", index=False)
    plan = {
        "schema_version": 2,
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": checksum(graph_path),
        "graph_metadata": graph.graph,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "nodes": sorted(graph),
        "frozen_files": {"features.csv": checksum(results_dir / "features.csv")},
        "feature_sources": sources,
        "feature_metadata": {
            "path": str(feature_metadata_path.resolve()),
            "sha256": checksum(feature_metadata_path),
        },
        "feature_columns": features.columns.tolist(),
        "fractions": fractions,
        "models": models,
        "realizations": realizations,
        "seed": seed,
        "max_steps": max_steps,
        **_provenance(),
    }
    write_json(results_dir / "plan.json", plan)
    print(f"Prepared all {len(features)} core nodes: {results_dir}")
    return plan


def _load_plan(results_dir: Path) -> dict:
    plan = json.loads((results_dir / "plan.json").read_text(encoding="utf-8"))
    if plan["schema_version"] != 2:
        raise ValueError("Unsupported impact plan version; run node-impact again")
    for filename, expected_checksum in plan["frozen_files"].items():
        if checksum(results_dir / filename) != expected_checksum:
            raise ValueError(f"Frozen input changed: {filename}; prepare a new study")
    return plan


def simulate_impact_study(results_dir: Path) -> None:
    """Run every core node using the frozen settings; never silently replace results."""
    plan = _load_plan(results_dir)
    graph_path = Path(plan["graph_path"])
    if checksum(graph_path) != plan["graph_sha256"]:
        raise ValueError("Core graph changed since preparation; prepare a new study")
    output = results_dir / "runs.csv"
    partial = results_dir / "runs.partial.csv"
    if output.exists() or partial.exists():
        raise ValueError("Run output already exists; use a new study directory")
    graph = load_graph(graph_path)
    if sorted(graph) != plan["nodes"]:
        raise ValueError("Core node order changed")
    network = load_matrices(graph_path)
    metadata = {"plan_sha256": checksum(results_dir / "plan.json"), **_provenance()}
    with partial.open("w", newline="", encoding="utf-8") as destination:
        writer = None
        for row in iter_node_impacts(
            graph,
            network,
            fractions=tuple(plan["fractions"]),
            models=tuple(plan["models"]),
            realizations=plan["realizations"],
            seed=plan["seed"],
            max_steps=plan["max_steps"],
            show_progress=True,
        ):
            if writer is None:
                writer = csv.DictWriter(destination, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            destination.flush()
    partial.rename(output)
    metadata["runs_sha256"] = checksum(output)
    write_json(results_dir / "simulation_metadata.json", metadata)
    print(f"Saved paired outcomes: {output}")


def save_impact_reports(
    results_dir: Path,
    *,
    bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Calculate impact statistics and save tables, metadata, and plots."""
    plan = _load_plan(results_dir)
    metadata = json.loads((results_dir / "simulation_metadata.json").read_text(encoding="utf-8"))
    if metadata["plan_sha256"] != checksum(results_dir / "plan.json"):
        raise ValueError("Plan changed after simulation")
    if metadata["runs_sha256"] != checksum(results_dir / "runs.csv"):
        raise ValueError("Simulation output changed")
    features = pd.read_csv(
        results_dir / "features.csv", dtype={"node": str}, keep_default_na=False, na_values=[""]
    )
    runs = pd.read_csv(
        results_dir / "runs.csv", dtype={"node": str}, keep_default_na=False, na_values=[""]
    )
    expected_rows = (
        plan["n_nodes"] * len(plan["models"]) * len(plan["fractions"]) * plan["realizations"]
    )
    if len(runs) != expected_rows:
        raise ValueError("Incomplete simulation output")
    if features.node.tolist() != plan["nodes"]:
        raise ValueError("Frozen features must follow the full core node order")
    tables = calculate_impact_statistics(runs, features, bootstrap=bootstrap, seed=seed)
    output = results_dir / "analysis"
    output.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    write_json(
        output / "metadata.json",
        {
            "plan_sha256": metadata["plan_sha256"],
            "runs_sha256": metadata["runs_sha256"],
            "bootstrap": bootstrap,
            "seed": seed,
            "predictors": PREDICTORS,
            **_provenance(),
        },
    )
    if "spillover_impact" in runs:
        spillover = calculate_impact_statistics(
            runs, features, bootstrap=bootstrap, seed=seed, outcome="spillover_impact"
        )
        for name, table in spillover.items():
            table.to_csv(output / f"spillover_{name}.csv", index=False)
        summary = impact_component_summary(runs)
        summary.to_csv(output / "impact_components.csv", index=False)
    save_impact_plots(tables["node_impact"], tables["correlations"], output)
    print(tables["regression_diagnostics"].to_string(index=False))
    print(f"Saved analysis: {output}")
    return tables


def impact_component_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Separate direct GCC fragmentation from changes in unpinned opinions."""
    eligible = runs[runs.eligible.astype(bool)]
    return eligible.groupby(["model", "f"], as_index=False).agg(
        runs=("node", "size"),
        mean_total_impact=("impact", "mean"),
        mean_direct_impact=("direct_impact", "mean"),
        mean_spillover_impact=("spillover_impact", "mean"),
        propagation_run_fraction=("unpinned_changes", lambda x: float((x > 0).mean())),
        mean_unpinned_changes=("unpinned_changes", "mean"),
        max_unpinned_changes=("unpinned_changes", "max"),
        mean_unpinned_minus_to_plus=("unpinned_minus_to_plus", "mean"),
        mean_unpinned_plus_to_minus=("unpinned_plus_to_minus", "mean"),
    )
