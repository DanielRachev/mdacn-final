"""Null Model B experiments: export rewired graphs and compare signed ICO Phi_c."""

import json
import zlib
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm

from mdacn import config
from mdacn.artifacts import checksum, write_json
from mdacn.centrality import leverage_centrality
from mdacn.centrality_pipeline import load_centrality_scores
from mdacn.contracts import NetworkMatrices, Strategy
from mdacn.data import build_matrices, edge_order_metadata, load_graph, load_matrices
from mdacn.dynamics import initialize_opinions, simulate_nco
from mdacn.evaluation import compute_opinion_gcc
from mdacn.interventions import select_contrarians, simulate_ico
from mdacn.rewiring import (
    Rule,
    Target,
    degree_table,
    homophily_bounds,
    homophily_counts,
    rewire,
    validate_rewiring,
)
from mdacn.temporal_analysis import paired_bootstrap_interval
from mdacn.thresholds import critical_budget_runs, summarize_thresholds

DEFAULT_BETAS = {target: tuple(beta) for target, beta in config.REWIRING["betas"].items()}
DEFAULT_BUDGETS = tuple(config.REWIRED_ICO["budgets"])
EMPIRICAL = "empirical"
REFERENCE = "reference"


@dataclass(frozen=True)
class RewiringSpec:
    """One rewired configuration: a name, steering target, acceptance rule, and beta."""

    name: str
    target: Target | None
    rule: Rule
    beta: float | None = None


def default_specs(
    betas: dict[str, tuple[float, ...]] | None = None,
) -> tuple[RewiringSpec, ...]:
    """Return the unbiased reference plus strict, non-worsening, and Metropolis runs.

    betas maps each target to its Metropolis betas; missing targets use the defaults.
    """
    betas = {**DEFAULT_BETAS, **(betas or {})}
    specs = [RewiringSpec(REFERENCE, None, "unbiased")]
    for target in ("insular", "integrated"):
        specs.append(RewiringSpec(f"{target}_strict", target, "strict"))
        specs.append(RewiringSpec(f"{target}_non_worsening", target, "non_worsening"))
        specs.extend(
            RewiringSpec(f"{target}_metropolis_beta{beta:g}", target, "metropolis", beta)
            for beta in betas[target]
        )
    return tuple(specs)


def stream_seed(seed: int, label: str, index: int) -> int:
    """Return an explicit child seed for a named stream and index.

    Keys use a stable hash of the label, so adding or reordering configurations
    does not change the seeds of the others.
    """
    sequence = np.random.SeedSequence(seed, spawn_key=(zlib.crc32(label.encode()), index))
    return int(sequence.generate_state(1)[0])


def run_rewiring_experiment(
    graph_path: Path,
    *,
    metrics_path: Path = Path("results/topology/metrics.json"),
    results_dir: Path = Path("results/rewiring"),
    specs: tuple[RewiringSpec, ...] | None = None,
    replicates: int = 5,
    reference_replicates: int | None = None,
    attempts_per_edge: int = 100,
    window_per_edge: float = 2.0,
    tolerance: float = 0.002,
    seed: int = 0,
    checkpoints: int = 20,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Generate, validate, and export rewired graphs for each spec.

    Every graph receives at most attempts_per_edge * |E| attempted swaps and stops
    earlier once its tracked score changes by at most tolerance * (signed edges) over
    a window of window_per_edge * |E| attempts; window_per_edge = 0 disables the
    stop. Checkpoints are saved once per window, or checkpoints times without
    a window. Return one summary row per graph, including the empirical graph.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if reference_replicates is not None and (
        isinstance(reference_replicates, bool)
        or not isinstance(reference_replicates, int)
        or reference_replicates < 1
    ):
        raise ValueError("reference_replicates must be a positive integer")
    if min(attempts_per_edge, window_per_edge, tolerance, checkpoints) < 0:
        raise ValueError("Attempt, window, tolerance, and checkpoint settings must be >= 0")
    specs = default_specs() if specs is None else tuple(specs)
    if len({spec.name for spec in specs}) != len(specs) or EMPIRICAL in {s.name for s in specs}:
        raise ValueError("Spec names must be unique and differ from 'empirical'")
    graph = load_graph(graph_path)
    nodes = load_matrices(graph_path).nodes
    if nodes != tuple(sorted(graph)):
        raise ValueError("Saved node order does not match the core graph")
    load_centrality_scores(graph_path)
    processed = config.processed_paths(graph_path.parent)
    attempts = attempts_per_edge * graph.number_of_edges()
    window = round(window_per_edge * graph.number_of_edges())
    interval = window or (max(1, attempts // checkpoints) if checkpoints and attempts else 0)
    bounds = homophily_bounds(graph)
    before = homophily_counts(graph)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _summary_row(
            EMPIRICAL,
            None,
            "empirical",
            None,
            0,
            None,
            graph,
            bounds,
            assortativity=_saved_assortativity(metrics_path, graph),
        )
    ]
    rows[0].update(
        connected=nx.is_connected(graph),
        connected_components=nx.number_connected_components(graph),
    )
    jobs = [
        (spec, replicate)
        for spec in specs
        for replicate in range(
            reference_replicates
            if spec.name == REFERENCE and reference_replicates is not None
            else replicates
        )
    ]
    for spec, replicate in tqdm(jobs, desc="Rewiring", unit="graph", disable=not show_progress):
        graph_seed = stream_seed(seed, f"rewire:{spec.name}", replicate)
        result = rewire(
            graph,
            attempts=attempts,
            seed=graph_seed,
            target=spec.target,
            rule=spec.rule,
            beta=spec.beta,
            checkpoint_interval=interval,
            window=window,
            tolerance=tolerance,
        )
        validation = validate_rewiring(graph, result.graph)
        network = build_matrices(result.graph)
        if network.nodes != nodes:
            raise AssertionError("Rewired node order differs from nodes.json")
        folder = results_dir / "graphs" / spec.name / f"replicate_{replicate}"
        folder.mkdir(parents=True, exist_ok=True)
        save_npz(folder / "unsigned.npz", network.unsigned)
        save_npz(folder / "signed.npz", network.signed)
        pd.DataFrame(degree_table(graph, result.graph)).to_csv(folder / "degrees.csv", index=False)
        _centrality_table(result.graph, nodes).to_csv(folder / "centrality.csv", index=False)
        homophily = homophily_counts(result.graph)
        metadata = {
            "spec": asdict(spec),
            "replicate": replicate,
            "seed": seed,
            "graph_seed": graph_seed,
            "swaps": result.stats,
            "validation": validation,
            "homophily_before": before,
            "homophily_after": homophily,
            "homophily_bounds": bounds,
        }
        write_json(folder / "metadata.json", metadata)
        row = _summary_row(
            spec.name,
            spec.target,
            spec.rule,
            spec.beta,
            replicate,
            graph_seed,
            result.graph,
            bounds,
        )
        row.update(
            max_attempts=attempts,
            attempts=result.stats["attempts"],
            converged=result.stats["converged"],
            accepted=result.stats["accepted"],
            acceptance_rate=result.stats["acceptance_rate"],
            **{f"rejected_{k}": v for k, v in result.stats["rejections"].items()},
            edges_changed=validation["edges_changed"],
            fraction_changed=validation["fraction_changed"],
            connected=validation["connected"],
            connected_components=validation["connected_components"],
            path=str(folder.relative_to(results_dir).as_posix()),
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "rewiring_summary.csv", index=False)
    disconnected = summary.loc[~summary["connected"].astype(bool), ["name", "replicate"]]
    if len(disconnected):
        # Disconnected graphs are kept with every node; nothing is dropped or relabelled.
        pairs = list(disconnected.itertuples(index=False, name=None))
        print(f"Warning: {len(pairs)} disconnected graphs: {pairs}", flush=True)
    run_metadata = {
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": checksum(graph_path),
        "nodes_path": str(processed["nodes"].resolve()),
        "nodes_sha256": checksum(processed["nodes"]),
        "empirical_centrality": str(processed["centrality"].resolve()),
        "empirical_centrality_sha256": checksum(processed["centrality"]),
        "graph_metadata": graph.graph,
        **edge_order_metadata(graph),
        "n_nodes": len(graph),
        "n_edges": graph.number_of_edges(),
        "specs": [asdict(spec) for spec in specs],
        "replicates": replicates,
        "reference_replicates": reference_replicates
        if reference_replicates is not None
        else replicates,
        "attempts_per_edge": attempts_per_edge,
        "max_attempts_per_graph": attempts,
        "window_per_edge": window_per_edge,
        "window_attempts": window,
        "tolerance": tolerance,
        "checkpoint_interval": interval,
        "seed": seed,
        "versions": {name: version(name) for name in ("mdacn-final", "networkx", "numpy")},
    }
    write_json(results_dir / "metadata.json", run_metadata)
    print(f"Saved {len(jobs)} rewired graphs to {results_dir}", flush=True)
    return summary


def run_rewired_ico_experiment(
    graph_path: Path = Path("data/processed/core.graphml"),
    *,
    rewired_dir: Path = Path("results/rewiring"),
    results_dir: Path = Path("results/rewiring_ico"),
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    negative_fraction: float = 0.5,
    initializations: int = 20,
    strategies: tuple[Strategy, ...] = ("degree", "random"),
    seed: int = 0,
    max_steps: int = 1000,
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run paired signed ICO on the empirical and every rewired graph.

    Each initial state settles under signed NCO before contrarians are pinned at +1.
    Initial states and random-targeting seeds depend only on the initialization
    index, so they match across graphs. Following the shared convention, g_minus
    and g_plus use the fixed empirical unsigned core and the full core N for every
    graph. Baseline and terminal states are saved as packed bits in final_states.npz
    so other measurements can be taken from stored results. Return raw runs,
    per-run Phi_c, and comparisons.
    """
    budgets = tuple(sorted(budgets))
    if not budgets or not all(np.isfinite(b) and 0 <= b <= 1 for b in budgets):
        raise ValueError("budgets must be a nonempty sequence of values in [0, 1]")
    if len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be unique")
    if isinstance(initializations, bool) or initializations < 1:
        raise ValueError("initializations must be a positive integer")
    if not 0 <= negative_fraction <= 1 or not 0 < threshold < 1:
        raise ValueError("negative_fraction must lie in [0, 1] and threshold in (0, 1)")
    rewiring = json.loads((rewired_dir / "metadata.json").read_text())
    processed = config.processed_paths(graph_path.parent)
    if checksum(processed["nodes"]) != rewiring["nodes_sha256"]:
        raise ValueError(f"{processed['nodes']} differs from the node order used for rewiring")
    empirical = load_matrices(graph_path)
    nodes = empirical.nodes
    centrality = _resolve(rewired_dir, rewiring["empirical_centrality"])
    if checksum(centrality) != rewiring["empirical_centrality_sha256"]:
        raise ValueError(f"{centrality} changed since rewiring")
    # Opinion components are measured on the fixed empirical core for every graph.
    core = nx.relabel_nodes(nx.from_scipy_sparse_array(empirical.unsigned), dict(enumerate(nodes)))
    graphs = [(EMPIRICAL, 0, empirical, centrality)]
    summary = pd.read_csv(rewired_dir / "rewiring_summary.csv")
    for row in summary.loc[summary["name"] != EMPIRICAL].itertuples(index=False):
        folder = rewired_dir / row.path
        network = _load_network(folder, nodes)
        graphs.append((row.name, int(row.replicate), network, folder / "centrality.csv"))
    connectivity = {
        (row.name, int(row.replicate)): (bool(row.connected), int(row.connected_components))
        for row in summary.itertuples(index=False)
    }
    disconnected = sorted(key for key, (connected, _) in connectivity.items() if not connected)
    if disconnected:
        # Disconnected graphs keep every node and the full-N denominator.
        print(f"Warning: {len(disconnected)} disconnected graphs: {disconnected}", flush=True)
    initial_seeds = [stream_seed(seed, "initial", i) for i in range(initializations)]
    target_seeds = [stream_seed(seed, "target", i) for i in range(initializations)]

    rows = []
    baseline_states, final_states = [], []
    for name, replicate, network, centrality_file in tqdm(
        graphs, desc="Rewired ICO", unit="graph", disable=not show_progress
    ):
        scores = _load_scores(centrality_file, nodes)
        for initialization, (initial_seed, target_seed) in enumerate(
            zip(initial_seeds, target_seeds, strict=True)
        ):
            initial = initialize_opinions(len(nodes), negative_fraction, seed=initial_seed)
            baseline = simulate_nco(network.signed, initial, model="signed", max_steps=max_steps)
            base = {
                "graph": name,
                "replicate": replicate,
                "initialization": initialization,
                "initial_seed": initial_seed,
                "target_seed": target_seed,
                "n_nodes": len(nodes),
                "connected": connectivity[(name, replicate)][0],
                "connected_components": connectivity[(name, replicate)][1],
                "f": negative_fraction,
                "baseline_steps": baseline.steps,
                "baseline_stop_reason": baseline.stop_reason,
                "baseline_negative_fraction": float(np.mean(baseline.final_state == -1)),
                "baseline_g_minus": compute_opinion_gcc(core, baseline.final_state, nodes=nodes)[
                    "g_minus"
                ],
                "baseline_index": len(baseline_states),
            }
            baseline_states.append(baseline.final_state)
            for strategy in strategies:
                for budget in budgets:
                    row = {**base, "strategy": strategy, "budget": budget, "state_index": -1}
                    if baseline.stop_reason != "steady_state":
                        # Post-baseline intervention needs a settled baseline.
                        rows.append(row)
                        continue
                    contrarians = select_contrarians(
                        nodes,
                        budget,
                        strategy=strategy,
                        seed=target_seed,
                        scores=scores.get(strategy),
                    )
                    result = simulate_ico(
                        network.signed,
                        baseline.final_state,
                        nodes=nodes,
                        contrarians=contrarians,
                        max_steps=max_steps,
                        model="signed",
                    )
                    row.update(
                        pinned_count=len(contrarians),
                        final_negative_fraction=float(np.mean(result.final_state == -1)),
                        **compute_opinion_gcc(core, result.final_state, nodes=nodes),
                        steps=result.steps,
                        stop_reason=result.stop_reason,
                        cycle_length=result.cycle_length,
                        state_index=len(final_states),
                    )
                    final_states.append(result.final_state)
                    rows.append(row)
    raw = pd.DataFrame(rows)
    for column in ("g_minus", "g_plus", "stop_reason", "cycle_length"):
        if column not in raw:
            raw[column] = np.nan if column.startswith("g_") else pd.NA
    raw["cycle_length"] = raw["cycle_length"].astype("Int64")
    phi_c = critical_budgets(raw, threshold=threshold)
    comparisons = compare_to_controls(raw, n_bootstrap=n_bootstrap, seed=seed)

    results_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(results_dir / "ico_runs.csv", index=False)
    np.savez_compressed(
        results_dir / "final_states.npz",
        baseline=_pack(baseline_states, len(nodes)),
        final=_pack(final_states, len(nodes)),
        n_nodes=len(nodes),
    )
    phi_c.to_csv(results_dir / "phi_c_runs.csv", index=False)
    summarize_phi_c(phi_c, rewiring_summary=summary).to_csv(
        results_dir / "phi_c_summary.csv", index=False
    )
    comparisons.to_csv(results_dir / "g_minus_comparisons.csv", index=False)
    metadata = {
        "graph_path": str(graph_path.resolve()),
        "matrix_sha256": {
            name: checksum(processed[name]) for name in ("unsigned", "signed", "nodes")
        },
        "empirical_centrality": str(centrality.resolve()),
        "rewired_dir": str(rewired_dir.resolve()),
        "rewiring_summary_sha256": checksum(rewired_dir / "rewiring_summary.csv"),
        "budgets": list(budgets),
        "negative_fraction": negative_fraction,
        "initializations": initializations,
        "strategies": list(strategies),
        "seed": seed,
        "initial_seeds": initial_seeds,
        "target_seeds": target_seeds,
        "max_steps": max_steps,
        "threshold": threshold,
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy")},
    }
    write_json(results_dir / "metadata.json", metadata)
    print(f"Saved rewired ICO results to {results_dir}", flush=True)
    return raw, phi_c, comparisons


def critical_budgets(raw: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    """Find steady crossings on each paired graph/replicate trajectory."""
    settled = raw.loc[raw.baseline_stop_reason.eq("steady_state")]
    frames = []
    for (graph, replicate), group in settled.groupby(["graph", "replicate"], sort=False):
        table = critical_budget_runs(
            group.rename(columns={"initialization": "realization"}), threshold=threshold
        )
        table = table.rename(columns={"realization": "initialization"})
        table["graph"], table["replicate"] = graph, replicate
        if "connected" in group:
            table["connected"] = bool(group.connected.iloc[0])
        frames.append(table)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "graph",
                "replicate",
                "strategy",
                "initialization",
                "phi_c",
                "threshold_reached",
                "reference_below_threshold",
                "all_steady",
                "max_tested_budget",
            ]
        )
    )


def summarize_phi_c(
    phi_c: pd.DataFrame, *, rewiring_summary: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Keep failed and zero-budget trials beside conditional threshold means."""
    summary = summarize_thresholds(phi_c, ["graph", "strategy"])
    summary["reached"] = summary.reached_runs
    if rewiring_summary is None:
        return summary
    achieved = (
        rewiring_summary.groupby("name", as_index=False, sort=False)
        .agg(
            h_insular_fraction_of_max=("h_insular_fraction_of_max", "mean"),
            h_integrated_fraction_of_max=("h_integrated_fraction_of_max", "mean"),
            fraction_changed=("fraction_changed", "mean"),
        )
        .rename(columns={"name": "graph"})
    )
    return summary.merge(achieved, on="graph", how="left")


def compare_to_controls(
    raw: pd.DataFrame, *, n_bootstrap: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Return paired g_minus differences of each graph against empirical and reference.

    Rows of each difference matrix are initializations and columns are replicates;
    incomplete matrices (unsettled baselines) are skipped and counted.
    """
    rng = np.random.default_rng(seed)
    settled = raw.loc[raw["baseline_stop_reason"] == "steady_state"]
    table = settled.pivot_table(
        index=["strategy", "budget", "initialization"],
        columns=["graph", "replicate"],
        values="g_minus",
    )
    names = [name for name in dict.fromkeys(raw["graph"]) if name != EMPIRICAL]
    rows = []
    for (strategy, budget), part in table.groupby(level=["strategy", "budget"]):
        for name in names:
            controls = [EMPIRICAL] + ([REFERENCE] if name != REFERENCE else [])
            for control in controls:
                if name not in part.columns.get_level_values(0) or control not in part:
                    continue
                values = part[name].to_numpy()
                reference = part[control].to_numpy()
                if control == EMPIRICAL:
                    reference = np.repeat(reference[:, :1], values.shape[1], axis=1)
                elif reference.shape != values.shape:
                    continue
                difference = values - reference
                complete = np.isfinite(difference).all(axis=1)
                row = {
                    "strategy": strategy,
                    "budget": budget,
                    "graph": name,
                    "control": control,
                    "paired_initializations": int(complete.sum()),
                    "skipped_initializations": int((~complete).sum()),
                    "replicates": values.shape[1],
                    "mean_difference": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                }
                if complete.any():
                    low, high = paired_bootstrap_interval(
                        difference[complete], rng=rng, n_bootstrap=n_bootstrap
                    )
                    row.update(
                        mean_difference=float(difference[complete].mean()),
                        ci_low=low,
                        ci_high=high,
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _pack(states: list[np.ndarray], n_nodes: int) -> np.ndarray:
    if not states:
        return np.zeros((0, (n_nodes + 7) // 8), dtype=np.uint8)
    return np.packbits(np.asarray(states) > 0, axis=1)


def _summary_row(
    name: str,
    target: Target | None,
    rule: str,
    beta: float | None,
    replicate: int,
    graph_seed: int | None,
    graph: nx.Graph,
    bounds: dict[str, int],
    *,
    assortativity: float | None = None,
) -> dict:
    counts = homophily_counts(graph)
    if assortativity is None:
        assortativity = (
            nx.degree_assortativity_coefficient(graph) if graph.number_of_edges() else float("nan")
        )
    return {
        "name": name,
        "target": target,
        "rule": rule,
        "beta": beta,
        "replicate": replicate,
        "graph_seed": graph_seed,
        **counts,
        "h_insular_fraction_of_max": counts["h_insular"] / bounds["h_insular_max"]
        if bounds["h_insular_max"]
        else float("nan"),
        "h_integrated_fraction_of_max": counts["h_integrated"] / bounds["h_integrated_max"]
        if bounds["h_integrated_max"]
        else float("nan"),
        "degree_assortativity": assortativity,
    }


def _centrality_table(graph: nx.Graph, nodes: tuple[str, ...]) -> pd.DataFrame:
    """Degree and leverage targeting scores in node order."""
    leverage = leverage_centrality(graph)
    return pd.DataFrame(
        {
            "node": nodes,
            "degree": [graph.degree(node) for node in nodes],
            "leverage": [leverage[node] for node in nodes],
        }
    )


def _saved_assortativity(path: Path, graph: nx.Graph) -> float | None:
    """Return saved core assortativity when the metrics describe this core's input."""
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    input_checksum = graph.graph.get("input_sha256")
    same_input = (
        input_checksum is not None and metrics.get("input", {}).get("sha256") == input_checksum
    )
    same_core = metrics.get("core", {}).get("edges") == graph.number_of_edges()
    value = metrics.get("metrics", {}).get("degree_assortativity")
    return float(value) if same_input and same_core and value is not None else None


def _load_network(folder: Path, nodes: tuple[str, ...]) -> NetworkMatrices:
    unsigned, signed = load_npz(folder / "unsigned.npz"), load_npz(folder / "signed.npz")
    if unsigned.shape != (len(nodes), len(nodes)) or signed.shape != unsigned.shape:
        raise ValueError(f"{folder} matrices do not match the node order")
    return NetworkMatrices(nodes, csr_matrix(unsigned), csr_matrix(signed))


def _load_scores(path: Path, nodes: tuple[str, ...]) -> dict[str, dict[str, float]]:
    table = pd.read_csv(path)
    if set(table["node"]) != set(nodes):
        raise ValueError(f"{path} does not cover the node order")
    return {
        column: dict(zip(table["node"], table[column].astype(float), strict=True))
        for column in ("degree", "leverage")
    }


def _resolve(root: Path, path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else root / path
