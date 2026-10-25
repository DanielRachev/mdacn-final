"""Null Model B experiments: export rewired graphs and compare signed ICO Phi_c."""

import hashlib
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

from mdacn.centrality import leverage_centrality
from mdacn.contracts import NetworkMatrices, SimulationResult, Strategy
from mdacn.data import build_matrices, load_graph
from mdacn.dynamics import _prepare_matrix, _prepare_state, _step, initialize_opinions, simulate_nco
from mdacn.evaluation import compute_opinion_gcc
from mdacn.interventions import select_contrarians
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

# Calibrated on the 10-core: insular betas span below, near, and above the empirical
# insular score; integrated betas span reference to near-saturation. Betas >= 8
# matched non-worsening in both directions.
DEFAULT_BETAS = {"insular": (1.0, 2.0, 4.0), "integrated": (0.5, 1.0, 2.0)}
DEFAULT_BUDGETS = (0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25)
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
    nodes_path: Path = Path("data/processed/nodes.json"),
    centrality_path: Path = Path("data/processed/centrality.csv"),
    metrics_path: Path = Path("results/topology/metrics.json"),
    results_dir: Path = Path("results/rewiring"),
    specs: tuple[RewiringSpec, ...] | None = None,
    replicates: int = 5,
    attempts_per_edge: int = 100,
    window_per_edge: float = 2.0,
    tolerance: float = 0.002,
    seed: int = 0,
    checkpoints: int = 20,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Owner: Jakub. Generate, validate, and export rewired graphs for each spec.

    Every graph receives at most attempts_per_edge * |E| attempted swaps and stops
    earlier once its tracked score changes by at most tolerance * (signed edges) over
    a window of window_per_edge * |E| attempts; window_per_edge = 0 disables the
    stop. Checkpoints are saved once per window, or checkpoints times without
    a window. Each replicate
    writes unsigned.npz, signed.npz, degrees.csv, and metadata.json under
    graphs/<spec>/replicate_<k>. Matrices follow the original nodes.json order,
    which is checked rather than rewritten. Each replicate also saves the degree
    and leverage scores used for targeting as centrality.csv.

    Saved empirical outputs are reused: centrality_path supplies empirical scores
    and metrics_path the empirical degree assortativity when present and matching
    the core. Missing empirical scores are computed once and saved under
    graphs/empirical. Return one summary row per graph, including the empirical graph.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if min(attempts_per_edge, window_per_edge, tolerance, checkpoints) < 0:
        raise ValueError("Attempt, window, tolerance, and checkpoint settings must be >= 0")
    specs = default_specs() if specs is None else tuple(specs)
    if len({spec.name for spec in specs}) != len(specs) or EMPIRICAL in {s.name for s in specs}:
        raise ValueError("Spec names must be unique and differ from 'empirical'")
    graph = load_graph(graph_path)
    nodes = _load_nodes(nodes_path, graph)
    attempts = attempts_per_edge * graph.number_of_edges()
    window = round(window_per_edge * graph.number_of_edges())
    interval = window or (max(1, attempts // checkpoints) if checkpoints and attempts else 0)
    bounds = homophily_bounds(graph)
    before = homophily_counts(graph)
    results_dir.mkdir(parents=True, exist_ok=True)
    empirical_centrality = _empirical_centrality(graph, centrality_path, results_dir)

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
    jobs = [(spec, replicate) for spec in specs for replicate in range(replicates)]
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
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
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
        "graph_sha256": _sha256(graph_path),
        "nodes_path": str(nodes_path.resolve()),
        "nodes_sha256": _sha256(nodes_path),
        "empirical_centrality": empirical_centrality,
        "empirical_centrality_sha256": _sha256(_resolve(results_dir, empirical_centrality)),
        "targeting_scores": "Degree and leverage per graph, saved once as centrality.csv",
        "graph_metadata": graph.graph,
        "n_nodes": len(graph),
        "n_edges": graph.number_of_edges(),
        "specs": [asdict(spec) for spec in specs],
        "replicates": replicates,
        "attempts_per_edge": attempts_per_edge,
        "max_attempts_per_graph": attempts,
        "window_per_edge": window_per_edge,
        "window_attempts": window,
        "tolerance": tolerance,
        "convergence": (
            "Stop when the tracked score changes by at most tolerance * (positive + "
            "negative edges) over one window; tracked score is the target score, or "
            "within-community signed edges for the reference"
        ),
        "checkpoint_interval": interval,
        "seed": seed,
        "seed_assignment": "SeedSequence(seed, spawn_key=(crc32('rewire:<spec>'), replicate))",
        "proposal": (
            "Uniform first edge, uniform second edge of the same sign, independent "
            "orientation flips; (u, v), (x, y) -> (u, y), (x, v)"
        ),
        "rejections": "Self loops, duplicate edges, and unchanged pairs; then the rule",
        "sign_rule": (
            "Same-sign swaps; each edge record (sign, n_positive, n_negative) follows "
            "its first endpoint: (u, y) takes (u, v), (x, v) takes (x, y)"
        ),
        "steering": (
            "h_insular = positive_within + negative_between; "
            "h_integrated = positive_between + negative_within; zero edges unsteered"
        ),
        "rules": {
            "unbiased": "accept every valid swap",
            "strict": "accept delta_h > 0",
            "non_worsening": "accept delta_h >= 0",
            "metropolis": "accept delta_h >= 0, else with probability exp(beta * delta_h)",
        },
        "invariants": (
            "Node set, unsigned degree, positive/negative/zero degree per node, global "
            "sign counts, edge-record multiset. Edge set and connectivity can change."
        ),
        "communities": "Fixed baseline labels stored in the core graph's 'community' attribute",
        "versions": {name: version(name) for name in ("mdacn-final", "networkx", "numpy")},
    }
    (results_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(f"Saved {len(jobs)} rewired graphs to {results_dir}", flush=True)
    return summary


def run_rewired_ico_experiment(
    processed_dir: Path = Path("data/processed"),
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
    """Owner: Jakub. Run paired signed ICO on the empirical and every rewired graph.

    Each initial state settles under signed NCO before contrarians are pinned at +1.
    Initial states and random-targeting seeds depend only on the initialization
    index, so they match across graphs. Following the shared convention, g_minus
    and g_plus use the fixed empirical unsigned core and the full core N for every
    graph. Baseline and terminal states are saved as packed bits in final_states.npz
    so other measurements can be taken from stored results. Return raw runs,
    per-run Phi_c, and comparisons.

    Nothing is recomputed: empirical matrices and node order come from
    processed_dir, rewired matrices and every targeting score from rewired_dir.
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
    nodes_path = processed_dir / "nodes.json"
    if _sha256(nodes_path) != rewiring["nodes_sha256"]:
        raise ValueError(f"{nodes_path} differs from the node order used for rewiring")
    nodes = tuple(json.loads(nodes_path.read_text(encoding="utf-8")))
    centrality = _resolve(rewired_dir, rewiring["empirical_centrality"])
    if _sha256(centrality) != rewiring["empirical_centrality_sha256"]:
        raise ValueError(f"{centrality} changed since rewiring")
    empirical = _load_network(processed_dir, nodes)
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
                    result = _simulate_signed_ico(
                        network.signed, baseline.final_state, nodes, contrarians, max_steps
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
        "processed_dir": str(processed_dir.resolve()),
        "matrix_sha256": {
            name: _sha256(processed_dir / name)
            for name in ("unsigned.npz", "signed.npz", "nodes.json")
        },
        "empirical_centrality": str(centrality.resolve()),
        "rewired_dir": str(rewired_dir.resolve()),
        "rewiring_summary_sha256": _sha256(rewired_dir / "rewiring_summary.csv"),
        "budgets": list(budgets),
        "negative_fraction": negative_fraction,
        "initializations": initializations,
        "strategies": list(strategies),
        "seed": seed,
        "initial_seeds": initial_seeds,
        "target_seeds": target_seeds,
        "seed_assignment": "SeedSequence(seed, spawn_key=(crc32('initial'|'target'), i))",
        "max_steps": max_steps,
        "threshold": threshold,
        "model": "Signed NCO baseline, then signed ICO with contrarians pinned at +1",
        "targeting_scores": "Loaded from each graph's saved centrality.csv",
        "g_minus_definition": (
            "Largest -1 component on the fixed empirical unsigned core / core N, for "
            "every graph (shared convention)"
        ),
        "final_states": (
            "final_states.npz: packed bits (1 = +1) in nodes.json order; 'baseline' rows "
            "match baseline_index and 'final' rows match state_index (-1 = not run)"
        ),
        "phi_c_definition": (
            "First tested budget with g_minus < threshold; 0 if already below at Phi = 0; "
            "NaN with threshold_reached = False when not reached in the tested range"
        ),
        "excluded_runs": "Initializations whose signed baseline did not reach a steady state",
        "comparisons": (
            "Paired g_minus differences per budget; bootstrap over initializations and "
            "replicates. Reference pairs share a replicate index only by convention."
        ),
        "versions": {name: version(name) for name in ("mdacn-final", "numpy", "scipy")},
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved rewired ICO results to {results_dir}", flush=True)
    return raw, phi_c, comparisons


def critical_budgets(raw: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    """Return each run's first tested budget with g_minus below threshold.

    A run is one graph, replicate, strategy, and initialization. Runs without a
    settled baseline are excluded. all_steady flags runs whose ICO states all settled.
    """
    keys = ["graph", "replicate", "strategy", "initialization"]
    settled = raw.loc[raw["baseline_stop_reason"] == "steady_state"]
    rows = []
    for key, group in settled.groupby(keys, sort=False):
        group = group.sort_values("budget")
        met = group.loc[group["g_minus"] < threshold, "budget"]
        row = {
            **dict(zip(keys, key, strict=True)),
            "phi_c": float(met.iloc[0]) if len(met) else float("nan"),
            "threshold_reached": bool(len(met)),
            "all_steady": bool(group["stop_reason"].eq("steady_state").all()),
        }
        if "connected" in group:
            row["connected"] = bool(group["connected"].iloc[0])
        rows.append(row)
    columns = [*keys, "phi_c", "threshold_reached", "all_steady"]
    return pd.DataFrame(rows, columns=columns + (["connected"] if "connected" in raw else []))


def summarize_phi_c(
    phi_c: pd.DataFrame, *, rewiring_summary: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Count reached thresholds and summarize reached Phi_c per graph and strategy.

    With rewiring_summary, add each configuration's mean achieved homophily (as a
    fraction of its bound) and changed-edge fraction, so Phi_c can be read against
    the configuration that was actually reached.
    """
    extra = (
        {"disconnected_runs": ("connected", lambda s: int((~s.astype(bool)).sum()))}
        if "connected" in phi_c
        else {}
    )
    summary = phi_c.groupby(["graph", "strategy"], as_index=False, sort=False).agg(
        runs=("phi_c", "size"),
        reached=("threshold_reached", "sum"),
        median_phi_c=("phi_c", "median"),
        mean_phi_c=("phi_c", "mean"),
        min_phi_c=("phi_c", "min"),
        max_phi_c=("phi_c", "max"),
        all_steady_runs=("all_steady", "sum"),
        **extra,
    )
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


def _simulate_signed_ico(
    signed: csr_matrix,
    initial_state: np.ndarray,
    nodes: tuple[str, ...],
    contrarians: tuple[str, ...],
    max_steps: int,
) -> SimulationResult:
    """Signed ICO from a settled state with contrarians pinned at +1.

    Mirrors simulate_ico(model='signed') on feat/impact-study; replace this helper
    with that function once it reaches main.
    """
    matrix = _prepare_matrix(signed, "signed")
    state = _prepare_state(initial_state, len(nodes))
    chosen = set(contrarians)
    fixed = np.array([i for i, node in enumerate(nodes) if node in chosen], dtype=int)
    state[fixed] = 1
    seen = {np.packbits(state > 0).tobytes(): 0}
    for step in range(1, max_steps + 1):
        state = _step(matrix, state)
        state[fixed] = 1
        key = np.packbits(state > 0).tobytes()
        previous = seen.get(key)
        if previous is not None:
            period = step - previous
            if period == 1:
                return SimulationResult(state, step, "steady_state")
            return SimulationResult(state, step, "limit_cycle", period)
        seen[key] = step
    return SimulationResult(state, max_steps, "max_steps")


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


def _empirical_centrality(graph: nx.Graph, path: Path, results_dir: Path) -> str:
    """Return the saved empirical score file, computing and saving it only if missing."""
    nodes = tuple(sorted(graph))
    if path.is_file():
        table = pd.read_csv(path)
        degrees = dict(zip(table["node"], table["degree"], strict=True))
        if set(degrees) != set(nodes) or any(degrees[n] != graph.degree(n) for n in nodes):
            raise ValueError(f"{path} does not match the core graph")
        if "leverage" not in table:
            raise ValueError(f"{path} has no leverage column")
        return str(path.resolve())
    target = results_dir / "graphs" / EMPIRICAL / "centrality.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    _centrality_table(graph, nodes).to_csv(target, index=False)
    return target.relative_to(results_dir).as_posix()


def _saved_assortativity(path: Path, graph: nx.Graph) -> float | None:
    """Return saved core assortativity when the metrics describe this core's input."""
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    checksum = graph.graph.get("input_sha256")
    same_input = checksum is not None and metrics.get("input", {}).get("sha256") == checksum
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


def _load_nodes(path: Path, graph: nx.Graph) -> tuple[str, ...]:
    nodes = tuple(json.loads(path.read_text(encoding="utf-8")))
    if nodes != tuple(sorted(graph)):
        raise ValueError(f"{path} does not match the core graph's node order")
    return nodes


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
