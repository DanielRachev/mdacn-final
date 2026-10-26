"""Replay the NCO and ICO protocol on nearby core extractions."""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from mdacn import config
from mdacn.artifacts import checksum, write_json
from mdacn.centrality_pipeline import save_centrality_scores
from mdacn.data import load_graph
from mdacn.ico_pipeline import compute_phi_c, run_ico_experiment
from mdacn.nco_pipeline import run_nco_experiment
from mdacn.sensitivity_plots import (
    plot_ico_by_configuration,
    plot_nco_by_configuration,
    plot_nco_raw_fraction,
    plot_phi_c_by_configuration,
    plot_topology_sign_fractions,
)
from mdacn.targeting_pipeline import save_targeting_scores

HEADLINE_FRACTION = 0.5
HEADLINE_BUDGETS = (0.05, 0.1, 0.25)


def _core_paths(extraction_dir: Path, configuration: str) -> dict[str, Path]:
    """Return the saved paths for one exported core configuration."""
    return config.processed_paths(extraction_dir / configuration)


def _require_configuration(extraction_dir: Path, configuration: str) -> Path:
    """Return the core graph and name the comparison command when inputs are missing."""
    paths = _core_paths(extraction_dir, configuration)
    required = ("core", "unsigned", "signed", "nodes", "communities")
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise ValueError(
            f"Missing sensitivity inputs for {configuration}: {', '.join(missing)}. "
            "Run compare-extractions first."
        )
    return paths["core"]


def _topology_table(
    extraction_dir: Path, configurations: tuple[str, ...], results_dir: Path
) -> pd.DataFrame:
    """Return node, edge, and sign summaries for the selected cores."""
    comparison = results_dir.parent / "extraction_comparison" / "comparison.csv"
    if comparison.is_file():
        table = pd.read_csv(comparison, dtype={"configuration": str}, keep_default_na=False)
        missing = sorted(set(configurations) - set(table["configuration"]))
        if missing:
            raise ValueError(f"Missing configurations in {comparison}: {missing}")
        order = {name: index for index, name in enumerate(configurations)}
        table = table.loc[table["configuration"].isin(configurations)].copy()
        table["_order"] = table["configuration"].map(order)
        return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    rows = []
    for configuration in configurations:
        graph = load_graph(extraction_dir / configuration / "core.graphml")
        signs = [data["sign"] for _, _, data in graph.edges(data=True)]
        edges = len(signs)
        rows.append(
            {
                "configuration": configuration,
                "nodes": graph.number_of_nodes(),
                "edges": edges,
                "non_self_interactions": int(
                    sum(
                        data.get("n_positive", 0) + data.get("n_negative", 0)
                        for _, _, data in graph.edges(data=True)
                    )
                ),
                "negative_edges": signs.count(-1),
                "zero_edges": signs.count(0),
                "negative_edge_fraction": signs.count(-1) / edges if edges else np.nan,
                "zero_edge_fraction": signs.count(0) / edges if edges else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _summarize_nco(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return outcome summaries and stop-reason counts for the NCO sweep."""
    records = []
    for keys, group in runs.groupby(["configuration", "model", "f"], sort=True):
        configuration, model, fraction = keys
        records.append(
            {
                "configuration": configuration,
                "model": model,
                "f": float(fraction),
                "runs": len(group),
                "n_nodes": int(group["n_nodes"].iloc[0]),
                "mean_g_minus": float(group["g_minus"].mean()),
                "std_g_minus": float(group["g_minus"].std(ddof=1)) if len(group) > 1 else np.nan,
                "q025_g_minus": float(group["g_minus"].quantile(0.025)),
                "q975_g_minus": float(group["g_minus"].quantile(0.975)),
                "mean_g_plus": float(group["g_plus"].mean()),
                "std_g_plus": float(group["g_plus"].std(ddof=1)) if len(group) > 1 else np.nan,
                "q025_g_plus": float(group["g_plus"].quantile(0.025)),
                "q975_g_plus": float(group["g_plus"].quantile(0.975)),
                "mean_final_negative_fraction": float(group["final_negative_fraction"].mean()),
                "std_final_negative_fraction": (
                    float(group["final_negative_fraction"].std(ddof=1))
                    if len(group) > 1
                    else np.nan
                ),
                "q025_final_negative_fraction": float(
                    group["final_negative_fraction"].quantile(0.025)
                ),
                "q975_final_negative_fraction": float(
                    group["final_negative_fraction"].quantile(0.975)
                ),
            }
        )
    summary = pd.DataFrame(records)
    stops = (
        runs.groupby(["configuration", "model", "f", "stop_reason"], as_index=False)
        .size()
        .rename(columns={"size": "runs"})
    )
    if not stops.empty:
        total = stops.groupby(["configuration", "model", "f"])["runs"].transform("sum")
        stops["fraction"] = stops["runs"] / total
    return summary, stops


def _summarize_ico(
    runs: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return outcome summaries, stop reasons, and per-run Phi_c for the ICO sweep."""
    records = []
    keys = ["configuration", "model", "f", "strategy", "budget"]
    for values, group in runs.groupby(keys, sort=True):
        configuration, model, fraction, strategy, budget = values
        records.append(
            {
                "configuration": configuration,
                "model": model,
                "f": float(fraction),
                "strategy": strategy,
                "budget": float(budget),
                "runs": len(group),
                "mean_g_minus": float(group["g_minus"].mean()),
                "std_g_minus": float(group["g_minus"].std(ddof=1)) if len(group) > 1 else np.nan,
                "q025_g_minus": float(group["g_minus"].quantile(0.025)),
                "q975_g_minus": float(group["g_minus"].quantile(0.975)),
                "mean_baseline_g_minus": float(group["baseline_g_minus"].mean()),
                "mean_delta_g_minus": float(group["delta_g_minus"].mean()),
                "std_delta_g_minus": (
                    float(group["delta_g_minus"].std(ddof=1)) if len(group) > 1 else np.nan
                ),
                "backfire_run_fraction": float(group["backfired"].mean()),
                "mean_negative_influence_links": float(group["negative_influence_links"].mean()),
                "mean_repelled_share": float(group["repelled_share"].mean()),
            }
        )
    summary = pd.DataFrame(records)
    stops = (
        runs.groupby(["configuration", "model", "f", "strategy", "stop_reason"], as_index=False)
        .size()
        .rename(columns={"size": "runs"})
    )
    if not stops.empty:
        total = stops.groupby(["configuration", "model", "f", "strategy"])["runs"].transform("sum")
        stops["fraction"] = stops["runs"] / total

    phi_tables = []
    for configuration, group in runs.groupby("configuration", sort=True):
        table = compute_phi_c(group, threshold=threshold)
        table.insert(0, "configuration", configuration)
        phi_tables.append(table)
    phi_c = pd.concat(phi_tables, ignore_index=True) if phi_tables else pd.DataFrame()
    return summary, stops, phi_c


def _pick_mean(table: pd.DataFrame, value: str, **conditions: object) -> float:
    """Return the mean of one column for matching rows, or NaN when absent."""
    subset = table
    for column, target in conditions.items():
        if column not in subset:
            return np.nan
        if column in ("f", "budget"):
            subset = subset.loc[np.isclose(subset[column].astype(float), float(target))]
        else:
            subset = subset.loc[subset[column] == target]
    if subset.empty or value not in subset:
        return np.nan
    return float(subset[value].mean())


def _headline_table(
    topology: pd.DataFrame,
    nco_summary: pd.DataFrame,
    ico_summary: pd.DataFrame,
    phi_c: pd.DataFrame,
) -> pd.DataFrame:
    """Return one compact row per configuration for the report and the deck."""
    rows = []
    for record in topology.to_dict("records"):
        configuration = str(record["configuration"])
        row = {
            name: record.get(name, np.nan)
            for name in (
                "configuration",
                "nodes",
                "edges",
                "negative_edge_fraction",
                "zero_edge_fraction",
                "average_degree",
                "modularity",
                "event_retention",
            )
        }
        for model in ("unsigned", "signed"):
            row[f"nco_{model}_g_minus_f{HEADLINE_FRACTION:g}"] = _pick_mean(
                nco_summary,
                "mean_g_minus",
                configuration=configuration,
                model=model,
                f=HEADLINE_FRACTION,
            )
            row[f"nco_{model}_g_plus_f{HEADLINE_FRACTION:g}"] = _pick_mean(
                nco_summary,
                "mean_g_plus",
                configuration=configuration,
                model=model,
                f=HEADLINE_FRACTION,
            )
            for budget in HEADLINE_BUDGETS:
                row[f"ico_{model}_degree_g_minus_phi{budget:g}"] = _pick_mean(
                    ico_summary,
                    "mean_g_minus",
                    configuration=configuration,
                    model=model,
                    strategy="degree",
                    budget=budget,
                )
            row[f"phi_c_{model}_degree"] = _pick_mean(
                phi_c,
                "mean_phi_c",
                configuration=configuration,
                model=model,
                strategy="degree",
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _strategy_ranking_table(
    ico_summary: pd.DataFrame,
    phi_c: pd.DataFrame,
    *,
    fraction: float = HEADLINE_FRACTION,
) -> pd.DataFrame:
    """Return strategies ranked by paired effect at the selected budgets."""
    if ico_summary.empty:
        return pd.DataFrame()
    subset = ico_summary.loc[np.isclose(ico_summary["f"].astype(float), fraction)]
    records = []
    for keys, group in subset.groupby(["configuration", "model", "budget"], sort=True):
        configuration, model, budget = keys
        if not any(np.isclose(float(budget), value) for value in HEADLINE_BUDGETS):
            continue
        phi_table = phi_c.loc[
            (phi_c["configuration"] == configuration)
            & (phi_c["model"] == model)
            & (phi_c["strategy"].isin(group["strategy"]))
            & np.isclose(phi_c["f"].astype(float), fraction)
        ].set_index("strategy")
        for rank, row in enumerate(
            group.sort_values("mean_delta_g_minus", kind="stable").itertuples(index=False),
            start=1,
        ):
            records.append(
                {
                    "configuration": configuration,
                    "model": model,
                    "f": float(fraction),
                    "budget": float(budget),
                    "rank": rank,
                    "strategy": row.strategy,
                    "mean_g_minus": float(row.mean_g_minus),
                    "mean_delta_g_minus": float(row.mean_delta_g_minus),
                    "backfire_run_fraction": float(row.backfire_run_fraction),
                    "mean_repelled_share": float(row.mean_repelled_share),
                    "mean_phi_c": (
                        float(phi_table.loc[row.strategy, "mean_phi_c"])
                        if row.strategy in phi_table.index
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _read_saved(path: Path, configuration: str) -> pd.DataFrame:
    """Read a previously saved sensitivity sweep and tag its configuration."""
    if not path.is_file():
        raise ValueError(f"Missing saved sensitivity table: {path}. Drop the skip flag first.")
    table = pd.read_csv(path)
    table["configuration"] = configuration
    return table


def run_sensitivity(
    extraction_dir: Path | str | None = None,
    *,
    results_dir: Path | str | None = None,
    configurations: tuple[str, ...] | None = None,
    fractions: tuple[float, ...] | None = None,
    ico_fractions: tuple[float, ...] | None = None,
    models: tuple[str, ...] | None = None,
    strategies: tuple[str, ...] | None = None,
    budgets: tuple[float, ...] | None = None,
    realizations: int | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    phi_c_threshold: float | None = None,
    skip_nco: bool = False,
    skip_ico: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run NCO and selected ICO outcomes on every exported core.

    Inputs come from compare-extractions. Each core keeps its own centrality
    and targeting exports. The protocol, seeds, realisations, and budget grid
    stay fixed across configurations. Return the written summary tables.
    """
    extraction_dir = Path(
        extraction_dir if extraction_dir is not None else config.SENSITIVITY["extraction_dir"]
    )
    results_dir = Path(
        results_dir if results_dir is not None else config.SENSITIVITY["results_dir"]
    )
    configurations = tuple(
        configurations if configurations is not None else config.SENSITIVITY["configurations"]
    )
    fractions = tuple(fractions if fractions is not None else config.SENSITIVITY["fractions"])
    ico_fractions = tuple(
        ico_fractions if ico_fractions is not None else config.SENSITIVITY["ico_fractions"]
    )
    models = tuple(models if models is not None else config.SENSITIVITY["models"])
    strategies = tuple(strategies if strategies is not None else config.SENSITIVITY["strategies"])
    budgets = tuple(budgets if budgets is not None else config.SENSITIVITY["budgets"])
    realizations = int(
        realizations if realizations is not None else config.SENSITIVITY["realizations"]
    )
    seed = int(seed if seed is not None else config.SENSITIVITY["seed"])
    max_steps = int(max_steps if max_steps is not None else config.SENSITIVITY["max_steps"])
    phi_c_threshold = float(
        phi_c_threshold if phi_c_threshold is not None else config.SENSITIVITY["phi_c_threshold"]
    )

    if not configurations:
        raise ValueError("configurations must be nonempty")
    if not skip_nco and not fractions:
        raise ValueError("fractions must be nonempty")
    if not skip_ico and (not ico_fractions or not models or not strategies or not budgets):
        raise ValueError("ICO fractions, models, strategies, and budgets must be nonempty")
    results_dir.mkdir(parents=True, exist_ok=True)

    nco_tables = []
    ico_tables = []
    for configuration in configurations:
        graph_path = _require_configuration(extraction_dir, configuration)
        print(f"Sensitivity: {configuration}", flush=True)
        if not skip_ico:
            save_centrality_scores(graph_path)
            # The ICO metadata checksums both proposed score files even when the
            # selected strategies do not use them, so export them for every core.
            save_targeting_scores(graph_path)

        if skip_nco:
            nco_tables.append(
                _read_saved(
                    results_dir / configuration / "nco" / "fraction_sweep.csv", configuration
                )
            )
        else:
            nco = run_nco_experiment(
                graph_path,
                results_dir=results_dir / configuration / "nco",
                fractions=fractions,
                realizations=realizations,
                seed=seed,
                max_steps=max_steps,
                save_reports=False,
            )
            nco["configuration"] = configuration
            nco_tables.append(nco)

        if skip_ico:
            ico_tables.append(
                _read_saved(
                    results_dir / configuration / "ico" / "intervention_sweep.csv", configuration
                )
            )
        else:
            ico = run_ico_experiment(
                graph_path,
                results_dir=results_dir / configuration / "ico",
                fractions=ico_fractions,
                budgets=budgets,
                models=models,
                strategies=strategies,
                realizations=realizations,
                seed=seed,
                max_steps=max_steps,
                phi_c_threshold=phi_c_threshold,
                save_reports=False,
            )
            ico["configuration"] = configuration
            ico_tables.append(ico)

    nco_runs = pd.concat(nco_tables, ignore_index=True) if nco_tables else pd.DataFrame()
    ico_runs = pd.concat(ico_tables, ignore_index=True) if ico_tables else pd.DataFrame()
    nco_runs.to_csv(results_dir / "nco_runs.csv", index=False)
    ico_runs.to_csv(results_dir / "ico_runs.csv", index=False)

    nco_summary, nco_stops = (
        _summarize_nco(nco_runs)
        if not nco_runs.empty
        else (
            pd.DataFrame(),
            pd.DataFrame(),
        )
    )
    ico_summary, ico_stops, phi_c = (
        _summarize_ico(ico_runs, phi_c_threshold)
        if not ico_runs.empty
        else (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    )
    topology = _topology_table(extraction_dir, configurations, results_dir)
    headline = _headline_table(topology, nco_summary, ico_summary, phi_c)
    ranking = _strategy_ranking_table(ico_summary, phi_c)

    topology.to_csv(results_dir / "topology.csv", index=False)
    nco_summary.to_csv(results_dir / "nco_summary.csv", index=False)
    nco_stops.to_csv(results_dir / "nco_stop_reasons.csv", index=False)
    ico_summary.to_csv(results_dir / "ico_summary.csv", index=False)
    ico_stops.to_csv(results_dir / "ico_stop_reasons.csv", index=False)
    phi_c.to_csv(results_dir / "phi_c.csv", index=False)
    headline.to_csv(results_dir / "headline.csv", index=False)
    ranking.to_csv(results_dir / "strategy_ranking.csv", index=False)
    write_json(
        results_dir / "metadata.json",
        {
            "extraction_dir": str(extraction_dir.resolve()),
            "configurations": list(configurations),
            "core_sha256": {
                name: checksum(extraction_dir / name / "core.graphml") for name in configurations
            },
            "fractions": list(fractions),
            "ico_fractions": list(ico_fractions),
            "models": list(models),
            "strategies": list(strategies),
            "budgets": list(budgets),
            "realizations": realizations,
            "seed": seed,
            "max_steps": max_steps,
            "phi_c_threshold": phi_c_threshold,
            "skip_nco": skip_nco,
            "skip_ico": skip_ico,
            "versions": {name: version(name) for name in ("mdacn-final", "numpy", "pandas")},
        },
    )

    if not nco_summary.empty:
        plot_nco_by_configuration(nco_summary, results_dir / "nco_g_minus_by_configuration.png")
        plot_nco_by_configuration(
            nco_summary,
            results_dir / "nco_g_plus_by_configuration.png",
            metric="mean_g_plus",
        )
        plot_nco_raw_fraction(nco_summary, results_dir / "nco_raw_negative_fraction.png")
    if not ico_summary.empty:
        plot_ico_by_configuration(
            ico_summary,
            results_dir / "ico_degree_g_minus_by_configuration.png",
            strategy="degree",
        )
    if not phi_c.empty:
        plot_phi_c_by_configuration(phi_c, results_dir / "phi_c_by_configuration.png")
    plot_topology_sign_fractions(topology, results_dir / "topology_sign_fractions.png")

    print(f"Saved sensitivity results to {results_dir}", flush=True)
    return {
        "topology": topology,
        "nco_summary": nco_summary,
        "ico_summary": ico_summary,
        "phi_c": phi_c,
        "headline": headline,
        "strategy_ranking": ranking,
    }
