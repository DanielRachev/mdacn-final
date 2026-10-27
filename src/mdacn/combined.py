"""Assemble saved experiment results into one combined report."""

import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from mdacn import config
from mdacn.artifacts import checksum, write_json
from mdacn.combined_plots import (
    plot_backfire,
    plot_ico_curves,
    plot_ico_delta,
    plot_impact_correlations,
    plot_phase_transition,
    plot_phi_c,
    plot_raw_fractions,
    plot_rewiring,
    plot_stop_reasons,
    plot_temporal_effects,
)
from mdacn.ico_pipeline import compute_phi_c

CURVE_METRICS = (
    "g_minus",
    "g_plus",
    "g_minus_second",
    "g_plus_second",
    "final_negative_fraction",
)


def _read_optional(path: Path, label: str) -> pd.DataFrame | None:
    """Read an input table or print a warning when it is missing."""
    if not path.is_file():
        print(f"Warning: missing combined input: {label} ({path})", flush=True)
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _require_columns(table: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    """Raise an actionable error when an existing input has the wrong shape."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def _register(
    manifest: dict,
    name: str,
    path: Path,
    table: pd.DataFrame | None = None,
    note: str = "",
) -> None:
    """Record one input path, row count, and checksum."""
    entry: dict[str, object] = {
        "path": str(path),
        "present": path.is_file(),
        "note": note,
    }
    if table is not None:
        entry["rows"] = int(len(table))
        entry["columns"] = table.columns.tolist()
    if path.is_file():
        entry["sha256"] = checksum(path)
    manifest["inputs"][name] = entry


def _resolve_temporal_dir(temporal_dir: Path | str | None) -> Path | None:
    """Return the chosen temporal report directory, or the latest one."""
    if temporal_dir is not None:
        path = Path(temporal_dir)
        return path if (path / "summary.csv").is_file() else None
    root = Path(config.COMBINED["temporal_root"])
    candidates = sorted(root.glob("*/summary.csv"), key=lambda path: path.stat().st_mtime)
    return candidates[-1].parent if candidates else None


def _summarize_curves(table: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return mean, spread, and percentile bands per source, model, and fraction."""
    table = table.copy()
    table["source"] = source
    metrics = [metric for metric in CURVE_METRICS if metric in table]
    records = []
    for keys, group in table.groupby(["source", "model", "f"], sort=True):
        source_value, model, fraction = keys
        row: dict[str, object] = {
            "source": source_value,
            "model": model,
            "f": float(fraction),
            "runs": len(group),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"q025_{metric}"] = float(values.quantile(0.025))
            row[f"q975_{metric}"] = float(values.quantile(0.975))
        records.append(row)
    return pd.DataFrame(records)


def _raw_fraction_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Return raw negative and positive opinion fractions with bands."""
    required = (
        "mean_final_negative_fraction",
        "std_final_negative_fraction",
        "q025_final_negative_fraction",
        "q975_final_negative_fraction",
    )
    if curves.empty or not set(required) <= set(curves):
        return pd.DataFrame()
    raw = curves[["source", "model", "f", "runs", *required]].rename(
        columns={
            "mean_final_negative_fraction": "mean_s1",
            "std_final_negative_fraction": "std_s1",
            "q025_final_negative_fraction": "q025_s1",
            "q975_final_negative_fraction": "q975_s1",
        }
    )
    raw["mean_s2"] = 1.0 - raw["mean_s1"]
    raw["std_s2"] = raw["std_s1"]
    raw["q025_s2"] = 1.0 - raw["q975_s1"]
    raw["q975_s2"] = 1.0 - raw["q025_s1"]
    return raw


def _stop_reason_table(table: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return stop-reason counts and within-group shares."""
    if table.empty or "stop_reason" not in table:
        return pd.DataFrame()
    table = table.copy()
    table["source"] = source
    if "strategy" not in table:
        table["strategy"] = ""
    grouped = (
        table.groupby(["source", "model", "f", "strategy", "stop_reason"], as_index=False)
        .size()
        .rename(columns={"size": "runs"})
    )
    total = grouped.groupby(["source", "model", "f", "strategy"])["runs"].transform("sum")
    grouped["fraction"] = grouped["runs"] / total
    return grouped


def _summarize_ico(table: pd.DataFrame) -> pd.DataFrame:
    """Return ICO outcome summaries per model, fraction, strategy, and budget."""
    records = []
    for keys, group in table.groupby(["model", "f", "strategy", "budget"], sort=True):
        model, fraction, strategy, budget = keys
        record: dict[str, object] = {
            "model": model,
            "f": float(fraction),
            "strategy": strategy,
            "budget": float(budget),
            "runs": len(group),
            "mean_g_minus": float(group["g_minus"].mean()),
            "std_g_minus": (float(group["g_minus"].std(ddof=1)) if len(group) > 1 else np.nan),
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
        if "g_plus" in group:
            record["mean_g_plus"] = float(group["g_plus"].mean())
        records.append(record)
    return pd.DataFrame(records)


def _summarize_rewiring(table: pd.DataFrame) -> pd.DataFrame:
    """Return one row per rewired configuration with mean achieved properties."""
    if table.empty or not {"name", "replicate"} <= set(table):
        return pd.DataFrame()
    columns = [
        column
        for column in (
            "h_insular",
            "h_integrated",
            "h_insular_fraction_of_max",
            "h_integrated_fraction_of_max",
            "fraction_changed",
            "acceptance_rate",
            "attempts",
            "converged",
        )
        if column in table
    ]
    aggregations: dict[str, tuple[str, str]] = {"replicates": ("replicate", "nunique")}
    for column in columns:
        aggregations[f"mean_{column}"] = (column, "mean")
    if "connected" in table:
        aggregations["connected_fraction"] = ("connected", "mean")
    return table.groupby("name", as_index=False).agg(**aggregations)


def _combine_temporal(
    summary: pd.DataFrame | None,
    static_effects: pd.DataFrame | None,
    shuffle_effects: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the temporal summary and labeled paired effects."""
    effects = []
    if static_effects is not None:
        table = static_effects.copy()
        table["comparison"] = "temporal_real_vs_static"
        effects.append(table)
    if shuffle_effects is not None:
        table = shuffle_effects.copy()
        table["comparison"] = "temporal_real_vs_shuffled"
        effects.append(table)
    combined = pd.concat(effects, ignore_index=True) if effects else pd.DataFrame()
    return summary if summary is not None else pd.DataFrame(), combined


def _impact_table(impact: pd.DataFrame | None, features: pd.DataFrame | None) -> pd.DataFrame:
    """Join node impact with the saved node scores when both exist."""
    if impact is None:
        return pd.DataFrame()
    if features is None or not {"node"} <= set(impact) or "node" not in features:
        return impact
    return impact.merge(features, on="node", how="left", suffixes=("", "_feature"))


def _write_context(
    output_dir: Path,
    manifest: dict,
    results_dir: Path,
) -> None:
    """Copy compact context tables used by the report and the deck."""
    specs = (
        ("extraction_comparison", results_dir / "extraction_comparison" / "comparison.csv"),
        ("label_audit", results_dir / "label_audit" / "filtering_stages.csv"),
        ("directed_roles", results_dir / "directed_roles" / "sign_fractions.csv"),
        ("incident_signs", results_dir / "incident_signs" / "r_neg_summary.csv"),
        ("signed_triads", results_dir / "signed_triads" / "triad_summary.csv"),
        ("triad_patterns", results_dir / "signed_triads" / "triad_sign_patterns.csv"),
    )
    for name, path in specs:
        table = _read_optional(path, name)
        _register(manifest, name, path, table)
        if table is not None:
            table.to_csv(output_dir / f"context_{name}.csv", index=False)


def run_combined_results(
    results_dir: Path | str | None = None,
    *,
    output_dir: Path | str | None = None,
    temporal_dir: Path | str | None = None,
    fraction: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Assemble available saved results into common tables and figures.

    Missing inputs are reported in ``inputs.json`` and ``warnings.txt`` and do
    not stop the report. Every table is derived from saved artifacts; no
    simulation is rerun. Return the written tables.
    """
    results_dir = Path(results_dir if results_dir is not None else config.COMBINED["results_dir"])
    output_dir = Path(output_dir if output_dir is not None else config.COMBINED["output_dir"])
    fraction = float(fraction if fraction is not None else config.COMBINED["fraction"])
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_dir = _resolve_temporal_dir(temporal_dir)

    manifest: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "temporal_dir": None if temporal_dir is None else str(temporal_dir.resolve()),
        "fraction": fraction,
        "inputs": {},
    }
    tables: dict[str, pd.DataFrame] = {}

    nco = _read_optional(results_dir / "nco" / "fraction_sweep.csv", "nco")
    if nco is not None:
        _require_columns(
            nco,
            results_dir / "nco" / "fraction_sweep.csv",
            ("model", "f", "realization", "g_minus", "stop_reason"),
        )
    _register(manifest, "nco", results_dir / "nco" / "fraction_sweep.csv", nco)

    sign_shuffle = _read_optional(
        results_dir / "sign_shuffle" / "sign_shuffle_sweep.csv", "sign_shuffle"
    )
    if sign_shuffle is not None:
        _require_columns(
            sign_shuffle,
            results_dir / "sign_shuffle" / "sign_shuffle_sweep.csv",
            ("model", "f", "realization", "g_minus", "stop_reason"),
        )
    _register(
        manifest,
        "sign_shuffle",
        results_dir / "sign_shuffle" / "sign_shuffle_sweep.csv",
        sign_shuffle,
    )

    ico = _read_optional(results_dir / "ico" / "intervention_sweep.csv", "ico")
    if ico is not None:
        _require_columns(
            ico,
            results_dir / "ico" / "intervention_sweep.csv",
            (
                "model",
                "f",
                "strategy",
                "budget",
                "realization",
                "g_minus",
                "baseline_g_minus",
                "delta_g_minus",
                "backfired",
                "negative_influence_links",
                "repelled_share",
                "stop_reason",
            ),
        )
        if "model" not in ico:
            ico["model"] = "unsigned"
    _register(manifest, "ico", results_dir / "ico" / "intervention_sweep.csv", ico)

    ico_metadata_path = results_dir / "ico" / "metadata.json"
    ico_threshold = float(config.ICO["phi_c_threshold"])
    if ico_metadata_path.is_file():
        metadata = json.loads(ico_metadata_path.read_text(encoding="utf-8"))
        ico_threshold = float(metadata.get("phi_c_threshold", ico_threshold))
    _register(manifest, "ico_metadata", ico_metadata_path)

    rewiring = _read_optional(results_dir / "rewiring" / "rewiring_summary.csv", "rewiring")
    _register(manifest, "rewiring", results_dir / "rewiring" / "rewiring_summary.csv", rewiring)
    rewiring_phi_path = results_dir / "rewiring_ico" / "phi_c_summary.csv"
    rewiring_phi = _read_optional(rewiring_phi_path, "rewiring_phi_c")
    _register(manifest, "rewiring_phi_c", rewiring_phi_path, rewiring_phi)
    comparisons_path = results_dir / "rewiring_ico" / "g_minus_comparisons.csv"
    rewiring_comparisons = _read_optional(comparisons_path, "rewiring_comparisons")
    _register(manifest, "rewiring_comparisons", comparisons_path, rewiring_comparisons)

    impact_path = results_dir / "node_impact" / "study" / "analysis" / "node_impact.csv"
    impact = _read_optional(impact_path, "node_impact")
    _register(manifest, "node_impact", impact_path, impact)
    correlations_path = results_dir / "node_impact" / "study" / "analysis" / "correlations.csv"
    correlations = _read_optional(correlations_path, "impact_correlations")
    _register(manifest, "impact_correlations", correlations_path, correlations)
    regression_path = results_dir / "node_impact" / "study" / "analysis" / "regression.csv"
    regression = _read_optional(regression_path, "impact_regression")
    _register(manifest, "impact_regression", regression_path, regression)
    features_path = results_dir / "node_impact" / "study" / "features.csv"
    features = _read_optional(features_path, "impact_features")
    _register(manifest, "impact_features", features_path, features)

    temporal_summary = None
    temporal_static = None
    temporal_shuffle = None
    if temporal_dir is not None:
        temporal_summary = _read_optional(temporal_dir / "summary.csv", "temporal_summary")
        temporal_static = _read_optional(
            temporal_dir / "paired_static_effects.csv", "temporal_static_effects"
        )
        temporal_shuffle = _read_optional(
            temporal_dir / "paired_shuffle_effects.csv", "temporal_shuffle_effects"
        )
        for name, filename, table in (
            ("temporal_summary", "summary.csv", temporal_summary),
            ("temporal_static_effects", "paired_static_effects.csv", temporal_static),
            ("temporal_shuffle_effects", "paired_shuffle_effects.csv", temporal_shuffle),
        ):
            _register(manifest, name, temporal_dir / filename, table)
    else:
        for name in (
            "temporal_summary",
            "temporal_static_effects",
            "temporal_shuffle_effects",
        ):
            _register(
                manifest,
                name,
                Path("reports/static_temporal/<timestamp>"),
                None,
                note="No temporal report directory was found.",
            )

    sensitivity_headline = _read_optional(
        results_dir / "sensitivity" / "headline.csv", "sensitivity_headline"
    )
    _register(
        manifest,
        "sensitivity_headline",
        results_dir / "sensitivity" / "headline.csv",
        sensitivity_headline,
    )
    sensitivity_ranking = _read_optional(
        results_dir / "sensitivity" / "strategy_ranking.csv", "sensitivity_ranking"
    )
    _register(
        manifest,
        "sensitivity_ranking",
        results_dir / "sensitivity" / "strategy_ranking.csv",
        sensitivity_ranking,
    )

    _write_context(output_dir, manifest, results_dir)

    available = [
        name
        for name, entry in manifest["inputs"].items()
        if entry["present"]  # type: ignore[index]
    ]
    if not available:
        raise ValueError(
            "No combined inputs found. Run nco, sign-shuffle, ico, node-impact, "
            "or another producing stage first."
        )

    curves = []
    if nco is not None:
        curves.append(_summarize_curves(nco, "nco"))
    if sign_shuffle is not None:
        curves.append(_summarize_curves(sign_shuffle, "sign_shuffle"))
    curve_table = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    raw = _raw_fraction_table(curve_table)

    stops = []
    if nco is not None:
        stops.append(_stop_reason_table(nco, "nco"))
    if sign_shuffle is not None:
        stops.append(_stop_reason_table(sign_shuffle, "sign_shuffle"))
    if ico is not None:
        stops.append(_stop_reason_table(ico, "ico"))
    stop_table = (
        pd.concat([table for table in stops if not table.empty], ignore_index=True)
        if any(not table.empty for table in stops)
        else pd.DataFrame()
    )

    ico_summary = _summarize_ico(ico) if ico is not None else pd.DataFrame()
    phi_c = compute_phi_c(ico, threshold=ico_threshold) if ico is not None else pd.DataFrame()
    if not ico_summary.empty:
        backfire = ico_summary.groupby(["model", "f", "strategy"], as_index=False).agg(
            max_backfire_run_fraction=("backfire_run_fraction", "max"),
            mean_repelled_share=("mean_repelled_share", "mean"),
            max_negative_influence_links=("mean_negative_influence_links", "max"),
        )
    else:
        backfire = pd.DataFrame()

    rewiring_by_config = _summarize_rewiring(rewiring) if rewiring is not None else pd.DataFrame()
    temporal_summary_table, temporal_effects = _combine_temporal(
        temporal_summary, temporal_static, temporal_shuffle
    )
    impact_table = _impact_table(impact, features)

    curve_table.to_csv(output_dir / "phase_transitions.csv", index=False)
    raw.to_csv(output_dir / "raw_fractions.csv", index=False)
    stop_table.to_csv(output_dir / "stop_reasons.csv", index=False)
    ico_summary.to_csv(output_dir / "ico_summary.csv", index=False)
    phi_c.to_csv(output_dir / "phi_c_by_strategy.csv", index=False)
    backfire.to_csv(output_dir / "backfire_summary.csv", index=False)
    rewiring_by_config.to_csv(output_dir / "rewiring_by_config.csv", index=False)
    if rewiring_phi is not None:
        rewiring_phi.to_csv(output_dir / "rewiring_phi_c.csv", index=False)
    if rewiring_comparisons is not None:
        rewiring_comparisons.to_csv(output_dir / "rewiring_g_minus_comparisons.csv", index=False)
    temporal_summary_table.to_csv(output_dir / "temporal_summary.csv", index=False)
    temporal_effects.to_csv(output_dir / "temporal_paired_effects.csv", index=False)
    impact_table.to_csv(output_dir / "impact_node_table.csv", index=False)
    if correlations is not None:
        correlations.to_csv(output_dir / "impact_correlations.csv", index=False)
    if regression is not None:
        regression.to_csv(output_dir / "impact_regression.csv", index=False)
    if sensitivity_headline is not None:
        sensitivity_headline.to_csv(output_dir / "sensitivity_headline.csv", index=False)
    if sensitivity_ranking is not None:
        sensitivity_ranking.to_csv(output_dir / "sensitivity_strategy_ranking.csv", index=False)

    if not curve_table.empty:
        plot_phase_transition(
            curve_table,
            output_dir / "phase_transition_g_minus.png",
            metric="g_minus",
            ylabel="Opposing cluster fraction g_minus",
        )
        plot_phase_transition(
            curve_table,
            output_dir / "phase_transition_g_plus.png",
            metric="g_plus",
            ylabel="Positive cluster fraction g_plus",
        )
    if not raw.empty:
        plot_raw_fractions(raw, output_dir / "raw_fractions.png")
    if not ico_summary.empty:
        plot_ico_curves(ico_summary, output_dir / "ico_g_minus.png", fraction=fraction)
        plot_ico_delta(ico_summary, output_dir / "ico_delta.png", fraction=fraction)
        plot_backfire(ico_summary, output_dir / "backfire.png", fraction=fraction)
    if not phi_c.empty:
        plot_phi_c(phi_c, output_dir / "phi_c_by_strategy.png")
    if not stop_table.empty:
        plot_stop_reasons(stop_table, output_dir / "stop_reasons.png")
    rewiring_phi_table = rewiring_phi if rewiring_phi is not None else pd.DataFrame()
    if not rewiring_by_config.empty or not rewiring_phi_table.empty:
        plot_rewiring(
            rewiring_by_config,
            rewiring_phi_table,
            output_dir / "rewiring_phi_c.png",
        )
    if not temporal_effects.empty:
        plot_temporal_effects(temporal_effects, output_dir / "temporal_effects.png")
    if correlations is not None and not correlations.empty:
        plot_impact_correlations(
            correlations, output_dir / "impact_correlations.png", fraction=fraction
        )

    missing = sorted(
        name
        for name, entry in manifest["inputs"].items()
        if not entry["present"]  # type: ignore[index]
    )
    manifest["available"] = available
    manifest["missing"] = missing
    manifest["versions"] = {name: version(name) for name in ("mdacn-final", "numpy", "pandas")}
    write_json(output_dir / "inputs.json", manifest)
    (output_dir / "warnings.txt").write_text(
        "".join(
            f"missing {name}: {manifest['inputs'][name]['path']}\n"  # type: ignore[index]
            for name in missing
        ),
        encoding="utf-8",
    )

    print(
        f"Combined results: {len(available)} inputs present, {len(missing)} missing.",
        flush=True,
    )
    if missing:
        print(f"Missing inputs: {', '.join(missing)}", flush=True)
    print(f"Saved combined results to {output_dir}", flush=True)
    tables.update(
        {
            "phase_transitions": curve_table,
            "raw_fractions": raw,
            "stop_reasons": stop_table,
            "ico_summary": ico_summary,
            "phi_c": phi_c,
            "backfire": backfire,
            "rewiring_by_config": rewiring_by_config,
            "temporal_summary": temporal_summary_table,
            "temporal_effects": temporal_effects,
            "impact": impact_table,
        }
    )
    return tables
