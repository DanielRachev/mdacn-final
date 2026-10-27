"""Figures for the combined results report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_COLORS = {
    "random": "#7f7f7f",
    "degree": "#2a78d6",
    "betweenness": "#eb6834",
    "eigenvector": "#1baf7a",
    "leverage": "#8c6bb1",
    "signed_leverage": "#e87ba4",
    "temporal_bridge": "#b8860b",
}
SOURCE_COLORS = {
    ("nco", "unsigned"): "#2a78d6",
    ("nco", "signed"): "#eb6834",
    ("sign_shuffle", "empirical"): "#1baf7a",
    ("sign_shuffle", "sign_shuffled"): "#8c6bb1",
}


def _strategy_color(strategy: str, index: int) -> str:
    """Return a stable colour for one targeting strategy."""
    return STRATEGY_COLORS.get(str(strategy), f"C{index % 10}")


def _strategy_label(strategy: str) -> str:
    """Return a readable strategy label."""
    return str(strategy).replace("_", " ")


def _save(fig: plt.Figure, path: Path) -> None:
    """Save one figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _closest(values, target: float) -> float | None:
    """Return the value nearest to target, or None when the sequence is empty."""
    values = np.asarray(list(values), dtype=float)
    if not values.size:
        return None
    return float(values[np.argmin(np.abs(values - target))])


def plot_phase_transition(
    curves: pd.DataFrame,
    path: Path,
    *,
    metric: str = "g_minus",
    ylabel: str = "Opposing cluster fraction g_minus",
) -> None:
    """Save NCO and sign-shuffle curves with a realisation band."""
    if curves.empty or f"mean_{metric}" not in curves:
        return
    low, high = f"q025_{metric}", f"q975_{metric}"
    fig, axis = plt.subplots(figsize=(8.5, 5))
    for (source, model), group in curves.groupby(["source", "model"], sort=True):
        group = group.sort_values("f")
        color = SOURCE_COLORS.get((source, model), None)
        label = f"{source.replace('_', ' ')} {model}"
        axis.plot(
            group["f"],
            group[f"mean_{metric}"],
            "o-",
            markersize=3,
            linewidth=1.6,
            color=color,
            label=label,
        )
        if {low, high} <= set(group):
            axis.fill_between(
                group["f"].to_numpy(float),
                group[low].to_numpy(float),
                group[high].to_numpy(float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    axis.set_xlabel("Initial negative fraction f")
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    axis.set_title("Phase transition by source and model")
    _save(fig, path)


def plot_raw_fractions(raw: pd.DataFrame, path: Path) -> None:
    """Save raw negative and positive opinion fractions."""
    if raw.empty or "mean_s1" not in raw:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for axis, prefix, title in zip(
        axes, ("s1", "s2"), ("Negative opinion s1", "Positive opinion s2"), strict=True
    ):
        for (source, model), group in raw.groupby(["source", "model"], sort=True):
            group = group.sort_values("f")
            color = SOURCE_COLORS.get((source, model), None)
            axis.plot(
                group["f"],
                group[f"mean_{prefix}"],
                "o-",
                markersize=3,
                linewidth=1.6,
                color=color,
                label=f"{source.replace('_', ' ')} {model}",
            )
            axis.fill_between(
                group["f"].to_numpy(float),
                group[f"q025_{prefix}"].to_numpy(float),
                group[f"q975_{prefix}"].to_numpy(float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(title)
        axis.set_xlabel("Initial negative fraction f")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Final opinion fraction")
    fig.suptitle("Raw opinion fractions by source and model")
    _save(fig, path)


def plot_ico_curves(summary: pd.DataFrame, path: Path, *, fraction: float = 0.5) -> None:
    """Save opposing cluster size against budget, one panel per model."""
    if summary.empty or "mean_g_minus" not in summary:
        return
    chosen = _closest(summary["f"].unique(), fraction)
    if chosen is None:
        return
    subset = summary.loc[np.isclose(summary["f"].astype(float), chosen)]
    models = sorted(subset["model"].unique())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models, strict=True):
        model_subset = subset.loc[subset["model"] == model]
        for index, strategy in enumerate(sorted(model_subset["strategy"].unique())):
            group = model_subset.loc[model_subset["strategy"] == strategy].sort_values("budget")
            if group.empty:
                continue
            color = _strategy_color(str(strategy), index)
            axis.plot(
                group["budget"],
                group["mean_g_minus"],
                "o-",
                markersize=3,
                linewidth=1.4,
                color=color,
                label=_strategy_label(str(strategy)),
            )
            spread = group["std_g_minus"].fillna(0).to_numpy(float)
            axis.fill_between(
                group["budget"].to_numpy(float),
                np.clip(group["mean_g_minus"].to_numpy(float) - spread, 0, 1),
                np.clip(group["mean_g_minus"].to_numpy(float) + spread, 0, 1),
                color=color,
                alpha=0.08,
                linewidth=0,
            )
        axis.set_title(f"{model.capitalize()} model")
        axis.set_xlabel("Contrarian budget Phi")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Opposing cluster fraction g_minus")
    fig.suptitle(f"ICO targeting at f = {chosen:g}")
    _save(fig, path)


def plot_ico_delta(summary: pd.DataFrame, path: Path, *, fraction: float = 0.5) -> None:
    """Save the paired change in opposing cluster size against budget."""
    if summary.empty or "mean_delta_g_minus" not in summary:
        return
    chosen = _closest(summary["f"].unique(), fraction)
    if chosen is None:
        return
    subset = summary.loc[np.isclose(summary["f"].astype(float), chosen)]
    models = sorted(subset["model"].unique())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models, strict=True):
        model_subset = subset.loc[subset["model"] == model]
        for index, strategy in enumerate(sorted(model_subset["strategy"].unique())):
            group = model_subset.loc[model_subset["strategy"] == strategy].sort_values("budget")
            if group.empty:
                continue
            color = _strategy_color(str(strategy), index)
            axis.plot(
                group["budget"],
                group["mean_delta_g_minus"],
                "o-",
                markersize=3,
                linewidth=1.4,
                color=color,
                label=_strategy_label(str(strategy)),
            )
            spread = group["std_delta_g_minus"].fillna(0).to_numpy(float)
            axis.fill_between(
                group["budget"].to_numpy(float),
                group["mean_delta_g_minus"].to_numpy(float) - spread,
                group["mean_delta_g_minus"].to_numpy(float) + spread,
                color=color,
                alpha=0.08,
                linewidth=0,
            )
        axis.axhline(0, color="0.5", linestyle="--", linewidth=1)
        axis.set_title(f"{model.capitalize()} model")
        axis.set_xlabel("Contrarian budget Phi")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Paired change in g_minus")
    fig.suptitle(f"ICO paired effect at f = {chosen:g}")
    _save(fig, path)


def plot_phi_c(phi_c: pd.DataFrame, path: Path) -> None:
    """Save Phi_c against the initial fraction, one panel per model."""
    if phi_c.empty or "mean_phi_c" not in phi_c:
        return
    models = sorted(phi_c["model"].unique())
    if not models:
        return
    finite = phi_c["mean_phi_c"].dropna()
    upper = max(0.05, float(finite.max()) * 1.2) if len(finite) else 0.1
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models, strict=True):
        model_subset = phi_c.loc[phi_c["model"] == model]
        for index, strategy in enumerate(sorted(model_subset["strategy"].unique())):
            group = model_subset.loc[model_subset["strategy"] == strategy].sort_values("f")
            if group.empty:
                continue
            axis.plot(
                group["f"],
                group["mean_phi_c"],
                "o-",
                markersize=4,
                linewidth=1.4,
                color=_strategy_color(str(strategy), index),
                label=_strategy_label(str(strategy)),
            )
        axis.set_title(f"{model.capitalize()} model")
        axis.set_xlabel("Initial negative fraction f")
        axis.set_ylim(-0.01, upper)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Critical budget Phi_c")
    fig.suptitle("Critical budget by targeting strategy")
    _save(fig, path)


def plot_backfire(summary: pd.DataFrame, path: Path, *, fraction: float = 0.5) -> None:
    """Save backfire and repelled-neighbour shares by strategy."""
    if summary.empty or "backfire_run_fraction" not in summary:
        return
    chosen = _closest(summary["f"].unique(), fraction)
    if chosen is None:
        return
    subset = summary.loc[
        np.isclose(summary["f"].astype(float), chosen) & summary["model"].eq("signed")
    ]
    if subset.empty:
        subset = summary.loc[np.isclose(summary["f"].astype(float), chosen)]
    aggregate = subset.groupby("strategy", as_index=False).agg(
        max_backfire=("backfire_run_fraction", "max"),
        mean_repelled=("mean_repelled_share", "mean"),
    )
    if aggregate.empty:
        return
    positions = np.arange(len(aggregate))
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        positions - 0.18,
        aggregate["max_backfire"],
        0.36,
        color="#eb6834",
        label="Runs with net backfire",
    )
    axis.bar(
        positions + 0.18,
        aggregate["mean_repelled"],
        0.36,
        color="#2a78d6",
        label="Mean repelled-neighbour share",
    )
    axis.set_xticks(positions, [_strategy_label(value) for value in aggregate["strategy"]])
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylabel("Share")
    axis.grid(alpha=0.2, axis="y")
    axis.legend()
    axis.set_title(f"Backfire at f = {chosen:g}, worst tested budget")
    _save(fig, path)


def plot_stop_reasons(stops: pd.DataFrame, path: Path) -> None:
    """Save stop-reason shares by stage and model."""
    if stops.empty or "stop_reason" not in stops:
        return
    stops = stops.copy()
    stops["group"] = stops["source"] + " " + stops["model"]
    pivot = stops.groupby(["group", "stop_reason"])["runs"].sum().unstack(fill_value=0).sort_index()
    if pivot.empty:
        return
    shares = pivot.div(pivot.sum(axis=1), axis=0)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    shares.plot(kind="bar", stacked=True, ax=axis, colormap="tab20", width=0.7)
    axis.set_ylabel("Share of runs")
    axis.set_xlabel("Stage and model")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(alpha=0.2, axis="y")
    axis.legend(title="Stop reason", fontsize=8)
    axis.set_title("Stop reasons by stage and model")
    _save(fig, path)


def plot_rewiring(
    by_config: pd.DataFrame,
    phi_c: pd.DataFrame,
    path: Path,
) -> None:
    """Save rewiring homophily and Phi_c when those tables are available."""
    if not phi_c.empty and {"graph", "strategy", "mean_phi_c"} <= set(phi_c):
        graphs = list(dict.fromkeys(phi_c["graph"]))
        positions = {name: index for index, name in enumerate(graphs)}
        fig, axis = plt.subplots(figsize=(10, 5))
        for index, strategy in enumerate(sorted(phi_c["strategy"].unique())):
            group = phi_c.loc[phi_c["strategy"] == strategy].copy()
            group["position"] = group["graph"].map(positions)
            axis.plot(
                group["position"],
                group["mean_phi_c"],
                "o-",
                markersize=5,
                linewidth=1.4,
                color=_strategy_color(str(strategy), index),
                label=_strategy_label(str(strategy)),
            )
            if {"min_phi_c", "max_phi_c"} <= set(group):
                axis.vlines(
                    group["position"],
                    group["min_phi_c"],
                    group["max_phi_c"],
                    color=_strategy_color(str(strategy), index),
                    alpha=0.25,
                    linewidth=4,
                )
        axis.set_xticks(list(positions.values()), graphs, rotation=20, ha="right")
        axis.set_ylabel("Critical budget Phi_c")
        axis.set_xlabel("Rewired configuration")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        axis.set_title("Phi_c on rewired configurations")
        _save(fig, path)
        return

    if by_config.empty:
        return
    columns = [
        name
        for name in ("mean_h_insular_fraction_of_max", "mean_h_integrated_fraction_of_max")
        if name in by_config
    ]
    if not columns:
        return
    positions = np.arange(len(by_config))
    width = 0.8 / len(columns)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for index, column in enumerate(columns):
        axis.bar(
            positions + (index - (len(columns) - 1) / 2) * width,
            by_config[column],
            width,
            label=column.removeprefix("mean_").replace("_", " "),
        )
    axis.set_xticks(positions, by_config["name"], rotation=20, ha="right")
    axis.set_ylabel("Achieved homophily / bound")
    axis.set_xlabel("Rewired configuration")
    axis.grid(alpha=0.2, axis="y")
    axis.legend(fontsize=8)
    axis.set_title("Achieved rewiring homophily")
    _save(fig, path)


def plot_temporal_effects(effects: pd.DataFrame, path: Path) -> None:
    """Save paired temporal effects with confidence intervals."""
    if effects.empty or not {"f", "metric", "difference"} <= set(effects):
        return
    metrics = list(dict.fromkeys(effects["metric"]))[:6]
    if not metrics:
        return
    columns = 2
    rows = int(np.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.6 * rows), squeeze=False)
    for axis, metric in zip(axes.ravel(), metrics, strict=False):
        subset = effects.loc[effects["metric"] == metric].sort_values("f")
        for index, (comparison, group) in enumerate(
            subset.groupby("comparison", sort=True) if "comparison" in subset else [("", subset)]
        ):
            axis.plot(
                group["f"],
                group["difference"],
                "o-",
                markersize=3,
                linewidth=1.4,
                color=f"C{index}",
                label=str(comparison).replace("_", " "),
            )
            if {"ci_low", "ci_high"} <= set(group):
                axis.fill_between(
                    group["f"].to_numpy(float),
                    group["ci_low"].to_numpy(float),
                    group["ci_high"].to_numpy(float),
                    color=f"C{index}",
                    alpha=0.15,
                    linewidth=0,
                )
        axis.axhline(0, color="0.5", linestyle="--", linewidth=1)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("Initial negative fraction f")
        axis.grid(alpha=0.2)
        if "comparison" in subset:
            axis.legend(fontsize=7)
    for axis in axes.ravel()[len(metrics) :]:
        axis.set_visible(False)
    fig.suptitle("Paired temporal effects")
    _save(fig, path)


def plot_impact_correlations(
    correlations: pd.DataFrame, path: Path, *, fraction: float = 0.5
) -> None:
    """Save predictor correlations with node impact."""
    if correlations.empty or not {"predictor", "coefficient"} <= set(correlations):
        return
    chosen = _closest(correlations["f"].unique(), fraction)
    subset = correlations.loc[np.isclose(correlations["f"].astype(float), chosen)]
    if subset.empty:
        return
    models = sorted(subset["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8), squeeze=False)
    for axis, model in zip(axes.ravel(), models, strict=True):
        model_subset = subset.loc[subset["model"] == model]
        methods = sorted(model_subset["method"].unique()) if "method" in model_subset else [""]
        predictors = sorted(model_subset["predictor"].unique())
        positions = np.arange(len(predictors))
        width = 0.8 / len(methods)
        for index, method in enumerate(methods):
            group = (
                model_subset if not method else model_subset.loc[model_subset["method"] == method]
            )
            group = group.set_index("predictor").reindex(predictors)
            lower = group["coefficient"] - group.get("ci_low", group["coefficient"])
            upper = group.get("ci_high", group["coefficient"]) - group["coefficient"]
            axis.bar(
                positions + (index - (len(methods) - 1) / 2) * width,
                group["coefficient"],
                width,
                yerr=[lower.fillna(0), upper.fillna(0)],
                capsize=3,
                label=method or "all",
            )
        axis.set_xticks(positions, [value.replace("_", " ") for value in predictors], rotation=25)
        axis.axhline(0, color="0.5", linewidth=1)
        axis.set_title(f"{model.capitalize()} model, f = {chosen:g}")
        axis.grid(alpha=0.2, axis="y")
        axis.legend(fontsize=8)
    axes.ravel()[0].set_ylabel("Correlation with node impact")
    fig.suptitle("Node-impact predictors")
    _save(fig, path)
