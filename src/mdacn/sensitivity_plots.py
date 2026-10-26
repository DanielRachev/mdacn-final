"""Figures for the extraction sensitivity comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8c6bb1", "#e87ba4", "#7f7f7f")


def _config_color(index: int) -> str:
    """Return a stable colour for one configuration."""
    return CONFIG_COLORS[index % len(CONFIG_COLORS)]


def _configurations(table: pd.DataFrame) -> list[str]:
    """Return configurations in first-seen order."""
    return [str(value) for value in pd.unique(table["configuration"])]


def _save(fig: plt.Figure, path: Path) -> None:
    """Save one figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_nco_by_configuration(
    summary: pd.DataFrame, path: Path, *, metric: str = "mean_g_minus"
) -> None:
    """Save mean NCO outcomes by core extraction, one panel per model.

    The shaded band is the 2.5 to 97.5 percentile range across realisations.
    """
    if summary.empty or metric not in summary:
        return
    suffix = metric.removeprefix("mean_")
    low, high = f"q025_{suffix}", f"q975_{suffix}"
    models = sorted(summary["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    configurations = _configurations(summary)
    for axis, model in zip(axes, models, strict=True):
        subset = summary.loc[summary["model"] == model]
        for index, configuration in enumerate(configurations):
            group = subset.loc[subset["configuration"] == configuration].sort_values("f")
            if group.empty:
                continue
            color = _config_color(index)
            axis.plot(
                group["f"],
                group[metric],
                "o-",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=configuration,
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
        axis.set_title(f"{model.capitalize()} model")
        axis.set_xlabel("Initial negative fraction f")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    label = (
        "Opposing cluster fraction g_minus"
        if suffix == "g_minus"
        else ("Positive cluster fraction g_plus")
    )
    axes[0].set_ylabel(label)
    fig.suptitle("NCO outcome by core extraction")
    _save(fig, path)


def plot_nco_raw_fraction(summary: pd.DataFrame, path: Path) -> None:
    """Save the raw negative fraction against f for every core extraction."""
    if summary.empty or "mean_final_negative_fraction" not in summary:
        return
    models = sorted(summary["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    configurations = _configurations(summary)
    for axis, model in zip(axes, models, strict=True):
        subset = summary.loc[summary["model"] == model]
        for index, configuration in enumerate(configurations):
            group = subset.loc[subset["configuration"] == configuration].sort_values("f")
            if group.empty:
                continue
            color = _config_color(index)
            axis.plot(
                group["f"],
                group["mean_final_negative_fraction"],
                "o-",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=configuration,
            )
            axis.fill_between(
                group["f"].to_numpy(float),
                group["q025_final_negative_fraction"].to_numpy(float),
                group["q975_final_negative_fraction"].to_numpy(float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(f"{model.capitalize()} model")
        axis.set_xlabel("Initial negative fraction f")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Final negative fraction s1")
    fig.suptitle("Raw negative fraction by core extraction")
    _save(fig, path)


def plot_ico_by_configuration(
    summary: pd.DataFrame,
    path: Path,
    *,
    strategy: str = "degree",
    fraction: float = 0.5,
) -> None:
    """Save mean opposing cluster size against budget, one panel per model."""
    if summary.empty or "strategy" not in summary:
        return
    subset = summary.loc[summary["strategy"] == strategy]
    if subset.empty:
        return
    fractions = np.sort(subset["f"].unique())
    chosen = fractions[np.argmin(np.abs(fractions - fraction))]
    subset = subset.loc[np.isclose(subset["f"].astype(float), float(chosen))]
    models = sorted(subset["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    configurations = _configurations(subset)
    for axis, model in zip(axes, models, strict=True):
        model_subset = subset.loc[subset["model"] == model]
        for index, configuration in enumerate(configurations):
            group = model_subset.loc[model_subset["configuration"] == configuration].sort_values(
                "budget"
            )
            if group.empty:
                continue
            color = _config_color(index)
            axis.plot(
                group["budget"],
                group["mean_g_minus"],
                "o-",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=configuration,
            )
            spread = group["std_g_minus"].fillna(0).to_numpy(float)
            axis.fill_between(
                group["budget"].to_numpy(float),
                np.clip(group["mean_g_minus"].to_numpy(float) - spread, 0, 1),
                np.clip(group["mean_g_minus"].to_numpy(float) + spread, 0, 1),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(f"{model.capitalize()} model")
        axis.set_xlabel("Contrarian budget Phi")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Opposing cluster fraction g_minus")
    fig.suptitle(f"ICO {strategy} targeting at f = {chosen:g}")
    _save(fig, path)


def plot_phi_c_by_configuration(phi_c: pd.DataFrame, path: Path, *, fraction: float = 0.5) -> None:
    """Save Phi_c against core extraction, one panel per model and line per strategy."""
    if phi_c.empty or "mean_phi_c" not in phi_c:
        return
    fractions = np.sort(phi_c["f"].unique())
    chosen = fractions[np.argmin(np.abs(fractions - fraction))]
    subset = phi_c.loc[np.isclose(phi_c["f"].astype(float), float(chosen))]
    models = sorted(subset["model"].unique())
    configurations = _configurations(subset)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    positions = np.arange(len(configurations))
    for axis, model in zip(axes, models, strict=True):
        model_subset = subset.loc[subset["model"] == model]
        strategies = sorted(model_subset["strategy"].unique())
        for index, strategy in enumerate(strategies):
            group = (
                model_subset.loc[model_subset["strategy"] == strategy]
                .set_index("configuration")
                .reindex(configurations)
            )
            axis.plot(
                positions,
                group["mean_phi_c"].to_numpy(float),
                "o-",
                markersize=4,
                linewidth=1.5,
                color=_config_color(index),
                label=strategy,
            )
        axis.set_xticks(positions, configurations, rotation=15, ha="right")
        axis.set_title(f"{model.capitalize()} model")
        axis.set_xlabel("Core extraction")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Critical budget Phi_c")
    fig.suptitle(f"Phi_c by core extraction at f = {chosen:g}")
    _save(fig, path)


def plot_topology_sign_fractions(topology: pd.DataFrame, path: Path) -> None:
    """Save negative and zero edge shares for every core extraction."""
    columns = {"negative_edge_fraction", "zero_edge_fraction"}
    if topology.empty or not columns <= set(topology):
        return
    configurations = [str(value) for value in topology["configuration"]]
    positions = np.arange(len(configurations))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        positions - width / 2,
        topology["negative_edge_fraction"],
        width,
        color="#eb6834",
        label="Negative edges",
    )
    axis.bar(
        positions + width / 2,
        topology["zero_edge_fraction"],
        width,
        color="#8c6bb1",
        label="Zero-sign edges",
    )
    axis.set_xticks(positions, configurations, rotation=15, ha="right")
    axis.set_ylabel("Share of undirected edges")
    axis.set_xlabel("Core extraction")
    axis.grid(alpha=0.2, axis="y")
    axis.legend()
    axis.set_title("Sign composition by core extraction")
    _save(fig, path)
