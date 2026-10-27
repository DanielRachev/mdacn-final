"""Figures drawn from saved node-impact statistics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mdacn.features import PREDICTORS

COLORS = {"unsigned": "#0072B2", "signed": "#D55E00"}
MARKERS = {"unsigned": "o", "signed": "s"}


def save_impact_components(summary: pd.DataFrame, output: Path, *, n_nodes: int) -> None:
    """Show direct GCC changes beside propagation beyond the pinned node."""
    for fraction, group in summary.groupby("f"):
        group = group.set_index("model").reindex([m for m in COLORS if m in set(group.model)])
        positions = np.arange(len(group))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
        for index, (column, label) in enumerate(
            [
                ("mean_total_impact", "total"),
                ("mean_direct_impact", "direct pinning / fragmentation"),
                ("mean_spillover_impact", "subsequent propagation"),
            ]
        ):
            values = group[column].to_numpy() * n_nodes
            bars = axes[0].bar(positions + (index - 1) * 0.23, values, 0.23, label=label)
            axes[0].bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
        axes[0].set_xticks(positions, group.index)
        axes[0].set_ylabel("Mean reduction in negative GCC nodes")
        axes[0].legend(fontsize=8)
        rates = 100 * group.propagation_run_fraction.to_numpy()
        bars = axes[1].bar(positions, rates, color=[COLORS[m] for m in group.index], width=0.5)
        axes[1].bar_label(bars, labels=[f"{r:.2f}%" for r in rates], padding=3)
        axes[1].set_xticks(positions, group.index)
        axes[1].set_ylabel("Trials with an unpinned opinion change (%)")
        axes[1].set_ylim(0, max(1, float(rates.max()) * 1.35))
        for axis in axes:
            axis.grid(axis="y", alpha=0.15)
        fig.suptitle(f"Single-node intervention: direct effect versus propagation (f={fraction:g})")
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / f"impact_components_f{fraction:g}.png", dpi=180)
        plt.close(fig)


def _save_distribution(summary: pd.DataFrame, fraction: float, output: Path) -> None:
    """Plot the distribution of node means, keeping every available node visible."""
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    models = [model for model in COLORS if model in summary.model.unique()]
    rng = np.random.default_rng(0)  # Stable display jitter; no simulation randomness.
    labels = []
    for position, model in enumerate(models):
        group = summary.loc[summary.model == model].sort_values("node")
        values = group.common_mean_impact.to_numpy(float)
        values = values[np.isfinite(values)]
        runs = int(group.common_runs.iloc[0])
        labels.append(f"{model.capitalize()}\n{len(values)} nodes, {runs} steady runs")
        if len(values):
            ax.boxplot(
                values,
                positions=[position],
                widths=0.4,
                patch_artist=True,
                showfliers=False,  # All nodes, including outliers, are drawn below.
                boxprops={"facecolor": COLORS[model], "alpha": 0.2},
                medianprops={"color": COLORS[model], "linewidth": 2},
            )
            ax.scatter(
                position + rng.uniform(-0.14, 0.14, len(values)),
                values,
                color=COLORS[model],
                marker=MARKERS[model],
                s=40,
                alpha=0.65,
                zorder=3,
            )
        else:
            ax.text(
                position,
                0.5,
                "No common steady outcomes",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=14,
            )
    ax.axhline(0, color="0.35", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(models)), labels, fontsize=16)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel("Mean node impact $I_i$", fontsize=18)
    ax.set_title(f"Single-node intervention impact (f={fraction:g})", fontsize=22, pad=18)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(pad=2)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def _save_correlations(correlations: pd.DataFrame, fraction: float, output: Path) -> None:
    """Draw stored coefficients and interval bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), sharey=True)
    models = [model for model in COLORS if model in correlations.model.unique()]
    positions = np.arange(len(PREDICTORS))
    offsets = np.linspace(-0.13, 0.13, len(models)) if len(models) > 1 else [0]
    for ax, method in zip(axes, ("pearson", "spearman"), strict=True):
        for model, offset in zip(models, offsets, strict=True):
            group = (
                correlations.loc[(correlations.model == model) & (correlations.method == method)]
                .set_index("predictor")
                .reindex(PREDICTORS)
            )
            estimates = group.coefficient.to_numpy(float)
            low, high = group.ci_low.to_numpy(float), group.ci_high.to_numpy(float)
            y = positions + offset
            intervals = np.isfinite(low) & np.isfinite(high)
            ax.hlines(y[intervals], low[intervals], high[intervals], color=COLORS[model], lw=2)
            ax.vlines(low[intervals], y[intervals] - 0.04, y[intervals] + 0.04, color=COLORS[model])
            ax.vlines(
                high[intervals], y[intervals] - 0.04, y[intervals] + 0.04, color=COLORS[model]
            )
            valid = np.isfinite(estimates)
            ax.scatter(
                estimates[valid],
                y[valid],
                color=COLORS[model],
                marker=MARKERS[model],
                s=60,
                zorder=3,
            )
            for missing_y in y[~valid]:
                ax.text(
                    0.98,
                    missing_y,
                    "n/a",
                    color=COLORS[model],
                    fontsize=12,
                    ha="right",
                    va="center",
                    transform=ax.get_yaxis_transform(),
                )
        ax.axvline(0, color="0.35", linewidth=1, linestyle="--")
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_ylim(len(PREDICTORS) - 0.5, -0.5)
        ax.set_yticks(positions, [name.replace("_", " ").capitalize() for name in PREDICTORS])
        ax.set_title(method.capitalize(), fontsize=18, pad=12)
        ax.set_xlabel("Correlation coefficient", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(axis="x", alpha=0.15)
    legend = []
    for model in models:
        runs = int(correlations.loc[correlations.model == model, "common_runs"].iloc[0])
        legend.append(
            Line2D(
                [],
                [],
                color=COLORS[model],
                marker=MARKERS[model],
                linewidth=2,
                label=f"{model.capitalize()} ({runs} steady runs)",
            )
        )
    fig.suptitle(
        f"Impact correlations (95% bootstrap intervals, f={fraction:g})", fontsize=22, y=0.97
    )
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.59, 0.91),
        ncol=len(models),
        frameon=False,
        fontsize=13,
    )
    fig.subplots_adjust(left=0.23, right=0.97, bottom=0.13, top=0.75, wspace=0.12)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def save_impact_plots(summary: pd.DataFrame, correlations: pd.DataFrame, output_dir: Path) -> None:
    """Save impact figures with two PNGs per initial fraction."""
    for pattern in (
        "impact_distribution_f*.png",
        "impact_correlations_f*.png",
        "impact_unsigned_f*.png",
        "impact_unsigned_f*.pdf",
        "impact_signed_f*.png",
        "impact_signed_f*.pdf",
    ):
        for path in output_dir.glob(pattern):
            path.unlink()
    for fraction, group in summary.groupby("f", sort=True):
        _save_distribution(group, fraction, output_dir / f"impact_distribution_f{fraction:g}.png")
        _save_correlations(
            correlations.loc[correlations.f == fraction],
            fraction,
            output_dir / f"impact_correlations_f{fraction:g}.png",
        )
