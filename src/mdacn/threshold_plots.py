"""Threshold figures that show censoring and the no-intervention reference."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_threshold_summary(
    table: pd.DataFrame,
    path: Path,
    *,
    x: str = "f",
    facet: str | None = "model",
    title: str = "Paired intervention thresholds",
) -> None:
    """Place conditional means beside success rates; show the zero-budget share."""
    required = {x, "strategy", "mean_phi_c", "runs", "reached_fraction", "zero_budget_fraction"}
    if missing := required - set(table):
        raise ValueError(f"Threshold plots require success counts: {sorted(missing)}")
    panels = sorted(table[facet].unique()) if facet else [None]
    categorical = not pd.api.types.is_numeric_dtype(table[x])
    labels = list(dict.fromkeys(table[x])) if categorical else sorted(table[x].unique())
    positions = np.arange(len(labels)) if categorical else np.asarray(labels)
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(max(9, 6 * len(panels)), 7.4),
        squeeze=False,
        sharey="row",
        layout="constrained",
    )
    for column, panel in enumerate(panels):
        subset = table[table[facet].eq(panel)] if facet else table
        for index, strategy in enumerate(sorted(subset.strategy.unique())):
            group = subset[subset.strategy.eq(strategy)].set_index(x).reindex(labels)
            label = str(strategy).replace("_", " ")
            for row, metric in enumerate(["mean_phi_c", "reached_fraction"]):
                axes[row, column].plot(
                    positions,
                    group[metric],
                    "o-",
                    markersize=4,
                    color=f"C{index % 10}",
                    label=label,
                )
        reference = subset.drop_duplicates(x).set_index(x).reindex(labels)
        axes[1, column].plot(
            positions,
            reference.zero_budget_fraction,
            "k--",
            linewidth=1.5,
            label="already below at budget 0",
        )
        for pos, count in zip(positions, reference.runs, strict=True):
            axes[1, column].annotate(
                f"n={int(count)}",
                (pos, 0),
                xytext=(0, -22),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        for row in range(2):
            axis = axes[row, column]
            axis.grid(alpha=0.2)
            axis.set_xticks(
                positions,
                labels,
                rotation=20 if categorical else 0,
                ha="right" if categorical else "center",
            )
        axes[0, column].set_title(f"{str(panel).capitalize()} model" if facet else "Thresholds")
        axes[0, column].set_ylim(bottom=-0.005)
        axes[1, column].set_ylim(-0.03, 1.08)
        axes[1, column].set_xlabel(
            "Initial negative fraction" if x == "f" else x.replace("_", " "), labelpad=23
        )
        axes[0, column].legend(fontsize=7, ncol=2)
        axes[1, column].legend(fontsize=7, ncol=2)
    axes[0, 0].set_ylabel("Mean tested threshold among reached runs")
    axes[1, 0].set_ylabel("Fraction of all runs reaching threshold")
    fig.suptitle(
        title + "\nMeans exclude unreached runs; zero-budget successes are shown separately",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
