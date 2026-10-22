"""Compact tables and figures for the extraction comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _label(configuration: str) -> str:
    return configuration.replace("k_core_", "k = ").replace("activity_lcc_", "Activity >= ")


def plot_summary_table(comparison: pd.DataFrame, path: Path) -> None:
    """Render selected metrics with units and a common set of columns."""
    metrics = [
        ("nodes", "Nodes", ",.0f"),
        ("edges", "Edges", ",.0f"),
        ("node_retention", "Nodes retained", ".1%"),
        ("edge_retention", "Edges retained", ".1%"),
        ("event_retention", "Events retained", ".1%"),
        ("average_degree", "Mean degree", ".2f"),
        ("negative_edge_fraction", "Negative edges", ".2%"),
        ("zero_edge_fraction", "Zero-sign edges", ".2%"),
        ("degree_assortativity", "Degree assortativity", ".3f"),
        ("community_count", "Louvain communities", ".0f"),
        ("modularity", "Modularity", ".3f"),
        ("average_clustering", "Mean clustering", ".3f"),
        ("spectral_radius", "Spectral radius", ".3f"),
        ("algebraic_connectivity", "Algebraic connectivity", ".3f"),
    ]
    cells = [
        [name] + ["--" if pd.isna(value) else format(value, spec) for value in comparison[key]]
        for key, name, spec in metrics
    ]
    fig, axis = plt.subplots(figsize=(max(9, 2 + len(comparison) * 1.8), 7.5))
    axis.set_axis_off()
    table = axis.table(
        cellText=cells,
        colLabels=["Metric"] + [_label(value) for value in comparison.configuration],
        cellLoc="center",
        bbox=(0, 0.07, 1, 0.85),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(range(len(comparison) + 1))
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#dce6ef" if row == 0 else ("#f0f3f6" if row % 2 else "white"))
        if row == 0 or column == 0:
            cell.set_text_props(weight="bold")
    axis.set_title("Core extraction: topology comparison", fontsize=15, pad=12)
    fig.text(
        0.5,
        0.035,
        "Retention uses the full body aggregate; events exclude self-links.\n"
        "Unsigned topology; Louvain seed recorded in metadata. -- means undefined.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_extraction_degrees(distributions: pd.DataFrame, path: Path) -> None:
    """Overlay distributions normalized within each graph, without fitted curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for label, distribution in distributions.groupby("configuration", sort=False):
        for index, axis in enumerate(axes):
            visible = distribution if index else distribution.loc[distribution.degree > 0]
            axis.scatter(
                visible.degree,
                visible.probability,
                s=12,
                alpha=0.7,
                label="Full aggregate" if label == "aggregate" else _label(label),
            )
    axes[0].set_xscale("log")
    if not (distributions.degree > 0).any():
        axes[0].set_xlim(0.5, 1.5)
        axes[0].text(0.5, 0.5, "No positive degrees", transform=axes[0].transAxes, ha="center")
    for axis, title in zip(axes, ("Log-log", "Log-linear"), strict=True):
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Degree within each graph")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    axes[0].set_ylabel("P(k), normalized within each graph")
    fig.suptitle("Full aggregate versus extracted cores")
    fig.text(
        0.5,
        0.015,
        "Degree zero is omitted only on the log-log axes; no scale-free claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_extraction_retention(comparison: pd.DataFrame, path: Path) -> None:
    """Show retained structure and aggregate signs on separate percentage scales."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(comparison))
    for offset, key, label in (
        (-0.25, "node_retention", "Nodes"),
        (0, "edge_retention", "Edges"),
        (0.25, "event_retention", "Events"),
    ):
        values = pd.to_numeric(comparison[key]).to_numpy(dtype=float) * 100
        axes[0].bar(x + offset, values, width=0.24, label=label)
    bottom = np.zeros(len(comparison))
    for key, label, color in (
        ("positive_edge_fraction", "Positive", "#4389b2"),
        ("negative_edge_fraction", "Negative", "#c75959"),
        ("zero_edge_fraction", "Zero sign", "#929da5"),
    ):
        values = pd.to_numeric(comparison[key]).fillna(0).to_numpy(dtype=float) * 100
        axes[1].bar(x, values, bottom=bottom, label=label, color=color, width=0.65)
        bottom += values
    for axis, title in zip(
        axes, ("Retained share of full aggregate", "Signs within each core"), strict=True
    ):
        axis.set_xticks(x, [_label(value) for value in comparison.configuration], fontsize=9)
        axis.set_ylabel("Percent")
        axis.set_ylim(0, 105)
        axis.set_title(title)
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        0.015,
        "Edges use majority event sign; ties remain in unsigned topology. "
        "Empty cores have no sign distribution.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)
