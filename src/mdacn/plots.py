"""Figure interfaces. Each function saves a figure to the supplied path."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mdacn.characterisation import degree_distribution, fitted_degree_probabilities


def plot_degree_distribution(distribution: pd.DataFrame, fits: pd.DataFrame, path: Path) -> None:
    """Owner: Jacek. Save empirical and fitted degree curves on log-log and log-linear axes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = distribution.loc[distribution["degree"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis in axes:
        axis.scatter(positive["degree"], positive["probability"], s=18, label="Empirical")
        axis.set_yscale("log")
        axis.set_xlabel("Degree k")
        axis.set_ylabel("P(k)")
        axis.grid(alpha=0.2)
    axes[0].set_xscale("log")
    axes[0].set_title("Log-log")
    axes[1].set_title("Log-linear")

    if not positive.empty:
        for axis in axes:
            axis.set_ylim(positive["probability"].min() / 5, 1)
        degrees = np.arange(1, int(positive["degree"].max()) + 1, dtype=np.int64)
        for fit in fits.itertuples(index=False):
            if fit.status != "ok":
                continue
            probabilities = fitted_degree_probabilities(
                degrees, model=fit.model, parameter=fit.parameter_value, k_min=fit.k_min
            )
            sample_fraction = fit.sample_size / distribution["count"].sum()
            probabilities *= sample_fraction
            visible = probabilities > 0
            name = "Power law" if fit.model == "power_law" else "Poisson"
            label = f"{name} ({fit.parameter}={fit.parameter_value:.2f})"

            for axis in axes:
                axis.plot(degrees[visible], probabilities[visible], label=label)
    for axis in axes:
        axis.legend()

    fig.suptitle("Core degree distribution and discrete baseline fits")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _core_label(core: nx.Graph) -> str:
    if core.graph.get("core_method") == "k_core":
        return f"Connected {core.graph['k']}-core"
    return "Activity-selected core"


def plot_degree_comparison(graph: nx.Graph, core: nx.Graph, path: Path) -> None:
    """Owner: Jacek. Compare degrees within the aggregate and the extracted core."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for network, label, color in (
        (graph, "Full aggregate", "tab:blue"),
        (core, _core_label(core), "tab:orange"),
    ):
        distribution = degree_distribution(network)
        for index, axis in enumerate(axes):
            visible = distribution if index == 1 else distribution.loc[distribution["degree"] > 0]
            axis.scatter(
                visible["degree"],
                visible["probability"],
                s=18,
                alpha=0.75,
                color=color,
                label=f"{label} (N={len(network):,})",
            )
    axes[0].set_xscale("log")
    for axis, title in zip(axes, ("Log-log", "Log-linear"), strict=True):
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Degree k within each graph")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("P(k), normalized within each graph")
    fig.suptitle("Full aggregate versus extracted core")
    node_retention = len(core) / len(graph)
    edge_retention = (
        core.number_of_edges() / graph.number_of_edges() if graph.number_of_edges() else 0
    )
    fig.text(
        0.5,
        0.02,
        f"Core retains {node_retention:.1%} of nodes and {edge_retention:.1%} of edges. "
        "Core degrees count retained neighbors.\nDegree zero is omitted on log-log axes.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_core_communities(core: nx.Graph, positions: dict[str, np.ndarray], path: Path) -> None:
    """Owner: Jacek. Show every core node and edge, coloured by saved Louvain labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 8))
    if core.number_of_edges():
        edges = nx.draw_networkx_edges(
            core, positions, ax=axis, width=0.3, alpha=0.035, edge_color="slategray"
        )
        edges.set_rasterized(True)
    communities = sorted({core.nodes[node]["community"] for node in core})
    palette = plt.get_cmap("tab10" if len(communities) <= 10 else "tab20")
    for index, community in enumerate(communities):
        members = [node for node in core if core.nodes[node]["community"] == community]
        nx.draw_networkx_nodes(
            core,
            positions,
            nodelist=members,
            ax=axis,
            node_size=12,
            linewidths=0,
            node_color=[palette(index % palette.N)],
            label=f"Community {community + 1}: {len(members):,}",
        )
    axis.set_title(
        f"{_core_label(core)}: Louvain communities\n"
        f"All {len(core):,} nodes and {core.number_of_edges():,} unsigned edges"
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, markerscale=2)
    axis.set_axis_off()
    fig.text(
        0.5,
        0.02,
        "Seeded unsigned layout; colors show structural communities, not opinions.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_phase_transition(results: pd.DataFrame, path: Path) -> None:
    """Owner: Daniel. Save unsigned versus signed steady negative fraction s_1 against f.

    Show variability across realizations. Identify cycles and update-limit runs
    separately from steady-state results.
    """
    if results.empty:
        raise ValueError("No sweep results to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    statuses = (
        ("steady_state", "Steady states"),
        ("limit_cycle", "Limit cycles: terminal phase"),
        ("max_steps", "Update limit: terminal state"),
    )
    statuses = [
        (status, title)
        for status, title in statuses
        if status == "steady_state" or (results["stop_reason"] == status).any()
    ]
    width = 7 if len(statuses) == 1 else 5 * len(statuses)
    fig, axes = plt.subplots(1, len(statuses), figsize=(width, 5.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lower, upper = 0.0, 1.0
    x_values = sorted(results["f"].unique())

    for axis, (status, title) in zip(axes, statuses, strict=True):
        subset = results.loc[results["stop_reason"] == status]
        for model, color in (("unsigned", "tab:blue"), ("signed", "tab:orange")):
            runs = subset.loc[subset["model"] == model]
            if runs.empty:
                continue
            summary = runs.groupby("f")["final_negative_fraction"].agg(["mean", "std"])
            spread = summary["std"].fillna(0)
            lower = min(lower, float((summary["mean"] - spread).min()))
            upper = max(upper, float((summary["mean"] + spread).max()))
            summary = summary.reindex(x_values)
            axis.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"].fillna(0),
                fmt="o-" if status == "steady_state" else "o--",
                color=color,
                capsize=3,
                label=f"{model.capitalize()} (n={len(runs)})",
            )
        if len(statuses) > 1:
            axis.set_title(title)
        axis.set_xlabel("Initial negative fraction f")
        xmin, xmax = min(x_values), max(x_values)
        if xmin == xmax:
            pad = 0.05 if xmin == 0 else max(0.05, abs(xmin) * 0.05)
            axis.set_xlim(xmin - pad, xmax + pad)
        else:
            axis.set_xlim(xmin, xmax)
        axis.grid(alpha=0.2)
        if subset.empty:
            axis.text(0.5, 0.5, "No runs", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.legend()

    axes[0].set_ylim(lower - 0.05, upper + 0.05)
    axes[0].set_ylabel("Final negative fraction")
    fig.suptitle("Unsigned versus signed NCO")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_opinion_gcc(
    results: pd.DataFrame, path: Path, *, opinion: Literal["minus", "plus"] = "minus"
) -> None:
    """Owner: Jacek. Show individual outcomes and fragmentation for one opinion.

    Rows show the largest component / N, the second-largest component / N,
    and nodes outside the largest component. Each row has its own scale.
    Models and terminal-state categories have separate columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    statuses = (
        ("steady_state", "Steady states"),
        ("limit_cycle", "Limit cycles: terminal phase"),
        ("max_steps", "Update limit: terminal state"),
    )
    statuses = [
        (status, title)
        for status, title in statuses
        if status == "steady_state" or (results["stop_reason"] == status).any()
    ]
    colors = {
        "unsigned": "tab:blue",
        "signed": "tab:orange",
        "empirical": "tab:orange",
        "sign_shuffled": "tab:green",
    }
    models = [model for model in colors if (results["model"] == model).any()]
    columns = [(status, title, model) for status, title in statuses for model in models]
    largest = results[f"g_{opinion}"]
    raw_fraction = results["final_negative_fraction"]
    if opinion == "plus":
        raw_fraction = 1 - raw_fraction
    measurements = results.assign(
        largest=largest,
        second=results[f"g_{opinion}_second"],
        outside=np.maximum(0, np.rint((raw_fraction - largest) * results["n_nodes"])),
    )
    fig, axes = plt.subplots(
        3,
        len(columns),
        figsize=(5.4 * len(columns), 9.5),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    fractions = sorted(results["f"].unique())
    jitter_width = min(np.diff(fractions), default=0.1) * 0.18
    for column, (status, title, model) in enumerate(columns):
        runs = measurements.loc[
            (measurements["stop_reason"] == status) & (measurements["model"] == model)
        ]
        jitter = np.random.default_rng(42).uniform(-jitter_width, jitter_width, len(runs))
        color = colors[model]
        for row, metric in enumerate(("largest", "second", "outside")):
            axis = axes[row, column]
            if not runs.empty:
                axis.scatter(
                    runs["f"] + jitter,
                    runs[metric],
                    s=22,
                    color=color,
                    alpha=0.5,
                    edgecolors="none",
                )
                summary = runs.groupby("f")[metric].mean().reindex(fractions)
                axis.plot(
                    summary.index,
                    summary,
                    color="black",
                    marker="D",
                    markersize=4,
                    linewidth=1.2,
                    linestyle="-" if status == "steady_state" else "--",
                )
            else:
                axis.text(0.5, 0.5, "No runs", ha="center", va="center", transform=axis.transAxes)
            axis.set_xlim(0, 1)
            axis.grid(alpha=0.2)
            label = model.replace("_", " ").capitalize()
            axes[0, column].set_title(f"{label} NCO\n{title} (n={len(runs)})")
            axes[2, column].set_xlabel("Initial negative fraction f")

        axes[0, 0].set_ylabel("Largest component\nfraction of original N")
        axes[1, 0].set_ylabel("Second-largest component\nfraction of original N")
        axes[2, 0].set_ylabel("Opinion nodes outside\nlargest component")
        axes[0, 0].set_ylim(-0.03, 1.03)
        min_fraction = 1 / results["n_nodes"].iloc[0]
        axes[1, 0].set_ylim(0, max(min_fraction, measurements["second"].max()) * 1.1)
        axes[2, 0].set_ylim(-0.5, max(1, measurements["outside"].max()) * 1.1)
        fig.legend(
            handles=[
                Line2D([], [], color="gray", marker="o", linestyle="none", label="Individual run"),
                Line2D([], [], color="black", marker="D", markersize=4, label="Mean"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=2,
            frameon=False,
        )
        name = "negative" if opinion == "minus" else "positive"
        fig.suptitle(f"Surviving {name} opinion components", fontsize=15, y=0.99)
        fig.text(
            0.5,
            0.02,
            f"Fixed unsigned core; N = {results['n_nodes'].iloc[0]:,}. "
            "Points are jittered horizontally for visibility.",
            ha="center",
            fontsize=9,
        )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.08, top=0.86, hspace=0.18, wspace=0.09)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sign_shuffle_phase_transition(results: pd.DataFrame, path: Path) -> None:
    """Owner: Polly. Save empirical signed versus sign-shuffled phase transitions."""
    if results.empty:
        raise ValueError("No sweep results to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    statuses = (
        ("steady_state", "Steady states"),
        ("limit_cycle", "Limit cycles: terminal phase"),
        ("max_steps", "Update limit: terminal state"),
    )
    statuses = [
        (status, title)
        for status, title in statuses
        if status == "steady_state" or (results["stop_reason"] == status).any()
    ]
    width = 7 if len(statuses) == 1 else 5 * len(statuses)
    fig, axes = plt.subplots(1, len(statuses), figsize=(width, 5.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lower, upper = 0.0, 1.0
    x_values = sorted(results["f"].unique())

    for axis, (status, title) in zip(axes, statuses, strict=True):
        subset = results.loc[results["stop_reason"] == status]
        for model, color in (("empirical", "tab:blue"), ("sign_shuffled", "tab:orange")):
            runs = subset.loc[subset["model"] == model]
            if runs.empty:
                continue
            summary = runs.groupby("f")["final_negative_fraction"].agg(["mean", "std"])
            spread = summary["std"].fillna(0)
            lower = min(lower, float((summary["mean"] - spread).min()))
            upper = max(upper, float((summary["mean"] + spread).max()))
            summary = summary.reindex(x_values)
            axis.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"].fillna(0),
                fmt="o-" if status == "steady_state" else "o--",
                color=color,
                capsize=3,
                label=f"{model.replace('_', ' ').title()} (n={len(runs)})",
            )
        if len(statuses) > 1:
            axis.set_title(title)
        axis.set_xlabel("Initial minority fraction f")
        xmin, xmax = min(x_values), max(x_values)
        if xmin == xmax:
            pad = 0.05 if xmin == 0 else max(0.05, abs(xmin) * 0.05)
            axis.set_xlim(xmin - pad, xmax + pad)
        else:
            axis.set_xlim(xmin, xmax)
        axis.grid(alpha=0.2)
        if subset.empty:
            axis.text(0.5, 0.5, "No runs", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.legend()

    axes[0].set_ylim(lower - 0.05, upper + 0.05)
    axes[0].set_ylabel("Final negative fraction s₁")
    fig.suptitle("Empirical signed network versus sign-shuffled null model")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_intervention_comparison(
    results: pd.DataFrame, path: Path, *, opinion: Literal["minus", "plus"] = "minus"
) -> None:
    """Owner: Polly. Save opposing cluster fraction S_1/N against contrarian budget Phi.

    Compare random, degree, and leverage targeting with realisation variability.
    Identify cycles and update-limit runs separately from steady-state results.
    Set opinion="plus" to show the positive GCC instead. The default also
    accepts older tables containing only cluster_size and n_nodes.
    """
    if results.empty:
        raise ValueError("No intervention results to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    if opinion == "minus":
        cluster_fraction = results["cluster_size"] / results["n_nodes"]
    elif opinion == "plus":
        cluster_fraction = results["g_plus"]
    else:
        raise ValueError("opinion must be 'minus' or 'plus'")
    results = results.assign(cluster_fraction=cluster_fraction)
    statuses = (
        ("steady_state", "Steady states"),
        ("limit_cycle", "Limit cycles: terminal phase"),
        ("max_steps", "Update limit: terminal state"),
    )
    statuses = [
        (status, title)
        for status, title in statuses
        if status == "steady_state" or (results["stop_reason"] == status).any()
    ]
    width = 7 if len(statuses) == 1 else 5 * len(statuses)
    fig, axes = plt.subplots(1, len(statuses), figsize=(width, 5.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    budgets = sorted(results["budget"].unique())
    lower, upper = 0.0, 1.0
    for axis, (status, title) in zip(axes, statuses, strict=True):
        subset = results.loc[results["stop_reason"] == status]
        for strategy, color in (
            ("random", "tab:blue"),
            ("degree", "tab:orange"),
            ("leverage", "tab:green"),
        ):
            runs = subset.loc[subset["strategy"] == strategy]
            if runs.empty:
                continue
            summary = runs.groupby("budget")["cluster_fraction"].agg(["mean", "std"])
            spread = summary["std"].fillna(0)
            lower = min(lower, float((summary["mean"] - spread).min()))
            upper = max(upper, float((summary["mean"] + spread).max()))
            # Do not connect across budgets with no runs in this status.
            summary = summary.reindex(budgets)
            axis.errorbar(
                summary.index,
                summary["mean"],
                yerr=summary["std"].fillna(0),
                fmt="o-" if status == "steady_state" else "o--",
                color=color,
                capsize=3,
                label=strategy.capitalize(),
            )
        axis.set_title(title)
        axis.set_xlabel("Contrarian budget Phi")
        axis.set_xlim(0, 1)
        axis.grid(alpha=0.2)
        if subset.empty:
            axis.text(0.5, 0.5, "No runs", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.legend()
    axes[0].set_ylabel(
        "Opposing cluster fraction S₁/N" if opinion == "minus" else "Positive GCC fraction g₊"
    )
    axes[0].set_ylim(lower - 0.05, upper + 0.05)
    cycles = int((results["stop_reason"] == "limit_cycle").sum())
    limited = int((results["stop_reason"] == "max_steps").sum())
    fig.suptitle(
        "Intervention comparison\n"
        f"Mean ± sample standard deviation; cycles: {cycles}, update-limit runs: {limited}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
