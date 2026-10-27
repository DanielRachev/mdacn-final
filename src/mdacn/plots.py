"""Figure interfaces. Each function saves a figure to the supplied path."""

import warnings
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


def _ensemble_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean and central 95% interval over realizations (axis 0), NaN-aware.

    A column without finite values (e.g. a wedge type absent from every null run)
    gives NaN, which is not drawn.
    """
    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanquantile(values, [0.025, 0.975], axis=0)
        return np.nanmean(values, axis=0), low, high


def _draw_interval(axis, x, center, low, high, *, color: str, capsize: int = 3, label=None):
    """Draw null-model points with asymmetric interval bars."""
    center, low, high = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (center, low, high))
    axis.errorbar(
        x,
        center,
        yerr=[center - low, high - center],
        fmt="o",
        color=color,
        capsize=capsize,
        label=label,
    )


def plot_negative_ratio_distribution(
    empirical: np.ndarray, reference: np.ndarray, distribution: pd.DataFrame, path: Path
) -> None:
    """Save P(r_neg) and its CCDF against the sign-shuffled reference.

    r_neg = 0 is a point mass, drawn separately from the density of positive values.
    The density uses a linear scale to show the bulk; the log-scale CCDF shows the tail.
    Bands are central 95% intervals across shuffles.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    observed_color, reference_color = "#2a78d6", "#eb6834"
    zero = distribution.iloc[0]
    positive = distribution.iloc[1:]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [0.6, 2, 2]})

    axes[0].bar(0, zero["probability"], width=0.6, color=observed_color, label="Empirical")
    _draw_interval(
        axes[0],
        [1],
        zero["reference_mean"],
        zero["reference_q025"],
        zero["reference_q975"],
        color=reference_color,
        capsize=4,
        label="Sign shuffled",
    )
    axes[0].set_xticks([0, 1], ["Empirical", "Shuffled"])
    axes[0].set_xlim(-0.6, 1.6)
    axes[0].set_ylim(0, max(zero["probability"], zero["reference_q975"]) * 1.2)
    axes[0].set_ylabel("Fraction of nodes")
    axes[0].set_title("P(r_neg = 0)")

    edges = np.concatenate((positive["left"].to_numpy()[:1], positive["right"].to_numpy()))
    axes[1].stairs(
        positive["density"], edges, fill=True, color=observed_color, alpha=0.75, label="Empirical"
    )
    axes[1].stairs(
        positive["reference_density_mean"],
        edges,
        color=reference_color,
        linewidth=2,
        label="Sign shuffled mean",
    )
    axes[1].fill_between(
        edges,
        np.append(positive["reference_density_q025"], positive["reference_density_q025"].iloc[-1]),
        np.append(positive["reference_density_q975"], positive["reference_density_q975"].iloc[-1]),
        step="post",
        color=reference_color,
        alpha=0.2,
        linewidth=0,
        label="Shuffled 95% band",
    )
    axes[1].set_xlim(0, 0.5)
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Negative incident ratio r_neg")
    axes[1].set_ylabel("Density of r_neg > 0")
    axes[1].set_title("P(r_neg), positive values; tail in CCDF")

    grid = np.unique(np.concatenate(([0.0], empirical, [1.0])))

    def ccdf(values: np.ndarray) -> np.ndarray:
        ordered = np.sort(values, axis=-1)
        below = np.stack(
            [np.searchsorted(row, grid, side="left") for row in np.atleast_2d(ordered)]
        )
        return 1 - below / ordered.shape[-1]

    shuffled = ccdf(reference)
    axes[2].step(
        grid, ccdf(empirical)[0], where="post", color=observed_color, linewidth=2, label="Empirical"
    )
    axes[2].step(
        grid,
        shuffled.mean(axis=0),
        where="post",
        color=reference_color,
        linewidth=2,
        label="Sign shuffled mean",
    )
    axes[2].fill_between(
        grid,
        np.quantile(shuffled, 0.025, axis=0),
        np.quantile(shuffled, 0.975, axis=0),
        step="post",
        color=reference_color,
        alpha=0.2,
        linewidth=0,
        label="Shuffled 95% band",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.5 / empirical.size, 1.05)
    axes[2].set_xlabel("Negative incident ratio x")
    axes[2].set_ylabel("P(r_neg ≥ x)")
    axes[2].set_title("Complementary CDF")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[2].set_xlim(0, 1)
    fig.suptitle(
        f"Negative incident ratio across {empirical.size:,} core nodes versus "
        f"{reference.shape[0]:,} sign shuffles"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_signed_triads(realizations: pd.DataFrame, path: Path) -> None:
    """Save signed-triad ratios and balanced fraction against null models.

    Ratios divide each empirical triad fraction by the null-model fraction; 1 means
    as expected. Fractions are among triangles whose edges are all nonzero. Intervals
    are central 95% ranges across realizations; Erdős–Rényi is descriptive only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    triads = ("+++", "++-", "+--", "---")
    styles = {
        "sign_shuffle": ("Sign shuffle", "#eb6834"),
        "degree_preserving_rewiring": ("Degree-preserving\nrewiring", "#1baf7a"),
        "erdos_renyi": ("Erdős–Rényi\n(descriptive)", "#eda100"),
    }
    empirical = realizations.loc[realizations["model"] == "empirical"].iloc[0]
    models = [model for model in styles if (realizations["model"] == model).any()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2, 1]})

    positions = np.arange(len(triads))
    observed = np.array([empirical[f"fraction_{triad}"] for triad in triads])
    offsets = np.linspace(-0.2, 0.2, len(models)) if len(models) > 1 else np.zeros(1)
    for offset, model in zip(offsets, models, strict=False):
        label, color = styles[model]
        group = realizations.loc[realizations["model"] == model]
        mean, low, high = _ensemble_interval(
            group[[f"fraction_{triad}" for triad in triads]].to_numpy(dtype=float)
        )
        # Dividing the empirical fraction reverses the order of the interval ends.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio, low, high = observed / mean, observed / high, observed / low
        ratio, low, high = (np.where(np.isfinite(x), x, np.nan) for x in (ratio, low, high))
        _draw_interval(
            axes[0],
            positions + offset,
            ratio,
            low,
            high,
            color=color,
            label=f"{label.replace(chr(10), ' ')} (n={len(group)})",
        )
    axes[0].axhline(1, color="0.4", linewidth=1)
    axes[0].set_xticks(
        positions,
        [
            f"({', '.join(triad)})\n{'balanced' if triad in ('+++', '+--') else 'unbalanced'}"
            for triad in triads
        ],
    )
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Empirical ÷ null fraction")
    axes[0].set_title("Signed triads relative to null models (1 = as expected)")

    axes[1].axhline(empirical["balanced_fraction"], color="#2a78d6", linewidth=2, label="Empirical")
    for position, model in enumerate(models):
        _, color = styles[model]
        values = realizations.loc[realizations["model"] == model, "balanced_fraction"]
        _draw_interval(axes[1], [position], *_ensemble_interval(values), color=color, capsize=4)
    axes[1].set_xticks(range(len(models)), [styles[model][0] for model in models])
    axes[1].set_xlim(-0.6, len(models) - 0.4)
    axes[1].set_ylabel("Balanced fraction")
    axes[1].set_title("Balanced fraction of nonzero triangles")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"{int(empirical['nonzero_triangles']):,} nonzero-sign triangles; "
        f"{int(empirical['zero_sign_triangles']):,} with a zero-sign edge reported separately"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_temporal_balance(
    metrics: dict,
    wedges: pd.DataFrame,
    nulls: pd.DataFrame,
    periods: pd.DataFrame,
    path: Path,
    *,
    min_period_triangles: int = 100,
) -> None:
    """Save P-O-X closing-sign rates, scores, and quarterly scores.

    Nulls show the ensemble mean and central 95% interval. Periods with fewer than
    min_period_triangles evaluated triangles are omitted from the time panel.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    observed_color = "#2a78d6"
    styles = {
        "post_label_shuffle": ("Post-label shuffle", "#eb6834"),
        "timestamp_shuffle": ("Timestamp shuffle", "#e87ba4"),
    }
    present = [null for null in styles if "null" in nulls and (nulls["null"] == null).any()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.4, 1.6]})
    offsets = np.linspace(-0.15, 0.15, len(present)) if len(present) > 1 else np.zeros(1)

    def null_points(axis, columns: list[str]) -> None:
        for offset, null in zip(offsets, present, strict=False):
            label, color = styles[null]
            _draw_interval(
                axis,
                np.arange(len(columns)) + offset,
                *_ensemble_interval(nulls.loc[nulls["null"] == null, columns]),
                color=color,
                label=f"{label} (n={int((nulls['null'] == null).sum())})",
            )

    wedge_labels = [
        f"({w[0]}, {w[1]})\npredict {'−' if w == '+-' else '+'}" for w in wedges["wedge_type"]
    ]
    axes[0].bar(
        range(len(wedges)),
        wedges["negative_rate"],
        width=0.6,
        color=observed_color,
        alpha=0.75,
        label="Empirical",
    )
    axes[0].axhline(
        metrics["negative_closing_rate"],
        color="0.4",
        linestyle="--",
        linewidth=1,
        label="All evaluated closings",
    )
    null_points(axes[0], [f"negative_rate_{wedge}" for wedge in wedges["wedge_type"]])
    axes[0].set_xticks(range(len(wedges)), wedge_labels)
    rates = [wedges["negative_rate"].max(), metrics["negative_closing_rate"]]
    if present:
        rates.append(
            np.nanmax(
                nulls[[f"negative_rate_{wedge}" for wedge in wedges["wedge_type"]]].to_numpy()
            )
        )
    axes[0].set_ylim(0, 1.35 * np.nanmax(rates))
    axes[0].set_ylabel("P(closing edge negative)")
    axes[0].set_title("Closing sign by wedge type")

    scores = ["balanced_accuracy", "mcc", "recall_negative", "precision_negative", "specificity"]
    names = [
        "Balanced\naccuracy",
        "MCC",
        "Recall\n(− closings)",
        "Precision\n(− predicted)",
        "Specificity",
    ]
    axes[1].bar(
        range(len(scores)),
        [metrics[score] for score in scores],
        width=0.6,
        color=observed_color,
        alpha=0.75,
        label="Empirical",
    )
    null_points(axes[1], scores)
    axes[1].set_xticks(range(len(scores)), names)
    axes[1].set_ylim(min(0, *(metrics[score] for score in scores)) - 0.05, 1.05)
    axes[1].set_title(f"Balance prediction scores ({int(metrics['status_evaluated']):,} triangles)")

    shown = periods.loc[periods["status_evaluated"] >= min_period_triangles]
    x = np.arange(len(shown))
    axes[2].plot(
        x,
        shown["balanced_accuracy"],
        "o-",
        color=observed_color,
        markersize=4,
        label="Balanced accuracy",
    )
    axes[2].plot(x, shown["mcc"], "s-", color="#1baf7a", markersize=4, label="MCC")
    step = max(1, len(shown) // 6)
    axes[2].set_xticks(x[::step], shown["period"].iloc[::step])
    axes[2].set_ylim(min(0, shown["mcc"].min() if len(shown) else 0) - 0.05, 1.05)
    axes[2].set_xlabel("Closing quarter")
    axes[2].set_title(f"Scores by closing period (≥{min_period_triangles} triangles)")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Temporal P-O-X: closing-edge sign predicted as the product of the wedge signs")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_directed_roles(
    roles: pd.DataFrame,
    reciprocity: pd.DataFrame,
    bridges: pd.DataFrame,
    path: Path,
    *,
    labelled: int = 6,
) -> None:
    """Save negative sender/receiver roles, reciprocity, and bridge rates.

    Roles are interaction roles from post labels, not judgements about aggression.
    Axes in the role panel are log(1 + events).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.4, 1, 0.8]})

    sent, received = roles["negative_out_strength"], roles["negative_in_strength"]
    axes[0].scatter(received + 1, sent + 1, s=10, alpha=0.4, color="#2a78d6", linewidths=0)
    limit = max(sent.max(), received.max()) + 1
    axes[0].plot(
        [1, limit], [1, limit], color="0.5", linewidth=1, linestyle="--", label="Sends = receives"
    )
    senders = [index for index in sent.nlargest(labelled).index if sent[index] > received[index]]
    receivers = [
        index for index in received.nlargest(labelled // 2).index if received[index] > sent[index]
    ]
    for rank, index in enumerate(senders):
        axes[0].annotate(
            roles.loc[index, "node"],
            (received[index] + 1, sent[index] + 1),
            fontsize=7,
            xytext=(4, 4 - 9 * (rank % 2)),
            textcoords="offset points",
        )
    for rank, index in enumerate(receivers):
        axes[0].annotate(
            roles.loc[index, "node"],
            (received[index] + 1, sent[index] + 1),
            fontsize=7,
            xytext=(-4, 8 + 10 * rank),
            textcoords="offset points",
            ha="right",
            arrowprops={"arrowstyle": "-", "color": "0.6", "linewidth": 0.5},
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Negative events received + 1")
    axes[0].set_ylabel("Negative events sent + 1")
    axes[0].set_title("Negative interaction roles (post labels)")

    signs = reciprocity.set_index("pair_sign")
    groups = [sign for sign in ("any", "positive", "negative") if sign in signs.index]
    reverse = (
        ("reverse_any_fraction", "Any reply", "#2a78d6"),
        ("reverse_positive_fraction", "Positive reply", "#1baf7a"),
        ("reverse_negative_fraction", "Negative reply", "#eb6834"),
    )
    width = 0.26
    for offset, (column, label, color) in zip((-width, 0, width), reverse, strict=True):
        axes[1].bar(
            np.arange(len(groups)) + offset,
            signs.loc[groups, column],
            width=width,
            color=color,
            label=label,
        )
    axes[1].axhline(
        reciprocity["density_reference"].iloc[0],
        color="0.4",
        linestyle=":",
        linewidth=1,
        label="Density reference",
    )
    axes[1].set_xticks(
        range(len(groups)), [f"{sign} pairs\n(n={signs.loc[sign, 'pairs']:,})" for sign in groups]
    )
    axes[1].set_ylabel("Fraction of ordered pairs with a reverse link")
    upper = max(0.05, 1.4 * float(signs.loc[groups, "reverse_any_fraction"].max()))
    axes[1].set_ylim(0, upper)
    axes[1].set_title("Directed reciprocity")

    rates = bridges.set_index("scope")
    scopes = [scope for scope in ("within", "between") if scope in rates.index]
    for offset, (column, label, color) in zip(
        (-0.18, 0.18),
        (
            ("negative_edge_rate", "Edges (majority sign)", "#2a78d6"),
            ("negative_event_rate", "Events", "#eb6834"),
        ),
        strict=True,
    ):
        axes[2].bar(
            np.arange(len(scopes)) + offset,
            rates.loc[scopes, column],
            width=0.36,
            color=color,
            label=label,
        )
    axes[2].set_xticks(
        range(len(scopes)),
        [
            f"{scope}\n{rates.loc[scope, 'edges']:,} edges\n{rates.loc[scope, 'events']:,} events"
            for scope in scopes
        ],
    )
    axes[2].set_ylabel("Negative fraction")
    axes[2].set_title("Fixed communities")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Directed roles, reciprocity, and community bridges in the core")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
