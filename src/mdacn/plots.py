"""Figure interfaces. Each function saves a figure to the supplied path."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdacn.characterisation import fitted_degree_probabilities


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


def plot_intervention_comparison(results: pd.DataFrame, path: Path) -> None:
    """Owner: Polly. Save opposing cluster fraction S_1/N against contrarian budget Phi.

    Compare random, degree, and leverage targeting with realization variability.
    Identify cycles and update-limit runs separately from steady-state results.
    """
    if results.empty:
        raise ValueError("No intervention results to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    results = results.assign(cluster_fraction=results["cluster_size"] / results["n_nodes"])
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
    axes[0].set_ylabel("Opposing cluster fraction S₁/N")
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
