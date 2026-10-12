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
    raise NotImplementedError("Daniel: plot_phase_transition")


def plot_intervention_comparison(results: pd.DataFrame, path: Path) -> None:
    """Owner: Polly. Save opposing cluster fraction S_1/N against contrarian budget Phi.

    Compare random, degree, and leverage targeting with realization variability.
    Identify cycles and update-limit runs separately from steady-state results.
    """
    raise NotImplementedError("Polly: plot_intervention_comparison")
