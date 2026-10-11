"""Figure interfaces. Each function saves a figure to the supplied path."""

from pathlib import Path

import pandas as pd


def plot_degree_distribution(distribution: pd.DataFrame, fits: pd.DataFrame, path: Path) -> None:
    """Owner: Jacek. Save empirical and fitted degree curves on log-log and log-linear axes."""
    raise NotImplementedError("Jacek: plot_degree_distribution")


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
