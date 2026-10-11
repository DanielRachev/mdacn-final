"""Experiment interfaces."""

import networkx as nx
import pandas as pd

from mdacn.contracts import NetworkMatrices


def run_fraction_sweep(
    network: NetworkMatrices,
    *,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """Owner: Daniel. Compare unsigned and signed NCO over initial negative fractions.

    Return one row per model, fraction, and realization, with reproducibility
    fields, opinion fractions, and stop status.
    Initial states match across models within each fraction and realization.
    """
    raise NotImplementedError("Daniel: run_fraction_sweep")


def run_intervention_sweep(
    graph: nx.Graph,
    network: NetworkMatrices,
    *,
    budgets: tuple[float, ...],
    initial_negative_fraction: float,
    realizations: int = 20,
    seed: int = 0,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """Owner: Polly. Compare random, degree, and leverage targeting on unsigned NCO.

    Return one row per strategy, budget, and realization, including opposing
    cluster size and stop status. Seeds and baseline states support comparison
    across strategies.
    """
    raise NotImplementedError("Polly: run_intervention_sweep")
