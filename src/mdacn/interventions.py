"""ICO interventions on the unsigned network."""

import networkx as nx
from scipy.sparse import csr_matrix

from mdacn.contracts import OpinionState, SimulationResult, Strategy


def select_contrarians(
    nodes: tuple[str, ...],
    budget_fraction: float,
    *,
    strategy: Strategy,
    seed: int,
    scores: dict[str, float] | None = None,
) -> tuple[str, ...]:
    """Owner: Polly. Select nodes for random, degree, or leverage targeting.

    budget_fraction lies in [0, 1]. Scores are required for degree and leverage.
    Return unique names from nodes; experiments record the realized budget.
    """
    raise NotImplementedError("Polly: select_contrarians")


def simulate_ico(
    adjacency: csr_matrix,
    initial_state: OpinionState,
    *,
    nodes: tuple[str, ...],
    contrarians: tuple[str, ...],
    max_steps: int,
) -> SimulationResult:
    """Owner: Polly. Run unsigned ICO with selected nodes fixed at +1 from time zero.

    Unselected nodes follow Daniel's unsigned NCO behavior. Result fields and
    stop reasons match SimulationResult. Node names match the matrix order.
    """
    raise NotImplementedError("Polly: simulate_ico")


def opposing_cluster_size(graph: nx.Graph, nodes: tuple[str, ...], state: OpinionState) -> int:
    """Owner: Polly. Return S_1, the largest connected cluster of -1 nodes.

    Return zero if no -1 nodes remain. State order matches nodes.
    """
    raise NotImplementedError("Polly: opposing_cluster_size")
