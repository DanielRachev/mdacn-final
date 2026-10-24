"""ICO interventions on unsigned and signed networks."""

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from mdacn.contracts import Model, OpinionState, SimulationResult, Strategy
from mdacn.dynamics import _prepare_matrix, _prepare_state, _step


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _candidate_order(nodes: tuple[str, ...], scores: dict[str, float]) -> tuple[str, ...]:
    missing = [node for node in nodes if node not in scores]
    if missing:
        raise ValueError(f"scores missing for nodes: {missing}")
    return tuple(sorted(nodes, key=lambda node: (-float(scores[node]), node)))


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
    if isinstance(budget_fraction, bool) or not np.isfinite(budget_fraction):
        raise ValueError("budget_fraction must be a finite float in [0, 1]")
    if not 0 <= budget_fraction <= 1:
        raise ValueError("budget_fraction must be in [0, 1]")
    if not nodes:
        return ()

    count = int(round(len(nodes) * budget_fraction))
    if count <= 0:
        return ()
    if budget_fraction >= 1:
        return tuple(dict.fromkeys(nodes))

    unique_nodes = tuple(dict.fromkeys(nodes))
    if strategy == "random":
        rng = np.random.default_rng(seed)
        selection = rng.choice(len(unique_nodes), size=min(count, len(unique_nodes)), replace=False)
        return tuple(unique_nodes[int(index)] for index in sorted(selection))
    if strategy in {"degree", "leverage"}:
        if scores is None:
            raise ValueError(f"scores are required for {strategy} targeting")
        ranked = _candidate_order(unique_nodes, scores)
        return tuple(ranked[:count])
    raise ValueError(f"Unknown strategy: {strategy}")


def simulate_ico(
    adjacency: csr_matrix,
    initial_state: OpinionState,
    *,
    nodes: tuple[str, ...],
    contrarians: tuple[str, ...],
    max_steps: int,
    model: Model = "unsigned",
) -> SimulationResult:
    """Run ICO with selected nodes fixed at +1 from the intervention start.

    The default retains Polly's unsigned interface. Signed runs use the same
    synchronous update, self vote, and tie convention as Daniel's signed NCO.
    Callers supply the settled baseline when applying post-baseline intervention.
    """
    _validate_nonnegative_integer(max_steps, "max_steps")
    if len(nodes) != len(set(nodes)):
        raise ValueError("nodes must be unique")
    if adjacency.shape != (len(nodes), len(nodes)):
        raise ValueError("adjacency shape must match nodes")
    matrix = _prepare_matrix(adjacency, model)
    state = _prepare_state(initial_state, len(nodes))

    contrarian_set = set(contrarians)
    if not set(nodes).issuperset(contrarian_set):
        raise ValueError("contrarians must be a subset of nodes")
    fixed = np.array(
        [index for index, node in enumerate(nodes) if node in contrarian_set], dtype=int
    )
    state[fixed] = 1

    seen = {np.packbits(state > 0).tobytes(): 0}
    for step in range(1, max_steps + 1):
        state = _step(matrix, state)
        state[fixed] = 1
        key = np.packbits(state > 0).tobytes()
        previous_step = seen.get(key)
        if previous_step is not None:
            period = step - previous_step
            if period == 1:
                return SimulationResult(state, step, "steady_state")
            return SimulationResult(state, step, "limit_cycle", period)
        seen[key] = step
    return SimulationResult(state, max_steps, "max_steps")


def opposing_cluster_size(graph: nx.Graph, nodes: tuple[str, ...], state: OpinionState) -> int:
    """Owner: Polly. Return S_1, the largest connected cluster of -1 nodes.

    Return zero if no -1 nodes remain. State order matches nodes.
    """
    negative_nodes = {node for node, opinion in zip(nodes, state, strict=True) if opinion == -1}
    if not negative_nodes:
        return 0
    subgraph = graph.subgraph(negative_nodes).copy()
    if subgraph.number_of_nodes() == 0:
        return 0
    return max(len(component) for component in nx.connected_components(subgraph))
