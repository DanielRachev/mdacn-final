"""ICO interventions on unsigned and signed networks."""

from heapq import nlargest

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from mdacn.contracts import Model, OpinionState, SimulationResult, Strategy
from mdacn.dynamics import _prepare_matrix, _prepare_state, _step

# Strategies that rank nodes by a per-node score.
RANKED_STRATEGIES = (
    "degree",
    "betweenness",
    "eigenvector",
    "leverage",
    "signed_leverage",
    "temporal_bridge",
)


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
    """Owner: Polly. Select nodes by random or score-ranked targeting.

    Phase 2 adds the ranked strategies. budget_fraction lies in [0, 1]. Scores
    are required for ranked strategies. Return unique names from nodes;
    experiments record the realized budget.
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
        selection = rng.permutation(len(unique_nodes))[: min(count, len(unique_nodes))]
        return tuple(unique_nodes[int(index)] for index in sorted(selection))
    if strategy in RANKED_STRATEGIES:
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
    return opinion_cluster_sizes(graph, nodes, state)[0]


def opinion_cluster_sizes(
    graph: nx.Graph, nodes: tuple[str, ...], state: OpinionState, *, opinion: int = -1
) -> tuple[int, int]:
    """Return the largest and second-largest components of one opinion.

    Use unsigned topology, including zero-sign edges. Missing components have
    size zero; equally sized components occupy separate ranks.
    """
    members = {node for node, value in zip(nodes, state, strict=True) if value == opinion}
    sizes = nlargest(2, (len(group) for group in nx.connected_components(graph.subgraph(members))))
    largest = sizes[0] if sizes else 0
    second_largest = sizes[1] if len(sizes) > 1 else 0
    return largest, second_largest
