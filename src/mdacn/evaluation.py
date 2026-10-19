"""Shared measurements of terminal opinion states on the fixed unsigned core."""

import networkx as nx
import numpy as np

from mdacn.contracts import OpinionState
from mdacn.interventions import opinion_cluster_sizes


def compute_opinion_gcc(
    graph: nx.Graph,
    opinion_vector: OpinionState,
    *,
    nodes: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Owner: Jacek. Return the largest -1 and +1 component fractions.

    Use the full unsigned core, including pinned nodes, as both the connectivity
    reference and denominator. Opinions follow alphabetical node order unless
    nodes is supplied explicitly. The caller labels the state's stop reason.
    """
    components = compute_opinion_components(graph, opinion_vector, nodes=nodes)
    return {name: components[name] for name in ("g_minus", "g_plus")}


def compute_opinion_components(
    graph: nx.Graph,
    opinion_vector: OpinionState,
    *,
    nodes: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return the first two component fractions for each opinion, divided by full N.

    g_minus_second and g_plus_second describe the second-largest component of
    the same opinion as g_minus and g_plus respectively. Node alignment and
    connectivity follow compute_opinion_gcc.
    """
    nodes = tuple(sorted(graph)) if nodes is None else tuple(nodes)
    state = np.asarray(opinion_vector)
    if state.shape != (len(nodes),) or len(nodes) != len(graph) or set(nodes) != set(graph):
        raise ValueError("Opinions and node order must match the full graph")
    measurements = {}
    for opinion, name in ((-1, "minus"), (1, "plus")):
        largest, second = opinion_cluster_sizes(graph, nodes, state, opinion=opinion)
        measurements[f"g_{name}"] = largest / len(graph) if nodes else 0.0
        measurements[f"g_{name}_second"] = second / len(graph) if nodes else 0.0
    return measurements
