"""Shared measurements of terminal opinion states on the fixed unsigned core."""

import networkx as nx
import numpy as np

from mdacn.contracts import OpinionState
from mdacn.interventions import opposing_cluster_size


def compute_opinion_gcc(
        graph:nx.Graph,
        opinion_vector: OpinionState,
        *,
        nodes: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Owner: Jacek. Return the largest -1 and +1 component fractions.

    Use the full unsigned core, including pinned nodes, as both the connectivity
    reference and denominator. Opinions follow alphabetical node order unless
    nodes is supplied explicitly. The caller labels the state's stop reason.
    """
    nodes = tuple(sorted(graph)) if nodes is None else tuple(nodes)
    state = np.asarray(opinion_vector)

    if not nodes:
        return {"g_minus": 0.0, "g_plus": 0.0}

    return {
        "g_minus": opposing_cluster_size(graph, nodes, state) / len(graph),
        "g_plus": opposing_cluster_size(graph, nodes, -state) / len(graph)
    }