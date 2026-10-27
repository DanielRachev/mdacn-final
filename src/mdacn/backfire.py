"""Backfire measurements for negative influence from pinned contrarians.

A pinned node contributes ``W_ji * sigma_i = (-1) * (+1) = -1`` through each
negative link. These measurements count that influence and whether neighbours
actually flipped, relative to the paired no-intervention baseline.
"""

import numpy as np

from mdacn.contracts import Model, NetworkMatrices, OpinionState

NEGATIVE_WEIGHT = -1.0


def backfire_metrics(
    network: NetworkMatrices,
    *,
    contrarians: tuple[str, ...],
    baseline_state: OpinionState,
    final_state: OpinionState,
    model: Model,
) -> dict[str, float | int]:
    """Return negative links from contrarians and the neighbours they moved.

    ``negative_influence_links`` counts signed negative edges incident to a pinned
    node. ``repelled_neighbours`` counts those links whose non-pinned neighbour
    held +1 at baseline and -1 at the terminal state. Unsigned runs have no
    reactance term, so every count is zero. These are model measurements, not
    observed responses.
    """
    nodes = network.nodes
    n_nodes = len(nodes)
    baseline = np.asarray(baseline_state)
    final = np.asarray(final_state)
    if baseline.shape != (n_nodes,) or final.shape != (n_nodes,):
        raise ValueError("Opinion vectors must match the node order length")
    missing = [node for node in contrarians if node not in set(nodes)]
    if missing:
        raise ValueError(f"Contrarians must be core nodes: {missing}")

    metrics: dict[str, float | int] = {
        "negative_influence_links": 0,
        "repelled_neighbours": 0,
        "repelled_share": 0.0,
    }
    if model != "signed" or not contrarians:
        return metrics

    index = {node: position for position, node in enumerate(nodes)}
    pinned = {index[node] for node in contrarians}
    signed = network.signed.tocsr()
    links = 0
    repelled = 0
    for contrarian in sorted(pinned):
        start, end = signed.indptr[contrarian], signed.indptr[contrarian + 1]
        for neighbour, weight in zip(
            signed.indices[start:end], signed.data[start:end], strict=True
        ):
            if float(weight) != NEGATIVE_WEIGHT:
                continue
            links += 1
            if neighbour in pinned:
                continue
            if baseline[neighbour] == 1 and final[neighbour] == -1:
                repelled += 1
    metrics["negative_influence_links"] = links
    metrics["repelled_neighbours"] = repelled
    metrics["repelled_share"] = repelled / links if links else 0.0
    return metrics
