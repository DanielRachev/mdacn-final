"""Null Model B: degree-preserving double-edge swaps with homophily steering."""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

Target = Literal["insular", "integrated"]
Rule = Literal["unbiased", "strict", "non_worsening", "metropolis"]
RejectionReason = Literal["pool_too_small", "self_loop", "no_op", "duplicate", "rule"]

SIGN_NAMES = {1: "positive", -1: "negative", 0: "zero"}
_POOLS = (1, -1, 0)
_BLOCK = 1 << 16


@dataclass(frozen=True)
class RewiringResult:
    """Rewired graph and swap statistics; the input graph is not modified."""

    graph: nx.Graph
    stats: dict = field(repr=False)


def homophily_counts(graph: nx.Graph, *, community: str = "community") -> dict[str, int]:
    """Count edges per sign inside and between fixed community labels.

    Return {sign}_within and {sign}_between for positive, negative, and zero
    edges, plus the steering scores h_insular and h_integrated.
    """
    labels = _labels(graph, community)
    counts = Counter()
    for u, v, sign in graph.edges(data="sign"):
        place = "within" if labels[u] == labels[v] else "between"
        counts[f"{SIGN_NAMES[sign]}_{place}"] += 1
    result = {
        f"{name}_{place}": counts[f"{name}_{place}"]
        for name in SIGN_NAMES.values()
        for place in ("within", "between")
    }
    result["h_insular"] = result["positive_within"] + result["negative_between"]
    result["h_integrated"] = result["positive_between"] + result["negative_within"]
    return result


def homophily_bounds(graph: nx.Graph, *, community: str = "community") -> dict[str, int]:
    """Return degree/community upper bounds on within and between edges per sign.

    Node i can have at most min(d_i^s, |C_i| - 1) internal and min(d_i^s, N - |C_i|)
    external edges of sign s. Each bound is halved and capped at the sign's edge
    count. The bounds hold for any same-sign rewiring; they need not be attainable.
    """
    labels = _labels(graph, community)
    sizes = Counter(labels.values())
    n_nodes = len(graph)
    signed_degree = {node: Counter() for node in graph}
    edge_counts = Counter()
    for u, v, sign in graph.edges(data="sign"):
        signed_degree[u][sign] += 1
        signed_degree[v][sign] += 1
        edge_counts[sign] += 1
    bounds = {}
    for sign, name in SIGN_NAMES.items():
        inside = sum(min(signed_degree[i][sign], sizes[labels[i]] - 1) for i in graph)
        outside = sum(min(signed_degree[i][sign], n_nodes - sizes[labels[i]]) for i in graph)
        bounds[f"{name}_within_max"] = min(edge_counts[sign], inside // 2)
        bounds[f"{name}_between_max"] = min(edge_counts[sign], outside // 2)
    bounds["h_insular_max"] = bounds["positive_within_max"] + bounds["negative_between_max"]
    bounds["h_integrated_max"] = bounds["positive_between_max"] + bounds["negative_within_max"]
    return bounds


def accept_swap(delta: int, *, rule: Rule, beta: float | None, uniform: float) -> bool:
    """Decide one valid swap from its homophily change and a U(0, 1) draw."""
    if rule == "unbiased":
        return True
    if rule == "strict":
        return delta > 0
    if rule == "non_worsening":
        return delta >= 0
    if rule == "metropolis":
        return delta >= 0 or uniform < math.exp(beta * delta)
    raise ValueError(f"Unknown rule: {rule}")


def rewire(
    graph: nx.Graph,
    *,
    attempts: int,
    seed: int,
    target: Target | None = None,
    rule: Rule = "unbiased",
    beta: float | None = None,
    community: str = "community",
    checkpoint_interval: int = 0,
    window: int = 0,
    tolerance: float = 0.0,
) -> RewiringResult:
    """Owner: Jakub. Apply attempted same-sign double-edge swaps to a copy of graph.

    Each attempt picks a uniform edge, then a second uniform edge with the same sign,
    and rewires (u, v), (x, y) to (u, y), (x, v). Swaps that create a self loop or a
    duplicate edge, or that leave both edges unchanged, are rejected. Every node keeps
    its unsigned degree and its positive, negative, and zero degree.

    Edge records (sign, n_positive, n_negative) follow the first endpoint's stub:
    (u, y) takes the record of (u, v), and (x, v) the record of (x, y).

    target and rule steer the positive and negative pools toward the insular or
    integrated score; zero-sign edges carry no signed influence and are always
    swapped without steering. rule='unbiased' accepts every valid swap and is the
    degree-preserving reference. beta is required only for rule='metropolis'.
    checkpoint_interval > 0 records counts every that many attempts.

    attempts is the maximum. With window > 0, the run also stops once the tracked
    score changes by at most tolerance * (positive + negative edges) over the last
    window attempts. The tracked score is the target score, or the number of
    within-community signed edges for the unbiased reference.
    """
    _validate_settings(attempts, target, rule, beta, checkpoint_interval)
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 0:
        raise ValueError("window must be a non-negative integer")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Rewiring requires a simple undirected graph without self loops")
    labels_by_node = _labels(graph, community)
    nodes = tuple(sorted(graph))
    index = {node: i for i, node in enumerate(nodes)}
    labels = [labels_by_node[node] for node in nodes]
    n_nodes = len(nodes)

    ends: dict[int, list[list[int]]] = {sign: [] for sign in _POOLS}
    records: dict[int, list[dict]] = {sign: [] for sign in _POOLS}
    for u, v, data in graph.edges(data=True):
        sign = data.get("sign")
        if sign not in _POOLS:
            raise ValueError("Every edge must have a sign of -1, 0, or +1")
        ends[sign].append([index[u], index[v]])
        records[sign].append(dict(data))
    keys = {min(a, b) * n_nodes + max(a, b) for pool in ends.values() for a, b in pool}
    sizes = [len(ends[sign]) for sign in _POOLS]
    total = sum(sizes)
    within = {sign: sum(labels[a] == labels[b] for a, b in ends[sign]) for sign in _POOLS}
    # Score of one edge by pool: (between, within). Zero edges never contribute.
    weight = {
        1: (0, 1) if target == "insular" else (1, 0),
        -1: (1, 0) if target == "insular" else (0, 1),
        0: (0, 0),
    }

    rng = np.random.default_rng(seed)
    rejections = Counter()
    attempts_by_sign = Counter()
    accepted_by_sign = Counter()
    checkpoints = []
    steered_edges = sizes[0] + sizes[1]
    window_start = _tracked_score(target, within, sizes)
    converged = False
    done = 0
    while done < attempts and not converged:
        block = min(_BLOCK, attempts - done)
        first = rng.integers(0, max(total, 1), size=block).tolist()
        second = rng.random(block).tolist()
        # Orientation bits flip (u, v) and (x, y) independently. An undirected edge
        # pair has two distinct rewirings; without the flips, the stored endpoint
        # order would decide which one is proposed and half the moves would never
        # occur (e.g. the within-community pairing of two crossing edges).
        flips = rng.integers(0, 4, size=block).tolist()
        uniforms = rng.random(block).tolist()
        for offset in range(block):
            done += 1
            position = first[offset]
            pool_index = 0 if position < sizes[0] else 1 if position < sizes[0] + sizes[1] else 2
            position -= sum(sizes[:pool_index])
            sign = _POOLS[pool_index]
            attempts_by_sign[sign] += 1
            pool = ends[sign]
            if len(pool) < 2:
                rejections["pool_too_small"] += 1
            else:
                other = int(second[offset] * (len(pool) - 1))
                other += other >= position
                u, v = pool[position] if not flips[offset] & 1 else pool[position][::-1]
                x, y = pool[other] if not flips[offset] & 2 else pool[other][::-1]
                new_first = min(u, y) * n_nodes + max(u, y)
                new_second = min(x, v) * n_nodes + max(x, v)
                if u == y or x == v:
                    rejections["self_loop"] += 1
                elif u == x or v == y:
                    rejections["no_op"] += 1
                elif new_first in keys or new_second in keys:
                    rejections["duplicate"] += 1
                else:
                    gain = weight[sign]
                    change = (
                        (labels[u] == labels[y])
                        + (labels[x] == labels[v])
                        - (labels[u] == labels[v])
                        - (labels[x] == labels[y])
                    )
                    delta = (gain[1] - gain[0]) * change
                    steered = sign != 0 and rule != "unbiased"
                    if steered and not accept_swap(
                        delta, rule=rule, beta=beta, uniform=uniforms[offset]
                    ):
                        rejections["rule"] += 1
                    else:
                        keys.discard(min(u, v) * n_nodes + max(u, v))
                        keys.discard(min(x, y) * n_nodes + max(x, y))
                        keys.add(new_first)
                        keys.add(new_second)
                        pool[position] = [u, y]
                        pool[other] = [x, v]
                        within[sign] += change
                        accepted_by_sign[sign] += 1
            if checkpoint_interval and done % checkpoint_interval == 0:
                checkpoints.append(_checkpoint(done, sum(accepted_by_sign.values()), within, sizes))
            if window and done % window == 0:
                score = _tracked_score(target, within, sizes)
                if abs(score - window_start) <= tolerance * steered_edges:
                    converged = True
                    break
                window_start = score

    rewired = nx.Graph()
    rewired.graph.update(graph.graph)
    rewired.add_nodes_from((node, dict(graph.nodes[node])) for node in nodes)
    for sign in _POOLS:
        for (a, b), record in zip(ends[sign], records[sign], strict=True):
            rewired.add_edge(nodes[a], nodes[b], **record)
    if any("n_interactions" in graph.nodes[node] for node in nodes):
        # Event-count records moved with their edges, so node activity is recomputed.
        for node in rewired:
            rewired.nodes[node]["n_interactions"] = sum(
                edge.get("n_positive", 0) + edge.get("n_negative", 0)
                for edge in rewired[node].values()
            )
    accepted = sum(accepted_by_sign.values())
    stats = {
        "seed": int(seed),
        "target": target,
        "rule": rule,
        "beta": beta,
        "max_attempts": attempts,
        "attempts": done,
        "window": window,
        "tolerance": tolerance,
        "converged": converged,
        "stop_reason": "converged" if converged else "max_attempts",
        "accepted": accepted,
        "acceptance_rate": accepted / done if done else 0.0,
        "rejections": {
            reason: rejections[reason]
            for reason in ("pool_too_small", "self_loop", "no_op", "duplicate", "rule")
        },
        "attempts_by_sign": {SIGN_NAMES[s]: attempts_by_sign[s] for s in _POOLS},
        "accepted_by_sign": {SIGN_NAMES[s]: accepted_by_sign[s] for s in _POOLS},
        "checkpoints": checkpoints,
    }
    return RewiringResult(rewired, stats)


def validate_rewiring(
    original: nx.Graph, rewired: nx.Graph, *, community: str = "community"
) -> dict:
    """Raise AssertionError if rewiring broke an invariant; otherwise summarize it.

    Checked: simple graph, node set, community labels, each node's unsigned and
    positive/negative/zero degree, global sign counts, and the multiset of edge
    records. Connectivity is reported, not enforced.
    """
    if rewired.is_directed() or rewired.is_multigraph() or nx.number_of_selfloops(rewired):
        raise AssertionError("Rewired graph is not simple and undirected")
    if set(original) != set(rewired):
        raise AssertionError("Node set changed")
    if _labels(original, community) != _labels(rewired, community):
        raise AssertionError("Community labels changed")
    if dict(original.degree()) != dict(rewired.degree()):
        raise AssertionError("Unsigned degree changed")
    if _signed_degrees(original) != _signed_degrees(rewired):
        raise AssertionError("Signed degree changed")
    if _records(original) != _records(rewired):
        raise AssertionError("Edge records or global sign counts changed")
    original_edges = {frozenset(edge) for edge in original.edges()}
    changed = sum(frozenset(edge) not in original_edges for edge in rewired.edges())
    sizes = sorted((len(part) for part in nx.connected_components(rewired)), reverse=True)
    signs = Counter(sign for _, _, sign in rewired.edges(data="sign"))
    return {
        "nodes": rewired.number_of_nodes(),
        "edges": rewired.number_of_edges(),
        "sign_counts": {SIGN_NAMES[s]: signs[s] for s in _POOLS},
        "edges_changed": changed,
        "fraction_changed": changed / rewired.number_of_edges() if len(rewired.edges) else 0.0,
        "connected": len(sizes) == 1,
        "connected_components": len(sizes),
        "largest_component_sizes": sizes[:5],
        "degree_preserved": True,
        "signed_degree_preserved": True,
        "edge_records_preserved": True,
    }


def degree_table(original: nx.Graph, rewired: nx.Graph) -> list[dict]:
    """Return each node's unsigned and signed degrees before and after rewiring."""
    before, after = _signed_degrees(original), _signed_degrees(rewired)
    rows = []
    for node in sorted(original):
        row = {"node": node, "degree_before": original.degree(node)}
        row["degree_after"] = rewired.degree(node)
        for sign, name in SIGN_NAMES.items():
            row[f"{name}_before"] = before[node][sign]
            row[f"{name}_after"] = after[node][sign]
        rows.append(row)
    return rows


def _labels(graph: nx.Graph, community: str) -> dict:
    missing = [node for node in graph if community not in graph.nodes[node]]
    if missing:
        raise ValueError(f"Nodes missing '{community}' labels: {missing[:5]}")
    return {node: graph.nodes[node][community] for node in graph}


def _signed_degrees(graph: nx.Graph) -> dict:
    degrees = {node: Counter({sign: 0 for sign in _POOLS}) for node in graph}
    for u, v, sign in graph.edges(data="sign"):
        degrees[u][sign] += 1
        degrees[v][sign] += 1
    return degrees


def _records(graph: nx.Graph) -> Counter:
    return Counter(tuple(sorted(data.items())) for _, _, data in graph.edges(data=True))


def _tracked_score(target: Target | None, within: dict, sizes: list[int]) -> int:
    if target == "insular":
        return within[1] + sizes[1] - within[-1]
    if target == "integrated":
        return sizes[0] - within[1] + within[-1]
    return within[1] + within[-1]


def _checkpoint(attempts: int, accepted: int, within: dict, sizes: list[int]) -> dict:
    n_positive, n_negative = sizes[0], sizes[1]
    return {
        "attempts": attempts,
        "accepted": accepted,
        "positive_within": within[1],
        "negative_within": within[-1],
        "zero_within": within[0],
        "h_insular": within[1] + n_negative - within[-1],
        "h_integrated": n_positive - within[1] + within[-1],
    }


def _validate_settings(
    attempts: int, target: Target | None, rule: Rule, beta: float | None, interval: int
) -> None:
    for value, name in ((attempts, "attempts"), (interval, "checkpoint_interval")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if rule not in ("unbiased", "strict", "non_worsening", "metropolis"):
        raise ValueError(f"Unknown rule: {rule}")
    if target not in (None, "insular", "integrated"):
        raise ValueError(f"Unknown target: {target}")
    if (rule == "unbiased") != (target is None):
        raise ValueError("Only rule='unbiased' runs without a target")
    if rule == "metropolis":
        if beta is None or not np.isfinite(beta) or beta <= 0:
            raise ValueError("metropolis requires a finite beta > 0")
    elif beta is not None:
        raise ValueError("beta applies only to rule='metropolis'")
