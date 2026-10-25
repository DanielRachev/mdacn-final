"""Null Model B: same-sign double-edge swaps, steering, validation, and exports."""

import json
import subprocess
import sys
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse import load_npz, save_npz

from mdacn.centrality_pipeline import save_centrality_scores
from mdacn.data import build_matrices, export_graph
from mdacn.evaluation import compute_opinion_gcc
from mdacn.rewiring import (
    accept_swap,
    homophily_bounds,
    homophily_counts,
    rewire,
    validate_rewiring,
)
from mdacn.rewiring_pipeline import (
    RewiringSpec,
    critical_budgets,
    run_rewired_ico_experiment,
    run_rewiring_experiment,
    stream_seed,
)


def two_communities(seed: int = 0) -> nx.Graph:
    """Two signed 6-node communities joined by crossing edges."""
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    for node in range(12):
        graph.add_node(f"n{node:02d}", community=node // 6)
    nodes = sorted(graph)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if rng.random() < 0.45:
                sign = int(rng.choice([1, 1, 1, -1, 0]))
                graph.add_edge(
                    u, v, sign=sign, n_positive=1 + (sign == 1), n_negative=1 + (sign == -1)
                )
    return graph


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    rule=st.sampled_from(["unbiased", "strict", "non_worsening", "metropolis"]),
    target=st.sampled_from(["insular", "integrated"]),
)
def test_rewiring_preserves_invariants(seed, rule, target):
    graph = two_communities(seed % 7)
    result = rewire(
        graph,
        attempts=300,
        seed=seed,
        target=None if rule == "unbiased" else target,
        rule=rule,
        beta=1.0 if rule == "metropolis" else None,
    )
    summary = validate_rewiring(graph, result.graph)
    assert summary["sign_counts"] == {
        name: sum(s == sign for _, _, s in graph.edges(data="sign"))
        for sign, name in ((1, "positive"), (-1, "negative"), (0, "zero"))
    }
    stats = result.stats
    assert stats["accepted"] + sum(stats["rejections"].values()) == stats["attempts"] == 300
    assert graph.number_of_edges() == result.graph.number_of_edges()


def test_rewiring_is_reproducible_and_leaves_input_unchanged():
    graph = two_communities()
    before = sorted(graph.edges(data="sign"))
    first = rewire(graph, attempts=500, seed=3)
    second = rewire(graph, attempts=500, seed=3)
    assert sorted(first.graph.edges(data=True)) == sorted(second.graph.edges(data=True))
    assert sorted(graph.edges(data="sign")) == before
    assert first.stats["accepted"] > 0


def test_orientation_flip_proposes_both_pairings():
    graph = nx.Graph()
    graph.add_nodes_from("abcd", community=0)
    graph.add_edge("a", "b", sign=1)
    graph.add_edge("c", "d", sign=1)
    outcomes = Counter()
    for seed in range(200):
        edges = rewire(graph, attempts=1, seed=seed).graph.edges()
        outcomes[frozenset(frozenset(edge) for edge in edges)] += 1
    pairings = {
        frozenset({frozenset("ad"), frozenset("bc")}),
        frozenset({frozenset("ac"), frozenset("bd")}),
    }
    assert pairings <= set(outcomes)


def test_edge_records_follow_first_endpoint():
    graph = nx.Graph()
    graph.add_nodes_from("abcd", community=0)
    graph.add_edge("a", "b", sign=1, n_positive=5, n_negative=0)
    graph.add_edge("c", "d", sign=1, n_positive=2, n_negative=1)
    swapped = 0
    for seed in range(50):
        result = rewire(graph, attempts=1, seed=seed)
        swapped += result.stats["accepted"]
        shared = {
            data["n_positive"]: len({u, v} & (set("ab") if data["n_positive"] == 5 else set("cd")))
            for u, v, data in result.graph.edges(data=True)
        }
        # Each record keeps one original endpoint after a swap, both otherwise.
        assert set(shared) == {5, 2}
        assert set(shared.values()) == ({1} if result.stats["accepted"] else {2})
    assert swapped > 0


def test_unbiased_swaps_sample_degree_sequence_uniformly():
    # The three labelled 4-cycles share the 2-regular degree sequence on four nodes.
    graph = nx.cycle_graph(["a", "b", "c", "d"])
    nx.set_node_attributes(graph, 0, "community")
    nx.set_edge_attributes(graph, 1, "sign")
    counts = Counter(
        frozenset(frozenset(edge) for edge in rewire(graph, attempts=25, seed=s).graph.edges())
        for s in range(3000)
    )
    assert len(counts) == 3
    assert all(abs(count / 3000 - 1 / 3) < 0.04 for count in counts.values())


@pytest.mark.parametrize("target", ["insular", "integrated"])
@pytest.mark.parametrize("rule", ["strict", "non_worsening"])
def test_greedy_rules_never_lower_score(target, rule):
    graph = two_communities()
    result = rewire(graph, attempts=2000, seed=1, target=target, rule=rule, checkpoint_interval=1)
    scores = [point[f"h_{target}"] for point in result.stats["checkpoints"]]
    start = homophily_counts(graph)[f"h_{target}"]
    assert all(b >= a for a, b in zip([start, *scores], scores, strict=False))
    assert scores[-1] == homophily_counts(result.graph)[f"h_{target}"]
    assert scores[-1] <= homophily_bounds(graph)[f"h_{target}_max"]


def test_strict_rule_rejects_signed_moves_but_swaps_zero_edges():
    graph = two_communities()
    result = rewire(graph, attempts=3000, seed=2, target="insular", rule="strict")
    assert result.stats["rejections"]["rule"] > 0
    assert result.stats["accepted_by_sign"]["zero"] > 0


def test_targets_move_in_opposite_directions():
    graph = two_communities()
    kwargs = dict(attempts=3000, seed=4, rule="metropolis", beta=2.0)
    insular = homophily_counts(rewire(graph, target="insular", **kwargs).graph)
    integrated = homophily_counts(rewire(graph, target="integrated", **kwargs).graph)
    assert insular["positive_within"] > integrated["positive_within"]
    assert insular["negative_between"] >= integrated["negative_between"]


def test_window_stop_ends_frozen_runs_early():
    graph = two_communities()
    result = rewire(graph, attempts=50_000, seed=5, target="insular", rule="strict", window=500)
    stats = result.stats
    assert stats["converged"] and stats["stop_reason"] == "converged"
    assert stats["attempts"] < stats["max_attempts"] and stats["attempts"] % 500 == 0
    assert stats["accepted"] + sum(stats["rejections"].values()) == stats["attempts"]
    validate_rewiring(graph, result.graph)


def test_window_stop_is_reproducible_and_optional():
    graph = two_communities()
    kwargs = dict(attempts=20_000, seed=6, window=1000, tolerance=0.05)
    first, second = rewire(graph, **kwargs), rewire(graph, **kwargs)
    assert first.stats == second.stats
    capped = rewire(graph, attempts=2000, seed=6)
    assert not capped.stats["converged"] and capped.stats["attempts"] == 2000


@pytest.mark.parametrize("kwargs", [dict(window=-1), dict(tolerance=-0.1)])
def test_invalid_window_settings(kwargs):
    with pytest.raises(ValueError):
        rewire(two_communities(), attempts=1, seed=0, **kwargs)


def test_accept_swap_rules():
    assert accept_swap(-2, rule="unbiased", beta=None, uniform=0.99)
    assert not accept_swap(0, rule="strict", beta=None, uniform=0.0)
    assert accept_swap(0, rule="non_worsening", beta=None, uniform=0.99)
    assert not accept_swap(-1, rule="non_worsening", beta=None, uniform=0.0)
    assert accept_swap(-1, rule="metropolis", beta=1.0, uniform=0.3)
    assert not accept_swap(-1, rule="metropolis", beta=1.0, uniform=0.4)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rule="strict"),
        dict(rule="unbiased", target="insular"),
        dict(rule="metropolis", target="insular"),
        dict(rule="strict", target="insular", beta=1.0),
        dict(rule="metropolis", target="insular", beta=0.0),
        dict(rule="other", target="insular"),
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        rewire(two_communities(), attempts=1, seed=0, **kwargs)


def test_missing_community_labels_are_rejected():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "sign")
    with pytest.raises(ValueError, match="community"):
        rewire(graph, attempts=1, seed=0)


def test_validation_detects_broken_invariants():
    graph = two_communities()
    broken = graph.copy()
    u, v, data = next(iter(graph.edges(data=True)))
    broken.remove_edge(u, v)
    with pytest.raises(AssertionError):
        validate_rewiring(graph, broken)
    flipped = graph.copy()
    flipped[u][v]["sign"] = -data["sign"] if data["sign"] else 1
    with pytest.raises(AssertionError):
        validate_rewiring(graph, flipped)


def test_stream_seeds_are_explicit_and_independent():
    assert stream_seed(0, "rewire:reference", 0) == stream_seed(0, "rewire:reference", 0)
    assert stream_seed(0, "rewire:reference", 0) != stream_seed(0, "rewire:reference", 1)
    assert stream_seed(0, "rewire:reference", 0) != stream_seed(1, "rewire:reference", 0)
    assert stream_seed(0, "initial", 0) != stream_seed(0, "target", 0)


def test_critical_budgets():
    raw = pd.DataFrame(
        {
            "graph": ["g"] * 6,
            "replicate": [0] * 6,
            "strategy": ["degree"] * 6,
            "initialization": [0, 0, 0, 1, 1, 1],
            "budget": [0.0, 0.1, 0.2] * 2,
            "g_minus": [0.9, 0.4, 0.1, 0.8, 0.7, 0.6],
            "baseline_stop_reason": ["steady_state"] * 6,
            "stop_reason": ["steady_state"] * 6,
        }
    )
    phi_c = critical_budgets(raw)
    assert phi_c["phi_c"].iloc[0] == 0.1 and phi_c["threshold_reached"].iloc[0]
    assert np.isnan(phi_c["phi_c"].iloc[1]) and not phi_c["threshold_reached"].iloc[1]


def write_inputs(tmp_path, graph=None):
    """Save a core graph plus the processed matrices and node order next to it."""
    graph = two_communities() if graph is None else graph
    graph_path, nodes_path = tmp_path / "core.graphml", tmp_path / "nodes.json"
    export_graph(graph, graph_path)
    network = build_matrices(graph)
    save_npz(tmp_path / "unsigned.npz", network.unsigned)
    save_npz(tmp_path / "signed.npz", network.signed)
    nodes_path.write_text(json.dumps(network.nodes))
    return graph, graph_path, nodes_path


def saved_inputs(tmp_path):
    """Keep tests away from the real data/processed and results/topology files."""
    return dict(centrality_path=tmp_path / "centrality.csv", metrics_path=tmp_path / "m.json")


def test_pipeline_exports_and_paired_ico(tmp_path):
    graph, graph_path, nodes_path = write_inputs(tmp_path)
    specs = (
        RewiringSpec("reference", None, "unbiased"),
        RewiringSpec("insular_strict", "insular", "strict"),
    )
    kwargs = dict(nodes_path=nodes_path, specs=specs, replicates=2, attempts_per_edge=5, seed=7)
    kwargs.update(saved_inputs(tmp_path))
    summary = run_rewiring_experiment(graph_path, results_dir=tmp_path / "rw", **kwargs)
    again = run_rewiring_experiment(graph_path, results_dir=tmp_path / "rw2", **kwargs)
    pd.testing.assert_frame_equal(summary.drop(columns="path"), again.drop(columns="path"))
    assert summary["name"].tolist() == ["empirical"] + ["reference"] * 2 + ["insular_strict"] * 2
    original = build_matrices(graph)
    for row in summary.iloc[1:].itertuples():
        folder = tmp_path / "rw" / row.path
        unsigned = load_npz(folder / "unsigned.npz")
        assert unsigned.shape == original.unsigned.shape
        assert (np.asarray(unsigned.sum(axis=1)) == np.asarray(original.unsigned.sum(axis=1))).all()
        metadata = json.loads((folder / "metadata.json").read_text())
        assert metadata["validation"]["degree_preserved"]
        degrees = pd.read_csv(folder / "degrees.csv")
        assert (degrees["degree_before"] == degrees["degree_after"]).all()
        scores = pd.read_csv(folder / "centrality.csv")
        assert scores["node"].tolist() == list(original.nodes)
        assert (scores["degree"] == degrees["degree_after"]).all()
    assert not (tmp_path / "rw" / "nodes.json").exists()
    # No saved empirical scores were supplied, so they are computed once and saved.
    assert (tmp_path / "rw" / "graphs" / "empirical" / "centrality.csv").is_file()

    raw, phi_c, comparisons = run_rewired_ico_experiment(
        tmp_path,
        rewired_dir=tmp_path / "rw",
        results_dir=tmp_path / "ico",
        budgets=(0.0, 0.25, 0.5),
        initializations=3,
        seed=1,
        max_steps=50,
        n_bootstrap=50,
    )
    assert set(raw["graph"]) == {"empirical", "reference", "insular_strict"}
    paired = raw.groupby("initialization")["initial_seed"].nunique()
    assert (paired == 1).all()
    assert (raw.groupby(["graph", "replicate"]).size() == 3 * 3 * 2).all()
    assert set(phi_c["strategy"]) == {"degree", "random"}
    assert raw["connected"].all() and "connected" in phi_c
    summary_ico = pd.read_csv(tmp_path / "ico" / "phi_c_summary.csv")
    assert (summary_ico["disconnected_runs"] == 0).all()
    assert summary_ico["h_insular_fraction_of_max"].notna().all()
    # g_minus uses the fixed empirical core for every graph; saved states replay it.
    states = np.load(tmp_path / "ico" / "final_states.npz")
    final = np.unpackbits(states["final"], axis=1, count=int(states["n_nodes"]))
    for row in raw.sample(20, random_state=0).itertuples():
        state = np.where(final[row.state_index] == 1, 1, -1).astype(np.int8)
        gcc = compute_opinion_gcc(graph, state, nodes=original.nodes)
        assert (row.g_minus, row.g_plus) == (gcc["g_minus"], gcc["g_plus"])
    assert len(states["baseline"]) == raw["baseline_index"].nunique()
    assert {"empirical", "reference"} >= set(comparisons["control"])
    for name in ("ico_runs.csv", "phi_c_runs.csv", "phi_c_summary.csv", "metadata.json"):
        assert (tmp_path / "ico" / name).is_file()


def test_pipeline_reuses_saved_empirical_outputs(tmp_path):
    graph = two_communities()
    graph.graph["input_sha256"] = "abc"
    graph, graph_path, nodes_path = write_inputs(tmp_path, graph)
    inputs = saved_inputs(tmp_path)
    nodes = sorted(graph)
    scores = pd.DataFrame(
        {"node": nodes, "degree": [graph.degree(n) for n in nodes], "leverage": 0.0}
    )
    scores.to_csv(inputs["centrality_path"], index=False)
    metrics = {
        "input": {"sha256": "abc"},
        "core": {"edges": graph.number_of_edges()},
        "metrics": {"degree_assortativity": 0.123},
    }
    inputs["metrics_path"].write_text(json.dumps(metrics))
    specs = (RewiringSpec("reference", None, "unbiased"),)
    summary = run_rewiring_experiment(
        graph_path,
        nodes_path=nodes_path,
        results_dir=tmp_path / "rw",
        specs=specs,
        replicates=1,
        attempts_per_edge=2,
        **inputs,
    )
    assert summary.loc[0, "degree_assortativity"] == 0.123
    assert not (tmp_path / "rw" / "graphs" / "empirical").exists()
    metadata = json.loads((tmp_path / "rw" / "metadata.json").read_text())
    assert metadata["empirical_centrality"] == str(inputs["centrality_path"].resolve())

    scores.assign(degree=scores["degree"] + 1).to_csv(inputs["centrality_path"], index=False)
    with pytest.raises(ValueError, match="does not match"):
        run_rewiring_experiment(
            graph_path, nodes_path=nodes_path, results_dir=tmp_path / "rw2", specs=specs, **inputs
        )
    with pytest.raises(ValueError, match="changed since rewiring"):
        run_rewired_ico_experiment(tmp_path, rewired_dir=tmp_path / "rw", initializations=1)


def test_ico_rejects_a_different_node_order(tmp_path):
    _, graph_path, nodes_path = write_inputs(tmp_path)
    run_rewiring_experiment(
        graph_path,
        nodes_path=nodes_path,
        results_dir=tmp_path / "rw",
        specs=(RewiringSpec("reference", None, "unbiased"),),
        replicates=1,
        attempts_per_edge=2,
        **saved_inputs(tmp_path),
    )
    nodes_path.write_text(json.dumps(sorted(json.loads(nodes_path.read_text()), reverse=True)))
    with pytest.raises(ValueError, match="node order"):
        run_rewired_ico_experiment(tmp_path, rewired_dir=tmp_path / "rw", initializations=1)


def test_pipeline_rejects_mismatched_nodes(tmp_path):
    _, graph_path, nodes_path = write_inputs(tmp_path)
    nodes_path.write_text(json.dumps(["other"]))
    with pytest.raises(ValueError, match="node order"):
        run_rewiring_experiment(
            graph_path, nodes_path=nodes_path, results_dir=tmp_path / "rw", **saved_inputs(tmp_path)
        )


def test_rewire_cli(tmp_path):
    _, graph_path, _ = write_inputs(tmp_path)
    # The CLI reads the centrality stage's export beside the graph, as other stages do.
    save_centrality_scores(graph_path)
    command = [sys.executable, "-m", "mdacn", "rewire", "--graph", str(graph_path)]
    command += ["--results-dir", str(tmp_path / "rw"), "--metrics", str(tmp_path / "m.json")]
    command += ["--replicates", "1", "--attempts-per-edge", "2"]
    command += ["--insular-betas", "1", "--integrated-betas", "1", "2"]
    subprocess.run(command, check=True, capture_output=True)
    summary = pd.read_csv(tmp_path / "rw" / "rewiring_summary.csv")
    assert len(summary) == 1 + 1 + (2 + 1) + (2 + 2)
    command = [sys.executable, "-m", "mdacn", "rewire-ico", "--processed-dir", str(tmp_path)]
    command += ["--rewired-dir", str(tmp_path / "rw"), "--results-dir", str(tmp_path / "ico")]
    command += ["--initializations", "1", "--budgets", "0", "0.5", "--max-steps", "20"]
    subprocess.run(command, check=True, capture_output=True)
    assert (tmp_path / "ico" / "phi_c_runs.csv").is_file()
