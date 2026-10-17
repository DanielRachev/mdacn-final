"""Paired Monte Carlo sweeps and saved phase-transition artifacts."""

import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mdacn.data import aggregate_edges, build_matrices, export_graph
from mdacn.dynamics import initialize_opinions, simulate_nco
from mdacn.experiments import (
    run_fraction_sweep,
    run_intervention_sweep,
    run_sign_shuffle_fraction_sweep,
    shuffle_edge_signs,
)
from mdacn.nco_pipeline import run_nco_experiment, run_sign_shuffle_experiment
from mdacn.plots import (
    plot_intervention_comparison,
    plot_phase_transition,
    plot_sign_shuffle_phase_transition,
)


def test_paired_sweep_reproducibility_and_replay():
    graph = nx.complete_graph(5)
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    kwargs = dict(fractions=(0.1, 0.5, 0.9), realizations=20, seed=12, max_steps=20)
    results = run_fraction_sweep(network, **kwargs)
    pd.testing.assert_frame_equal(results, run_fraction_sweep(network, **kwargs))
    assert len(results) == 120
    paired = results.groupby(["f", "realization"])
    assert (paired["initial_seed"].nunique() == 1).all()
    assert (paired["final_negative_fraction"].nunique() == 1).all()
    assert set(results["stop_reason"]) == {"steady_state"}
    for row in results.itertuples(index=False):
        initial = initialize_opinions(5, row.f, seed=row.initial_seed)
        replay = simulate_nco(network.unsigned, initial, model=row.model, max_steps=20)
        assert row.initial_negative_fraction == np.mean(initial == -1)
        assert row.final_negative_fraction == np.mean(replay.final_state == -1)


def test_signed_divergence_and_nonsteady_statuses():
    graph = nx.complete_graph(3)
    nx.set_edge_attributes(graph, -1, "sign")
    network = build_matrices(graph)
    results = run_fraction_sweep(network, fractions=(0,), realizations=1, max_steps=3)
    assert results["stop_reason"].tolist() == ["steady_state", "limit_cycle"]
    assert results.loc[1, "cycle_length"] == 2
    limited = run_fraction_sweep(network, fractions=(0,), realizations=1, max_steps=1)
    assert limited["stop_reason"].tolist() == ["steady_state", "max_steps"]
    assert limited["final_negative_fraction"].tolist() == [0, 1]


@pytest.mark.parametrize("kwargs", [{"realizations": 0}, {"fractions": ()}, {"fractions": (1.1,)}])
def test_invalid_sweep_rejected(kwargs):
    graph = nx.path_graph(2)
    nx.set_edge_attributes(graph, 1, "sign")
    with pytest.raises(ValueError):
        run_fraction_sweep(build_matrices(graph), **kwargs)


def test_nco_pipeline_and_cli(tmp_path):
    graph = aggregate_edges(
        pd.DataFrame(
            [("a", "b", 1), ("b", "a", -1), ("a", "c", -1), ("b", "c", -1)],
            columns=["source", "target", "sentiment"],
        )
    )
    path = tmp_path / "core.graphml"
    export_graph(graph, path)
    output = tmp_path / "nco"
    results = run_nco_experiment(path, results_dir=output, realizations=2, max_steps=10)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["zero_edges"] == 1
    assert metadata["negative_edges"] == 2
    assert metadata["n_nodes"] == 3
    assert len(metadata["graph_sha256"]) == 64
    assert json.loads((output / "nodes.json").read_text()) == ["a", "b", "c"]
    assert len(pd.read_csv(output / "fraction_sweep.csv")) == len(results) == 36
    summary = pd.read_csv(output / "fraction_summary.csv")
    assert summary["runs"].sum() == 36
    assert (summary.groupby(["model", "f"])["run_fraction"].sum() == 1).all()
    for extension in ("png", "pdf"):
        assert (output / f"phase_transition.{extension}").stat().st_size > 1000
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mdacn",
            "nco",
            "--graph",
            str(path),
            "--results-dir",
            str(tmp_path / "cli"),
            "--realizations",
            "1",
            "--fractions",
            "0.2",
            "0.8",
            "--max-steps",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Saved NCO results" in completed.stdout
    cli_results = pd.read_csv(tmp_path / "cli" / "fraction_sweep.csv")
    assert len(cli_results) == 4
    assert set(cli_results["stop_reason"]) == {"max_steps"}


def test_sign_shuffle_pipeline_and_cli(tmp_path):
    graph = aggregate_edges(
        pd.DataFrame(
            [
                ("a", "b", 1),
                ("b", "a", -1),
                ("a", "c", 1),
                ("c", "a", -1),
                ("b", "c", 1),
                ("c", "b", -1),
            ],
            columns=["source", "target", "sentiment"],
        )
    )
    path = tmp_path / "core.graphml"
    export_graph(graph, path)
    output = tmp_path / "sign_shuffle"
    results = run_sign_shuffle_experiment(
        path, results_dir=output, fractions=(0.2, 0.8), realizations=1, seed=7, max_steps=10
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["null_model"] == "sign_shuffling"
    assert metadata["n_nodes"] == 3
    assert len(pd.read_csv(output / "sign_shuffle_sweep.csv")) == len(results) == 4
    summary = pd.read_csv(output / "sign_shuffle_summary.csv")
    assert summary["runs"].sum() == 4
    assert (output / "sign_shuffle_phase_transition.png").stat().st_size > 1000
    assert (output / "sign_shuffle_phase_transition.pdf").stat().st_size > 1000
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mdacn",
            "sign-shuffle",
            "--graph",
            str(path),
            "--results-dir",
            str(tmp_path / "cli_sign_shuffle"),
            "--fractions",
            "0.2",
            "0.8",
            "--realizations",
            "1",
            "--max-steps",
            "10",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Saved sign-shuffle results" in completed.stdout
    cli_results = pd.read_csv(tmp_path / "cli_sign_shuffle" / "sign_shuffle_sweep.csv")
    assert len(cli_results) == 4
    assert set(cli_results["model"]) == {"empirical", "sign_shuffled"}


def test_plot_separates_stop_statuses(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    calls = []
    original = Axes.errorbar

    def capture(self, x, y, **kwargs):
        calls.append(np.asarray(y).tolist())
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture)
    results = pd.DataFrame(
        {
            "model": ["unsigned"] * 3,
            "f": [0.5] * 3,
            "stop_reason": ["steady_state", "limit_cycle", "max_steps"],
            "final_negative_fraction": [0.1, 0.8, 0.4],
        }
    )
    plot_phase_transition(results, tmp_path / "statuses.png")
    assert calls == [[0.1], [0.8], [0.4]]
    assert not plt.get_fignums()


def test_plot_intervention_comparison_summarises_strategy_curves(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    calls = []
    original = Axes.errorbar

    def capture(self, x, y, **kwargs):
        calls.append((list(x), list(y), kwargs.get("label")))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture)
    results = pd.DataFrame(
        {
            "strategy": ["random", "degree", "leverage", "random", "degree", "leverage"],
            "budget": [0.2, 0.2, 0.2, 0.4, 0.4, 0.4],
            "cluster_size": [10, 8, 6, 4, 2, 1],
            "n_nodes": [20, 20, 20, 20, 20, 20],
            "stop_reason": ["steady_state"] * 6,
        }
    )

    plot_intervention_comparison(results, tmp_path / "intervention.png")

    assert len(calls) == 3
    assert {call[2] for call in calls} == {"Random", "Degree", "Leverage"}
    assert not plt.get_fignums()


def test_sign_shuffle_rejects_empty_graph(tmp_path):
    path = tmp_path / "empty.graphml"
    export_graph(nx.Graph(), path)
    output = tmp_path / "sign_shuffle"

    with pytest.raises(ValueError, match="at least one node"):
        run_sign_shuffle_experiment(path, results_dir=output, fractions=(0.5,), realizations=1)

    assert not output.exists()


@pytest.mark.parametrize("sign", [0.5, -0.5, 1.9, None])
def test_sign_shuffle_rejects_invalid_signs(sign):
    graph = nx.Graph()
    graph.add_edge("a", "b", sign=sign)

    with pytest.raises(ValueError, match="Every edge must have a sign"):
        shuffle_edge_signs(graph)


def test_sign_shuffle_preserves_topology_and_sign_counts():
    graph = nx.path_graph(6)
    signs = [1, -1, 0, 1, -1]
    for (u, v), sign in zip(graph.edges(), signs, strict=True):
        graph[u][v]["sign"] = sign

    shuffled = shuffle_edge_signs(graph, seed=7)

    assert set(shuffled.edges()) == set(graph.edges())
    assert sorted(edge_data["sign"] for _, _, edge_data in shuffled.edges(data=True)) == sorted(
        signs
    )
    assert [edge_data["sign"] for _, _, edge_data in graph.edges(data=True)] == signs


def test_sign_shuffle_fraction_sweep_is_reproducible_and_plots(tmp_path):
    import matplotlib.pyplot as plt

    graph = nx.complete_graph(4)
    signs = [1, -1, 1, -1, 1, -1]
    for (u, v), sign in zip(graph.edges(), signs, strict=True):
        graph[u][v]["sign"] = sign

    kwargs = dict(fractions=(0.2, 0.8), realizations=2, seed=11, max_steps=10)
    results = run_sign_shuffle_fraction_sweep(graph, **kwargs)
    pd.testing.assert_frame_equal(results, run_sign_shuffle_fraction_sweep(graph, **kwargs))
    assert set(results["model"]) == {"empirical", "sign_shuffled"}
    assert len(results) == 8
    assert (results["n_nodes"] == 4).all()
    paired = results.groupby(["f", "realization"])["initial_seed"].nunique()
    assert (paired == 1).all()

    path = tmp_path / "sign_shuffle.png"
    plot_sign_shuffle_phase_transition(results, path)
    assert path.stat().st_size > 1000
    assert not plt.get_fignums()


def test_intervention_sweep_plots_without_extra_columns(tmp_path):
    import matplotlib.pyplot as plt

    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    kwargs = dict(budgets=(0, 0.25, 1), initial_negative_fraction=0.75, realizations=2, seed=3)
    results = run_intervention_sweep(graph, network, **kwargs)
    pd.testing.assert_frame_equal(results, run_intervention_sweep(graph, network, **kwargs))
    assert (results["n_nodes"] == 4).all()
    assert (results.loc[results["budget"] == 1, "cluster_size"] == 0).all()
    path = tmp_path / "intervention.png"
    plot_intervention_comparison(results, path)
    assert path.stat().st_size > 1000
    assert not plt.get_fignums()


@pytest.mark.parametrize("include_steady", [True, False])
def test_intervention_plot_separates_stop_statuses(tmp_path, monkeypatch, include_steady):
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    calls = []
    original = Axes.errorbar

    def capture(self, x, y, **kwargs):
        calls.append((self, np.asarray(y), np.asarray(kwargs["yerr"])))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture)
    results = pd.DataFrame(
        {
            "strategy": ["degree"] * 5,
            "budget": [0.25, 0.25, 0.25, 0.25, 0.5],
            "cluster_size": [0, 2, 2, 4, 0],
            "n_nodes": [4] * 5,
            "stop_reason": [
                "steady_state",
                "steady_state",
                "limit_cycle",
                "max_steps",
                "max_steps",
            ],
        }
    )
    if not include_steady:
        results = results.loc[results["stop_reason"] != "steady_state"]
    plot_intervention_comparison(results, tmp_path / "intervention_statuses.png")
    expected = (
        [[0.25, np.nan], [0.5, np.nan], [1, 0]] if include_steady else [[0.5, np.nan], [1, 0]]
    )
    assert len(calls) == len(expected)
    assert len({id(axis) for axis, _, _ in calls}) == len(expected)
    for (_, values, _), target in zip(calls, expected, strict=True):
        np.testing.assert_allclose(values, target, equal_nan=True)
    if include_steady:
        assert calls[0][2][0] == pytest.approx(np.std([0, 0.5], ddof=1))
    assert not plt.get_fignums()
