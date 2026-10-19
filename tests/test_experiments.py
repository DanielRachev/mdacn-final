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
from mdacn.evaluation import compute_opinion_gcc
from mdacn.experiments import run_fraction_sweep, run_intervention_sweep
from mdacn.nco_pipeline import run_nco_experiment
from mdacn.plots import plot_intervention_comparison, plot_opinion_gcc, plot_phase_transition


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
        gcc = compute_opinion_gcc(graph, replay.final_state, nodes=network.nodes)
        assert (row.g_minus, row.g_plus) == (gcc["g_minus"], gcc["g_plus"])


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
    assert len(pd.read_csv(output / "fraction_sweep.csv")) == len(results) == 68
    summary = pd.read_csv(output / "fraction_summary.csv")
    assert summary["runs"].sum() == 68
    assert (summary.groupby(["model", "f"])["run_fraction"].sum() == 1).all()
    assert set(summary) >= {"mean_g_minus", "std_g_minus", "mean_g_plus", "std_g_plus"}
    assert set(summary) >= {"mean_g_minus_second", "mean_g_plus_second"}
    for extension in ("png", "pdf"):
        assert (output / f"phase_transition.{extension}").stat().st_size > 1000
        assert (output / f"opinion_gcc_phase_transition.{extension}").stat().st_size > 1000
        assert (output / f"positive_opinion_gcc_phase_transition.{extension}").stat().st_size > 1000
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
            "g_minus": [0.05, 0.7, 0.3],
            "g_plus": [0.85, 0.1, 0.2],
            "g_minus_second": [0.01, 0.02, 0.005],
            "g_plus_second": [0.01, 0.02, 0.01],
            "n_nodes": [100] * 3,
        }
    )
    plot_phase_transition(results, tmp_path / "statuses.png")
    assert calls == [[0.1], [0.8], [0.4]]
    component_calls = []
    original_plot = Axes.plot

    def capture_components(self, x, y, **kwargs):
        component_calls.append((self, np.asarray(y).tolist()))
        return original_plot(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_components)
    plot_opinion_gcc(results, tmp_path / "gcc_statuses.png")
    assert [values for _, values in component_calls] == [
        [0.05],
        [0.01],
        [5],
        [0.7],
        [0.02],
        [10],
        [0.3],
        [0.005],
        [10],
    ]
    assert len({id(axis) for axis, _ in component_calls}) == 9
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
    np.testing.assert_allclose(results["g_minus"], results["cluster_size"] / 4)
    assert (results.loc[results["budget"] == 1, "g_plus"] == 1).all()
    path = tmp_path / "intervention.png"
    plot_intervention_comparison(results, path)
    assert path.stat().st_size > 1000
    positive_path = tmp_path / "positive_intervention.png"
    plot_intervention_comparison(results, positive_path, opinion="plus")
    assert positive_path.stat().st_size > 1000
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


def test_refined_default_preserves_coarse_sweep_seeds_and_outcomes():
    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    network = build_matrices(graph)
    coarse_fractions = tuple(i / 10 for i in range(1, 10))
    settings = dict(realizations=2, seed=0, max_steps=20)
    coarse = run_fraction_sweep(network, fractions=coarse_fractions, **settings)
    refined = run_fraction_sweep(network, **settings)
    assert len(refined) == 68
    assert set(refined["f"]) - set(coarse_fractions) == {
        0.42,
        0.44,
        0.46,
        0.48,
        0.52,
        0.54,
        0.56,
        0.58,
    }
    pd.testing.assert_frame_equal(coarse, refined.loc[refined["f"].isin(coarse_fractions)])
