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
from mdacn.experiments import run_fraction_sweep
from mdacn.nco_pipeline import run_nco_experiment
from mdacn.plots import plot_phase_transition


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
