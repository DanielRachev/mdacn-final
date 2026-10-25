import json
from copy import deepcopy

import networkx as nx
import pandas as pd
import pytest

from mdacn.config import impact_settings
from mdacn.impact_workflow import run_node_impact


def fixture_config(tmp_path, saved_features):
    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    graph_path = tmp_path / "core.graphml"
    nx.write_graphml(graph, graph_path)
    features, metadata = saved_features(graph, graph_path)
    config = deepcopy(impact_settings())
    config.update(
        graph=str(graph_path),
        results_dir=str(tmp_path / "output"),
        features=str(features),
        feature_metadata=str(metadata),
    )
    config["simulation"].update(
        fractions=[1.0], models=["signed"], realizations=2, seed=7, max_steps=10
    )
    config["analysis"].update(bootstrap=10, seed=9)
    return config, tmp_path / "output"


def test_one_command_workflow_runs_fixture_and_freezes_settings(tmp_path, saved_features):
    config, output = fixture_config(tmp_path, saved_features)
    original = deepcopy(config)
    assert run_node_impact(config) == output
    assert config == original
    assert json.loads((output / "config.json").read_text()) == original
    plan = json.loads((output / "study/plan.json").read_text())
    assert plan["n_nodes"] == 4
    assert plan["nodes"] == ["a", "b", "c", "d"]
    assert {path.name for path in output.iterdir()} == {"config.json", "study"}
    assert plan["fractions"] == [1.0] and plan["models"] == ["signed"]
    assert plan["seed"] == 7
    assert (output / "study/runs.csv").is_file()
    runs = pd.read_csv(output / "study/runs.csv")
    assert len(runs) == 4 * 2
    assert runs.groupby("realization").node.nunique().eq(4).all()
    assert (output / "study/analysis/correlations.csv").is_file()
    assert (output / "study/analysis/impact_distribution_f1.png").is_file()
    metadata = json.loads((output / "study/analysis/metadata.json").read_text())
    assert metadata["bootstrap"] == 10 and metadata["seed"] == 9
    (output / "study/runs.partial.csv").write_text("old partial run")
    (output / "study/analysis/stale.csv").write_text("old analysis")
    (output / "study/sample.csv").write_text("old sample")
    (output / "study/sample_coverage.csv").write_text("old coverage")
    config["simulation"]["models"] = ["unsigned"]
    config["simulation"]["seed"] = 11
    run_node_impact(config)
    assert json.loads((output / "config.json").read_text()) == config
    assert not (output / "study/runs.partial.csv").exists()
    assert not (output / "study/analysis/stale.csv").exists()
    assert not (output / "study/sample.csv").exists()
    assert not (output / "study/sample_coverage.csv").exists()
    assert (output / "study/analysis/impact_correlations_f1.png").is_file()
    assert json.loads((output / "study/plan.json").read_text())["seed"] == 11


@pytest.mark.parametrize("section", ["power", "sampling"])
def test_obsolete_settings_are_rejected_before_work(tmp_path, saved_features, section):
    config, output = fixture_config(tmp_path, saved_features)
    config[section] = {}
    with pytest.raises(ValueError, match="Invalid root config"):
        run_node_impact(config)
    assert not output.exists()


@pytest.mark.parametrize("issue", ["max_steps", "missing_core"])
def test_invalid_run_preserves_existing_results(tmp_path, saved_features, issue):
    config, output = fixture_config(tmp_path, saved_features)
    if issue == "max_steps":
        config["simulation"]["max_steps"] = 0
    else:
        (tmp_path / "core.graphml").unlink()
    study = output / "study"
    study.mkdir(parents=True)
    old = study / "runs.csv"
    old.write_text("previous results")
    with pytest.raises(ValueError, match="max_steps|Missing input"):
        run_node_impact(config)
    assert old.read_text() == "previous results"


def test_overwrite_rejects_inputs_inside_study(tmp_path, saved_features):
    config, output = fixture_config(tmp_path, saved_features)
    study = output / "study"
    study.mkdir(parents=True)
    graph_path = study / "core.graphml"
    graph_path.write_bytes((tmp_path / "core.graphml").read_bytes())
    config["graph"] = str(graph_path)
    with pytest.raises(ValueError, match="Input files"):
        run_node_impact(config)
    assert graph_path.is_file()


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("analysis", "bootstrap", 1),
        ("simulation", "max_steps", 0),
        ("simulation", "realizations", 0),
        ("simulation", "seed", -1),
    ],
)
def test_bad_settings_fail_before_work(tmp_path, saved_features, section, key, value):
    config, output = fixture_config(tmp_path, saved_features)
    config[section][key] = value
    with pytest.raises(ValueError):
        run_node_impact(config)
    assert not output.exists()


def test_cli_runs_workflow_using_python_config(monkeypatch):
    import mdacn.__main__ as cli

    calls = []
    monkeypatch.setattr(cli, "run_stage", calls.append)
    monkeypatch.setattr("sys.argv", ["mdacn", "node-impact"])
    cli.main()
    assert calls == ["node-impact"]
