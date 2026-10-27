"""Null-model and signed-structure stages through the config-driven CLI and workflow."""

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from mdacn import __main__ as cli
from mdacn import config, workflow

ANALYSES = {
    "incident-signs": "run_incident_sign_analysis",
    "signed-triads": "run_signed_triad_analysis",
    "temporal-balance": "run_temporal_balance_analysis",
    "directed-roles": "run_directed_role_analysis",
    "label-audit": "run_label_audit",
}


def _custom_paths(monkeypatch) -> None:
    monkeypatch.setitem(config.PATHS, "dataset", "custom/body.tsv")
    monkeypatch.setitem(config.PATHS, "processed_dir", "custom/processed")
    monkeypatch.setitem(config.PATHS, "results_dir", "custom/results")
    monkeypatch.setattr(workflow, "_require", lambda *args: None)


@pytest.mark.parametrize("stage", list(ANALYSES))
def test_signed_stages_use_config_defaults(monkeypatch, stage):
    _custom_paths(monkeypatch)
    calls = []
    monkeypatch.setattr(workflow, ANALYSES[stage], lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr("sys.argv", ["mdacn", stage])
    cli.main()
    args, options = calls[0]
    reads_events = workflow.SIGNED_ANALYSES[stage][1]
    graph = Path("custom/processed/core.graphml")
    assert args == ((Path("custom/body.tsv"), graph) if reads_events else (graph,))
    settings = workflow.SIGNED_ANALYSES[stage][0]
    expected = {"results_dir": config.stage_paths()[stage], **getattr(config, settings or "", {})}
    if stage == "signed-triads":
        expected["rewired_dir"] = Path("custom/results/rewiring/graphs/reference")
    if stage == "directed-roles":
        expected["feature_path"] = Path("custom/processed/incident_signs.csv")
    assert options == expected


def test_signed_stage_options_override_config(monkeypatch):
    _custom_paths(monkeypatch)
    calls = []
    monkeypatch.setattr(workflow, "run_incident_sign_analysis", lambda *a, **k: calls.append(k))
    argv = ["mdacn", "incident-signs", "--realizations", "7", "--bins", "quantile"]
    argv += ["--n-bins", "4", "--graph", "other/core.graphml", "--results-dir", "out"]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    assert calls[0] == {
        "results_dir": Path("out"),
        "realizations": 7,
        "seed": 0,
        "bins": "quantile",
        "bin_width": 0.025,
        "n_bins": 4,
    }


@pytest.mark.parametrize("override", [False, True])
def test_rewiring_stages_use_config_defaults(monkeypatch, override):
    _custom_paths(monkeypatch)
    calls = []
    monkeypatch.setattr(workflow, "run_rewiring_experiment", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        workflow, "run_rewired_ico_experiment", lambda *a, **k: calls.append((a, k))
    )
    rewire = ["mdacn", "rewire"] + (
        ["--replicates", "2", "--insular-betas", "3"] if override else []
    )
    monkeypatch.setattr("sys.argv", rewire)
    cli.main()
    ico = ["mdacn", "rewire-ico"] + (
        ["--strategies", "leverage", "--budgets", "0", "0.1"] if override else []
    )
    monkeypatch.setattr("sys.argv", ico)
    cli.main()
    (rewire_args, rewire_options), (ico_args, ico_options) = calls
    assert rewire_args == ico_args == (Path("custom/processed/core.graphml"),)
    assert rewire_options["replicates"] == (2 if override else config.REWIRING["replicates"])
    insular = {spec.beta for spec in rewire_options["specs"] if spec.rule == "metropolis"}
    integrated = set(config.REWIRING["betas"]["integrated"])
    expected_betas = {3.0} if override else set(config.REWIRING["betas"]["insular"])
    assert insular == expected_betas | integrated
    assert rewire_options["metrics_path"] == Path("custom/results/topology/metrics.json")
    assert ico_options["rewired_dir"] == Path("custom/results/rewiring")
    assert ico_options["strategies"] == (("leverage",) if override else ("degree", "random"))
    assert ico_options["budgets"] == (
        (0.0, 0.1) if override else tuple(config.REWIRED_ICO["budgets"])
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A small event log with triangles, a multi-target post, and text features."""
    matplotlib.use("Agg")
    rows = [
        ("a", "b", "p1", 1, 1),
        ("a", "c", "p1", 1, 1),
        ("b", "c", "p2", 2, -1),
        ("c", "d", "p3", 3, -1),
        ("d", "e", "p4", 4, 1),
        ("e", "f", "p5", 5, 1),
        ("d", "f", "p6", 6, -1),
        ("a", "d", "p7", 7, 1),
        ("b", "a", "p8", 8, 1),
        ("f", "e", "p9", 9, -1),
    ]
    dataset = tmp_path / "body.tsv"
    pd.DataFrame(
        [
            (source, target, post, f"2015-01-{day:02d} 00:00:00", sign, f"{day}.0,1.0")
            for source, target, post, day, sign in rows
        ],
        columns=[
            "SOURCE_SUBREDDIT",
            "TARGET_SUBREDDIT",
            "POST_ID",
            "TIMESTAMP",
            "LINK_SENTIMENT",
            "PROPERTIES",
        ],
    ).to_csv(dataset, sep="\t", index=False)
    monkeypatch.setattr(
        config,
        "PATHS",
        {
            "dataset": str(dataset),
            "processed_dir": str(tmp_path / "processed"),
            "results_dir": str(tmp_path / "results"),
        },
    )
    monkeypatch.setattr(
        config,
        "CHARACTERISATION",
        {"core_method": "k_core", "k": 1, "min_interactions": 1, "seed": 0},
    )
    monkeypatch.setattr(
        config, "REWIRING", {**config.REWIRING, "replicates": 1, "attempts_per_edge": 5}
    )
    monkeypatch.setattr(config, "REWIRED_ICO", {**config.REWIRED_ICO, "initializations": 2})
    monkeypatch.setattr(config, "INCIDENT_SIGNS", {**config.INCIDENT_SIGNS, "realizations": 20})
    monkeypatch.setattr(
        config, "SIGNED_TRIADS", {**config.SIGNED_TRIADS, "realizations": 20, "er_realizations": 2}
    )
    monkeypatch.setattr(config, "TEMPORAL_BALANCE", {**config.TEMPORAL_BALANCE, "realizations": 3})
    return config.stage_paths()


def test_contributor_stages_run_end_to_end(project):
    workflow.run_stage("characterise")
    workflow.run_stage("centrality")
    with pytest.raises(ValueError, match="Run directed-roles first"):
        workflow.run_stage("features")
    for stage in ("rewire", "rewire-ico", *ANALYSES):
        workflow.run_stage(stage)
    assert (project["rewire"] / "rewiring_summary.csv").is_file()
    assert (project["rewire-ico"] / "phi_c_runs.csv").is_file()
    triads = pd.read_csv(project["signed-triads"] / "triad_realizations.csv")
    assert "degree_preserving_rewiring" in set(triads["model"])
    assert (project["temporal-balance"] / "prediction_metrics.csv").is_file()
    assert (project["label-audit"] / "label_audit.json").is_file()
    scores = pd.read_csv(project["incident_signs"])
    assert scores.columns.tolist() == ["node", "negative_incident_ratio", "signed_bridge_ratio"]
    assert scores["node"].tolist() == list("abcdef")
    with pytest.raises(ValueError, match="contributor's export stage"):
        workflow.run_stage("features")


def test_signed_stage_needs_the_core(project):
    with pytest.raises(ValueError, match="Run characterise first"):
        workflow.run_stage("incident-signs")
    with pytest.raises(ValueError, match="Run characterise first"):
        workflow.run_stage("rewire")
