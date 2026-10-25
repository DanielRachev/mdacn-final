import matplotlib
import pandas as pd
import pytest

from mdacn import config, impact_pipeline, workflow


@pytest.fixture
def project(tmp_path, monkeypatch):
    matplotlib.use("Agg")
    dataset = tmp_path / "body.tsv"
    pd.DataFrame(
        [(f"n{i}", f"n{i + 1}", 1, "2020-01-01 00:00:00") for i in range(5)],
        columns=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT", "TIMESTAMP"],
    ).to_csv(
        dataset,
        sep="\t",
        index=False,
    )
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
        {
            "core_method": "k_core",
            "k": 1,
            "min_interactions": 1,
            "seed": 0,
        },
    )
    monkeypatch.setattr(
        config,
        "SIMULATION",
        {
            "fractions": [1.0],
            "realizations": 2,
            "seed": 3,
            "max_steps": 10,
        },
    )
    scores = pd.DataFrame(
        {
            "node": [f"n{i}" for i in range(6)],
            "degree": [1, 2, 2, 2, 2, 1],
            "leverage": [-1 / 3, 1 / 6, 0, 0, 1 / 6, -1 / 3],
            "signed_leverage": [0, 1 / 6, 0, 0, 1 / 6, 0],
            "betweenness": [0, 0.4, 0.6, 0.6, 0.4, 0],
            "negative_incident_ratio": [0] * 6,
            "signed_bridge_ratio": [0] * 6,
            "temporal_activity": [1] * 6,
        }
    )
    paths = config.stage_paths()
    paths["processed"].mkdir(parents=True)
    for output, columns in (
        ("signed_leverage", ["signed_leverage"]),
        ("incident_signs", ["negative_incident_ratio", "signed_bridge_ratio"]),
        ("temporal_activity", ["temporal_activity"]),
    ):
        scores[["node", *columns]].to_csv(paths[output], index=False)
    monkeypatch.setattr(config, "ICO", {"budgets": [0.0, 0.25]})
    monkeypatch.setattr(
        config,
        "NODE_IMPACT",
        {
            "simulation": {**config.SIMULATION, "models": ["unsigned", "signed"]},
            "analysis": {"bootstrap": 10, "seed": 0},
        },
    )
    monkeypatch.setattr(
        "mdacn.data.urlopen", lambda *a, **k: pytest.fail("No network in fixture tests")
    )
    return config.stage_paths()


def test_full_pipeline_and_individual_stages_produce_same_saved_runs(project, monkeypatch):
    sources = [project[name] for name in ("signed_leverage", "incident_signs", "temporal_activity")]
    original_files = [path.read_bytes() for path in sources]
    original_scores = pd.concat(
        [pd.read_csv(path).set_index("node") for path in sources], axis=1
    ).reset_index()
    workflow.run_project()
    pd.testing.assert_frame_equal(
        pd.read_csv(project["features"])[original_scores.columns], original_scores, check_like=True
    )
    assert project["features"].is_file() and project["feature_metadata"].is_file()
    assert (project["topology"] / "metrics.json").is_file()
    assert (project["topology"] / "core_layout.csv").is_file()
    artifacts = [
        project["nco"] / "fraction_sweep.csv",
        project["sign-shuffle"] / "sign_shuffle_sweep.csv",
        project["ico"] / "intervention_sweep.csv",
        project["node-impact"] / "study/runs.csv",
    ]
    expected = [pd.read_csv(path) for path in artifacts]
    assert (project["nco"] / "fraction_summary.csv").is_file()
    assert (project["sign-shuffle"] / "sign_shuffle_phase_transition.png").is_file()
    assert (project["node-impact"] / "study/analysis/correlations.csv").is_file()
    for stage in ("nco", "sign-shuffle"):
        for name in ("opinion_gcc_phase_transition", "positive_opinion_gcc_phase_transition"):
            assert (project[stage] / f"{name}.png").is_file()
    for stage in workflow.STAGES[:-1]:
        workflow.run_stage(stage)
    assert [path.read_bytes() for path in sources] == original_files
    assert not (project["nco"] / "fraction_summary.csv").exists()
    assert not (project["node-impact"] / "study/analysis").exists()
    for stage in ("nco", "sign-shuffle"):
        for name in ("opinion_gcc_phase_transition", "positive_opinion_gcc_phase_transition"):
            assert not (project[stage] / f"{name}.png").exists()
    for path, frame in zip(artifacts, expected, strict=True):
        pd.testing.assert_frame_equal(pd.read_csv(path), frame)
    before = [path.read_bytes() for path in artifacts]
    for function in (
        "run_nco_experiment",
        "run_sign_shuffle_experiment",
        "run_node_impact",
        "run_ico_experiment",
        "save_centrality_scores",
        "validate_features",
    ):
        monkeypatch.setattr(
            workflow, function, lambda *a, **k: pytest.fail("Analysis must reuse saved runs")
        )
    workflow.run_stage("reports")
    assert [path.read_bytes() for path in artifacts] == before
    assert (project["nco"] / "fraction_summary.csv").is_file()
    assert (project["node-impact"] / "study/analysis/correlations.csv").is_file()


def test_missing_stage_input_has_actionable_error(project):
    with pytest.raises(ValueError, match="Run characterise first"):
        workflow.run_stage("features")
    with pytest.raises(ValueError, match="Run characterise first"):
        workflow.run_stage("node-impact")
    with pytest.raises(ValueError, match="No saved experiment results"):
        workflow.run_stage("reports")


def test_missing_contributor_scores_error_and_preserve_saved_exports(project):
    workflow.run_stage("characterise")
    workflow.run_stage("centrality")
    workflow.run_stage("features")
    before = project["features"].read_bytes()
    project["centrality"].unlink()
    with pytest.raises(ValueError, match="Missing centrality input: .*centrality.csv"):
        workflow.run_stage("features")
    assert project["features"].read_bytes() == before


def test_analysis_accepts_a_single_completed_experiment_branch(project, monkeypatch):
    workflow.run_stage("characterise")
    workflow.run_stage("nco")
    monkeypatch.setattr(
        workflow, "run_nco_experiment", lambda *a, **k: pytest.fail("No simulation")
    )
    workflow.run_stage("reports")
    assert (project["nco"] / "fraction_summary.csv").is_file()
    assert not project["node-impact"].exists()


def test_stale_features_are_rejected(project, monkeypatch):
    workflow.run_stage("characterise")
    workflow.run_stage("centrality")
    workflow.run_stage("features")
    core = project["core"]
    core.write_bytes(core.read_bytes() + b"\n")
    monkeypatch.setattr(
        impact_pipeline, "simulate_impact_study", lambda *a: pytest.fail("Stale inputs")
    )
    with pytest.raises(ValueError, match="run features again"):
        workflow.run_stage("node-impact")


def test_feature_table_tampering_is_rejected(project):
    workflow.run_stage("characterise")
    workflow.run_stage("centrality")
    workflow.run_stage("features")
    with project["features"].open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="Feature table changed"):
        workflow.run_stage("node-impact")


def test_cli_run_dispatches_entire_pipeline(monkeypatch):
    import mdacn.__main__ as cli

    calls = []
    monkeypatch.setattr(cli, "run_project", lambda: calls.append("run"))
    monkeypatch.setattr("sys.argv", ["mdacn", "run"])
    cli.main()
    assert calls == ["run"]


def test_consumers_reuse_exported_matrices_and_centrality(project, monkeypatch):
    import mdacn.centrality_pipeline as centrality
    import mdacn.data as data

    workflow.run_stage("characterise")
    workflow.run_stage("centrality")
    workflow.run_stage("features")
    source = project["centrality"].read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Consumers must load saved outputs")

    monkeypatch.setattr(data, "build_matrices", forbidden)
    for name in (
        "degree_centrality",
        "betweenness_centrality",
        "eigenvector_centrality",
        "leverage_centrality",
    ):
        monkeypatch.setattr(centrality, name, forbidden)
    for stage in ("nco", "sign-shuffle", "ico", "node-impact"):
        workflow.run_stage(stage)
    assert project["centrality"].read_bytes() == source


@pytest.mark.parametrize("name", ["unsigned", "signed", "nodes", "network_metadata"])
def test_missing_matrices_error_without_rebuilding(project, monkeypatch, name):
    import mdacn.data as data

    workflow.run_stage("characterise")
    project[name].unlink()
    monkeypatch.setattr(data, "build_matrices", lambda *a: pytest.fail("No fallback"))
    with pytest.raises(ValueError, match="Run characterise first"):
        workflow.run_stage("nco")
    assert not project["nco"].exists()


@pytest.mark.parametrize("name", ["unsigned", "signed", "nodes"])
def test_changed_matrix_artifact_is_rejected(project, name):
    workflow.run_stage("characterise")
    with project[name].open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match=f"Saved {name} changed"):
        workflow.run_stage("nco")
    assert not project["nco"].exists()


def test_changed_centrality_is_rejected(project):
    workflow.run_stage("characterise")
    workflow.run_stage("centrality")
    with project["centrality"].open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="Centrality table changed"):
        workflow.run_stage("features")
    assert not project["features"].exists()


def test_ico_sweeps_saved_inputs_and_reports_without_simulation(project, monkeypatch):
    workflow.run_stage("characterise")
    workflow.run_stage("centrality")
    scores = pd.read_csv(project["centrality"])
    assert scores.node.tolist() == [f"n{i}" for i in range(6)]
    assert scores.degree.tolist() == [1, 2, 2, 2, 2, 1]
    assert scores.degree_centrality.tolist() == pytest.approx([0.2, 0.4, 0.4, 0.4, 0.4, 0.2])
    assert {"betweenness", "eigenvector", "leverage"} <= set(scores)
    workflow.run_stage("ico", fractions=[0.0, 1.0])
    path = project["ico"] / "intervention_sweep.csv"
    before = path.read_bytes()
    outcomes = pd.read_csv(path)
    assert len(outcomes) == 2 * 2 * 2 * 3
    assert set(outcomes.f) == {0.0, 1.0}
    assert not (project["ico"] / "intervention_summary.csv").exists()
    monkeypatch.setattr(workflow, "run_ico_experiment", lambda *a, **k: pytest.fail("No rerun"))
    workflow.run_stage("reports")
    assert path.read_bytes() == before
    summary = pd.read_csv(project["ico"] / "intervention_summary.csv")
    assert summary.runs.sum() == len(outcomes)
    assert (project["ico"] / "intervention_comparison_f0.png").is_file()
    assert (project["ico"] / "intervention_comparison_f1.png").is_file()
