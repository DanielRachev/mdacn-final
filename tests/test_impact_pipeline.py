import json

import networkx as nx
import pandas as pd
import pytest

from mdacn.impact_pipeline import (
    prepare_impact_study,
    save_impact_reports,
    simulate_impact_study,
)


def prepare_fixture(tmp_path, saved_features):
    graph = nx.path_graph(["a", "b", "c", "d"])
    nx.set_edge_attributes(graph, 1, "sign")
    path = tmp_path / "core.graphml"
    nx.write_graphml(graph, path)
    output = tmp_path / "study"
    features, metadata = saved_features(graph, path)
    prepare_impact_study(
        path,
        results_dir=output,
        features_path=features,
        feature_metadata_path=metadata,
        realizations=2,
        fractions=(1.0,),
        max_steps=10,
    )
    return path, output


def test_prepare_never_simulates_and_freezes_all_core_features(
    tmp_path, monkeypatch, saved_features
):
    import mdacn.impact_pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "iter_node_impacts", lambda *a, **k: pytest.fail("No simulations")
    )
    path, output = prepare_fixture(tmp_path, saved_features)
    assert not (output / "runs.csv").exists()
    plan = json.loads((output / "plan.json").read_text())
    assert plan["n_nodes"] == 4
    assert plan["nodes"] == ["a", "b", "c", "d"]
    assert pd.read_csv(output / "features.csv").node.tolist() == plan["nodes"]
    assert set(plan["frozen_files"]) == {"features.csv"}
    assert not (output / "sample.csv").exists()
    assert not (output / "sample_coverage.csv").exists()
    with pytest.raises(ValueError, match="new or empty"):
        prepare_impact_study(
            path,
            results_dir=output,
            features_path=tmp_path / "features.csv",
            feature_metadata_path=tmp_path / "features.json",
        )
    with (output / "features.csv").open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        simulate_impact_study(output)


def test_changed_graph_is_rejected(tmp_path, saved_features):
    path, output = prepare_fixture(tmp_path, saved_features)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Core graph changed"):
        simulate_impact_study(output)


def test_older_sampled_studies_require_a_new_run(tmp_path, saved_features):
    _, output = prepare_fixture(tmp_path, saved_features)
    path = output / "plan.json"
    plan = json.loads(path.read_text())
    plan["schema_version"] = 1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="run node-impact again"):
        simulate_impact_study(output)


def test_reports_reject_incomplete_core_outcomes(tmp_path, saved_features):
    from mdacn.artifacts import checksum

    _, output = prepare_fixture(tmp_path, saved_features)
    simulate_impact_study(output)
    runs_path = output / "runs.csv"
    runs = pd.read_csv(runs_path)
    runs.loc[runs.node != "d"].to_csv(runs_path, index=False)
    metadata_path = output / "simulation_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["runs_sha256"] = checksum(runs_path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Incomplete simulation output"):
        save_impact_reports(output, bootstrap=10)
    assert not (output / "analysis").exists()


def test_missing_saved_score_is_rejected_without_calculating_it(tmp_path, saved_features):
    from mdacn.artifacts import checksum

    path, _ = prepare_fixture(tmp_path, saved_features)
    features = tmp_path / "features.csv"
    pd.read_csv(features).drop(columns="leverage").to_csv(features, index=False)
    metadata_path = tmp_path / "features.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["features_sha256"] = checksum(features)
    metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / "missing-score"
    with pytest.raises(ValueError, match="Missing required node scores: leverage"):
        prepare_impact_study(
            path,
            results_dir=output,
            features_path=features,
            feature_metadata_path=metadata_path,
        )
    assert not output.exists()


def test_fixture_pipeline_exports_results_and_analysis_without_rerunning(
    tmp_path, monkeypatch, saved_features
):
    import mdacn.impact_pipeline as pipeline

    _, output = prepare_fixture(tmp_path, saved_features)
    simulate_impact_study(output)
    runs = pd.read_csv(output / "runs.csv")
    assert len(runs) == 4 * 2 * 2
    assert runs.groupby(["model", "realization"]).node.nunique().eq(4).all()
    assert not (output / "runs.partial.csv").exists()
    with pytest.raises(ValueError, match="already exists"):
        simulate_impact_study(output)
    monkeypatch.setattr(pipeline, "iter_node_impacts", lambda *a, **k: pytest.fail("No rerun"))
    tables = save_impact_reports(output, bootstrap=10)
    assert tables["node_impact"].set_index("node").loc["a", "temporal_activity"].eq(4).all()
    assert (output / "analysis" / "impact_distribution_f1.png").is_file()
    assert (output / "analysis" / "impact_correlations_f1.png").is_file()
    assert (output / "analysis" / "correlations.csv").is_file()
    assert (output / "analysis" / "regression_diagnostics.csv").is_file()
    metadata = json.loads((output / "analysis" / "metadata.json").read_text())
    assert "signed_bridge_ratio" in metadata["predictors"]
    assert "temporal_activity" in metadata["predictors"]
