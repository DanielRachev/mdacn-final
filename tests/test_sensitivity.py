"""Tests for the extraction sensitivity comparison."""

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import pytest

from mdacn.pipeline import characterise_core
from mdacn.sensitivity import run_sensitivity


def _write_core(path, edges):
    """Export one tiny core with signs and interaction counts."""
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d", "e"])
    for source, target, sign in edges:
        graph.add_edge(
            source,
            target,
            sign=sign,
            n_positive=1 if sign == 1 else 0,
            n_negative=1 if sign == -1 else 0,
        )
    for node in graph:
        graph.nodes[node]["n_interactions"] = sum(
            data["n_positive"] + data["n_negative"] for _, _, data in graph.edges(node, data=True)
        )
    characterise_core(graph, seed=0, processed_dir=path)


@pytest.fixture
def extraction(tmp_path):
    """Return two exported tiny cores with the expected files."""
    edges = [
        ("a", "b", 1),
        ("b", "c", -1),
        ("c", "d", 1),
        ("d", "e", -1),
        ("e", "a", 1),
        ("a", "c", -1),
    ]
    base = tmp_path / "extraction"
    for name in ("k_core_8", "activity_lcc_100"):
        _write_core(base / name, edges)
    return base


def test_sensitivity_writes_summaries_for_every_configuration(extraction, tmp_path):
    results = run_sensitivity(
        extraction,
        results_dir=tmp_path / "results" / "sensitivity",
        configurations=("k_core_8", "activity_lcc_100"),
        fractions=(0.0, 0.5),
        ico_fractions=(0.5,),
        models=("unsigned", "signed"),
        strategies=("random", "degree"),
        budgets=(0.0, 0.25),
        realizations=1,
        seed=0,
        max_steps=50,
        phi_c_threshold=0.5,
    )

    output = tmp_path / "results" / "sensitivity"
    for name in (
        "topology.csv",
        "nco_runs.csv",
        "ico_runs.csv",
        "nco_summary.csv",
        "ico_summary.csv",
        "phi_c.csv",
        "headline.csv",
        "strategy_ranking.csv",
        "metadata.json",
        "nco_g_minus_by_configuration.png",
        "ico_degree_g_minus_by_configuration.png",
        "phi_c_by_configuration.png",
    ):
        assert (output / name).is_file(), name

    assert set(results["topology"]["configuration"]) == {"k_core_8", "activity_lcc_100"}
    assert set(results["nco_summary"]["configuration"]) == {"k_core_8", "activity_lcc_100"}
    assert set(results["ico_summary"]["strategy"]) == {"random", "degree"}
    assert results["headline"]["configuration"].tolist() == ["k_core_8", "activity_lcc_100"]
    assert set(results["strategy_ranking"]["strategy"]) == {"random", "degree"}
    assert results["strategy_ranking"]["rank"].min() == 1


def test_sensitivity_names_the_comparison_command_when_a_core_is_missing(tmp_path):
    with pytest.raises(ValueError, match="compare-extractions"):
        run_sensitivity(
            tmp_path / "missing",
            results_dir=tmp_path / "results",
            configurations=("k_core_8",),
            fractions=(0.0,),
            ico_fractions=(0.5,),
            models=("unsigned",),
            strategies=("random",),
            budgets=(0.0, 0.25),
            realizations=1,
            seed=0,
            max_steps=10,
            phi_c_threshold=0.5,
        )
