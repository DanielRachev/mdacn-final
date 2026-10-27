"""Tests for the combined results report."""

import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from mdacn.combined import run_combined_results


def _write(path, table):
    """Write one small CSV input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _nco_table():
    rows = []
    for model in ("unsigned", "signed"):
        for fraction in (0.5, 0.6):
            for realization in (0, 1):
                rows.append(
                    {
                        "model": model,
                        "f": fraction,
                        "realization": realization,
                        "g_minus": 0.4 + 0.1 * realization,
                        "g_plus": 0.5 - 0.1 * realization,
                        "g_minus_second": 0.0,
                        "g_plus_second": 0.0,
                        "final_negative_fraction": 0.4 + 0.1 * realization,
                        "initial_negative_fraction": fraction,
                        "stop_reason": "steady_state",
                        "n_nodes": 10,
                    }
                )
    return pd.DataFrame(rows)


def _sign_shuffle_table():
    table = _nco_table()
    table["model"] = table["model"].map({"unsigned": "empirical", "signed": "sign_shuffled"})
    table["shuffle_seed"] = 1
    return table


def _ico_table():
    rows = []
    for model in ("unsigned", "signed"):
        for strategy in ("degree", "random"):
            for budget in (0.0, 0.25):
                for realization in (0, 1):
                    baseline = 0.5
                    value = baseline if budget == 0 else (0.1 if strategy == "degree" else 0.45)
                    rows.append(
                        {
                            "model": model,
                            "f": 0.5,
                            "strategy": strategy,
                            "budget": budget,
                            "realization": realization,
                            "g_minus": value,
                            "g_plus": 1 - value,
                            "baseline_g_minus": baseline,
                            "delta_g_minus": value - baseline,
                            "backfired": False,
                            "negative_influence_links": 10,
                            "repelled_neighbours": 0,
                            "repelled_share": 0.01,
                            "stop_reason": "steady_state",
                            "n_nodes": 10,
                            "contrarian_count": 0 if budget == 0 else 3,
                        }
                    )
    return pd.DataFrame(rows)


def _fixture_tree(tmp_path):
    """Create a small results tree with every optional group present."""
    results = tmp_path / "results"
    _write(results / "nco" / "fraction_sweep.csv", _nco_table())
    _write(results / "sign_shuffle" / "sign_shuffle_sweep.csv", _sign_shuffle_table())
    _write(results / "ico" / "intervention_sweep.csv", _ico_table())
    (results / "ico" / "metadata.json").write_text(json.dumps({"phi_c_threshold": 0.5}) + "\n")
    _write(
        results / "rewiring" / "rewiring_summary.csv",
        pd.DataFrame(
            {
                "name": ["reference", "reference", "insular_strict", "insular_strict"],
                "replicate": [0, 1, 0, 1],
                "connected": [True] * 4,
                "h_insular_fraction_of_max": [0.2, 0.21, 0.8, 0.79],
                "h_integrated_fraction_of_max": [0.3, 0.31, 0.1, 0.11],
                "fraction_changed": [0.5, 0.51, 0.6, 0.59],
                "acceptance_rate": [0.2, 0.21, 0.1, 0.11],
                "attempts": [100] * 4,
                "converged": [True] * 4,
            }
        ),
    )
    _write(
        results / "rewiring_ico" / "phi_c_summary.csv",
        pd.DataFrame(
            {
                "graph": ["reference", "insular_strict"],
                "strategy": ["degree", "degree"],
                "mean_phi_c": [0.05, 0.25],
                "min_phi_c": [0.01, 0.1],
                "max_phi_c": [0.1, 0.25],
            }
        ),
    )
    _write(
        results / "node_impact" / "study" / "analysis" / "node_impact.csv",
        pd.DataFrame(
            {
                "model": ["signed", "signed"],
                "f": [0.5, 0.5],
                "node": ["a", "b"],
                "mean_impact": [0.1, 0.2],
                "ci_low": [0.0, 0.1],
                "ci_high": [0.2, 0.3],
            }
        ),
    )
    _write(
        results / "node_impact" / "study" / "analysis" / "correlations.csv",
        pd.DataFrame(
            {
                "model": ["signed", "signed"],
                "f": [0.5, 0.5],
                "predictor": ["degree", "leverage"],
                "method": ["pearson", "pearson"],
                "coefficient": [0.3, 0.4],
                "ci_low": [0.1, 0.2],
                "ci_high": [0.5, 0.6],
                "status": ["ok", "ok"],
            }
        ),
    )
    _write(
        results / "node_impact" / "study" / "features.csv",
        pd.DataFrame({"node": ["a", "b"], "degree": [1, 2], "leverage": [0.1, 0.2]}),
    )
    _write(
        results / "sensitivity" / "headline.csv",
        pd.DataFrame({"configuration": ["k_core_10"], "nodes": [10], "edges": [20]}),
    )
    _write(
        results / "sensitivity" / "strategy_ranking.csv",
        pd.DataFrame(
            {
                "configuration": ["k_core_10"],
                "model": ["signed"],
                "budget": [0.05],
                "rank": [1],
                "strategy": ["degree"],
                "mean_delta_g_minus": [-0.5],
            }
        ),
    )
    temporal = tmp_path / "temporal"
    _write(
        temporal / "summary.csv",
        pd.DataFrame(
            {
                "f": [0.5],
                "variant": ["temporal_real"],
                "metric": ["minority_fraction"],
                "mean": [0.2],
                "std": [0.01],
                "n_initializations": [5],
                "n_shuffles": [0],
            }
        ),
    )
    _write(
        temporal / "paired_static_effects.csv",
        pd.DataFrame(
            {
                "f": [0.5],
                "metric": ["minority_fraction"],
                "difference": [0.01],
                "ci_low": [0.0],
                "ci_high": [0.02],
            }
        ),
    )
    _write(
        temporal / "paired_shuffle_effects.csv",
        pd.DataFrame(
            {
                "f": [0.5],
                "metric": ["minority_fraction"],
                "difference": [0.005],
                "ci_low": [0.0],
                "ci_high": [0.01],
            }
        ),
    )
    return results, temporal


def test_combined_report_writes_available_sections(tmp_path):
    results, temporal = _fixture_tree(tmp_path)
    output = tmp_path / "combined"

    tables = run_combined_results(
        results,
        output_dir=output,
        temporal_dir=temporal,
        fraction=0.5,
    )

    for name in (
        "inputs.json",
        "warnings.txt",
        "phase_transitions.csv",
        "raw_fractions.csv",
        "stop_reasons.csv",
        "ico_summary.csv",
        "phi_c_by_strategy.csv",
        "backfire_summary.csv",
        "rewiring_by_config.csv",
        "rewiring_phi_c.csv",
        "temporal_summary.csv",
        "temporal_paired_effects.csv",
        "impact_node_table.csv",
        "impact_correlations.csv",
        "sensitivity_headline.csv",
        "sensitivity_strategy_ranking.csv",
        "phase_transition_g_minus.png",
        "ico_g_minus.png",
        "ico_delta.png",
        "phi_c_by_strategy.png",
        "temporal_effects.png",
        "impact_correlations.png",
    ):
        assert (output / name).is_file(), name

    assert len(tables["phase_transitions"]) == 8
    assert len(tables["ico_summary"]) == 8
    assert len(tables["phi_c"]) == 4
    assert len(tables["temporal_effects"]) == 2
    assert "degree" in tables["impact"].columns
    manifest = json.loads((output / "inputs.json").read_text())
    assert "nco" in manifest["available"]
    assert "ico" in manifest["available"]
    assert "extraction_comparison" in manifest["missing"]


def test_combined_report_warns_for_missing_sections(tmp_path):
    results = tmp_path / "results"
    _write(results / "nco" / "fraction_sweep.csv", _nco_table())
    output = tmp_path / "combined"

    tables = run_combined_results(
        results,
        output_dir=output,
        temporal_dir=tmp_path / "no_temporal",
    )

    assert not tables["phase_transitions"].empty
    manifest = json.loads((output / "inputs.json").read_text())
    for name in ("sign_shuffle", "ico", "node_impact", "rewiring"):
        assert name in manifest["missing"]
        assert f"missing {name}" in (output / "warnings.txt").read_text()


def test_combined_report_requires_at_least_one_input(tmp_path):
    with pytest.raises(ValueError, match="No combined inputs"):
        run_combined_results(
            tmp_path / "empty",
            output_dir=tmp_path / "combined",
            temporal_dir=tmp_path / "no_temporal",
        )
