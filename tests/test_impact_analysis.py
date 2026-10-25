import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

from mdacn.features import PREDICTORS
from mdacn.impact_analysis import calculate_impact_statistics


def fixture_tables(n=16):
    rng = np.random.default_rng(13)
    features = pd.DataFrame(rng.uniform(size=(n, len(PREDICTORS))), columns=PREDICTORS)
    features.insert(0, "node", [f"n{i}" for i in range(n)])
    beta = np.array([0.2, -0.1, 0.05, 0.03, -0.07, 0.04, -0.02])
    means = features[list(PREDICTORS)].to_numpy() @ beta
    runs = pd.DataFrame(
        [
            {
                "node": node,
                "model": "signed",
                "f": 0.5,
                "realization": r,
                "initial_seed": r + 10,
                "impact": means[i] * (1 + 0.1 * r),
                "baseline_stop_reason": "steady_state",
                "intervention_stop_reason": "steady_state",
            }
            for r in range(4)
            for i, node in enumerate(features.node)
        ]
    )
    return features, runs, beta


def test_associations_and_standardized_ols_match_known_linear_relation():
    features, runs, beta = fixture_tables()
    tables = calculate_impact_statistics(runs, features, bootstrap=100, seed=8)
    expected = beta * features[list(PREDICTORS)].std(ddof=0).to_numpy() * 1.15
    fit = tables["regression"].set_index("term")
    np.testing.assert_allclose(fit.loc[list(PREDICTORS), "coefficient"], expected, atol=1e-12)
    assert tables["regression_diagnostics"].iloc[0].r_squared == pytest.approx(1)
    assert tables["regression_diagnostics"].iloc[0].complete_nodes == len(features)
    assert (fit.ci_low <= fit.coefficient).all()
    assert (fit.ci_high >= fit.coefficient).all()
    means = runs.groupby("node").impact.mean().reindex(features.node)
    corr = tables["correlations"].query("predictor == 'degree'").set_index("method")
    assert corr.loc["pearson", "coefficient"] == pytest.approx(
        pearsonr(features.degree, means).statistic
    )
    assert corr.loc["spearman", "coefficient"] == pytest.approx(
        spearmanr(features.degree, means).statistic
    )
    again = calculate_impact_statistics(runs, features, bootstrap=100, seed=8)
    pd.testing.assert_frame_equal(tables["correlations"], again["correlations"])


def test_analysis_uses_common_seeds_and_keeps_node_specific_coverage():
    features, runs, _ = fixture_tables()
    mask = (runs.node == "n0") & (runs.realization == 0)
    runs.loc[mask, "intervention_stop_reason"] = "limit_cycle"
    runs.loc[mask, "impact"] = 999  # Must not enter either steady estimate.
    tables = calculate_impact_statistics(runs, features, bootstrap=30)
    summary = tables["node_impact"].set_index("node")
    assert summary.common_runs.eq(3).all()
    assert summary.loc["n0", "steady_runs"] == 3
    assert summary.loc["n1", "steady_runs"] == 4
    assert summary.mean_impact.abs().max() < 1
    expected = runs.loc[(runs.node == "n1") & (runs.realization > 0), "impact"].mean()
    assert summary.loc["n1", "common_mean_impact"] == pytest.approx(expected)
    assert tables["stop_summary"].node_runs.sum() == len(runs)


def test_seed_bootstrap_preserves_shared_baseline_dependence():
    features, runs, _ = fixture_tables()
    # Add a common random offset. Shared resampling must not change fitted slopes.
    means = runs.loc[runs.realization == 0].set_index("node").impact
    runs["impact"] = runs.node.map(means) + runs.realization * 0.05
    fit = calculate_impact_statistics(runs, features, bootstrap=50)["regression"].set_index("term")
    np.testing.assert_allclose(
        fit.loc[list(PREDICTORS), "ci_low"], fit.loc[list(PREDICTORS), "ci_high"], atol=1e-12
    )
    assert fit.loc["intercept", "ci_high"] > fit.loc["intercept", "ci_low"]


@pytest.mark.parametrize(
    "issue,status",
    [
        ("nonsteady", "no_common_steady_runs"),
        ("constant", "constant_impact"),
        ("collinear", "rank_deficient"),
    ],
)
def test_undefined_analysis_is_explicit(issue, status):
    features, runs, _ = fixture_tables()
    if issue == "nonsteady":
        runs["baseline_stop_reason"] = "limit_cycle"
        runs["intervention_stop_reason"] = "not_run"
        runs["impact"] = np.nan
    elif issue == "constant":
        runs["impact"] = 0.0
    else:
        features["leverage"] = features.degree
    tables = calculate_impact_statistics(runs, features, bootstrap=10)
    assert tables["regression_diagnostics"].iloc[0].status == status
    assert tables["regression"].empty
    assert "coefficient" in tables["regression"]


def test_one_realization_has_no_uncertainty_interval():
    features, runs, _ = fixture_tables()
    tables = calculate_impact_statistics(runs.loc[runs.realization == 0], features, bootstrap=10)
    assert tables["node_impact"].common_ci_low.isna().all()
    assert tables["correlations"].ci_low.isna().all()
    assert tables["regression"].ci_low.isna().all()


def test_analysis_rejects_duplicate_or_missing_pairs():
    features, runs, _ = fixture_tables()
    with pytest.raises(ValueError, match="Duplicate"):
        calculate_impact_statistics(pd.concat([runs, runs.iloc[[0]]]), features)
    with pytest.raises(ValueError, match="Every realization"):
        calculate_impact_statistics(runs.iloc[1:], features)


@pytest.mark.parametrize("column", ["signed_bridge_ratio", "temporal_activity"])
@pytest.mark.parametrize("issue", ["missing", "nan"])
def test_analysis_requires_all_scores(column, issue):
    features, runs, _ = fixture_tables()
    if issue == "missing":
        features = features.drop(columns=column)
    else:
        features[column] = np.nan
    with pytest.raises(ValueError, match=column):
        calculate_impact_statistics(runs, features, bootstrap=10)
