"""Core-node associations with paired-seed uncertainty."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mdacn.features import PREDICTORS, validate_features


def _interval(values: np.ndarray, *, enough_runs: bool) -> tuple[float, float, int]:
    """Return 95% percentile bounds over finite bootstrap estimates."""
    finite = np.asarray(values)[np.isfinite(values)]
    if not enough_runs or len(finite) < 2:
        return np.nan, np.nan, len(finite)
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), len(finite)


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def _fit_regression(
    features: pd.DataFrame,
    means: np.ndarray,
    boot_means: np.ndarray,
    context: dict,
    common_runs: int,
    attempted_runs: int,
) -> tuple[list[dict], dict]:
    """Fit OLS with an intercept and standardized predictors."""
    x = features[list(PREDICTORS)].to_numpy(float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(means)
    x, y = x[valid], means[valid]
    n, p = x.shape
    diagnostic = {
        **context,
        "n_nodes": len(features),
        "complete_nodes": n,
        "dropped_nodes": len(features) - n,
        "common_runs": common_runs,
        "attempted_runs": attempted_runs,
        "predictors": ",".join(PREDICTORS),
        "rank": None,
        "condition_number": None,
        "r_squared": None,
        "rmse": None,
        "status": "ok",
    }
    coefficients = []
    if not common_runs:
        diagnostic["status"] = "no_common_steady_runs"
    elif n <= p + 1:
        diagnostic["status"] = "insufficient_nodes"
    elif np.ptp(y) == 0:
        diagnostic["status"] = "constant_impact"
    else:
        centers, scales = x.mean(axis=0), x.std(axis=0)
        design = np.column_stack((np.ones(n), (x - centers) / np.where(scales > 0, scales, 1)))
        rank = int(np.linalg.matrix_rank(design))
        diagnostic["rank"] = rank
        diagnostic["condition_number"] = float(np.linalg.cond(design))
        if rank != p + 1:
            diagnostic["status"] = "rank_deficient"
        else:
            inverse = np.linalg.pinv(design)
            beta = inverse @ y
            residuals = y - design @ beta
            diagnostic["r_squared"] = float(1 - residuals @ residuals / np.sum((y - y.mean()) ** 2))
            diagnostic["rmse"] = float(np.sqrt(np.mean(residuals**2)))
            boot_beta = boot_means[:, valid] @ inverse.T
            for index, term in enumerate(["intercept", *PREDICTORS]):
                low, high, count = _interval(boot_beta[:, index], enough_runs=common_runs >= 2)
                coefficients.append(
                    {
                        **context,
                        "term": term,
                        "coefficient": beta[index],
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_valid": count,
                        "feature_center": 0.0 if index == 0 else centers[index - 1],
                        "feature_scale": 1.0 if index == 0 else scales[index - 1],
                        "n_nodes": n,
                        "common_runs": common_runs,
                    }
                )
    return coefficients, diagnostic


def calculate_impact_statistics(
    runs: pd.DataFrame,
    node_features: pd.DataFrame,
    *,
    bootstrap: int = 1000,
    seed: int = 0,
    outcome: str = "impact",
) -> dict[str, pd.DataFrame]:
    """Calculate statistics using common steady runs and shared seed resampling."""
    if outcome not in runs:
        raise ValueError(f"Missing impact outcome: {outcome}")
    if bootstrap < 2:
        raise ValueError("bootstrap must be at least 2")
    if runs.empty or node_features.empty:
        raise ValueError("Runs and node features must be nonempty")
    if (
        node_features.node.duplicated().any()
        or runs.duplicated(["model", "f", "realization", "node"]).any()
    ):
        raise ValueError("Duplicate core nodes or repeated run keys")
    nodes = node_features.node.tolist()
    if set(runs.node) != set(nodes):
        raise ValueError("Run nodes must match the core feature table")
    features = validate_features(node_features, nodes).set_index("node").reindex(nodes)
    rng = np.random.default_rng(seed)
    summaries, correlations, coefficients, diagnostics = [], [], [], []
    stops = (
        runs.groupby(
            ["model", "f", "baseline_stop_reason", "intervention_stop_reason"], as_index=False
        )
        .size()
        .rename(columns={"size": "node_runs"})
    )
    stops["group_node_runs"] = stops.groupby(["model", "f"]).node_runs.transform("sum")
    stops["fraction"] = stops.node_runs / stops.group_node_runs

    for (model, fraction), group in runs.groupby(["model", "f"], sort=True):
        context = {"model": model, "f": fraction}
        if not group.groupby("realization").node.nunique().eq(len(nodes)).all():
            raise ValueError("Every realization must contain every core node")
        if not group.groupby("realization").initial_seed.nunique().eq(1).all():
            raise ValueError("Nodes in a realization must share an initial seed")
        steady = (group.baseline_stop_reason == "steady_state") & (
            group.intervention_stop_reason == "steady_state"
        )
        steady_values = group.loc[steady, outcome].to_numpy(float)
        if not np.isfinite(steady_values).all() or np.any(np.abs(steady_values) > 1):
            raise ValueError("Steady impacts must be finite GCC differences in [-1, 1]")
        values = (
            group.assign(steady_impact=group[outcome].where(steady))
            .pivot(index="node", columns="realization", values="steady_impact")
            .reindex(nodes)
            .astype(float)
        )
        common = values.dropna(axis=1)
        n_common = common.shape[1]
        common_means = common.mean(axis=1).to_numpy()
        # All node means in a replicate use the SAME resampled initial states.
        boot_means = np.empty((bootstrap, len(nodes)))
        boot_means[:] = np.nan
        if n_common:
            indices = rng.integers(0, n_common, size=(bootstrap, n_common))
            boot_means = common.to_numpy()[:, indices].mean(axis=2).T

        for index, node in enumerate(nodes):
            available = values.loc[node].dropna().to_numpy()
            node_low = node_high = np.nan
            if len(available) >= 2:
                draws = rng.integers(0, len(available), size=(bootstrap, len(available)))
                node_low, node_high, _ = _interval(available[draws].mean(axis=1), enough_runs=True)
            low, high, _ = _interval(boot_means[:, index], enough_runs=n_common >= 2)
            summaries.append(
                {
                    **context,
                    "node": node,
                    "attempted_runs": values.shape[1],
                    "steady_runs": len(available),
                    "mean_impact": float(available.mean()) if len(available) else np.nan,
                    "ci_low": node_low,
                    "ci_high": node_high,
                    "common_runs": n_common,
                    "common_mean_impact": common_means[index],
                    "common_ci_low": low,
                    "common_ci_high": high,
                }
            )

        for predictor in PREDICTORS:
            x = features[predictor].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(common_means)
            for method in ("pearson", "spearman"):
                estimate = _correlation(x[valid], common_means[valid], method)
                draws = np.array([_correlation(x[valid], y[valid], method) for y in boot_means])
                low, high, count = _interval(draws, enough_runs=n_common >= 2)
                correlations.append(
                    {
                        **context,
                        "predictor": predictor,
                        "method": method,
                        "n_nodes": int(valid.sum()),
                        "common_runs": n_common,
                        "coefficient": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_valid": count,
                        "status": "ok" if np.isfinite(estimate) else "undefined",
                    }
                )

        fitted, diagnostic = _fit_regression(
            features,
            common_means,
            boot_means,
            context,
            n_common,
            values.shape[1],
        )
        coefficients.extend(fitted)
        diagnostics.append(diagnostic)

    summary = pd.DataFrame(summaries).merge(node_features, on="node", validate="many_to_one")
    return {
        "node_impact": summary,
        "correlations": pd.DataFrame(correlations),
        "regression": pd.DataFrame(
            coefficients,
            columns=[
                "model",
                "f",
                "term",
                "coefficient",
                "ci_low",
                "ci_high",
                "bootstrap_valid",
                "feature_center",
                "feature_scale",
                "n_nodes",
                "common_runs",
            ],
        ),
        "regression_diagnostics": pd.DataFrame(diagnostics),
        "stop_summary": stops,
    }
