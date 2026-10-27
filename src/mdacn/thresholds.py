"""Paired intervention thresholds, with unsuccessful trials kept explicit."""

import numpy as np
import pandas as pd


def critical_budget_runs(results: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    """Find first steady crossing on each fixed-baseline budget trajectory.

    A missing threshold is right-censored only if all tested outcomes settled.
    Otherwise it is unresolved. Reject historical budget-wise initializations.
    """
    if "g_minus" not in results:
        raise ValueError("Phi_c requires g_minus measurements")
    required = {"budget", "realization", "strategy", "initial_seed", "baseline_g_minus"}
    if missing := required - set(results):
        raise ValueError(f"Paired Phi_c requires {sorted(missing)}; rerun the ICO experiment")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and in (0, 1)")
    keys = [key for key in ("model", "f", "strategy", "realization") if key in results]
    rows = []
    for key, group in results.groupby(keys, sort=True):
        if group.initial_seed.isna().any() or group.initial_seed.nunique() != 1:
            raise ValueError("Phi_c requires one initial seed across budgets; rerun paired ICO")
        if group.budget.duplicated().any():
            raise ValueError("Repeated budget in one ICO trajectory")
        reference = group.baseline_g_minus.to_numpy(float)
        if not np.isfinite(reference).all() or not np.allclose(
            reference, reference[0], rtol=0, atol=1e-12
        ):
            raise ValueError("Phi_c requires the same settled baseline across budgets")
        steady = (
            group.stop_reason.eq("steady_state")
            if "stop_reason" in group
            else pd.Series(True, index=group.index)
        )
        below = bool(reference[0] < threshold)
        crossed = group.loc[steady & group.g_minus.lt(threshold), "budget"]
        reached = below or not crossed.empty
        value = 0.0 if below else float(crossed.min()) if reached else np.nan
        rows.append(
            {
                **dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True)),
                "initial_seed": int(group.initial_seed.iloc[0]),
                "baseline_g_minus": reference[0],
                "reference_below_threshold": below,
                "phi_c": value,
                "threshold_reached": reached,
                "all_steady": bool(steady.all()),
                "max_tested_budget": float(group.budget.max()),
                "status": "reference below threshold"
                if below
                else "reached"
                if reached
                else "right censored"
                if steady.all()
                else "unresolved nonsteady outcomes",
            }
        )
    return pd.DataFrame(rows)


def summarize_thresholds(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Report conditional thresholds together with success and baseline counts."""
    rows = []
    for key, group in runs.groupby(keys, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        reached = group.threshold_reached.astype(bool)
        below = group.reference_below_threshold.astype(bool)
        eligible = ~below
        n, nr, nz, ne = len(group), int(reached.sum()), int(below.sum()), int(eligible.sum())
        intervention = group.loc[eligible & reached, "phi_c"]
        row = {
            **dict(zip(keys, values, strict=True)),
            "runs": n,
            "reached_runs": nr,
            "unreached_runs": n - nr,
            "zero_budget_runs": nz,
            "eligible_runs": ne,
            "intervention_reached_runs": len(intervention),
            "reached_fraction": nr / n,
            "zero_budget_fraction": nz / n,
            "intervention_reached_fraction": len(intervention) / ne if ne else np.nan,
            "mean_phi_c": float(group.phi_c.mean()),
            "median_phi_c": float(group.phi_c.median()),
            "std_phi_c": float(group.phi_c.std()),
            "mean_intervention_phi_c": float(intervention.mean()),
            "median_intervention_phi_c": float(intervention.median()),
            "min_phi_c": float(group.phi_c.min()),
            "max_phi_c": float(group.phi_c.max()),
            "max_tested_budget": float(group.max_tested_budget.max()),
            "all_steady_runs": int(group.all_steady.sum()),
            "censored_runs": int((~reached & group.all_steady).sum()),
            "unresolved_runs": int((~reached & ~group.all_steady).sum()),
            "status": "reference below threshold at Phi = 0"
            if nz == n
            else "not reached in the tested range"
            if nr == 0
            else "partially reached in the tested range"
            if nr < n
            else "reached",
        }
        if "connected" in group:
            row["disconnected_runs"] = int((~group.connected.astype(bool)).sum())
        rows.append(row)
    return pd.DataFrame(rows)
