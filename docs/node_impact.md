# Node impact

Run the entire implemented project from the repository root:

```powershell
uv run python -m mdacn run
```

Or, after `setup` and `characterise`, run `features`, `node-impact`, then `reports`
using `uv run python -m mdacn <stage>`. `features` saves scores; `node-impact`
loads them and records paired interventions; `reports` saves statistics and plots.
Edit [config.py](../src/mdacn/config.py) for paths and settings. `stage_paths()`
names the expected contributor CSV files under `processed_dir`:
`centrality.csv`, `signed_leverage.csv`, `incident_signs.csv`, and
`temporal_activity.csv`. All seven scores must cover every core node.
Reruns overwrite generated stage outputs. `features` merges and validates these
exports; missing scores produce errors. Neither stage calculates replacements.

Defaults: target `|r|=0.3`, power 80%, family alpha 5%, fourteen primary Pearson tests
(seven scores x two models), `f=0.5`, and 20 realizations. Adjust the comparison
count if the primary test family changes. Power gives a reference budget under
independence assumptions, not validated network power; associations remain exploratory.
Any sampled node can be pinned to +1. Nonsteady baselines, including cycles, are skipped.

Scores are stored in `data/processed/node_features.csv`. Under `results/node_impact/`, inspect:

- `config.json`: run settings, including power-analysis inputs.
- `study/sample.csv`, `sample_coverage.csv`, `plan.json`: selected nodes and provenance.
- `study/runs.csv`: paired impacts and stop reasons.
- `study/analysis/`: node means, correlations, regression diagnostics, intervals, and plots.

The `reports` stage writes two 2560 x 1440 PNGs at the default `f=0.5`:
`impact_distribution_f0.5.png` (box plots and one point per node mean) and
`impact_correlations_f0.5.png` (Pearson/Spearman coefficients with 95% intervals).
