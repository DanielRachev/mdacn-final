# Node impact

Run the entire implemented project from the repository root:

```powershell
uv run python -m mdacn run
```

Or, after `setup`, `characterise`, and `centrality`, run `features`, `node-impact`, then `reports`
using `uv run python -m mdacn <stage>`. `features` saves scores; `node-impact`
loads them and records paired interventions; `reports` saves statistics and plots.
Edit [config.py](../src/mdacn/config.py) for paths and settings. `stage_paths()`
names the expected contributor CSV files under `processed_dir`:
`centrality.csv`, `signed_leverage.csv`, `incident_signs.csv`, and
`temporal_activity.csv`. All seven scores must cover every core node.
`centrality` creates the first file. The remaining three require the respective
contributor implementations. Simulations load the saved core matrices and node order.
Reruns overwrite generated stage outputs. `features` merges and validates these
exports; missing scores produce errors. Neither stage calculates replacements.

Every core node is tested separately, including nodes already at +1. Defaults:
`f=0.5`, unsigned/signed models, and 20 paired realizations. Each node uses the same
settled baseline within a model/realization. Nonsteady baselines, including cycles,
are skipped. Intervals describe seed uncertainty on this core.

Scores are stored in `data/processed/node_features.csv`. Under `results/node_impact/`, inspect:

- `config.json`: simulation and bootstrap settings.
- `study/features.csv`, `plan.json`: frozen scores for every core node and provenance.
- `study/runs.csv`: paired impacts and stop reasons.
- `study/analysis/`: node means, correlations, regression diagnostics, intervals, and plots.

At the defaults, `runs.csv` contains `core_nodes * 40` rows, including unsuccessful
outcomes. For 2,896 core nodes, expect 115,840 rows. Rerun `node-impact` to replace
older sampled studies before running `reports`.

The `reports` stage writes two 2560 x 1440 PNGs at the default `f=0.5`:
`impact_distribution_f0.5.png` (box plots and one point per node mean) and
`impact_correlations_f0.5.png` (Pearson/Spearman coefficients with 95% intervals).
