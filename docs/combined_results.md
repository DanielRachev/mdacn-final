# Combined results

This stage assembles saved experiment results into one report and one set of
figures. It does not rerun simulations. Missing inputs are reported in
`inputs.json` and `warnings.txt`, and the remaining sections are still written.

## Inputs

| Input | Produced by | Path | Required |
| --- | --- | --- | --- |
| NCO sweep | `nco` | `results/nco/fraction_sweep.csv` | No, but needed for the phase-transition curves |
| Sign-shuffle sweep | `sign-shuffle` | `results/sign_shuffle/sign_shuffle_sweep.csv` | No |
| ICO sweep | `ico` | `results/ico/intervention_sweep.csv` | No, but needed for the budget and `Phi_c` figures |
| ICO metadata | `ico` | `results/ico/metadata.json` | No |
| Rewiring summary | `rewire` (PR 13) | `results/rewiring/rewiring_summary.csv` | No |
| Rewiring `Phi_c` | `rewire-ico` (PR 13) | `results/rewiring_ico/phi_c_summary.csv` | No |
| Rewiring comparisons | `rewire-ico` (PR 13) | `results/rewiring_ico/g_minus_comparisons.csv` | No |
| Node impact | `node-impact` | `results/node_impact/study/analysis/node_impact.csv` | No |
| Impact correlations | `node-impact` | `results/node_impact/study/analysis/correlations.csv` | No |
| Impact regression | `node-impact` | `results/node_impact/study/analysis/regression.csv` | No |
| Node features | `features` | `results/node_impact/study/features.csv` | No |
| Temporal summary | Antreas's scripts | `reports/static_temporal/<timestamp>/summary.csv` | No |
| Temporal paired effects | Antreas's scripts | `reports/static_temporal/<timestamp>/paired_*.csv` | No |
| Sensitivity headline | `sensitivity` | `results/sensitivity/headline.csv` | No |
| Sensitivity ranking | `sensitivity` | `results/sensitivity/strategy_ranking.csv` | No |
| Filtering stages | `label-audit` (PR 13) | `results/label_audit/filtering_stages.csv` | No |
| Sign fractions | `directed-roles` (PR 13) | `results/directed_roles/sign_fractions.csv` | No |
| Incident-sign summary | `incident-signs` | `results/incident_signs/r_neg_summary.csv` | No |

Run each producing stage first when its output is wanted:

```bash
uv run python -m mdacn nco
uv run python -m mdacn sign-shuffle
uv run python -m mdacn ico --fractions 0.3 0.4 0.5 0.6 0.7
```

The temporal directory is discovered from the latest
`reports/static_temporal/*/summary.csv` when `--temporal-dir` is not supplied.
Pass `--temporal-dir` to pin a specific run.

## Command

```bash
uv run python -m mdacn combined-results
```

Useful overrides:

```bash
uv run python -m mdacn combined-results --temporal-dir reports/static_temporal/20261026_120000
uv run python -m mdacn combined-results --results-dir results --output-dir results/combined
```

## Outputs

Under `results/combined/`:

| File | Contents |
| --- | --- |
| `inputs.json` | Every expected input, whether it was present, row count, columns, and checksum |
| `warnings.txt` | One line per missing input |
| `phase_transitions.csv` | Mean, spread, and 2.5 to 97.5 percentile range of `g_minus` and `g_plus` by source, model, and `f` |
| `raw_fractions.csv` | Raw negative and positive opinion fractions with bands |
| `stop_reasons.csv` | Steady-state, cycle, and update-limit counts by stage, model, and fraction |
| `ico_summary.csv` | Opposing cluster size, paired delta, backfire share, and negative influence links by model, fraction, strategy, and budget |
| `phi_c_by_strategy.csv` | Per-run and aggregated critical budget |
| `backfire_summary.csv` | Worst backfire share and repelled-neighbour share by model, fraction, and strategy |
| `temporal_summary.csv` | Antreas's mean temporal outcomes |
| `temporal_paired_effects.csv` | Temporal-versus-static and temporal-versus-shuffled paired effects |
| `impact_node_table.csv` | Node impact joined with the saved node scores |
| `impact_correlations.csv` | Correlation coefficients and intervals from the node-impact analysis |
| `impact_regression.csv` | Regression coefficients and intervals |
| `rewiring_by_config.csv` | Mean achieved homophily and changed-edge share per rewired configuration |
| `rewiring_phi_c.csv` | `Phi_c` per rewired graph and strategy |
| `rewiring_g_minus_comparisons.csv` | Paired `g_minus` differences against empirical and reference graphs |
| `sensitivity_headline.csv` | Compact per-core sensitivity table |
| `sensitivity_strategy_ranking.csv` | Sensitivity strategy ranking at selected budgets |
| `context_extraction_comparison.csv` | Extraction comparison, when available |
| `context_filtering_stages.csv` | Label-audit filtering stages, when available |
| `context_directed_roles.csv` | Directed-role sign fractions, when available |
| `context_incident_signs.csv` | Incident-sign summary, when available |

Figures:

| File | Contents |
| --- | --- |
| `phase_transition_g_minus.png` | Opposing cluster fraction against `f` |
| `phase_transition_g_plus.png` | Positive cluster fraction against `f` |
| `raw_fractions.png` | Raw negative and positive opinion fractions |
| `ico_g_minus.png` | Opposing cluster fraction against `Phi` by strategy |
| `ico_delta.png` | Paired change in opposing cluster size against `Phi` |
| `phi_c_by_strategy.png` | `Phi_c` against `f` by strategy |
| `backfire.png` | Backfire and repelled-neighbour shares |
| `stop_reasons.png` | Stop-reason composition |
| `rewiring_phi_c.png` | `Phi_c` on rewired configurations, when available |
| `temporal_effects.png` | Paired temporal effects with intervals, when available |
| `impact_correlations.png` | Predictor correlations with node impact, when available |

## Missing inputs

A missing file does not stop the report. Each one prints a warning and is
recorded in `inputs.json` and `warnings.txt`. The corresponding table is written
empty and that section's figures are skipped. If no input at all is present, the
command stops with an error naming the producing stages.

Some inputs are not produced by a stage on `main`:

- `results/rewiring/` and `results/rewiring_ico/` come from PR 13.
- `results/label_audit/` and `results/directed_roles/` come from PR 13.
- `reports/static_temporal/` comes from Antreas's scripts.
- `results/node_impact/study/` needs `incident_signs.csv` and
  `temporal_activity.csv`, which are external exports on `main`.

In those cases, copy the saved directory from the owner, or run the producing
branch. The manifest names exactly what is missing.

## Measured run

Run on 26 October 2026 against the default connected 10-core, with the ICO
sweep at `f = 0.3, 0.4, 0.5, 0.6, 0.7` and all seven strategies.

**Available:** NCO, sign-shuffle, ICO, sensitivity, the extraction comparison,
node-impact, and a local temporal reproduction. Fourteen inputs were present.

**Missing:** rewiring, label audit, directed roles, and incident signs. Six
entries are listed in `inputs.json` and `warnings.txt`.

The available results:

- NCO and sign-shuffle have the phase transition at `f` about 0.5. The
  sign-shuffle empirical and shuffled curves nearly overlap: `g_minus` at
  `f = 0.5` is 0.402 empirical against 0.400 shuffled, both with a realisation
  range from about 0.005 to 0.99. The specific placement of negative signs does
  not move the transition in this comparison.
- One sign-shuffled run out of 1,000 stopped in a limit cycle. Every NCO and ICO
  run reached steady state.
- ICO at `f = 0.5` and `Phi = 0.25`, signed model: `g_minus` is 0.000 for
  leverage, 0.004 for betweenness, 0.008 for signed leverage, 0.017 for degree,
  0.023 for the temporal-activity bridge, 0.026 for eigenvector, and 0.570 for
  random. The unsigned model has the same order.
- `Phi_c` at `f = 0.5` is 0.0095 to 0.0100 for every ranked strategy and 0.020
  to 0.022 for random. All runs reached the threshold.
- No run backfired. The signed repelled-neighbour share is 0.008 to 0.014 for
  the ranked strategies and 0.002 for random. Signed leverage does not reduce it
  relative to leverage.
- The default-core strategy order matches the sensitivity comparison.
- The temporal reproduction uses three fractions, 0.3, 0.5, and 0.7, with 20
  initialisations and 20 shuffles. It is a local reduced reproduction, not
  Antreas's full nine-fraction run.
  - Real temporal versus static: at `f = 0.3`, absolute magnetization is 0.106
    lower (95% interval -0.109 to -0.103), minority fraction 0.053 higher,
    minority-LCC fraction 0.156 higher, and switches 1,382 higher. At `f = 0.5`
    the magnetization gap is -0.150, minority fraction +0.075, minority-LCC
    fraction -0.053 with an interval of -0.125 to 0.022, and switches +443. At
    `f = 0.7` the pattern is similar to `f = 0.3`.
  - Real temporal versus shuffled timestamps: at `f = 0.3`, magnetization is
    0.041 higher, minority fraction 0.020 lower, minority-LCC fraction 0.147
    higher, and switches 252 higher. At `f = 0.5`, magnetization is 0.116 higher,
    minority fraction 0.058 lower, minority-LCC fraction 0.041 higher with an
    interval that includes zero, and switches 321 lower. At `f = 0.7` the pattern
    is similar to `f = 0.3`, with minority-LCC fraction 0.179 higher.
  - The temporal ordering changes outcomes relative to both the static baseline
    and the timestamp-shuffled control. The strongest magnetization and minority
    effects are at `f = 0.5`. The minority-LCC difference at `f = 0.5` is not
    distinguishable from zero.

- The node-impact reproduction used all 2,896 core nodes, 20 realisations, and
  locally derived `incident_signs.csv` and `temporal_activity.csv` files. At
  `f = 0.5`, signed model, the strongest Pearson correlation with mean impact is
  the negative incident ratio at -0.232 (95% interval -0.253 to -0.145), followed
  by degree at -0.155 (-0.238 to -0.053), betweenness at -0.134 (-0.217 to
  -0.033), and temporal activity at -0.119 (-0.207 to -0.015). Signed leverage
  is -0.058 with an interval of -0.117 to 0.002, leverage is +0.010 with an
  interval that includes zero, and the signed bridge ratio is -0.065 with an
  interval that includes zero. The unsigned model has no strong Pearson
  correlation; leverage is +0.108 and the signed bridge ratio is -0.145, both
  with intervals that include zero.
- The node-impact regression has an R squared of 0.101 for the signed model and
  0.060 for the unsigned model. The mean impact scale is small, about 0.0004 for
  the top nodes, and many `f = 0.5` baselines are already fragmented. The
  structural predictors are weak.

The combined report and deck still need the rewiring section and the PR13
diagnostics: label audit, directed roles, and incident signs. Their paths are
named in `inputs.json`, so they can be added without changing the command.

## Interpretation

- The phase-transition figures use the full realisation spread. Differences
  smaller than that band are not established effects.
- The ICO figures use the paired `delta_g_minus` for strategy comparisons; the
  raw `g_minus` curve mixes independent budget-wise initial states.
- `Phi_c` is a per-realisation minimum over the tested budget grid. Report the
  status strings and the reached fraction, not only the mean.
- The rewiring figure reads `Phi_c` against the configuration actually reached,
  not against the nominal target.
- The sensitivity table shows whether the core choice changes the conclusions.
- The node-impact table is a join, not a new regression. It reuses Daniel's saved
  analysis and adds the scores for the deck.

## Reproduce the shown results

Record the commands and settings beside the slides:

```bash
uv run python -m mdacn setup
uv run python -m mdacn characterise
uv run python -m mdacn centrality
uv run python -m mdacn targeting-scores
uv run python -m mdacn nco
uv run python -m mdacn sign-shuffle
uv run python -m mdacn ico --fractions 0.3 0.4 0.5 0.6 0.7
uv run python -m mdacn compare-extractions --seed 0
uv run python -m mdacn sensitivity --strategies random degree betweenness eigenvector leverage signed_leverage temporal_bridge --skip-nco
uv run python -m mdacn combined-results
```

Add the producing commands for the temporal, rewiring, node-impact, label-audit,
and directed-role outputs when those are present.
