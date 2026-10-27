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
| Signed-triad summary | `signed-triads` (PR 13) | `results/signed_triads/triad_summary.csv` | No |
| Triad sign patterns | `signed-triads` (PR 13) | `results/signed_triads/triad_sign_patterns.csv` | No |

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
| `context_signed_triads.csv` | Signed-triad summary, when available |
| `context_triad_patterns.csv` | Empirical signed-triad sign patterns, when available |

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

Run on 26 and 27 October 2026 against the default connected 10-core, with the
ICO sweep at `f = 0.3, 0.4, 0.5, 0.6, 0.7` and all seven strategies.

**Available:** all 22 inputs: NCO, sign-shuffle, ICO, rewiring and rewire-ICO,
node-impact, temporal, sensitivity, extraction comparison, label audit,
directed roles, incident signs, and signed triads. `warnings.txt` is empty.

**Missing:** none in the integration run. On the PR3 branch alone, the
PR13-owned outputs are still absent until PR13 merges.

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

- The node-impact reproduction used all 2,896 core nodes and 20 realisations.
  The canonical `incident_signs.csv` from `directed-roles` is identical to the
  locally derived file used for this run; `temporal_activity.csv` remains a
  locally derived lifetime interaction count. At
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

- The rewiring run produced 55 graphs in 6:52. Every graph stayed connected and
  converged. Achieved insular homophily ranges from 0.896 for
  `insular_non_worsening` to 0.016 for `integrated_strict`; achieved integrated
  homophily runs from 0.120 to 0.985. The unbiased reference reaches 0.230
  insular and 0.774 integrated. The empirical graph is 0.531 and 0.479.
- `Phi_c` for degree targeting is 0.030 on the empirical graph, 0.054 on the
  unbiased reference, 0.026 to 0.042 across the insular configurations, and
  0.028 to 0.054 across the integrated configurations. Random targeting stays
  at 0.000 for the empirical, reference, and most integrated graphs, and rises
  to 0.060 to 0.084 only for the most insular configurations, with wide
  intervals. Random remains much worse.
- Paired against the empirical graph at `Phi = 0.1`, degree targeting has a
  `g_minus` difference of -0.0204 (95% interval -0.0265 to -0.0151) on most
  rewired graphs. The two most insular configurations are not distinguishable:
  `insular_strict` is +0.0009 (-0.0020 to 0.0031) and
  `insular_non_worsening` is +0.0095 (-0.0005 to 0.0205). Rewiring does not make
  degree targeting harder; most configurations are slightly easier.
- Signed triads: the empirical balanced fraction is 0.8319 among nonzero
  triangles, against 0.8104 under sign shuffling (z = 6.4, p = 0.001), 0.7999
  under degree-preserving rewiring (five replicates, so p is coarse), and
  0.8112 for Erdős–Rényi. Among nonzero triangles the empirical graph has 81.0%
  `+++`, 16.7% `++-`, 2.15% `+--`, and 0.09% `---`; it has more `+++` and `+--`
  and fewer `++-` than the shuffled null. It also has 42,403 zero-sign triangles
  against 27,357 under shuffling.
- Incident signs are strongly concentrated: zero fraction 0.357 against 0.219,
  variance 0.0065 against 0.0034, and a top-5% share of negative edge ends of
  0.438 against 0.339. All three have p = 0.001.
- Directed roles: negative edges are more common between communities than
  within, 8.36% against 6.03%; negative events are 11.94% between against 8.65%
  within. Negative pairs are slightly more reciprocal than positive pairs, 24.2%
  against 23.1%, while the density reference is 0.78%.
- The label audit reproduces the published filtering table exactly: 286,561 body
  rows, 259,092 posts, 35,776 nodes, 124,330 undirected edges, 1,983 zero-sign
  edges; the core has 159,180 rows, 140,761 posts, 2,896 nodes, 58,466 edges,
  and 1,469 zero-sign edges. No post has mixed sentiment.

Every expected input is present in the integration run and `warnings.txt` is
empty. The scratch branch `scratch/full-integration` contains the resolved PR13
merge plus PR1, PR2, and PR3. The PR3 branch on its own still reports the six
PR13-owned inputs as missing until PR13 merges. The missing stages took about
1.5 hours in total: `rewire` 6:52, `rewire-ico` 13:59, and `node-impact` 1:00:32.

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
uv run python -m mdacn directed-roles
uv run python -m mdacn targeting-scores
uv run python -m mdacn features
uv run python -m mdacn nco
uv run python -m mdacn sign-shuffle
uv run python -m mdacn ico --fractions 0.3 0.4 0.5 0.6 0.7
uv run python -m mdacn node-impact
uv run python -m mdacn rewire
uv run python -m mdacn rewire-ico
uv run python -m mdacn incident-signs
uv run python -m mdacn signed-triads
uv run python -m mdacn label-audit
uv run python -m mdacn reports
uv run python -m mdacn compare-extractions --seed 0
uv run python -m mdacn sensitivity --strategies random degree betweenness eigenvector leverage signed_leverage temporal_bridge --skip-nco
uv run python scripts/run_static_temporal_comparison.py --fractions 0.3 0.5 0.7 --initializations 20 --shuffles 20 --bootstrap 1000
uv run python -m mdacn combined-results
```

`directed-roles` produces `incident_signs.csv` for `features`. The temporal
activity export remains external. The temporal script writes its run under
`reports/static_temporal/`, which `combined-results` discovers automatically.
