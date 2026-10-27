# Contrarian targeting benchmark (ICO)

The `ico` stage pins a budget fraction `Phi` of core nodes at `+1` and measures
how far the opposing `-1` cluster shrinks. It runs every strategy on both the
unsigned and the signed core.

## Inputs

| Input | Produced by | Use |
| --- | --- | --- |
| `core.graphml`, `unsigned.npz`, `signed.npz`, `nodes.json` | `characterise` | Topology, connectivity reference, and node order |
| `centrality.csv` | `centrality` | Raw `degree`, `betweenness`, `eigenvector`, `leverage` |
| `signed_leverage.csv` | `targeting-scores` | Proposed signed leverage |
| `temporal_bridge_centrality.csv` | `targeting-scores` | Proposed temporal-activity bridge |

Every table must cover the same `nodes.json` order exactly once. A missing score
file stops the run and names the stage to execute.

## Strategies

| Strategy | Score |
| --- | --- |
| `random` | Uniform sample without replacement (baseline) |
| `degree` | Raw core degree `d_i` |
| `betweenness` | Shortest-path betweenness `B_i` |
| `eigenvector` | Eigenvector centrality `x_i` |
| `leverage` | Node leverage `l_ni = (1/d_i) * sum_j (d_i - d_j)/(d_i + d_j)` |
| `signed_leverage` | `(1/d_i) * sum_j W_ij * max(0, (d_i - d_j)/(d_i + d_j))` |
| `temporal_bridge` | `sqrt(bridge_fraction * activity / max(activity))` |

`d_i` is the unsigned core degree, so zero-sign edges stay in the denominator.
The bridge fraction is the share of a node's neighbours outside its baseline
community. Each strategy takes the top `round(N * Phi)` nodes by score, with
equal scores ordered by node name.

Both proposed scores are hypotheses. Signed leverage is meant to reduce backfire
by avoiding negative links to less connected nodes, and the bridge score favours
nodes that cross community boundaries and interact often. Neither guarantees
better targeting. A full-window activity score is retrospective.

## Procedure

1. Assign a fraction `f` of nodes to `-1` with a recorded seed.
2. Settle one baseline per model, fraction, and realization, and reuse it at every
   budget and strategy. Random targets use prefixes of one seeded permutation,
   so larger budgets retain the targets from smaller budgets. A cycle or update-limit
   baseline is not a steady state, so the run stops and reports its seeds.
3. Pin the selected contrarians at `+1` from the settled state.
4. Record `g_minus = S_minus_GCC / N`, `g_plus`, the paired baseline fractions,
   their difference, and the terminal status.

`Phi = 0` is the paired no-intervention reference. The grid comes from
`ICO["budgets"]`.

## Backfire

A pinned `+1` node contributes `W_ji * sigma_i = -1` through each negative link.

| Column | Meaning |
| --- | --- |
| `negative_influence_links` | Signed negative edges incident to a pinned node |
| `repelled_neighbours` | Those links whose neighbour held `+1` at baseline and `-1` at the end |
| `repelled_share` | `repelled_neighbours / negative_influence_links` |
| `delta_g_minus` | `g_minus` minus the paired baseline `g_minus` |
| `backfired` | `delta_g_minus > 0`, so the opposing cluster grew |

Unsigned runs have no reactance term, so their backfire columns are zero.
Negative influence is a model mechanism; a net backfire and any advantage of hub
targeting are measured, not assumed.

## Critical budget

`Phi_c` is the smallest tested budget with `g_minus < ICO["phi_c_threshold"]`,
0.5 by default. Only steady outcomes count as crossings. It is measured on a
fixed-baseline trajectory and then aggregated across realizations. Historical
budget-wise independent seed tables are rejected and must be rerun. This is an
operational threshold on a finite graph, not a thermodynamic critical point.

`phi_c_runs.csv` preserves each trajectory and censoring status. Summaries
include `reached_runs`, `unreached_runs`, `zero_budget_runs`, `eligible_runs`,
`intervention_reached_fraction`, and `mean_intervention_phi_c`. The latter two
exclude baselines already below the criterion. `mean_phi_c` remains conditional
on reaching the criterion; it is always plotted with the reached fraction and
the zero-budget share. Unreached steady trajectories are censored beyond the
maximum tested budget; nonsteady outcomes remain unresolved.

| Status | Meaning |
| --- | --- |
| `reached` | All trajectories crossed; the mean and median cover those runs |
| `partially reached in the tested range` | Some trajectories did not cross; the conditional mean excludes them |
| `reference below threshold at Phi = 0` | The no-intervention reference was already below the threshold, so `Phi_c = 0` |
| `not reached in the tested range` | No budget crossed and `mean_phi_c` is empty |

The status is separate from a threshold on the raw negative fraction `s_1`.

## Runtime

The suite costs `fractions x models x budgets x realizations x strategies` runs,
with one baseline settle per model/fraction/realization. The default 50-point grid gives about 98,000 runs,
five times the unsigned three-strategy sweep. Use a narrower grid and refine near
an observed threshold:

```bash
uv run python -m mdacn ico --fractions 0.3 0.4 0.5 0.6 0.7
```

## Outputs

| Location | Contents |
| --- | --- |
| `results/ico/intervention_sweep.csv` | One row per model, fraction, strategy, budget, realisation |
| `results/ico/metadata.json` | Settings, threshold, and input checksums |
| `results/ico/nodes.json` | Node order of the run |
| `results/ico/intervention_summary.csv` | Means and variability per model, `f`, strategy, budget, status |
| `results/ico/phi_c.csv` | Per-run and aggregated critical budget |
| `results/ico/intervention_comparison_<model>_f<f>.png` | `g_minus` against `Phi` per model and fraction |
| `results/ico/phi_c_comparison.png` | `Phi_c` against `f` per model and strategy |

`reports` rebuilds the summary, `phi_c.csv`, and both figures from
`intervention_sweep.csv` without rerunning the simulation.
