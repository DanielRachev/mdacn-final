# Core extraction sensitivity

This stage checks whether the default connected 10-core changes the reported
conclusions. It is a robustness check, not parameter tuning. The graph
configuration changes; the simulation protocol, seeds, realisation count,
budget grid, and node-level score definitions stay fixed.

## Inputs

Run the extraction comparison first. It exports four independent cores with
their matching matrices, node order, and community labels:

```bash
uv run python -m mdacn compare-extractions --seed 0
```

| Configuration | Nodes | Edges | Negative edges | Zero-sign edges |
| --- | ---: | ---: | ---: | ---: |
| `k_core_8` | 3,672 | 64,952 | 6.98% | 2.37% |
| `k_core_10` | 2,896 | 58,466 | 7.17% | 2.51% |
| `k_core_12` | 2,346 | 52,859 | 7.27% | 2.67% |
| `activity_lcc_100` | 957 | 25,864 | 6.90% | 3.38% |

The node, edge, and sign counts come from
`results/extraction_comparison/comparison.csv`. The activity-LCC graph uses a
different selection rule and is much smaller, so it is a stress test rather
than a like-for-like neighbour of the k-core values.

## Protocol

For every configuration:

1. Export the existing centrality scores for that core.
2. Export the two proposed targeting scores. The ICO metadata records their
   checksums even when the selected strategies do not use them.
3. Run NCO with the existing `f` sweep, 20 realisations, and both models.
4. Run ICO at `f = 0.5` with the selected strategies, the full budget grid,
   20 realisations, and both models.
5. Compute `Phi_c` per run and aggregate across realisations.

The default ICO subset is `random`, `degree`, and `leverage`. The proposed
`signed_leverage` and `temporal_bridge` strategies can be added with
`--strategies`; the score files are already exported for every core.

## Command

```bash
uv run python -m mdacn sensitivity
```

Useful overrides:

```bash
uv run python -m mdacn sensitivity --strategies random degree leverage signed_leverage temporal_bridge
uv run python -m mdacn sensitivity --ico-fractions 0.3 0.5 0.7
uv run python -m mdacn sensitivity --skip-nco
uv run python -m mdacn sensitivity --skip-ico
```

`--skip-nco` and `--skip-ico` read the saved sweep from the output directory
instead of rerunning it. They fail with the missing path when no earlier run
exists.

## Outputs

Under `results/sensitivity/`:

| File | Contents |
| --- | --- |
| `topology.csv` | Per-configuration node, edge, sign, and retention summary |
| `nco_runs.csv` | One row per configuration, model, fraction, and realisation |
| `nco_summary.csv` | Mean, spread, and 2.5 to 97.5 percentile range per configuration, model, and fraction |
| `nco_stop_reasons.csv` | Steady, cycle, and update-limit counts for the NCO sweep |
| `ico_runs.csv` | One row per configuration, model, fraction, strategy, budget, and realisation |
| `ico_summary.csv` | Mean opposing cluster fraction, paired delta, backfire share, and negative influence links |
| `ico_stop_reasons.csv` | Stop-reason counts for the ICO sweep |
| `phi_c.csv` | Per-run and aggregated critical budget per configuration, model, fraction, and strategy |
| `headline.csv` | One row per configuration with selected NCO and ICO outcomes |
| `metadata.json` | Settings, core checksums, and versions |
| `nco_g_minus_by_configuration.png` | Opposing cluster fraction against `f`, one line per core |
| `nco_g_plus_by_configuration.png` | Positive cluster fraction against `f` |
| `nco_raw_negative_fraction.png` | Raw final negative fraction against `f` |
| `ico_degree_g_minus_by_configuration.png` | Opposing cluster fraction against `Phi` at `f = 0.5` |
| `phi_c_by_configuration.png` | Critical budget by core, one panel per model |
| `topology_sign_fractions.png` | Negative and zero-sign edge shares by core |

Each configuration also keeps its own `nco/` and `ico/` directories with the
raw sweep, metadata, and node order.

## Measured findings

Run on 26 October 2026 with the default protocol, seed 0, and 20 realisations.

### NCO

| Configuration | Signed crossing `f` | Unsigned crossing `f` |
| --- | ---: | ---: |
| `k_core_8` | 0.498 | 0.500 |
| `k_core_10` | 0.498 | 0.498 |
| `k_core_12` | 0.501 | 0.501 |
| `activity_lcc_100` | 0.502 | 0.505 |

The transition is at `f` about 0.50 in every configuration. The mean `g_minus`
at exactly `f = 0.5` varies more, from 0.358 to 0.543, but the 95% interval at
that point is about ±0.21 and the curve rises from near zero at `f = 0.44` to
near one at `f = 0.56`. The level at the crossing is a mixture of all +1 and all
-1 runs, so it is not a stable comparison. The signed and unsigned curves nearly
overlap; the largest gap is k = 12 at `f = 0.5`, 0.403 signed against 0.358
unsigned, inside the realisation spread.

### ICO

| Configuration | Degree delta at `Phi = 0.05`, signed | Degree delta at `Phi = 0.05`, unsigned | First strong budget |
| --- | ---: | ---: | ---: |
| `k_core_8` | -0.424 | -0.333 | 0.05 |
| `k_core_10` | -0.561 | -0.610 | 0.05 |
| `k_core_12` | -0.517 | -0.566 | 0.05 |
| `activity_lcc_100` | -0.042 | -0.039 | 0.10 |

"First strong budget" is the first tested budget with a mean paired
`delta_g_minus` of at most -0.10. Degree and leverage reach that effect at
`Phi = 0.05` on the three k-cores and at `Phi = 0.10` on the activity-LCC graph.
Random targeting needs `Phi = 0.20` to 0.25 in every configuration and reduces
`g_minus` less. Degree and leverage have nearly identical first crossings;
leverage is slightly better at larger budgets.

No run backfired in the 2,880 intervention runs: `delta_g_minus` was never
positive. The repelled-neighbour share was at most 1.4%, so some neighbours moved
to -1, but the opposing cluster still shrank overall.

### Phi_c

All k-core runs reached the threshold. Mean `Phi_c` for degree targeting ranges
from 0.0095 to 0.0195 across the three k-cores, within one budget-grid step. The
activity-LCC mean is lower but not comparable: many of its runs are already
below the threshold at `Phi = 0`. Use the first strong budget for that graph.

### Verdict

The k = 10 default is not driving the qualitative conclusions. The NCO
transition, the strategy ranking, and the large 5% degree or leverage effect hold
for k = 8, 10, and 12. The activity-LCC graph is the exception: it needs a
larger budget, and its topology and sign composition differ enough that it is
reported as a stress test. Small differences in the mean at `f = 0.5` and in
mean `Phi_c` are within realisation uncertainty and the budget grid.

## Interpretation

Compare the four configurations on three levels:

1. **Topology and signs.** Node and edge counts, density, negative-edge share,
   and zero-sign share. A smaller graph has less statistical power and a
   different community structure.
2. **NCO outcomes.** Whether `g_minus`, `g_plus`, and the raw negative fraction
   curves move relative to the realisation spread. Absolute cluster sizes are
   not comparable because `N` differs; compare fractions.
3. **ICO outcomes.** Whether the strategy ranking, the opposing cluster
   fraction at a fixed `Phi`, or `Phi_c` changes. `headline.csv` gives a compact
   selection at `f = 0.5` and `Phi = 0.05, 0.1, 0.25`.

If the k = 8, 10, and 12 curves overlap within uncertainty and the ranking is
stable, the k = 10 default is justified for the reported conclusions. If they
diverge, report the divergence and avoid presenting the k = 10 result as
universal. The activity-LCC graph is reported separately because its smaller
node set and different selection rule make a direct comparison imperfect.

`Phi_c` is measured per realisation over the tested budget grid. The existing
sweep assigns an independent initial state to each budget, so a per-run `Phi_c`
is a minimum over budget-wise runs, not the first crossing of one fixed
trajectory. Report it as an interval across realisations and keep the status
strings (`reached`, `reference below threshold at Phi = 0`, `not reached in the
tested range`) visible.

## Runtime

The default protocol runs 8,000 NCO simulations and 3,360 ICO rows across the
four cores, of which 480 are budget-zero baselines. The measured run took about
15 minutes, including the centrality and targeting exports. The betweenness
export is the main fixed cost per core. Use the skip flags to rerun one half;
the all-seven ICO subset is roughly twice the ICO work.
