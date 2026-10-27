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
`--strategies`; the score files are already exported for every core. The
measured all-seven comparison used:

```bash
uv run python -m mdacn sensitivity \
  --strategies random degree betweenness eigenvector leverage signed_leverage temporal_bridge \
  --skip-nco
```

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
| `strategy_ranking.csv` | Strategies ranked by paired delta at `Phi = 0.05`, 0.1, and 0.25 |
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

The all-seven run produced 7,840 ICO rows, of which 1,120 are budget-zero
baselines. All 7,840 reached steady state.

At `Phi = 0.05`, the six ranked strategies are effectively tied on the three
k-cores: mean paired `delta_g_minus` lies between -0.333 and -0.562. Random
reaches only -0.018 to -0.032. On the activity-LCC graph no strategy has a strong
effect at 0.05: the ranked strategies lie between -0.036 and -0.044, random
between -0.032 and -0.035. The activity effect appears at 0.10, where the ranked
strategies reach -0.420 to -0.469.

At `Phi = 0.25`, the ranking is the same on all four cores and both models:

| Rank | Strategy | Mean `delta_g_minus` range |
| ---: | --- | ---: |
| 1 | `leverage` | -0.399 to -0.871 |
| 2 | `betweenness` | -0.396 to -0.867 |
| 3 | `signed_leverage` | -0.394 to -0.862 |
| 4 | `degree` | -0.387 to -0.855 |
| 5 | `temporal_bridge` | -0.385 to -0.850 |
| 6 | `eigenvector` | -0.381 to -0.848 |
| 7 | `random` | -0.109 to -0.226 |

Ranks 1 to 6 differ by at most about 0.02. Paired over realisations, leverage is
ahead of degree by 0.0165 on average at `Phi = 0.25`, and the 95% interval
excludes zero in all eight core and model groups. That gap is small next to the
random-versus-ranked gap of about 0.44. At `Phi = 0.05` leverage is only 0.0007
ahead of degree, so that ranking is not useful beyond random being much worse.
The order is stable across k = 8, 10, and 12. The activity-LCC graph follows the
same order but needs a larger budget.

The proposed scores do not show an advantage. `signed_leverage` is slightly
better than degree at 0.25 (-0.0062 paired) but worse at 0.05 (+0.0196), driven
by k = 12 unsigned where it is clearly worse: `delta_g_minus` -0.408 against
-0.566, and `g_minus` 0.190 against 0.032. It recovers by 0.10.
`temporal_bridge` is indistinguishable from degree at 0.05 and slightly worse at
0.25 (+0.0032). Neither score reduces the repelled-neighbour share: the largest
values were leverage 1.18%, betweenness 1.32%, signed leverage 1.36%, temporal
bridge 1.39%, degree 1.43%, and random 0.42%.

No run backfired in the 6,720 intervention runs: `delta_g_minus` was never
positive.

### Phi_c

All 56 configuration, model, and strategy combinations reached the threshold.
Mean `Phi_c` is identical for all ranked strategies within a core, 0.0095 to
0.0195 on the k-cores, except `signed_leverage` on k = 12 unsigned, 0.0180
against 0.0155. Random is higher, 0.020 to 0.0445 on the k-cores. The
activity-LCC mean is not comparable because many runs are already below the
threshold at `Phi = 0`; use the first strong budget there.

### Verdict

The k = 10 default is not driving the qualitative conclusions. The NCO
transition, the ICO strategy ranking, and the budget needed for a strong effect
hold for k = 8, 10, and 12. The activity-LCC graph is the exception: it needs
`Phi` about 0.10, and its smaller node set and different selection rule make the
comparison imperfect, so it is reported as a stress test.

The two proposed scores do not beat the classical strategies. This is a null
result. Leverage is the best overall at large budgets, with betweenness second;
signed leverage and the temporal-activity bridge are not better, and signed
leverage does not reduce the repelled-neighbour share. Report this as measured,
not as a failed implementation.

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
   selection at `f = 0.5` and `Phi = 0.05, 0.1, 0.25`; `strategy_ranking.csv`
   gives the full order for every configuration and model.

If the k = 8, 10, and 12 curves overlap within uncertainty and the ranking is
stable, the k = 10 default is justified for the reported conclusions. If they
diverge, report the divergence and avoid presenting the k = 10 result as
universal. The activity-LCC graph is reported separately because its smaller
node set and different selection rule make a direct comparison imperfect.

`Phi_c` is measured per realisation over the tested budget grid. The corrected sweep reuses one initial state and settled baseline across all
budgets, with nested random target sets. Historical measurements above used
budget-wise independent initializations and are superseded by corrected runs. Report it as an interval across realisations and keep the status
strings (`reached`, `reference below threshold at Phi = 0`, `not reached in the
tested range`) visible.

## Runtime

The default three-strategy protocol runs 8,000 NCO simulations and 3,360 ICO
rows across the four cores, of which 480 are budget-zero baselines. The measured
run took about 15 minutes, including the centrality and targeting exports. The
all-seven ICO rerun produced 7,840 rows and took about 25 minutes with
`--skip-nco`, including the score exports and summaries. The betweenness export
is the main fixed cost per core. Use the skip flags to rerun one half;
`--skip-nco --skip-ico` regenerates summaries, the ranking table, and figures
from saved sweeps without rerunning simulations or score exports.
