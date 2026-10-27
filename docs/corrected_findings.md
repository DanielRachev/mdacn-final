# Corrected experiment findings, 27 October 2026

These measurements supersede the independently seeded, budget-wise ICO
thresholds in the original result bundle. They use the integrated PRs #11–13,
the original report/sensitivity implementation, a fixed reconstructed body-link
graph, and the corrected paired-baseline protocol. Generated data remain in
the separately retained `mdacn-corrected-results` artifact bundle.

## Protocol and measured changes

The main intervention sweep now settles one initial state per realization and
reuses it across budgets and strategies. Random selections are prefixes of one
seeded permutation. Threshold tables preserve censoring and zero-budget
successes; plots show the mean among successful interventions on eligible
baselines, alongside overall success and zero-budget rates. Seeded null models
use canonical endpoint order and record its checksum. Degree-preserving
reference graphs increase from 5 to 100; the ten targeted configurations retain
five graphs each. Single-node impact is split into direct pinning/fragmentation
and subsequent propagation, with unpinned opinion changes counted separately.

At initial negative fraction 0.5 on the default 2,896-node core:

| Finding | Corrected measurement |
| --- | --- |
| Ranked-strategy success among baselines needing intervention | 8/8 signed; 7/7 unsigned, by a tested 5% budget |
| Random-strategy success on those baselines | 0/8 signed; 0/7 unsigned, through 25% |
| Baselines already below the GCC criterion | 12/20 signed; 13/20 unsigned |
| Mean ranked threshold including zero-budget successes | 0.0200 signed; 0.0175 unsigned, previously 0.0100 and 0.0095 |
| Best mean residual negative GCC at 25% | Leverage: 0.000483 signed; 0 unsigned |
| Mean impact attributable to direct pinning/fragmentation | 99.79% signed; 99.33% unsigned |
| Single-node trials changing any unpinned opinion | 1.03% signed; 0.27% unsigned |
| Degree-preserving balance-null mean | 79.78%, versus empirical 83.19% |
| One-sided balance p-value, 100 reference graphs | 0.0099, previously 0.1667 with five graphs |

The 5% crossing is a grid result: the tested threshold lies in (1%, 5%], not
at a precisely estimated critical point. The old random conditional mean of
zero in some tables described only already-successful baselines. A missing
intervention threshold now makes its failure on eligible runs explicit.

## Interpretation

Leverage has the lowest mean residual GCC across all four extractions and both
models at a 25% budget. Signed leverage beats degree targeting on the default
core but falls short of ordinary leverage; temporal bridge also falls short
of degree there. Degree requires a tested 10% budget on the activity extraction,
compared with 5% on the three k-cores. These results support targeting highly
connected, locally influential nodes, but do not establish that the proposed
scores improve on the best existing score.

Single-node total impacts and their original regressions are unchanged. Their
small size is predominantly mechanical. Signed propagation is more frequent
and sometimes harmful: 34 signed single-node trials have a net negative total
impact. This does not contradict effective simultaneous multi-node targeting.

Topology, NCO, temporal comparisons, and sign-concentration findings are
unchanged against the preceding canonical replay. The dense, mostly positive
core amplifies small initial majorities; a mean near one half at initial
fraction 0.5 mixes mostly dominant outcomes across runs. Time-resolved dynamics
retain more minority opinion than static aggregation, while real timestamps
produce more consensus than shuffled timestamps. Those controls answer
different questions, so their directions are consistent.

Negative ties are concentrated, and signed balance exceeds all three tested
null ensembles. This is evidence about network structure, conditional on this
selected core and these null generators. SNAP's positive label includes neutral
links; communities are not measured ideological camps. Rewired experiments
support degree targeting but show no simple monotonic relation between the
manipulated homophily scores and intervention cost. Their GCC metric uses the
fixed original unsigned core, even when the influence graph is rewired.

The remaining limits are one dataset, overlapping extractions, 20 initial
states, a coarse budget grid, five graphs per targeted rewiring configuration,
and retrospective full-window activity features. Simulation findings do not
establish causal effects on real beliefs or a thermodynamic phase transition.
