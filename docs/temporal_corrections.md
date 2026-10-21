# Temporal experiment corrections

Recomputed on 21 October 2026 using Antreas' PR #8 and the corrected implementation.
This note supersedes the biweekly result in section 5.3 and the ICO confidence
intervals in section 6.3 of `Temporal_LLM_Summary.pdf`.

## What changed

The old `2W-SUN` construction kept events from alternate weeks instead of merging
two weeks. All temporal calculations now use the same non-overlapping calendar,
anchored at the first observed base period. Inactive windows and the full node
order remain present. The weekly and monthly observed matrices are exactly
unchanged.

The shuffled timelines are shared across opinion initializations. Confidence
intervals now resample both initializations and timeline IDs, keeping the same
initialization and timeline paired between strategies. This applies to the
resolution comparison and ICO. It changes uncertainty estimates; it does not
change the weekly simulations or their mean outcomes.

## Input and preservation checks

The 2,896-node core contains 159,180 directed events and 58,466 undirected edges.
Rebuilding its static matrices from the raw events exactly matches the supplied
`unsigned.npz`, `signed.npz`, and `nodes.json` order.

| Resolution | Windows | Events before | Events after |
| --- | ---: | ---: | ---: |
| Weekly | 174 | 159,180 | 159,180 |
| Biweekly | 87 | 79,536 | 159,180 |
| Monthly | 41 | 159,180 | 159,180 |

Every corrected snapshot was independently rebuilt from event counts and net
signs. Timestamp permutations retain the same calendar and global counts in
each window, as well as the full-window topology and signs.

## Resolution comparison

At initial negative fraction `f = 0.5`, with 20 paired initializations, 20 global
timestamp shuffles, seed 42, and 2,000 bootstrap draws:

| Resolution | Real mean switches | Shuffled mean switches | Real minus shuffled | Corrected 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Weekly | 3,675.85 | 4,043.13 | -367.28 | [-470.33, -265.77] |
| Biweekly | 3,060.15 | 3,367.13 | -306.98 | [-395.36, -215.85] |
| Monthly | 2,670.15 | 3,015.31 | -345.16 | [-449.69, -238.98] |

The invalid biweekly difference was -283.51. Correctly retaining the second week
increases switching in both conditions. Real timing still produces fewer switches
than shuffled timing at each tested resolution. These are counts of simulated
opinion changes, not counts of distinct nodes or observed Reddit opinion changes.
Different resolutions provide different numbers of update opportunities, so
compare real versus shuffled within each row.

A separate 100-shuffle rerun gives differences of -370.39, -310.47, and -351.38
for weekly, biweekly, and monthly windows. Its intervals also exclude zero.
The intervals for the final raw positive-opinion share include zero at all three
resolutions; this experiment does not establish a directional change in that share.

## Static versus weekly NCO

The 180 paired initializations across `f = 0.1, ..., 0.9` were rerun. All static
runs reached fixed points. At `f = 0.5`, the mean final minority remains 3.85%
for static NCO and 10.51% for weekly NCO. Weekly NCO retains a larger mean
minority at every tested fraction. This compares different contact availability
as well as timing: the lifetime static graph exposes all edges at every update.

## Temporal ICO

The complete original grid was rerun with 20 initializations, 20 shared global
timestamp shuffles, seed 42, and 2,000 bootstrap draws. The grid contains three
initial fractions, eight budgets, and random/degree targeting. The metric is
`g_minus = largest negative-opinion component / 2,896`, on the fixed unsigned core.

All 20,160 simulations completed, producing 2,520 per-run critical-budget records.
The 840 individual outcomes checked against the original implementation match.
At `f = 0.7`, the mean outcomes on real timing are:

| Pinned fraction | Degree targeting `g_minus` | Random targeting `g_minus` |
| --- | ---: | ---: |
| 0% | 0.94019 | 0.94019 |
| 1% | 0.23838 | 0.92619 |
| 3% | 0.09491 | 0.89560 |
| 5% | 0.06379 | 0.86435 |
| 10% | 0.02345 | 0.74137 |
| 15% | 0.01145 | 0.37895 |
| 20% | 0.00998 | 0.03425 |
| 25% | 0.00775 | 0.02034 |

At a 5% budget, the degree-targeted mean is 0.06379 on real timing and 0.14246
on shuffled timing. The paired difference is -0.07867, with corrected 95% interval
[-0.08571, -0.07117]. A separate 20,000-draw calculation gives
[-0.08584, -0.07118]. The timing effect remains supported after accounting for
timeline variation. Random targeting has a smaller effect in the opposite
direction: real minus shuffled is +0.01816, interval [0.01492, 0.02197].

The real degree-minus-random difference is -0.80055, interval
[-0.80931, -0.79090]. Degree targeting crosses `g_minus < 0.5` at the first
nonzero tested budget, 1%, in all 20 real runs. Random targeting first crosses
at 10% in one run, 15% in ten, and 20% in nine. These are thresholds on the
tested grid, not precise critical points.

The simulations support a strong advantage for degree targeting under these
settings. The component fraction uses the full original core as denominator,
including pinned nodes. It measures simulated opinions, not observed beliefs
or an empirical effect of real interventions.

## Reproduction and checks

From a checkout with the raw body TSV and the default core exported under
`data/`, run:

```bash
uv run python scripts/run_temporal_analysis.py --repetitions 20 --shuffles 20 \
  --output reports/verification/analysis-20
uv run python scripts/run_temporal_analysis.py --repetitions 20 --shuffles 100 \
  --output reports/verification/analysis
uv run python scripts/run_temporal_ico.py --initializations 20 --shuffles 20 \
  --bootstrap 2000 --output reports/verification/ico
```

The body TSV SHA-256 is
`9bbf57511dc0da7369c2e896d865b68f170136def4f223f4c9e9e10818f4bfb5`.
The node-order SHA-256 is
`a07411cc292ea632f690c4633c72c3d94233c3d5307afaf3aa5b7b858993cd0a`.
The local verification directory includes raw ICO rows, summaries, bootstrap
intervals, per-run threshold crossings, static/weekly replay rows, and input
hashes/package versions in `provenance.json`. Generated outputs are not committed.

All 137 tests pass with warnings treated as errors. The new regressions fail on
the original implementation and cover event retention, cross-module window
alignment, shared-timeline uncertainty, and paired strategy comparisons. Ruff,
formatting, and whitespace checks pass. Formatting/import cleanup in the other
temporal files does not change their function bodies.

The separate burstiness, reciprocity, and sign-persistence permutation suites
were not rerun; their weekly path is unchanged. Signed-leverage targeting still
requires Georgi's node-aligned score export.
