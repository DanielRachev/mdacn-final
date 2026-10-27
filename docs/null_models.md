# Null models: invariants

Three controls test different parts of the empirical signal. Each preserves
some properties exactly and leaves others free.

| Property | A: sign shuffle | B: static rewiring | C: timestamp shuffle |
| --- | --- | --- | --- |
| Node set and core size `N` | ✓ | ✓ | ✓ |
| Aggregate edge set (unsigned topology) | ✓ | ✗ | ✓ |
| Unsigned degree per node | ✓ | ✓ | ✓ (full window) |
| Global positive/negative/zero edge counts | ✓ | ✓ | ✓ |
| Positive/negative/zero degree per node | ✗ | ✓ (same-sign swaps) | ✓ (full window) |
| Full-window event totals per dyad (`n_positive`, `n_negative`) | ✓ | ✗ (multiset ✓; records move to new dyads) | ✓ |
| Sign and event counts on the same edge | ✗ | ✓ (records move together) | ✓ |
| Sign placement relative to communities | ✗ | ✗ (steered) | ✓ |
| Connectivity | ✓ | not guaranteed; reported | ✓ (aggregate) |
| Per-snapshot node degree | — | ✗ | ✗ |
| Weekly node activity | — | — | ✗; global timestamp histogram ✓ |

None of the controls preserves per-snapshot node degrees. Timestamp shuffling
changes weekly node activity while keeping the global timestamp histogram; its
`within_source` mode also keeps each source's timestamp multiset.

Checks in code:

- A: `tests/test_experiments.py::test_sign_shuffle_preserves_topology_and_sign_counts`.
- B: `validate_rewiring` in `src/mdacn/rewiring.py`, run on every exported graph.
- C: `validate_aggregate_preservation` in `src/mdacn/temporal_nulls.py`.

## Controls in the signed-structure analyses

The incident-sign, triad, and P-O-X analyses reuse A, B, and C and add two
controls.

| Analysis | Controls |
| --- | --- |
| `incident-signs` | A (1,000 shuffles); exact hypergeometric tail per node under A |
| `signed-triads` | A (same seeds); B unbiased reference replicates; Erdős–Rényi (descriptive) |
| `temporal-balance` | Post-label shuffle; C in `global` mode |

| Property | Post-label shuffle | Erdős–Rényi G(N, M) |
| --- | --- | --- |
| Node set and `N` | ✓ | ✓ |
| Edge count and global sign counts | event timing and endpoints ✓; number of negative posts ✓ | ✓ (signs placed at random) |
| Unsigned topology and degrees | ✓ | ✗ (different degree sequence and clustering) |
| One label per source post | ✓ (labels move with whole posts) | — |
| Per-node signed activity | ✗ | ✗ |

The post-label shuffle is the event-level analogue of A: it permutes
`LINK_SENTIMENT` across source posts, because every row of a post shares one label.
Erdős–Rényi is descriptive only: it changes topology and sign placement together,
so it cannot separate them. B's unbiased reference keeps each node's positive and
negative degree, which A does not; `signed-triads` checks this for every replicate
it loads.

Checks in code:

- Post-label shuffle: `tests/test_temporal_balance.py::test_post_label_shuffle_keeps_posts_consistent`.
- Erdős–Rényi: `tests/test_signed_triads.py::test_erdos_renyi_keeps_size_and_sign_multiset`.
- B as used by `signed-triads`: `signed_degrees_preserved` in `results/signed_triads/metadata.json`.

## Null Model B: degree-preserving homophily rewiring

**Move**: Pick a uniform edge, then a uniform second edge with the same sign.
Flip each edge's orientation with probability 1/2 (to allow A-to-B, as well as B-to-A node extraction, because the node order in the edge is fixed), then rewire `(u, v), (x, y)`
to `(u, y), (x, v)`. Reject self loops, duplicate edges, and unchanged pairs.
Without the orientation flips, stored endpoint order would decide which of the two
possible rewirings is proposed.

**Signs**: Only same-sign edges swap. Each edge record (`sign`, `n_positive`,
`n_negative`) follows its first endpoint: `(u, y)` takes the record of `(u, v)`,
and `(x, v)` takes the record of `(x, y)`. Node `n_interactions` is recomputed.

**Steering**: Community labels are the fixed Louvain labels of the empirical core;
simulated opinions play no role.

- Insular: `h = positive_within + negative_between`.
- Integrated: `h = positive_between + negative_within`.

A swap changes `h` by at most 2. Zero-sign edges are swapped without steering.

| Rule | Accepts a valid swap when |
| --- | --- |
| `unbiased` (reference) | always |
| `strict` | `delta_h > 0` |
| `non_worsening` | `delta_h >= 0` |
| `metropolis` | `delta_h >= 0`, else with probability `exp(beta * delta_h)` |

**Stopping**: Each graph stops when its tracked score changes by at most
`tolerance * (positive + negative edges)` over a window of attempts, or at the
attempt cap. The tracked score is `h` for steered runs and the number of
within-community signed edges for the reference. Metadata records whether each
graph converged.

**Calibration**: One replicate per rule on the 10-core (seed 0, window `2|E|`,
tolerance 0.005, cap `50|E|`). Values are achieved `h` as a fraction of its bound;
the empirical core scores 0.531 insular and 0.478 integrated, the reference 0.229
and 0.775.

| Rule | Insular | Integrated |
| --- | --- | --- |
| strict | 0.838 | 0.981 |
| non-worsening | 0.865 | 0.983 |
| beta 0.5 / 1 / 2 | 0.306 / 0.388 / 0.574 | 0.833 / 0.880 / 0.938 |
| beta 4 / 8 / 16 | 0.802 / 0.859 / 0.864 | 0.977 / 0.982 / 0.983 |

Defaults follow from this: insular betas 1, 2, 4 (below, near, and above the
empirical score) and integrated betas 0.5, 1, 2; betas of 8 and above match
non-worsening. With tolerance 0.005, strict and beta 4 stopped while still
improving slowly, so the default tolerance is 0.002 (strict then reaches 0.861 and
beta 4 0.822). Non-worsening keeps drifting through neutral moves and reached the
`50|E|` cap at 0.893, so the default cap is `100|E|`. Every calibrated graph stayed
connected.

**Feasibility**: `homophily_bounds` gives upper bounds from degrees and community
sizes: node `i` has at most `min(d_i^s, |C_i| - 1)` internal edges of sign `s`.
Achieved `h` is reported as a fraction of this bound; the bound need not be
attainable.

**Evaluation**: `rewire-ico` replays the same initial states on every graph,
settles signed NCO, pins contrarians at +1, and measures `g_minus` on the fixed
empirical unsigned core with the full core `N`, for every graph (shared
convention). Baseline and terminal states are saved in `final_states.npz`. `Phi_c` is each run's first tested budget with
`g_minus < 0.5`, or "not reached". Degree targeting pins the same nodes on every
graph because degrees are preserved.

**Saved inputs**: Nothing computed earlier is recomputed. `rewire` reads the core
graph, the matrices and `nodes.json` from `characterise`, and the empirical
`centrality.csv` from `centrality`; a missing or changed input is an error naming the
stage to rerun. The empirical degree assortativity comes from `metrics.json` when it
describes this core, and is otherwise computed from the core. Each rewired graph
saves its own degree and leverage scores.
`rewire-ico` loads the empirical matrices from `data/processed`, the rewired
matrices and all scores from `results/rewiring`, and checks `nodes.json` and the
empirical score file against the hashes recorded by `rewire`.

## Null Model B: Phi_c findings

**Setup**: Default run with `--seed 0`: the empirical 10-core, the unbiased reference,
and 10 steered configurations with 5 replicates each (55 rewired graphs). Each graph
ran 20 shared initial states at `f = 0.5`, degree and random targeting, and budgets
0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25. `g_minus` is measured on the fixed
empirical core.

- All 55 graphs converged before the attempt cap, stayed connected, and passed
  validation.
- All 17,920 baselines and ICO runs reached steady states, so none were excluded.

**Already-fragmented baselines**: At `f = 0.5`, many initial states end with
`g_minus < 0.5` before any intervention, so their `Phi_c` is 0:

- empirical: 8 of 20;
- rewired graphs: 41-48% of runs.

The table below uses only the remaining polarized runs (baseline `g_minus >= 0.5`).

| Graph | Insular `h` / max | Degree targeting `Phi_c` | Random targeting reached |
| --- | --- | --- | --- |
| empirical | 0.53 | 0.05 (12 of 12) | 0 of 12 |
| reference | 0.23 | 0.10 (54 of 54) | 0 of 54 |
| insular, beta 1 | 0.39 | 0.10 (43), 0.05 (11) | 0 of 54 |
| insular, beta 2 | 0.57 | 0.05 (53), 0.03 (2) | 1 of 55 |
| insular, beta 4 | 0.83 | 0.05 (28), 0.03 (9), 0.10 (19) | 22 of 56 |
| insular, strict | 0.87 | 0.05 (31), 0.03 (7), 0.10 (18) | 22 of 56 |
| insular, non-worsening | 0.90 | 0.05 (32), 0.03 (12), 0.10 (15) | 27 of 59 |
| integrated, beta 0.5 / 1 | 0.17 / 0.12 | 0.10 (all) | 0 |
| integrated, beta 2 | 0.06 | 0.05 (31), 0.10 (21) | 0 of 52 |
| integrated, strict | 0.02 | 0.10 (all) | 0 of 53 |
| integrated, non-worsening | 0.01 | 0.05 (42), 0.10 (10) | 0 of 52 |

Random targeting, when it reached the threshold, did so at budgets 0.15 to 0.25.

**Paired `g_minus` differences** for degree targeting at `Phi = 0.05`, with 95%
bootstrap intervals over initial states and replicates:

| Comparison | Mean difference | 95% CI |
| --- | --- | --- |
| reference minus empirical | +0.415 | 0.250 to 0.581 |
| insular beta 2 minus empirical | -0.020 | -0.027 to -0.015 |
| insular strict minus reference | -0.185 | -0.299 to -0.072 |
| insular beta 4 minus reference | -0.228 | -0.340 to -0.114 |
| integrated strict minus reference | -0.002 | -0.071 to 0.069 |

**Interpretation**: These are observations from one initial fraction and a coarse
budget grid, not established effects.

- **The degree sequence alone does not reproduce the empirical result**: The
  degree-preserving reference needs twice the empirical budget (0.10 vs 0.05).
- **Homophily at the empirical level does**: Insular beta 2 reaches about the
  empirical homophily and gives the empirical `Phi_c` and `g_minus`. This suggests
  the community/sign placement explains the gap between empirical and reference.
- **Strong insularity makes the negative cluster easier to break**: Only
  configurations near the insular bound (0.83 to 0.90) let random contrarians reach
  the threshold, in about 40% of polarized runs. This is a threshold effect: for
  random targeting at `Phi = 0.25`, the mean `g_minus` differences from the reference
  have 95% intervals that include zero (e.g. strict -0.047, -0.134 to 0.043).
- **Integrated configurations mostly behave like the reference**: Exception:
  integrated non-worsening and beta 2 cross at 0.05 in many runs. This pattern is
  observed, not explained.

**Limitations**

- Only `f = 0.5` was run.
- Most `Phi_c` differences are one grid step (0.03, 0.05, 0.10). A finer grid between
  0.03 and 0.10 would sharpen them.
- Reference comparisons pair replicates by index, which is a convention, not a
  matched design.
- Leverage, betweenness, eigenvector, and signed-leverage targeting were not run.

**Reproduce**

```bash
uv run python -m mdacn rewire
uv run python -m mdacn rewire-ico
```

Outputs are in `results/rewiring/` and `results/rewiring_ico/` (`phi_c_runs.csv`,
`phi_c_summary.csv`, `g_minus_comparisons.csv`, `final_states.npz`).


## Reproducible edge order and reference ensemble

Seeded sign permutations and rewiring pools now use a canonical ordering of
undirected endpoints. Metadata records `edge_order` and the SHA256 of the signed
edge sequence, so insertion order cannot silently change the experiment. Existing
serialized graphs and raw matrix outputs remain part of the experiment bundle.

The default unbiased reference has 100 independent seeded graph runs; targeted
configurations retain five replicates each. Set `reference_replicates` separately
from `replicates`. This yields 150 rewired graphs in total under the default
configuration list. The reference Monte Carlo p-value resolution improves from
1/6 to 1/101. This increases ensemble resolution but does not establish that the
swap chain samples the constrained graph space uniformly or has mixed fully.
