# Reddit network analysis

Network characterisation and unsigned/signed Non-Consensus Opinion (NCO)
simulation on the SNAP Reddit body hyperlink dataset.

## Run

Requires Python 3.12+ and uv. Run from this directory:

```bash
uv sync --locked
uv run python -m mdacn run
```

Or run each stage separately. Config supplies all input and output paths:

```bash
uv run python -m mdacn setup
uv run python -m mdacn characterise
uv run python -m mdacn centrality
uv run python -m mdacn targeting-scores
uv run python -m mdacn directed-roles
uv run python -m mdacn features
uv run python -m mdacn nco
uv run python -m mdacn sign-shuffle
uv run python -m mdacn ico
uv run python -m mdacn node-impact
uv run python -m mdacn rewire
uv run python -m mdacn rewire-ico
uv run python -m mdacn incident-signs
uv run python -m mdacn signed-triads
uv run python -m mdacn label-audit
uv run python -m mdacn reports
```

`directed-roles` runs before `features` because it writes `incident_signs.csv`.
`signed-triads` runs after `rewire` to use its reference replicates; run on its own
without them, it skips that model.

Temporal P-O-X prediction is an optional extension and is not part of `run`:

```bash
uv run python -m mdacn temporal-balance
```

| Command | Action |
| --- | --- |
| `run` | Run the stages in the first block above, in that order (not the optional `temporal-balance`). |
| `setup` | Download the body TSV (or title TSV with `--dataset title`); reuse an existing file. |
| `characterise` | Aggregate interactions, extract a core, compute metrics, and export graphs and matrices. |
| `compare-extractions` | Compare k = 8, 10, 12 and activity >= 100, exporting each core separately. |
| `compare-link-sources` | Compare body/title coverage, overlap, signs, and 10-core membership. |
| `sensitivity` | Replay NCO and selected ICO outcomes on the exported cores; compare topology, signs, and outcomes. |
| `combined-results` | Assemble saved NCO, sign-shuffle, ICO, temporal, rewiring, impact, and sensitivity outputs into one report. |
| `centrality` | Run the existing degree, betweenness, eigenvector, and leverage implementations and save aligned scores. |
| `targeting-scores` | Save the proposed signed-leverage and temporal-activity bridge scores used by `ico`. |
| `features` | Read expected contributor output files, then merge and validate their scores. |
| `nco` | Save unsigned and signed simulation outcomes on the exported core. |
| `sign-shuffle` | Save paired empirical/sign-shuffled simulation outcomes. |
| `ico` | Load saved matrices and scores, then run all seven targeting strategies on the unsigned and signed cores and save the budget sweep, `Phi_c`, and backfire columns. |
| `node-impact` | Load saved scores and save paired single-node intervention outcomes for every core node. |
| `reports` | Save summaries, impact statistics, and plots from stored experiment results. |
| `rewire` | Export degree-preserving rewired graphs: unbiased reference, insular, and integrated. |
| `rewire-ico` | Compare signed ICO Phi_c on the empirical and rewired graphs. |
| `incident-signs` | Compare the per-node negative incident ratio, P(r_neg), and community sign rates with sign-shuffled cores. |
| `signed-triads` | Count signed triads and the balanced fraction against sign shuffling, degree-preserving rewiring, and a descriptive Erdős–Rényi graph. |
| `temporal-balance` | Optional, outside `run`: predict each triangle's closing-edge sign from its earlier wedge (P-O-X), against post-label and timestamp shuffles. |
| `directed-roles` | Directed signed roles, reciprocity, `r_neg`, and signed bridge ratios; writes `incident_signs.csv` for `features`. |
| `label-audit` | Audit post-level labels, repeated rows per `POST_ID`, and counts at each filtering stage. |

Stages overwrite their generated outputs. Missing prerequisites produce an error
naming the stage to run first. After changing the core, rerun `centrality` and
the contributor exports, then `features` before
`node-impact`. `reports` can be rerun independently and uses saved experiment
settings for summaries and current analysis settings for bootstrap calculations.

Phase 2 node-impact commands, input tables, and output interpretation:
[Node-impact guide](docs/node_impact.md).
Targeting strategies, the backfire measurement, and `Phi_c`:
[Contrarian targeting benchmark](docs/interventions.md).

Alternative core: filter by original incident interaction count, then keep the
largest connected component.

```bash
uv run python -m mdacn characterise --core-method activity_lcc --min-interactions 100
```

Use each command's `--help` for input paths, output paths, and experiment options.
Shared paths, core extraction settings, and simulation defaults live in
[config.py](src/mdacn/config.py). Command-line options override these defaults;
`NODE_IMPACT` contains the node-impact workflow's overrides, `REWIRING` and
`REWIRED_ICO` the Null Model B settings, and `INCIDENT_SIGNS`, `SIGNED_TRIADS`, and
`TEMPORAL_BALANCE` the signed-structure settings.
Contributor filenames are defined in `stage_paths()` in `config.py`,
relative to `PATHS["processed_dir"]`. The default files and required columns are:

| File | Columns |
| --- | --- |
| `centrality.csv` | `node`, raw-count `degree`, `leverage`, `betweenness` |
| `signed_leverage.csv` | `node`, `signed_leverage` (written by `targeting-scores`) |
| `incident_signs.csv` | `node`, `negative_incident_ratio`, `signed_bridge_ratio` |
| `temporal_activity.csv` | `node`, `temporal_activity` |

`centrality` also exports normalized `degree_centrality` and `eigenvector`.
`targeting-scores` writes `signed_leverage.csv` and the bridge score it consumes
from `ICO`. `directed-roles` writes `incident_signs.csv`; `temporal_activity.csv`
still comes from the pending contributor implementation.
All seven impact scores are required for every core node.
The `features` stage writes the shared score table consumed by `node-impact`.
Scores are kept unchanged. Missing or invalid required scores produce an error;
neither `features` nor `node-impact` calculates replacements. Contributor score
exports for temporal activity remain prerequisites until their producing stage
is connected to `run`.

Signed leverage is `l_signed(i) = (1/d_i) * sum_j W_ij * max(0, (d_i - d_j)/(d_i + d_j))`
with the unsigned degree `d_i`, so zero-sign edges count in the denominator.
The bridge score is `sqrt(bridge_fraction * activity / max(activity))`, where the
bridge fraction is the share of a node's neighbours outside its community.
Activity is the core graph's `n_interactions` count unless `--activity` supplies
a per-node export; a full-window score is retrospective. `targeting-scores` also
writes `targeting_scores.json` with the input checksums and both definitions.

Simulation stages load `unsigned.npz`, `signed.npz`, and `nodes.json` produced by
`characterise`. Missing or changed matrix artifacts cause an error. A custom
`--graph` requires the matching artifacts and `network_metadata.json` beside it.
`ico` takes its budget grid, models, strategies, and `Phi_c` threshold from
`ICO`; fractions and repeated-run settings use `SIMULATION`. `reports` reads its
saved results without rerunning it.
`rewire`, `rewire-ico`, and the signed-structure stages take the same `--graph`; the
stages that read raw events also take `--input`.

`signed-triads` reads Null Model B replicates from `--rewired-dir`
(default `results/rewiring/graphs/reference`, written by the rewiring command) and
skips that model if the directory is missing.

Run Jacek's structural comparisons separately from the simulation experiments:

```bash
uv run python -m mdacn compare-extractions
uv run python -m mdacn setup --dataset title
uv run python -m mdacn compare-link-sources
```

The extraction comparison uses seed 0 and writes to its own output directories.
It exports each graph with its matching matrices and `nodes.json`; changing
extraction changes the node set. The title audit measures feasibility without
running opinion dynamics. See [measured findings and handoff](docs/extraction_comparison.md).

Replay the simulation protocol on those exported cores to check whether the
default 10-core changes the conclusions:

```bash
uv run python -m mdacn sensitivity
```

The default run uses the full `f` sweep for NCO and `random`, `degree`, and
`leverage` at `f = 0.5` for ICO, with both models and 20 realisations. It writes
`results/sensitivity/` tables, figures, and a per-core outcome summary. Use
`--strategies` to add the proposed scores, `--ico-fractions` to widen the ICO
grid, and `--skip-nco` or `--skip-ico` to rerun one half. See
[core extraction sensitivity](docs/sensitivity.md).

Assemble the saved experiment outputs into one report:

```bash
uv run python -m mdacn combined-results
```

The command reads the saved NCO, sign-shuffle, ICO, temporal, rewiring,
node-impact, and sensitivity outputs. Missing inputs are listed in
`results/combined/inputs.json` and `warnings.txt`; the command still writes the
sections it can build. See [combined results](docs/combined_results.md).

`sign-shuffle` shows a progress bar with elapsed time and ETA. Each step completes
one empirical/shuffled pair. The sweep reuses the core topology to avoid copying
the graph for each realization.

NCO, sign-shuffle, and ICO use the same 50-point fraction grid: 0.02 to 1.00 in
steps of 0.02. Set `--fractions` to supply another grid. Seeds are assigned by
recorded fraction order, then realization; changing the grid changes which
initial seeds are assigned to shared fraction points.

The full ICO suite costs about 98,000 runs on that grid, five times the earlier
unsigned three-strategy sweep. Narrow `--fractions` and refine near an observed
threshold instead; see [the targeting benchmark guide](docs/interventions.md).

Sign shuffling also saves the largest and second-largest components for each
opinion, measured on the fixed unsigned core and divided by the full core size.
Its component figures show individual runs, means, and fragmentation, with
steady states, cycles, and update-limit outcomes separated. Initial states are
paired within each experiment; sign shuffling retains its original seed schedule,
which differs from the unsigned/signed baseline even with the same root seed.

Rewiring (Null Model B) uses same-sign double-edge swaps, `(u, v), (x, y)` to
`(u, y), (x, v)`, with random orientation. Each node keeps its unsigned and
positive/negative/zero degree; each edge record follows its first endpoint. Swaps
are steered toward fixed baseline communities: insular puts positive edges inside
and negative edges between communities, integrated does the opposite. Acceptance
rules are unbiased (the reference), strict improvement, non-worsening, and
Metropolis with `--insular-betas` and `--integrated-betas` (defaults in `REWIRING`). Zero-sign edges are
never steered. Each graph stops once its steering score changes by at most
`--tolerance` times the signed edge count over a `--window-per-edge` window, or
at `--attempts-per-edge` times the edge count. Seeds are explicit:
each graph, initial state, and random-targeting draw has a child seed derived from
`--seed` and a named stream. `rewire-ico` replays the same initial states on every
graph, settles signed NCO, pins contrarians at +1, and reports per-run Phi_c, the
first tested budget with `g_minus < 0.5`; its settings are `REWIRED_ICO`. Both
commands reuse saved outputs: `rewire` reads the `centrality` stage's scores and the
core assortativity from `results/topology/metrics.json`, and saves each rewired
graph's targeting scores; `rewire-ico` loads the saved empirical matrices and all
scores from `--rewired-dir` instead of recomputing them.
Invariants and calibration: [Null models](docs/null_models.md).

## Data and methods

- Undirected aggregation merges both directions and removes self-links.
- Edge signs follow the majority of interaction sentiments; equal counts give zero.
- Zero-sign edges remain in unsigned topology and have no signed influence.
- Core extraction uses a k-core or activity threshold, followed by the largest connected component.
- Matrices are symmetric CSR with zero diagonals and a shared alphabetical node order.
- Characterisation uses unsigned topology: degree distribution, assortativity,
  Louvain communities, modularity, clustering, spectral radius, and algebraic connectivity.
- Degree fits compare discrete power-law and Poisson models on positive degrees,
  using maximum likelihood, AIC, and KS distance. No bootstrap goodness-of-fit test.

NCO rules, experiment settings, and measured results: [NCO baseline](docs/nco_baseline.md).
Dataset label, repeated rows per post, filtering counts, structural balance, and
references: [Methods](docs/methods.md).

## Outputs

| Location | Contents |
| --- | --- |
| `data/raw/` | Downloaded body and optional title TSVs. |
| `data/processed/aggregate.graphml`, `core.graphml` | Graphs with interaction counts and edge signs; core includes community labels. |
| `data/processed/unsigned.npz`, `signed.npz`, `nodes.json`, `network_metadata.json` | Aligned core matrices, node order, and file/core identity checks. |
| `data/processed/centrality.csv`, `centrality.json` | Existing centrality scores, raw degrees, and input/output checksums. |
| `data/processed/communities.csv` | Node community labels. |
| `data/processed/signed_leverage.csv`, `temporal_bridge_centrality.csv`, `targeting_scores.json` | Proposed targeting scores, the bridge score consumed by `ico`, and their definitions and checksums. |
| `data/processed/node_features.csv`, `node_features.json` | Node scores and graph/input checksums for reuse by later stages. |
| `results/topology/metrics.json` | Network metrics, input metadata, and layout settings. |
| `results/topology/degree_distribution.csv`, `degree_fits.csv`, `degree_distribution.png`, `.pdf` | Core degree distribution, descriptive fits, and figures. |
| `results/topology/aggregate_degree_distribution.csv`, `degree_comparison.png`, `.pdf` | Full aggregate degrees and comparison with the selected core. |
| `results/topology/core_layout.csv`, `core_communities.png`, `.pdf` | Full-core layout coordinates, community labels, and figures. |
| `results/nco/` | Simulation results, raw-fraction and opinion-component summaries/figures, and metadata. |
| `results/sign_shuffle/` | Sign-shuffle results, component summaries/figures, metadata, and node order. |
| `results/ico/` | `intervention_sweep.csv`, `phi_c.csv`, run settings, and node order; `reports` adds `intervention_summary.csv`, per-model comparison PNGs per initial fraction, and `phi_c_comparison.png`. |
| `results/node_impact/` | Run config and `study/` containing frozen core features, paired outcomes, and `analysis/`. |
| `data/processed/extraction_comparison/<configuration>/` | Separate core graphs, aligned matrices and metadata, communities, and node orders for Georgi. |
| `results/extraction_comparison/` | Comparison CSV/JSON, per-core metrics, degree distributions, tables, and figures. |
| `results/link_sources/` | Body/title summaries, overlap CSV/JSON, and a core-membership table. |
| `results/rewiring/` | Rewired matrices, per-node degrees, and validation metadata per graph; summary. Node order is the original `nodes.json`. |
| `results/rewiring_ico/` | Signed ICO runs, packed baseline/terminal states, per-run Phi_c, Phi_c summary, paired `g_minus` comparisons (fixed empirical core), and metadata. |
| `results/incident_signs/` | Per-node `r_neg` with shuffle p-values, binned P(r_neg), summary statistics, community sign rates, figures, and metadata. |
| `results/signed_triads/` | Triad counts per model and realization, null summaries, empirical sign patterns including zero signs, figures, and metadata. |
| `results/temporal_balance/` | Per-triangle closures, prediction metrics overall, per wedge and per quarter, temporal null comparisons, figures, and metadata. |
| `results/directed_roles/` | `node_roles.csv` in `nodes.json` order (directed roles, `r_neg`, bridges), reciprocity, sign fractions, community bridge rates, figures, and metadata. |
| `results/label_audit/` | Label audit for the body file and the core, rows per post, and filtering-stage counts. |

Generated files are ignored by Git. Reruns replace outputs in the selected
directories. Rerun preprocessing after changing data rules.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## References

- [SNAP Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html):
  Kumar et al., *Community Interaction and Conflict on the Web*, WWW 2018.
- [Power-law distributions in empirical data](https://arxiv.org/abs/0706.1062):
  Clauset, Shalizi, and Newman, SIAM Review 51 (2009).
