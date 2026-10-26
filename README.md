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
uv run python -m mdacn features
uv run python -m mdacn nco
uv run python -m mdacn sign-shuffle
uv run python -m mdacn ico
uv run python -m mdacn node-impact
uv run python -m mdacn reports
```

| Command | Action |
| --- | --- |
| `run` | Run all implemented stages in the order shown above. |
| `setup` | Download the body TSV (or title TSV with `--dataset title`); reuse an existing file. |
| `characterise` | Aggregate interactions, extract a core, compute metrics, and export graphs and matrices. |
| `compare-extractions` | Compare k = 8, 10, 12 and activity >= 100, exporting each core separately. |
| `compare-link-sources` | Compare body/title coverage, overlap, signs, and 10-core membership. |
| `centrality` | Run the existing degree, betweenness, eigenvector, and leverage implementations and save aligned scores. |
| `features` | Read expected contributor output files, then merge and validate their scores. |
| `nco` | Save unsigned and signed simulation outcomes on the exported core. |
| `sign-shuffle` | Save paired empirical/sign-shuffled simulation outcomes. |
| `ico` | Load saved matrices and scores, then save the existing unsigned random/degree/leverage budget sweep for each configured initial fraction. |
| `node-impact` | Load saved scores and save paired single-node intervention outcomes for every core node. |
| `reports` | Save summaries, impact statistics, and plots from stored experiment results. |

Stages overwrite their generated outputs. Missing prerequisites produce an error
naming the stage to run first. After changing the core, rerun `centrality` and
the contributor exports, then `features` before
`node-impact`. `reports` can be rerun independently and uses saved experiment
settings for summaries and current analysis settings for bootstrap calculations.

Phase 2 node-impact commands, input tables, and output interpretation:
[Node-impact guide](docs/node_impact.md).

Alternative core: filter by original incident interaction count, then keep the
largest connected component.

```bash
uv run python -m mdacn characterise --core-method activity_lcc --min-interactions 100
```

Use each command's `--help` for input paths, output paths, and experiment options.
Shared paths, core extraction settings, and simulation defaults live in
[config.py](src/mdacn/config.py). Command-line options override these defaults;
`NODE_IMPACT` contains the node-impact workflow's overrides.
Contributor filenames are defined in `stage_paths()` in `config.py`,
relative to `PATHS["processed_dir"]`. The default files and required columns are:

| File | Columns |
| --- | --- |
| `centrality.csv` | `node`, raw-count `degree`, `leverage`, `betweenness` |
| `signed_leverage.csv` | `node`, `signed_leverage` |
| `incident_signs.csv` | `node`, `negative_incident_ratio`, `signed_bridge_ratio` |
| `temporal_activity.csv` | `node`, `temporal_activity` |

`centrality` also exports normalized `degree_centrality` and `eigenvector`.
The other three files still come from the pending contributor implementations.
All seven impact scores are required for every core node.
The `features` stage writes the shared score table consumed by `node-impact`.
Scores are kept unchanged. Missing or invalid required scores produce an error;
neither `features` nor `node-impact` calculates replacements. Contributor score
exports for signed leverage, incident signs/bridges, and temporal activity remain
prerequisites until their producing stages are connected to `run`.

Simulation stages load `unsigned.npz`, `signed.npz`, and `nodes.json` produced by
`characterise`. Missing or changed matrix artifacts cause an error. A custom
`--graph` requires the matching artifacts and `network_metadata.json` beside it.
`ico` takes its budget grid from `ICO["budgets"]`; fractions and repeated-run
settings use `SIMULATION`. `reports` reads its saved results without rerunning it.

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

`sign-shuffle` shows a progress bar with elapsed time and ETA. Each step completes
one empirical/shuffled pair. The sweep reuses the core topology to avoid copying
the graph for each realization.

NCO, sign-shuffle, and ICO use the same 50-point fraction grid: 0.02 to 1.00 in
steps of 0.02. Set `--fractions` to supply another grid. Seeds are assigned by
recorded fraction order, then realization; changing the grid changes which
initial seeds are assigned to shared fraction points.

Sign shuffling also saves the largest and second-largest components for each
opinion, measured on the fixed unsigned core and divided by the full core size.
Its component figures show individual runs, means, and fragmentation, with
steady states, cycles, and update-limit outcomes separated. Initial states are
paired within each experiment; sign shuffling retains its original seed schedule,
which differs from the unsigned/signed baseline even with the same root seed.

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

## Outputs

| Location | Contents |
| --- | --- |
| `data/raw/` | Downloaded body and optional title TSVs. |
| `data/processed/aggregate.graphml`, `core.graphml` | Graphs with interaction counts and edge signs; core includes community labels. |
| `data/processed/unsigned.npz`, `signed.npz`, `nodes.json`, `network_metadata.json` | Aligned core matrices, node order, and file/core identity checks. |
| `data/processed/centrality.csv`, `centrality.json` | Existing centrality scores, raw degrees, and input/output checksums. |
| `data/processed/communities.csv` | Node community labels. |
| `data/processed/node_features.csv`, `node_features.json` | Node scores and graph/input checksums for reuse by later stages. |
| `results/topology/metrics.json` | Network metrics, input metadata, and layout settings. |
| `results/topology/degree_distribution.csv`, `degree_fits.csv`, `degree_distribution.png`, `.pdf` | Core degree distribution, descriptive fits, and figures. |
| `results/topology/aggregate_degree_distribution.csv`, `degree_comparison.png`, `.pdf` | Full aggregate degrees and comparison with the selected core. |
| `results/topology/core_layout.csv`, `core_communities.png`, `.pdf` | Full-core layout coordinates, community labels, and figures. |
| `results/nco/` | Simulation results, raw-fraction and opinion-component summaries/figures, and metadata. |
| `results/sign_shuffle/` | Sign-shuffle results, component summaries/figures, metadata, and node order. |
| `results/ico/` | `intervention_sweep.csv`, run settings and node order; `reports` adds `intervention_summary.csv` and comparison PNGs per initial fraction. |
| `results/node_impact/` | Run config and `study/` containing frozen core features, paired outcomes, and `analysis/`. |
| `data/processed/extraction_comparison/<configuration>/` | Separate core graphs, aligned matrices and metadata, communities, and node orders for Georgi. |
| `results/extraction_comparison/` | Comparison CSV/JSON, per-core metrics, degree distributions, tables, and figures. |
| `results/link_sources/` | Body/title summaries, overlap CSV/JSON, and a core-membership table. |

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
