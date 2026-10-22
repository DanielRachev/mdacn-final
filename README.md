# Reddit network analysis

Network characterisation and unsigned/signed Non-Consensus Opinion (NCO)
simulation on the SNAP Reddit body hyperlink dataset.

## Run

Requires Python 3.12+ and uv. Run from this directory:

```bash
uv sync --locked
uv run python -m mdacn setup
uv run python -m mdacn characterise --k 10 --seed 0
uv run python -m mdacn nco --realizations 20 --seed 0 --max-steps 1000
uv run python -m mdacn sign-shuffle --realizations 20 --seed 0 --max-steps 1000
```

| Command        | Action                                                                                   |
|----------------|------------------------------------------------------------------------------------------|
| `setup`        | Download the body TSV (or title TSV with `--dataset title`); reuse an existing file.     |
| `characterise` | Aggregate interactions, extract a core, compute metrics, and export graphs and matrices. |
| `compare-extractions` | Compare k = 8, 10, 12 and activity >= 100, exporting each core separately.           |
| `compare-link-sources` | Compare body/title coverage, overlap, signs, and 10-core membership.               |
| `nco`          | Compare unsigned and signed dynamics on the exported core.                               |
| `sign-shuffle` | Compare empirical signed dynamics with shuffled edge signs on the same core. |

Alternative core: filter by original incident interaction count, then keep the
largest connected component.

```bash
uv run python -m mdacn characterise --core-method activity_lcc --min-interactions 100
```

Use each command's `--help` for input paths, output paths, and experiment options.

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

Both experiment commands use the same 50-point fraction grid: 0.02 to 1.00 in
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

| Location                                                                                         | Contents                                                                          |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `data/raw/`                                                                                      | Downloaded body TSV.                                                              |
| `data/processed/aggregate.graphml`, `core.graphml`                                               | Graphs with interaction counts and edge signs; core includes community labels.    |
| `data/processed/unsigned.npz`, `signed.npz`, `nodes.json`                                        | Aligned core matrices and node order.                                             |
| `data/processed/communities.csv`                                                                 | Node community labels.                                                            |
| `results/topology/metrics.json`                                                                  | Network metrics, input metadata, and layout settings.                             |
| `results/topology/degree_distribution.csv`, `degree_fits.csv`, `degree_distribution.png`, `.pdf` | Core degree distribution, descriptive fits, and figures.                          |
| `results/topology/aggregate_degree_distribution.csv`, `degree_comparison.png`, `.pdf`            | Full aggregate degrees and comparison with the selected core.                     |
| `results/topology/core_layout.csv`, `core_communities.png`, `.pdf`                               | Full-core layout coordinates, community labels, and figures.                      |
| `results/nco/`                                                                                   | Simulation results, raw-fraction and opinion-GCC summaries/figures, and metadata. |
| `results/sign_shuffle/` | Sign-shuffle results, component summaries/figures, metadata, and node order. |
| `data/processed/extraction_comparison/<configuration>/` | Separate core graphs, matrices, communities, and node orders for Georgi. |
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
