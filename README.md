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
uv run python -m mdacn rewire --replicates 5 --seed 0
uv run python -m mdacn rewire-ico --initializations 20 --f 0.5 --seed 0
```

| Command        | Action                                                                                   |
|----------------|------------------------------------------------------------------------------------------|
| `setup`        | Download the body TSV; reuse an existing file.                                           |
| `characterise` | Aggregate interactions, extract a core, compute metrics, and export graphs and matrices. |
| `nco`          | Compare unsigned and signed dynamics on the exported core.                               |
| `sign-shuffle` | Compare empirical signed dynamics with shuffled edge signs on the same core. |
| `rewire`       | Export degree-preserving rewired graphs: unbiased reference, insular, and integrated. |
| `rewire-ico`   | Compare signed ICO Phi_c on the empirical and rewired graphs. |

Alternative core: filter by original incident interaction count, then keep the
largest connected component.

```bash
uv run python -m mdacn characterise --core-method activity_lcc --min-interactions 100
```

Use each command's `--help` for input paths, output paths, and experiment options.

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

Rewiring (Null Model B) uses same-sign double-edge swaps, `(u, v), (x, y)` to
`(u, y), (x, v)`, with random orientation. Each node keeps its unsigned and
positive/negative/zero degree; each edge record follows its first endpoint. Swaps
are steered toward fixed baseline communities: insular puts positive edges inside
and negative edges between communities, integrated does the opposite. Acceptance
rules are unbiased (the reference), strict improvement, non-worsening, and
Metropolis with `--insular-betas` and `--integrated-betas`. Zero-sign edges are
never steered. Each graph stops once its steering score changes by at most
`--tolerance` times the signed edge count over a `--window-per-edge` window, or
at `--attempts-per-edge` times the edge count. Seeds are explicit:
each graph, initial state, and random-targeting draw has a child seed derived from
`--seed` and a named stream. `rewire-ico` replays the same initial states on every
graph, settles signed NCO, pins contrarians at +1, and reports per-run Phi_c, the
first tested budget with `g_minus < 0.5`. Both commands reuse saved outputs:
`rewire` reads `data/processed/centrality.csv` (`--centrality`) and the core
assortativity from `--metrics`, and saves each rewired graph's targeting scores;
`rewire-ico` loads empirical matrices from `--processed-dir` and all scores from
`--rewired-dir` instead of recomputing them.
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
| `results/rewiring/` | Rewired matrices, per-node degrees, and validation metadata per graph; summary. Node order is the original `nodes.json`. |
| `results/rewiring_ico/` | Signed ICO runs, packed baseline/terminal states, per-run Phi_c, Phi_c summary, paired `g_minus` comparisons (fixed empirical core), and metadata. |

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
