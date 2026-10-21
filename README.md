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
```

| Command | Action |
| --- | --- |
| `setup` | Download the body TSV; reuse an existing file. |
| `characterise` | Aggregate interactions, extract a core, compute metrics, and export graphs and matrices. |
| `nco` | Compare unsigned and signed dynamics on the exported core. |

Alternative core: filter by original incident interaction count, then keep the
largest connected component.

```bash
uv run python -m mdacn characterise --core-method activity_lcc --min-interactions 100
```

Use each command's `--help` for input paths, output paths, and experiment options.

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

Corrected temporal windowing, uncertainty estimates, and rerun results:
[temporal experiment corrections](docs/temporal_corrections.md).

## Outputs

| Location | Contents |
| --- | --- |
| `data/raw/` | Downloaded body TSV. |
| `data/processed/aggregate.graphml`, `core.graphml` | Graphs with interaction counts and edge signs; core includes community labels. |
| `data/processed/unsigned.npz`, `signed.npz`, `nodes.json` | Aligned core matrices and node order. |
| `data/processed/communities.csv` | Node community labels. |
| `results/metrics.json` | Network metrics and input metadata. |
| `results/degree_distribution.csv`, `degree_fits.csv`, `degree_distribution.pdf` | Degree distribution, fits, and figure. |
| `results/nco/` | Simulation results, summaries, figures, and metadata. |

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
