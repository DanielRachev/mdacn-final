# Reddit network project

## Setup
Jacek's pipeline downloads the Reddit body hyperlink dataset, builds signed
interaction graphs, and characterizes an extracted core. Daniel's NCO engine and
Polly's centrality/intervention modules retain their separate interfaces.

Run these commands from this repository with `uv` installed:
## Setup and analysis

Run from the repository directory with Python 3.12+ and `uv` installed:

```bash
uv sync --locked
uv run python -c "import mdacn; print('Package import OK')"
uv run python -m mdacn setup
uv run python -m mdacn characterise
```

`setup` downloads only the raw body TSV from SNAP. It reuses an existing file and
writes through a `.part` file so interrupted downloads do not leave a completed
TSV. To download again, remove the existing TSV and rerun `setup`. An alternative
location can be supplied with `setup --output PATH`.

`characterise` defaults to `data/raw/soc-RedditHyperlinks-body.tsv`, a 10-core,
its largest connected component, and community seed `0`. This keeps a densely
connected subgraph for the simulations. Thresholds affect which communities are
represented; the core is not a random sample of the full network.

```bash
uv run python -m mdacn characterise \
  --input data/raw/soc-RedditHyperlinks-body.tsv \
  --core-method k_core --k 10 --seed 0

# Alternative: filter by incident event count, then keep the largest component.
uv run python -m mdacn characterise \
  --core-method activity_lcc --min-interactions 100 \
  --processed-dir data/processed/activity --results-dir results/activity
```

Activity counts repeated interactions in both directions on the original
aggregate graph. It is measured before the threshold filters nodes. Unlike a
k-core, this method does not require each retained node to have a minimum number
of distinct retained neighbours. `--help` lists the arguments for either command.
For a headless machine, set `MPLBACKEND=Agg` when running characterisation.

## Saved artifacts

Raw data, processed data, and results are ignored by Git and generated locally.
Files in the selected output directories are replaced when characterisation runs.

| Location                                             | Contents                                                                                                                |
|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `data/processed/aggregate.graphml`                   | Full undirected aggregate, including isolated nodes.                                                                    |
| `data/processed/core.graphml`                        | Connected extracted core, with community labels.                                                                        |
| `data/processed/unsigned.npz`, `signed.npz`          | Binary and signed core CSR matrices.                                                                                    |
| `data/processed/nodes.json`                          | Alphabetical node order shared by both matrices.                                                                        |
| `data/processed/communities.csv`                     | Subreddit names and Louvain labels in matrix order.                                                                     |
| `results/metrics.json`                               | Graph sizes, topological/spectral metrics, thresholds, seed, input checksum, observed date range, and package versions. |
| `results/degree_distribution.csv`, `degree_fits.csv` | Empirical degree probabilities and discrete fit summaries.                                                              |
| `results/degree_distribution.png`, `.pdf`            | Empirical and fitted degree curves on log-log and log-linear axes.                                                      |

Both GraphML files can be loaded by NetworkX without running characterisation
again. Node IDs are subreddit names. Edges have integer `n_positive`, `n_negative`,
and `sign` attributes. Node `n_interactions` retains the incident event count from
the full aggregate graph, even in the saved core. Core node `community` is the
seeded Louvain label. Graph attributes record the input checksum and source; the
core also records its extraction method and threshold.

## Interface with Daniel and Polly

Load either saved graph for NetworkX analysis:

```python
from pathlib import Path

import networkx as nx

# Polly: graph topology and signed edge attributes for centrality/interventions.
graph = nx.read_graphml(Path("data/processed/core.graphml"))
full_graph = nx.read_graphml(Path("data/processed/aggregate.graphml"))
```

Daniel can build matrices using the existing shared contract:

```python
from mdacn.data import build_matrices

network = build_matrices(graph)
# network.nodes: alphabetical subreddit names
# network.unsigned: symmetric binary CSR adjacency
# network.signed: symmetric +/-1 CSR adjacency
```

Both matrices have a zero diagonal and `float64` values to prevent overflow in
neighbourhood sums. Opinion states, scores, and contrarian masks must follow
`network.nodes`. Daniel passes `network.unsigned` or `network.signed` to his NCO
simulator. Polly uses the same graph and node order for her ICO simulator and
cluster measurements. The graph's signs/counts do not change unsigned topology.

The matrix files can also be loaded directly:

```python
import json
from scipy.sparse import load_npz
from mdacn.contracts import NetworkMatrices

network = NetworkMatrices(
    nodes=tuple(json.loads(Path("data/processed/nodes.json").read_text())),
    unsigned=load_npz("data/processed/unsigned.npz"),
    signed=load_npz("data/processed/signed.npz"),
)
```

`load_interactions(Path(...))` preserves directed and repeated events in `source`,
`target`, `sentiment`, and parsed `timestamp` columns for later temporal analysis.
Timestamps keep the source's unspecified timezone. Invalid required values are
reported with TSV line numbers. Aggregation discards self-links, merges both
directions, and assigns `+1` when positive and negative counts tie.

## Characterisation methods

All characterisation uses unsigned, unweighted topology. Louvain labels use the
specified seed; modularity evaluates that partition. Spectral radius is the
largest adjacency eigenvalue. Algebraic connectivity is the second-smallest
eigenvalue of the combinatorial Laplacian `D - A`; it is zero for disconnected or
singleton graphs. Larger eigenvalue calculations use sparse matrices.

Degree baselines use discrete maximum likelihood: a power law with exponent
`alpha` and a Poisson distribution with rate `lambda`. Both use the same observed
positive-degree sample and are conditioned on `k >= k_min`, where `k_min` is the
smallest positive observed core degree. Their support has no upper truncation.
The CSV records sample size, excluded isolates, log likelihood, AIC, and discrete
KS distance. Lower AIC compares these two models; KS distance describes their
mismatch with the empirical CDF. This is not a bootstrap goodness-of-fit test or
automatic tail-cutoff selection, and does not establish scale-free structure.
Constant-degree or empty positive-degree samples are explicitly marked as
unfitted. Other fitting/eigensolver failures stop the command with an error.

Undefined metrics, such as assortativity on a regular graph, are `null` in JSON;
unavailable fit values are blank in CSV. High modularity measures community
separation; it does not establish an opinion echo chamber. Dataset `+1` labels
mean neutral or positive interaction, and `-1` labels mean negative interaction.
Neither label measures factual correctness.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The focused checks cover parsing/aggregation, graph extraction and reloading,
matrix alignment and large-hub sums, known spectra, baseline fits, and a toy
pipeline run. Daniel's six existing placeholder tests remain skipped; this is
not evidence that the NCO simulator has been implemented or verified.

## Dataset and fitting references

Dataset: [SNAP Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html),
body file only. Citation: S. Kumar, W. L. Hamilton, J. Leskovec, and D. Jurafsky,
*Community Interaction and Conflict on the Web*, WWW 2018. Cite this dataset in
the later slides.

Fitting background: A. Clauset, C. R. Shalizi, and M. E. J. Newman,
[Power-law distributions in empirical data](https://arxiv.org/abs/0706.1062),
SIAM Review 51, 661–703 (2009).