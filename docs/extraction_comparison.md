# Core extraction and title-link feasibility

The connected body-link 10-core remains a reasonable computational baseline:
it retains 55.5% of body events using 8.1% of the aggregate nodes. The nearby
k-core settings provide a small sensitivity range. Activity >= 100 selects a
much smaller, denser graph, but does not impose an internal minimum degree.

Title links offer additional coverage, but replacing body links changes most
of the observed edges. Keep body links for the existing experiments. Treat a
combined body/title dataset as an optional extension requiring new preprocessing
and matched experiments before making claims about opinion dynamics.

These are Jacek's Phase 2 tasks 4-6, measured on 22 October 2026 from an
implementation based on PR7 (`b7726b6`). Georgi owns the subsequent NCO/ICO
configuration comparisons. No opinion-dynamics sweep was run for this note.

## Reproduce

Run from the repository root with the locked dependencies:

```bash
uv sync --locked
uv run python -m mdacn setup
MPLBACKEND=Agg uv run python -m mdacn compare-extractions --seed 0
uv run python -m mdacn setup --dataset title
uv run python -m mdacn compare-link-sources
```

`setup` downloads only, keeps an existing file, and defaults to body links.
`--output` overrides its destination. Neither comparison downloads data itself.
Use `--input`, `--ks`, `--min-interactions`, `--processed-dir`, and `--results-dir`
to customize extraction. The source audit accepts `--body-input`, `--title-input`,
`--k`, and `--results-dir`. Default output roots are separate from the original
`data/processed/core.graphml` and `results/topology/`.

Both inputs come from the [SNAP Reddit Hyperlink Network](https://snap.stanford.edu/data/soc-RedditHyperlinks.html),
described by Kumar et al., *Community Interaction and Conflict on the Web*,
WWW 2018. File identities are recorded in the generated JSON and the source
snapshot CSV. SHA-256 values for this run:

```text
body  9bbf57511dc0da7369c2e896d865b68f170136def4f223f4c9e9e10818f4bfb5
title b6ea49dcbc83f948b177b341cace8c37fb8f12fc1b0e18bbfd1ba4630b699893
```

## Extraction results

All four configurations use the same body input: 286,561 events, 35,776 nodes,
and 124,330 undirected edges. No self-links occur in this input. Aggregation
merges both directions; edge sign is the sign of positive minus negative event
counts. Ties remain in unsigned topology with zero signed influence.

| Extraction | Nodes | Edges | Nodes retained | Edges retained | Events retained | Negative edges | Zero-sign edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k = 8 | 3,672 | 64,952 | 10.26% | 52.24% | 61.00% | 6.98% | 2.37% |
| k = 10 | 2,896 | 58,466 | 8.09% | 47.02% | 55.55% | 7.17% | 2.51% |
| k = 12 | 2,346 | 52,859 | 6.56% | 42.52% | 51.30% | 7.27% | 2.67% |
| Activity >= 100 | 957 | 25,864 | 2.67% | 20.80% | 38.34% | 6.90% | 3.38% |

Node and edge retention divide by full-aggregate counts. Event retention counts
all non-self events whose two endpoints survive, divided by all non-self input
events. Repeated events therefore contribute repeatedly. Each method keeps its
largest connected component. Activity uses each node's incident event count in
the original aggregate, before filtering.

| Extraction | Minimum degree | Mean degree | Modularity | Mean clustering | Spectral radius | Algebraic connectivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k = 8 | 8 | 35.38 | 0.333 | 0.316 | 127.381 | 0.986 |
| k = 10 | 10 | 40.38 | 0.312 | 0.322 | 126.676 | 1.075 |
| k = 12 | 12 | 45.06 | 0.301 | 0.322 | 125.800 | 3.275 |
| Activity >= 100 | 1 | 54.05 | 0.253 | 0.396 | 112.318 | 0.582 |

The activity-selected graph has the highest average degree, yet retains leaves
and has the lowest algebraic connectivity. A large number of repeated events
does not imply many distinct neighbors. The k-core alternatives have similar
negative-edge proportions, while graph size and connectivity still change.
These structural results support keeping k = 10 as the existing middle setting;
they do not establish that NCO or ICO outcomes are insensitive to extraction.

The comparison uses unsigned, unweighted topology and Louvain seed 0. Nodes and
edges are inserted in a fixed alphabetical/lexicographic order before community
detection. This order was not fixed in older characterization outputs, so their
community labels and modularity may differ even with the same seed. The new
10-core's node set, edges, and signs reproduce the saved original core.
Community labels are local to each extraction. See [graph methods](graph_analytics.md)
for definitions and references; these measurements do not establish ideological
communities or scale-free structure.

Full-precision results are in `results/extraction_comparison/comparison.csv`
and `comparison.json`. The compact committed [extraction snapshot](extraction_comparison.csv)
rounds floating-point values to twelve significant digits. The generated PNG/PDF
artifacts are `topology_table`, `degree_comparison`, and `retention_and_signs`.
Degree distributions are normalized separately within each graph.

## Title-link coverage and overlap

| Quantity | Body | Title |
| --- | ---: | ---: |
| Directed event rows | 286,561 | 571,927 |
| Subreddit names | 35,776 | 54,075 |
| Directed source-target pairs | 137,821 | 234,792 |
| Undirected edges | 124,330 | 220,151 |
| Negative event fraction | 7.35% | 10.69% |
| Negative aggregate-edge fraction | 5.68% | 6.13% |
| Zero-sign aggregate-edge fraction | 1.59% | 1.83% |
| Connected 10-core nodes | 2,896 | 4,677 |
| Connected 10-core edges | 58,466 | 122,379 |

Both files span 31 December 2013 through 30 April 2017. The label describes
whether the source post is explicitly negative toward its target; +1 includes
neutral and positive posts. It is not an observed subreddit opinion.

| Quantity | Shared | Body covered by title | Title covered by body |
| --- | ---: | ---: | ---: |
| Subreddit names | 22,671 | 63.37% | 41.93% |
| Directed source-target pairs | 32,970 | 23.92% | 14.04% |
| Undirected edges | 34,814 | 28.00% | 15.81% |
| Connected 10-core members | 2,414 | 83.36% | 51.61% |
| `(source, target, POST_ID)` keys | 0 | 0.00% | 0.00% |

Each coverage fraction divides the shared count by its column's original count.
The source audit also exports union counts, source-only counts, and Jaccard
fractions. On shared aggregate edges, 30,515 of 34,814 signs agree (87.65%);
4,299 differ, including transitions involving zero. Agreement is descriptive
and includes the common positive class.

There are no repeated post-link keys within either downloaded file and no
identical keys across the two files. This concerns the full three-field key,
not just POST_ID. The code preserves repeated events and separately measures
shared occurrences using the minimum multiplicity for each key.

On the fixed 2,896 body-core nodes, title links cover 2,842 node names (98.14%)
but only 19,583 of the original 58,466 edges (33.49%). The title-induced graph
on this fixed node set has 76,040 edges, with a 2,828-node largest component and
68 isolates. Its additional edges do not replace the missing body edges. A
separately extracted title 10-core also removes 482 body-core members and adds
2,263 members, so replacing the input changes both topology and population.

The downloaded files contain 858,488 event keys and a union of 67,180 subreddit
names. SNAP's published summary lists 858,490 edges and 55,863 nodes; those
headline totals do not match the downloaded-file union. The counts here use the
checksummed TSVs directly. All node names in both files are already lowercase.

The [source summary](link_source_summary.csv) and [overlap snapshot](link_source_overlap.csv)
are committed with this note. Detailed JSON, CSVs, and `core_membership.csv`
are generated under `results/link_sources/`. Title data provide useful extra
observations, but neither size nor overlap establishes better intervention
performance. Any combined-input extension should define duplicate handling and
regenerate aggregates, cores, matrices, centralities, and experiments together.

## Handoff to Georgi

Each configuration has `core.graphml`, `unsigned.npz`, `signed.npz`,
`communities.csv`, and `nodes.json` under
`data/processed/extraction_comparison/<configuration>/`. Its results directory
contains `metrics.json` with input checksum, extraction settings, seed, node-file
checksum, output location, and measurements. Empty selections export empty
graphs/matrices, status `empty`, and missing unsupported metrics.

```python
import json
from pathlib import Path

import networkx as nx
from scipy.sparse import load_npz

root = Path("data/processed/extraction_comparison/k_core_8")
graph = nx.read_graphml(root / "core.graphml")
nodes = tuple(json.loads((root / "nodes.json").read_text()))
unsigned = load_npz(root / "unsigned.npz")
signed = load_npz(root / "signed.npz")
```

Opinion vectors and targeting arrays must follow that configuration's node
order. Different extractions have different populations; matching a root seed
does not make their initial states a node-for-node paired comparison. The
structural comparison's seed controls community detection only. Generated data
and figures remain ignored; rerun the commands to recreate them after cloning.
