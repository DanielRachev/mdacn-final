# Graph analytics: extraction, figures, and references

## Extraction and degree comparison

The static aggregate is a simple undirected graph: both event directions are
merged and self-links removed. Equal positive/negative event counts give sign
zero. These edges remain in the unsigned topology used for all measurements here.

We use NetworkX's k-core decomposition, then keep the largest connected component.
Each retained node has at least k neighbours within the k-core; this differs from
filtering on degree in the original graph. Batagelj and Zaveršnik [1] describe the
core decomposition algorithm. The alternative activity-LCC method filters on
original incident event counts and then keeps the largest component.

The default connected 10-core retains 2,896 of 35,776 aggregate nodes (8.1%) and
58,466 of 124,330 edges (47.0%). These figures describe the saved body-link input,
not every possible extraction. The core is a computational subset, not a random
sample of Reddit; the process conclusions apply to this subset.

`degree_comparison.png` and `.pdf` compare the full aggregate and selected core
on log-log and log-linear axes. Each distribution is normalized by its own node
count. Core degrees count only retained neighbors; the core distribution is not
the original degree histogram cut at ten. Degree-zero nodes are omitted only
from log-log axes and remain in the exported distributions.

The existing core degree-fit figure compares descriptive discrete power-law and
Poisson fits. Neither a straight-looking tail nor the smaller AIC establishes a
scale-free network. We do not compute a bootstrap goodness-of-fit test.

## Communities and layout

Newman [2] explains modularity as a community quality function. Our implementation
uses the Louvain heuristic of Blondel et al. [3], with unsigned, unweighted edges
and a recorded seed. Louvain is not Newman's spectral community algorithm, and
its output is not a guaranteed global modularity maximum.

`core_communities.png` and `.pdf` show all core nodes and edges, coloured by the
same labels exported in the core GraphML and community table. Layout uses an
unsigned force-directed spring calculation with 100 iterations and the pipeline
seed. One set of coordinates is reused for both figures and saved with node
names and community labels in `core_layout.csv`. Dense edges are drawn faintly;
node positions and colours do not identify observed opinions or political groups.

[Community interpretations](community_interpretation.md) describe the nine saved
groups using representative members and selective source checks, with confidence
and an annotated copy of the same layout.

## Spectral measurements and opinion GCCs

The spectral radius is the largest eigenvalue of the unsigned adjacency matrix.
Algebraic connectivity is the second-smallest eigenvalue of the combinatorial
Laplacian `L = D - A`, following Fiedler [4]. Van Mieghem [5] covers adjacency and
Laplacian spectra in network analysis. These characterise unsigned topology;
they do not directly predict convergence of our signed opinion model.

`compute_opinion_gcc` measures the largest connected component induced by each
opinion on the fixed unsigned core. Both `g_minus` and `g_plus` divide by the
original core size, including pinned nodes. Empty opinion subgraphs give zero.
These differ from raw opinion fractions when a state is fragmented. Keep each
measurement alongside its simulation stop reason; a terminal cycle phase is not
a steady state.

`compute_opinion_components` additionally measures the second-largest component
of each opinion, using the same denominator and node order. Component plots show
individual runs, both component ranks, and nodes outside the largest component
on separate scales. See the [NCO baseline](nco_baseline.md#component-measurements-and-refinement).

## References

1. Batagelj, V., and Zaveršnik, M. (2003). *An O(m) Algorithm for Cores
   Decomposition of Networks*. arXiv:cs/0310049.
   [Paper](https://arxiv.org/abs/cs/0310049),
   [DOI](https://doi.org/10.48550/arXiv.cs/0310049).
2. Newman, M. E. J. (2006). *Modularity and community structure in networks*.
   Proceedings of the National Academy of Sciences, 103(23), 8577–8582.
   [DOI](https://doi.org/10.1073/pnas.0601602103),
   [Author preprint](https://arxiv.org/abs/physics/0602124).
3. Blondel, V. D., Guillaume, J.-L., Lambiotte, R., and Lefebvre, E. (2008).
   *Fast unfolding of communities in large networks*. Journal of Statistical
   Mechanics: Theory and Experiment, 2008(10), P10008.
   [DOI](https://doi.org/10.1088/1742-5468/2008/10/P10008),
   [Author preprint](https://arxiv.org/abs/0803.0476).
4. Fiedler, M. (1973). *Algebraic connectivity of graphs*. Czechoslovak
   Mathematical Journal, 23(2), 298–305.
   [DOI](https://doi.org/10.21136/CMJ.1973.101168),
   [Journal archive](https://dml.cz/handle/10338.dmlcz/101168).
5. Van Mieghem, P. (2011). *Graph Spectra for Complex Networks*. Cambridge
   University Press, first edition.
   [DOI](https://doi.org/10.1017/CBO9780511921681).

Bibliographic details checked against the linked primary sources on 18 October
2026. The [publisher's front matter](https://assets.cambridge.org/97810093/66809/frontmatter/9781009366809_frontmatter.pdf)
confirms the first edition's 2011 publication; the second edition is from 2023.
