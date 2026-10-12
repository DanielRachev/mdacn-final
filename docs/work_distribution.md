# Work Distribution

## 1\. Modular Coding Work Distribution

### Jacek: Data Ingestion, Graph Extraction & Topological/Spectral Characterization

* **Objective:** Ingest the raw Reddit hyperlink dataset, extract a well-defined community core network, and compute foundational static topological and spectral metrics to characterize the topology.  
* **Scope of Deliverables (\~30% Project Progress):**  
  1. **Data Parsing & Edge Aggregation:**  
     * Ingest `soc-RedditHyperlinks-body.tsv` (columns: `SOURCE_SUBREDDIT`, `TARGET_SUBREDDIT`, `LINK_SENTIMENT`, `TIMESTAMP`).  
     * Extract an active core subgraph (e.g., top \$k\$-core or the largest connected component of subreddits with at least \$N\_{\\text{min}}\$ interactions) to ensure fast simulation iterations.  
     * Implement the **Net Sign Edge Collapse**: \$\$\\text{Sign}(u, v) \= \\begin{cases} \+1 & \\text{if } N\_{\\text{pos}}(u, v) \\ge N\_{\\text{neg}}(u, v) \\ \-1 & \\text{if } N\_{\\text{pos}}(u, v) \< N\_{\\text{neg}}(u, v) \\end{cases}\$\$  
     * Export the processed network in a standard format (e.g., NetworkX graph, `.gml`, or SciPy sparse adjacency matrices) for immediate use by Daniel and Polly.  
  2. **Topological Characterization:**  
     * **Degree Distribution:** Fit empirical degree distribution \$P(k)\$ against Power-Law (\$P(k) \\propto k^{-\\gamma}\$) and Poisson baselines on log-log and log-linear scales to confirm heavy-tailed scale-free structure.  
     * **Degree Assortativity (\$\\rho\_D\$):** Calculate degree correlation coefficient \$\\rho\_D\$ to determine whether high-degree subreddits predominantly link to each other.  
     * **Echo Chamber / Community Clustering:** Calculate Newman’s modularity (\$m\$) and average clustering coefficient (\$C\$) to quantify baseline community isolation.  
  3. **Spectral Metrics:**  
     * Construct adjacency matrix \$A\$ and compute its spectral radius (largest eigenvalue \$\\lambda\_1(A)\$).  
     * Construct the combinatorial Laplacian \$Q \= \\Delta \- A\$ and compute the algebraic connectivity: \$\$a(G) \= \\mu\_{N-1}\$\$ (the second-smallest eigenvalue of \$Q\$) to evaluate overall graph cohesion and resistance to disconnection.

### Daniel: Opinion Dynamics Engine & Phase Transition Baseline (NCO Models)

* **Objective:** Implement the synchronous Non-Consensus Opinion (NCO) dynamics simulation framework and demonstrate the divergence in opinion equilibrium between unsigned and signed interaction networks.  
* **Scope of Deliverables (\~30% Project Progress):**  
  1. **Vectorized Simulation Framework (Unit-Tested on Mock Graphs):**  
     * Build an autonomous simulator supporting discrete states \$\\sigma\_i(t) \\in {-1, \+1}\$.  
     * Implement parallel synchronous state updating across all nodes at time \$t \\to t+1\$.  
     * Add absorbing/steady-state detection (halting when state configurations stabilize or when periodic flipping limit cycles occur).  
  2. **Model 1 — Course Baseline Unsigned NCO:**  
     * Implement the course majority-rule update (local neighborhood \+ self): \$\$\\sigma\_i(t+1) \= \\operatorname{sign}\\left(\\sigma\_i(t) \+ \\sum\_{j \\in \\mathcal{N}\_i} \\sigma\_j(t)\\right)\$\$ *(with inertia tie-breaking \$\\sigma\_i(t+1) \= \\sigma\_i(t)\$ when the sum is 0\)*.  
  3. **Model 2 — Signed NCO (Introducing Psychological Reactance):**  
     * Incorporate signed interaction weights \$W\_{ji} \\in {+1, \-1}\$: \$\$\\sigma\_i(t+1) \= \\operatorname{sign}\\left(\\sigma\_i(t) \+ \\sum\_{j \\in \\mathcal{N}*i} W*{ji} \\cdot \\sigma\_j(t)\\right)\$\$ *(where \$W\_{ji} \= \+1\$ drives consensus/solidarity, and \$W\_{ji} \= \-1\$ triggers oppositional stance/reactance)*.  
  4. **Initial Result Generation:**  
     * Run synthetic fraction sweeps \$f \\in \[0.1, 0.9\]\$ (initial fraction of minority opinion \$\\sigma\_-\$) across 20+ Monte Carlo realizations.  
     * Generate comparative phase transition plots: final stable fraction \$s\_1\$ vs. initial fraction \$f\$ for **Unsigned vs. Signed NCO** on the extracted Reddit graph to show how negative links distort the classical phase transition point \$f\_c\$.

### Polly: Intervention Framework (Course ICO Baseline) & Centrality Profiling

* **Objective:** Implement the Inflexible Contrarian Opinion (ICO) framework, profile structural node centrality metrics, and evaluate baseline intervention strategies to disrupt polarized consensus.  
* **Scope of Deliverables (\~30% Project Progress):**  
  1. **Centrality Metric Computation:**  
     * Compute node centrality rankings across the core graph:  
       * Degree Centrality (\$d\_i\$)  
       * Shortest-Path Betweenness Centrality (\$B\_i\$)  
       * Eigenvector Centrality (\$x\_i\$)  
       * **Dr. Wang’s Node Leverage Centrality (\$l\_{ni}\$):** \$\$l\_{ni} \= \\frac{1}{d\_i} \\sum\_{j \\in \\mathcal{N}\_i} \\frac{d\_i \- d\_j}{d\_i \+ d\_j}\$\$  
  2. **ICO Intervention Simulator:**  
     * Extend Daniel's steady-state simulation to seed a budget fraction \$\\Phi\$ of nodes as permanent, unchangeable contrarians (\$\\sigma \= \+1\$).  
     * Track the decay/collapse of the opposing cluster (\$S\_1\$) as a function of contrarian budget \$\\Phi\$.  
  3. **Initial Strategy Comparison (Proof-of-Concept Results):**  
     * Run baseline intervention comparisons on the unsigned network for:  
       1. *Random Selection*  
       2. *Degree Targeting* (Hub targeting)  
       3. *Dr. Wang’s Leverage Centrality Targeting*  
     * Generate initial plots showing the destruction of majority opinion clusters as \$\\Phi\$ increases, verifying whether high-leverage nodes outperform simple high-degree hubs.  
  2. **Formalization of Phase 2 Formulations:**  
     * Mathematically formalize the planned novel extension metrics for the presentation:  
       * **Signed Leverage Centrality (\$l\_{\\text{signed}}\$)** to prevent backfire effects.  
       * **Sign-Shuffling Null Model** protocols for testing structural placement.

## 3\. Summary of Roles for Phase 1

| Team Member | Functional Coding Area | Slide Deck Responsibility |
| :---- | :---- | :---- |
| **Jacek** | Raw TSV parsing, core subgraph extraction, signed edge collapse, degree fitting, modularity, assortativity, and spectral metrics (\$\\lambda\_1(A), a(G)\$). | **Slides 1, 2, 3** (Motivation, Data Ingestion, Network Characterization) |
| **Daniel** | Synchronous NCO simulation engine, unsigned baseline, signed reactance model, sweep experiments, and phase transition plots (\$s\_1\$ vs. \$f\$). | **Slides 4, 5** (Opinion Process Modeling, Phase Transition Results) |
| **Polly** | Centrality profiling (Degree, Betweenness, Dr. Wang's Leverage \$l\_{ni}\$), ICO intervention engine, strategy comparison curves (\$S\_1\$ vs. \$\\Phi\$), and Phase 2 methodology formalization. | **Slides 6, 7, 8** (ICO Intervention, Initial Strategy Results, Phase 2 Roadmap) |

