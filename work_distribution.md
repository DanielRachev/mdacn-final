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

## 2\. 5–10 Minute Presentation Plan (Feedback Meeting)

**Target Length:** 8 slides total (approx. 1 minute per slide to finish comfortably within the 8–9 minute range and leave ample time for feedback).

### Slide 1: Motivation & Problem Definition (Jacek)

* **Title:** Echo Chambers and Opinion Polarization: Can Targeted Interventions Break Consensus? (Jacek)  
* **Visuals:** High-level schematic contrasting positive consensus (echo chamber) with signed antagonism and hostile cross-links.  
* **Talking Points:**  
  * Real-world social networks are plagued by tribal polarization, echo chambers, and misinformation.  
  * Research Question: How do opinions diffuse across real signed community topologies, and can strategic fact-checkers/moderators de-polarize an entrenched majority?  
  * Framing: Bridging course theory (Non-Consensus Opinion & Inflexible Contrarian models) with empirical social dynamics.

### Slide 2: Dataset & Structural Pre-processing (Jacek)

* **Title:** The SNAP Reddit Hyperlink Network: From Raw Logs to an Interaction Topology (Jacek)  
* **Visuals:** Schema showing directed hyperlink extraction, net sentiment collapse (\$+1\$ vs. \$-1\$), and sub-community graph size.  
* **Talking Points:**  
  * Dataset: SNAP Reddit Hyperlinks (`soc-RedditHyperlinks-body.tsv`) tracking community-to-community interactions with explicit sentiment.  
  * Graph Formulation: Extraction of the active core subgraph; collapsing multiple timestamped interactions via majority sentiment rule into an aggregate signed graph (\$W\_{ij} \\in {+1, \-1}\$).  
  * Defense Guardrail: Acknowledging that signs represent structural sentiment/conflict rather than ground-truth factuality.

### Slide 3: Aggregate Network Characterization (Jacek)

* **Title:** Network Topology: Scale-Free Traits, Modularity, and Algebraic Connectivity (Jacek)  
* **Visuals:** Empirical degree distribution plot (\$P(k)\$ vs. \$k\$ on log-log scale with power-law fit line) alongside a summary table of topological metrics (\$N, L, \\langle k \\rangle, \\rho\_D, m, \\lambda\_1, a(G)\$).  
* **Talking Points:**  
  * Confirmation of heavy-tailed, scale-free structure (presence of major hub subreddits).  
  * High modularity (\$m\$) confirms strong clustering into insular community bubbles (echo chambers).  
  * Spectral radius \$\\lambda\_1(A)\$ and algebraic connectivity \$a(G) \= \\mu\_{N-1}\$ quantify overall graph structural cohesion and robustness to disconnection.

### Slide 4: Modeling Dynamic Opinion Processes: Unsigned vs. Signed NCO (Daniel)

* **Title:** Modeling Opinion Dynamics: Incorporating Psychological Reactance into NCO (Daniel)  
* **Visuals:** Diagram comparing the classical majority-rule update against the signed update rule where negative links flip neighbor influence.  
* **Talking Points:**  
  * Course Baseline (Model 1): Unsigned NCO where all neighbors exert positive attractive pull.  
  * Structural Realism (Model 2): Signed NCO where negative links (\$W\_{ji} \= \-1\$) model psychological reactance—pushing communities to adopt opposing stances ("the enemy of my enemy is my friend").  
  * Vectorized simulation architecture and absorbing/steady-state criteria.

### Slide 5: Initial Results: Phase Transitions Under Reactance (Daniel)

* **Title:** Initial Results: Distortions of the Phase Transition in Signed Networks (Daniel)  
* **Visuals:** Comparative plot showing final steady-state opinion fraction (\$s\_1\$) as a function of initial minority seed fraction (\$f\$) for both Unsigned NCO and Signed NCO.  
* **Talking Points:**  
  * In the standard unsigned graph, we observe the characteristic sharp phase transition around \$f\_c \\approx 0.5\$.  
  * Introducing signed links alters consensus stability: hostile ties fragment consensus and allow minority stances to survive at lower initial fractions due to cross-cluster repulsion.  
  * Highlights why classical opinion models mispredict equilibrium when negative ties are ignored.

### Slide 6: Process Intervention: Inflexible Contrarians & Backfire Effects (Polly)

* **Title:** Strategic Intervention: The ICO Model, Centrality Metrics, and Backfire Effects (Polly)  
* **Visuals:** Diagram of the Inflexible Contrarian Opinion (ICO) model showing stubborn nodes (\$\\Phi\$), along with mathematical definitions of Degree, Betweenness, and Dr. Wang’s Leverage Centrality (\$l\_{ni}\$).  
* **Talking Points:**  
  * Introducing stubborn fact-checkers/moderators (\$\\Phi\$) holding immutable opinions (\$\\sigma \= \+1\$) to dismantle entrenched clusters.  
  * The **Backfire Effect**: In signed networks, standard hub targeting can fail—if a hub is connected via hostile ties, its fact-checking pushes neighbors further into opposition\!  
  * Evaluating structural targeting: comparing degree hubs against Dr. Wang’s node leverage centrality (\$l\_{ni}\$).

### Slide 7: Initial Intervention Results: Hubs vs. Leverage (Polly)

* **Title:** Initial Results: Majority Cluster Disruption Under Budget Constraints (Polly)  
* **Visuals:** Curve showing dominant cluster size (\$S\_1\$) versus contrarian budget fraction (\$\\Phi\$) comparing Random, Degree, and Node Leverage strategies.  
* **Talking Points:**  
  * Random intervention requires a prohibitively high budget to break majority consensus.  
  * Targeted interventions substantially lower the critical intervention threshold \$\\Phi\_c\$.  
  * Leverage centrality performs competitively against simple degree hubs by focusing on nodes that dominate their immediate local neighborhood rather than just global degree count.

### Slide 8: Next Steps & Phase 2 Roadmap for the Full Team (Polly)

* **Title:** Phase 2 Roadmap: Temporal Dynamics, Novel Signed Metrics, and Null Models (Polly)  
* **Visuals:** Gantt-style work breakdown mapping the remaining \~70% of tasks across all 6 team members.  
* **Talking Points:**  
  * Expansion to **Temporal Signed NCO** (weekly time-slices \$\\tau\$) to account for transient activity windows.  
  * Developing **Signed Leverage Centrality (\$l\_{\\text{signed}}\$)**: selectively targeting high-leverage nodes with positive cross-community ties to avoid backfire effects.  
  * **Null Models**: Degree-preserving rewiring to systematically test the impact of opinion homophily and random sign shuffling.  
  * Onboarding the remaining 3 team members to scale up experiments and prepare the final presentation.

## 3\. Summary of Roles for Phase 1

| Team Member | Functional Coding Area | Slide Deck Responsibility |
| :---- | :---- | :---- |
| **Jacek** | Raw TSV parsing, core subgraph extraction, signed edge collapse, degree fitting, modularity, assortativity, and spectral metrics (\$\\lambda\_1(A), a(G)\$). | **Slides 1, 2, 3** (Motivation, Data Ingestion, Network Characterization) |
| **Daniel** | Synchronous NCO simulation engine, unsigned baseline, signed reactance model, sweep experiments, and phase transition plots (\$s\_1\$ vs. \$f\$). | **Slides 4, 5** (Opinion Process Modeling, Phase Transition Results) |
| **Polly** | Centrality profiling (Degree, Betweenness, Dr. Wang's Leverage \$l\_{ni}\$), ICO intervention engine, strategy comparison curves (\$S\_1\$ vs. \$\\Phi\$), and Phase 2 methodology formalization. | **Slides 6, 7, 8** (ICO Intervention, Initial Strategy Results, Phase 2 Roadmap) |

