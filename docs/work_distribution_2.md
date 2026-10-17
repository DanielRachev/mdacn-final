# Final Project: Work Distribution 2

## Scope and Workload

Keep all existing assignments below. Jacek, Daniel, and Polly already own the Phase 1 work in [work_distribution.md](work_distribution.md); that document stays unchanged. Their Phase 2 work reuses those implementations and remains focused on the tasks already assigned here. New main-scope tasks go to Antreas, Georgi, and Jakub, who also own the final presentation. This keeps additional implementation, experiment, and integration work with the three final presenters.

| Team members | Phase 2 focus |
| --- | --- |
| Polly | Existing sign-shuffling experiment using the Phase 1 simulator. |
| Jacek | Existing graph visuals, shared opinion-GCC helper, limited extraction comparisons, title-link feasibility check, and graph references. |
| Daniel | Existing sampled node-impact analysis/regression and dynamics references, reusing the simulator and supplied metrics. |
| Antreas | Temporal model/control plus new temporal diagnostics, matched evaluations, and temporal ICO support. |
| Georgi | Intervention suite plus new configuration checks, combined results, and final-presentation assembly. |
| Jakub | Rewiring and signed motifs plus new null-model validation, directed/signed diagnostics, and dataset-methods documentation. |

## Shared Conventions

* Reuse the Phase 1 graph, matrices, simulator, centrality tables, and node order before implementing anything again. Every matrix, opinion vector, metric table, and targeting list must map to the same `nodes.json`; temporal snapshots retain the same full node set, including inactive nodes.
* Follow the current pipeline’s tie rule: equal positive/negative aggregate counts give sign zero. Such an edge remains in unsigned topology and contributes no signed influence. This clarifies the older Phase 1 positive-tie formula without editing that document.
* Keep `f` as the initial negative fraction, and `s_1`/`s_2` as raw negative/positive node fractions. Use separate names `g_minus = S_minus_GCC / N` and `g_plus = S_plus_GCC / N` for the opinion-induced largest-component fractions. Compute these on the fixed unsigned aggregate core, including for temporal states, so the connectivity reference is comparable. Keep the original core size `N`, including pinned nodes, as denominator.
* Use the same initialization and paired seeds for each empirical/control or intervention/no-intervention comparison on a fixed node set. Save configurations, input/core identity, node order, individual outcomes, variability, and stop reasons. A cycle or update-limit terminal state is not a steady state; temporal comparisons use the full snapshot horizon.
* Use `Phi = 0` as the no-intervention reference. Define `Phi_c` by the first tested budget where `g_minus < 0.5`; report zero if the reference is already below the threshold, and report "not reached in the tested range" when appropriate. Keep this distinct from a threshold on the raw negative fraction.
* The narrow initial sensitivity check covers existing extraction methods and nearby k-core thresholds only. Broader configuration searches and optional extensions are listed separately under TBD.

## Main Work Distribution

### Polly & Jakub: Controlled Null Models A & B (Sign Shuffling & Degree-Preserving Homophily Rewiring)

* **Objective:** Implement rigorous null models to isolate whether system polarization and intervention efficacy stem from true topological structure, sign placement, or simple interaction volume.
* **Tasks:**
  1. **\[Polly\] Null Model A — Sign Shuffling (Testing Placement vs. Volume):**
     * Implement an exact sign permutation on existing edges, preserving unsigned topology/degrees and the global counts of positive, negative, and zero signs. Per-node signed degrees and the original relationship between signs and event-count weights are not preserved.
     * Run Monte Carlo sweeps across the initial negative fraction \$f \\in \[0.1, 0.9\]\$ using the Signed NCO update rule.
     * Generate empirical versus sign-shuffled phase-transition curves across the existing f sweep, using matched initial states and the existing realization count. Distinguish raw opinion fractions from the shared GCC measurements and show variability.
     * Analyze whether empirical negative links create structural firebreaks along community boundaries or act as uniformly dispersed noise.
  2. **\[Jakub\] Null Model B — Degree-Preserving Homophily Rewiring (Testing Echo-Chamber Insulation):**
     * Implement a degree-preserving double-edge swap algorithm: randomly select independent edges \$(u, v)\$ and \$(x, y)\$ and rewire them to \$(u, y)\$ and \$(x, v)\$, strictly preserving each individual node's degree \$d\_i\$ (as covered in Lecture 2, Slide 35).
     * Add a rejection/acceptance sampling rule to steer community/sign homophily using fixed baseline community labels (not changing simulated opinions):
       * *Configuration 1 (Insular Echo Chamber):* Maximize internal positive ties within communities while routing negative ties across community boundaries.
       * *Configuration 2 (Cross-Cutting Integration):* Maximize inter-community positive ties and disperse negative ties internally.
     * Evaluate how feasible insular versus integrated configurations affect Phi_c under contrarian targeting. Keep the global positive/negative/zero sign counts fixed, state how signs follow swapped edges, and do not assume that the extreme target configurations can be reached.
  3. **[Jakub] Unbiased Rewiring Reference and Invariant Checks (Added):**
     * Reuse the double-edge-swap implementation for an unbiased degree-preserving reference before comparing the homophily-biased configurations. This tests whether effects exceed those explained by the observed degree sequence.
     * Reject self-loops/duplicate edges; record attempted and accepted swaps, achieved homophily, connectivity, each node's unsigned degree before/after, and global sign counts. Report disconnected outputs explicitly rather than silently changing the node set or denominator.
     * Provide a short invariant table: sign shuffling preserves topology/unsigned degrees and global sign counts; static rewiring preserves unsigned degrees and global sign counts; timestamp shuffling preserves full-window signed event totals and aggregate topology. None guarantees per-snapshot node degrees; timestamp shuffling also changes weekly node activity while retaining the global timestamp histogram.
* **Decoupling / Interfaces:**
  * Can run independently on the Phase 1 static 10-core graph.
  * Polly exports the sign-shuffled signed matrices; Jakub exports the rewired unsigned and signed matrices, node-order mapping, and validation metadata. Reuse the original `nodes.json`. Georgi consumes these outputs; neither person needs to duplicate the other’s implementation.

---

### Antreas: Temporal Dynamic Modeling (Model 3\) & Controlled Null Model C (Timestamp Shuffling)

* **Objective:** Extend opinion dynamics from static graphs to time-varying networks, testing the role of temporal burstiness, event ordering, and interaction recency.
* **Tasks:**
  1. **Temporal Network Snapshot Slicing:**
     * Discretize the 10-core Reddit interaction log into sequential, non-overlapping weekly time slices \$\\tau \\in {1, 2, \\dots, T}\$.
     * For each window, construct a symmetric signed matrix with the shared node order, aggregating both directions by majority rule within that window; ties give zero influence. Keep the directed event log separately for temporal reciprocity and pattern diagnostics.
  2. **Model 3 Implementation — Temporal Signed NCO Simulator:**
     * Reuse the Phase 1 synchronous rule with the current snapshot: `h_i(tau) = sigma_i(tau) + sum_j W_ji(tau) * sigma_j(tau)`. Set the next state to the sign of h_i, retaining the current state for a zero sum or no active neighbors. Use one update per weekly snapshot initially.
     * Track trajectories over the full calendar sequence, including persistent polarization and flipping. A temporarily unchanged state does not establish absorption while future snapshots remain; report outcomes at the specified final snapshot.
  3. **Null Model C — Temporal Timestamp Shuffling (Testing Temporal Burstiness):**
     * Implement a timestamp permutation algorithm: keep the static network topology, edge signs, and each node's total lifetime interaction count completely invariant, but randomly shuffle the timestamps across all edges.
     * *Preservation Validation:* Verify full-window directed event totals by node/sign and aggregate degrees/net signs. Also report that weekly node degrees/activity need not be preserved, while the global timestamp histogram and its calendar-wide spikes remain fixed.
     * Run comparative simulations on real temporal snapshots vs. the timestamp-shuffled null model.
     * Compare empirical node/dyad timing with the timestamp-permuted control. Do not describe the permutation as a generated Poisson process or claim that it removes every form of burstiness.
  4. **Temporal Signed-Pattern Diagnostics (Added):**
     * Summarize negative-event inter-arrival times, weekly sign fractions, repeated negative exchanges, and negative reciprocity within a fixed seven-day lag. Compare against the same timestamp-shuffled controls used for the simulations.
     * Measure whether negative events incident on the same node co-occur in time more often than under the control. Categorize simple patterns such as one-way negative activity, reciprocal exchanges, and concentrated bursts, with counts/denominators; avoid inferring coordination from timing alone.
     * Export per-node temporal activity and a compact diagnostic figure/table for Georgi’s targeting score, Daniel’s existing impact analysis, and the final explanation of temporal results.
  5. **Matched Temporal Evaluation and ICO Support (Added):**
     * Reuse the existing simulator interface and Jacek’s GCC helper. Compare static signed and temporal signed outcomes after the same number of updates from paired initial states; show static steady-state results separately.
     * Accept Georgi’s selected stubborn-node mask and keep those nodes at +1 throughout the snapshot sequence. Start temporal intervention comparisons with random, degree, and signed-leverage targeting on the agreed budget grid; the seven-strategy static benchmark stays as assigned. An exhaustive temporal strategy/parameter cross-product belongs to TBD.
* **Decoupling / Interfaces:**
  * Uses the raw TSV's `TIMESTAMP` field independently of the static metrics.
  * Temporal entry point takes the directed event log plus the fixed core/node order, initial state, and optional stubborn-node mask. Export snapshots, timing metadata, activity metrics, outcomes, and diagnostics for reuse; do not add a separate data/core extraction pipeline.

---

### Georgi: Advanced Intervention Suite, Proposed Centralities & Backfire Effect Analysis

* **Objective:** Implement and benchmark the complete suite of contrarian targeting strategies (including two proposed formulations), analyze backfire effects, and evaluate the critical budget \$\\Phi\_c\$.
* **Tasks:**
  1. **Metric Engineering & Proposed Centrality Formulations:**
     * Reuse Polly’s Phase 1 degree, betweenness, eigenvector, and leverage rankings. Georgi checks availability, node alignment, and settings, and computes only missing outputs rather than repeating completed centrality work.
     * **Implement Proposed Strategy 1 — Signed Leverage Centrality:** `l_signed(i) = (1/d_i) * sum_j W_ij * max(0, (d_i - d_j)/(d_i + d_j))`. Treat improved targeting/backfire reduction as a hypothesis; the score does not guarantee bridge selection. Check related work before calling the formulation novel.
     * **Implement Proposed Strategy 2 — Temporal-Activity Bridge Centrality:** Define a reproducible score combining inter-community bridge importance with Antreas’s measured interaction frequency, stating normalization and observation window. A full-window ranking is retrospective; prospective claims need earlier-window features.
  2. **Comprehensive ICO Intervention Benchmark:**
     * Reuse the Phase 1 ICO simulator and its post-baseline intervention timing; document the temporal intervention start with Antreas. Run static unsigned/signed budget sweeps over `Phi in [0.01, 0.25]`, adding `Phi = 0` as the paired reference. Start with an explicit manageable grid and refine near an observed threshold. Compare all seven strategies:
       1. Random Selection (Baseline)
       2. Degree Centrality
       3. Betweenness Centrality
       4. Eigenvector Centrality
       5. Dr. Wang’s Node Leverage Centrality (\$l\_{ni}\$)
       6. Signed Leverage Centrality (\$l\_{\\text{signed}}\$)
       7. Temporal-Activity Bridge Centrality
  3. **Backfire Effect Quantification & \$\\Phi\_c\$ Determination:**
     * Quantify instances of the **Backfire Effect**: measure cases where placing a stubborn contrarian (\$\\sigma \= \+1\$) connected via negative links (\$W\_{ji} \= \-1\$) repels neighbors and solidifies opposing \$-1\$ consensus: \$\$W\_{ji} \\cdot \\sigma\_{\\text{contrarian}} \= (-1) \\cdot (+1) \= \-1\$\$
     * Compute and plot Phi_c using the shared opposing-opinion GCC threshold across all seven strategies, with uncertainty and threshold-not-reached cases. Test whether signed leverage improves outcomes; do not assume superiority.
  4. **Narrow Configuration Sensitivity and Justification (Added):**
     * Use Jacek’s existing extraction comparison outputs: the two implemented core methods, with a small `k = 8, 10, 12` pilot around the default and the current activity-LCC setting. Keep input data, sign/tie policy, initialization procedure, and simulation settings fixed. Jacek’s assigned extraction task stays unchanged; Georgi owns the added process comparisons.
     * Reuse the existing f sweep and realization count. Summarize node/edge counts, negative/zero sign fractions, and selected NCO/ICO outcomes to justify the default; report when topology size changes make a comparison imperfect. Document the weekly snapshot, update-limit, and budget-grid choices rather than opening a broad search initially.
  5. **Combined Results and Final Presentation (Added):**
     * Assemble the matched unsigned/signed static comparisons, temporal/control results from Antreas, sign-shuffled runs from Polly, and static rewiring results from Jakub. Include component-size curves, the existing raw-fraction curves, uncertainty, and stop reasons so modest or null differences remain visible.
     * Combine the existing node-impact analysis with supplied degree/leverage, bridge, incident-sign, and activity metrics to explain process outcomes. Use precomputed outputs rather than duplicating Daniel’s regression work or asking Phase 1 contributors for another experiment suite.
     * Coordinate the final 15-minute deck with Antreas and Jakub as presenters. Include dataset/label interpretation, justified settings, null-model invariants, key findings/limitations, and verified references. Assemble the existing contributions and record the commands/configurations needed to reproduce the displayed results.
* **Decoupling / Interfaces:**
  * Operates on the static 10-core graph and the NCO simulator established in Phase 1\.
  * Consumes the giant connected component calculation helper provided by Jacek to track cluster fragmentation.

---

### Jacek: 10-Core Extraction Visualizations, Surviving Opinion GCC & Graph Analysis Literature

* **Objective:** Address feedback regarding core filtering verification, implement giant component tracking for opinion clusters, and compile foundational graph literature.
* **Tasks:**
  1. **10-Core Extraction Verification & Layout Visualizations:**
     * Generate the assigned full-aggregate versus connected-10-core degree plots. Describe the retained/removed structure without claiming that the tail must be scale-free; reuse Phase 1 metrics and plotting outputs.
     * Produce a clean sub-network visualization of the 10-core backbone highlighting major community clusters.
  2. **Surviving Opinion Giant Connected Component (GCC) Tracking:**
     * At a reported steady state or clearly labeled terminal state, extract each opinion-induced subgraph. Keep the full original core N as denominator and align the opinion vector to the graph node order.
     * Compute `g_minus = S_minus_GCC / N` and `g_plus = S_plus_GCC / N`, with zero for an empty opinion subgraph. Keep these separate from the existing raw fractions s_1 and s_2.
     * Provide this function as a shared evaluation helper so the existing experiment owners can evaluate the structural shattering of echo chambers (rather than only tracking disconnected raw node fractions).
  3. **Literature & Citations — Graph Analytics:**
     * Identify and document formal academic citations for graph processing techniques used:
       * \$k\$-core decomposition (Batagelj & Zaveršnik).
       * Modularity maximization and community detection (Newman).
       * Spectral graph theory and algebraic connectivity (Fiedler; Van Mieghem).
  4. Additional graphs/rendered tables for the topology metrics
     * Reuse the computed topology metrics for a compact table and selected figures that explain the extraction comparison; no additional metric suite is required.
  5. Experiment on the outcomes provided by the two topology extraction methods. Keep the existing comparison small: extraction/characterization outputs for Georgi’s added process sensitivity check, rather than a separate broad experiment suite.
  6. Investigate whether using title links is a good idea instead of just body links. Keep this as the assigned feasibility recommendation (coverage and overlap); a full title/body rerun is optional TBD.
* **Decoupling / Interfaces:**
  * Generates visual artifacts and exports `compute_opinion_gcc(graph, opinion_vector)` for both states. Reuse the existing `opposing_cluster_size` negative-state calculation in `src/mdacn/interventions.py` rather than implementing the same component search again.

---

### Daniel: Node Empirical Impact (\$I\_i\$) Simulation & Topological Regression Analysis

* **Objective:** Rigorously link node topological properties to their dynamical role in process intervention via empirical impact modeling.
* **Tasks:**
  1. **Empirical Node Impact (\$I\_i\$) Quantification:**
     * For a small, predeclared sample spanning node degree and negative-incident-ratio ranges, reuse the existing single-node intervention simulator. Measure `I_i = g_minus(no intervention) - g_minus(only i pinned to +1)` with paired initial states/seeds. Positive impact means a smaller opposing GCC; state the sampling limits.
  2. **Statistical Correlation & Regression Modeling:**
     * Run Pearson and Spearman rank correlation between empirical impact \$I\_i\$ and the candidate node's structural metrics:
       * Node Degree (\$d\_i\$)
       * Leverage Centrality (\$l\_{ni}\$)
       * Proposed Signed Leverage (\$l\_{\\text{signed}}\$)
       * Shortest-Path Betweenness (\$B\_i\$)
       * Negative Incident Ratio (\$d\_{\\text{neg}}(i) / d(i)\$)
     * Fit a small multivariable linear model for continuous impact; use logistic regression only for a defined binary outcome. Reuse signed-bridge and temporal-activity columns supplied by Jakub and Antreas where available. Report uncertainty and associations without causal or population-wide claims from the sampled nodes.
  3. **Literature & Citations — Dynamic Processes on Networks:**
     * Identify and document formal academic citations for opinion models and spreading dynamics:
       * Verify the original NCO source and match the self vote, synchronous update, and tie convention to the implemented rule.
       * Verify the original ICO/contrarian sources and classical targeting strategies; distinguish those models from general zealot models and this project’s signed/temporal extensions.
* **Decoupling / Interfaces:**
  * Runs single-node intervention scripts on the existing Phase 1 NCO simulator using precomputed node centrality tables.

---

### Jakub: Incident Sign Heterogeneity, Static Signed Triadic Balance & Structural Citations

* **Objective:** Characterize sign placement, directed interaction roles, and static signed-triangle balance, and document the dataset/model distinction. Temporal P-O-X prediction remains optional TBD.
* **Tasks:**
  1. **Incident Sign Heterogeneity Analysis:**
     * For every node \$i\$ in the 10-core graph, compute its negative incident ratio: \$\$r\_{\\text{neg}}(i) \= \\frac{d\_{\\text{neg}}(i)}{d(i)}\$\$
     * Plot the probability density function \$P(r\_{\\text{neg}})\$ to test whether hostility concentrates within specific pariah/conflict-focused subreddits or disperses uniformly across communities.
  2. **Static Signed Triadic Motifs & Structural Balance:**
     * Count the frequency of the 4 fundamental signed triadic motifs (Lecture 3, Slide 25):
       * Balanced: \$(+, \+, \+)\$ and \$(+, \-, \-)\$ ("the enemy of my enemy is my friend").
       * Unbalanced: \$(+, \+, \-)\$ and \$(-, \-, \-)\$.
     * Calculate the balanced-triangle fraction among triangles with nonzero edge signs and report zero-sign cases separately. Compare with the sign-shuffled empirical topology as the main sign-placement control. A matched-size/density/sign-fraction Erdős–Rényi comparison can remain descriptive, with its different degree sequence stated; no result is assumed in advance.
     * Temporal P-O-X triangle closure is optional, with ownership and scheduling TBD in the separate section below; it is not a required static-balance deliverable.
  3. **Literature & Citations — Structural Balance & Centrality:**
     * Identify and document formal academic citations for structural balance and leverage:
       * Heider's Structural Balance Theory & Cartwright-Harary graph theorem.
       * Node Leverage Centrality in opinion dynamics (Li, Li, Van Mieghem, Stanley, Wang, 2015).
  4. **Directed Roles and Signed Community Bridges (Added):**
     * On the raw directed core events, compute in/out-degree, positive/negative in/out-strength, and directed reciprocity. Separate sender/receiver interaction roles without claiming that the labels identify real aggressors or factual correctness.
     * Report negative-edge rates inside versus between fixed baseline communities with counts/denominators, plus global negative-event versus negative-edge fractions. Export per-node negative incident ratio and a clearly defined signed bridge ratio for Daniel’s existing analysis and Georgi’s interpretation.
  5. **Dataset Label Audit and Methods Documentation (Added):**
     * Document the supplied source-post/target label and its difference from the separate text features and simulated opinions, using the [SNAP dataset documentation](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) and Kumar et al. (2018). Audit repeated POST_ID/target records before assuming that separate hyperlink rows are independent text classifications; do not add a new NLP pipeline.
     * Report body-file event/node counts and core filtering separately from SNAP’s headline network size. Assemble the null-model invariant checks above and the existing graph/dynamics/structural references into the final methods material; do not claim unverified citations are already checked.
* **Decoupling / Interfaces:**
  * Directly processes the edge list and triangle patterns from the 10-core graph without requiring simulation outputs.

---

## Feedback and Outline Coverage

| Required item | Existing or added owner/deliverable |
| --- | --- |
| Sentiment granularity and dataset provenance | Jakub, dataset-label audit and methods documentation (added). |
| Citations for graph methods, NCO/ICO, targeting, balance, and null methods | Existing Jacek/Daniel/Jakub literature tasks; Jakub assembles dataset/null-method sources. |
| Justification of configuration numbers | Georgi, narrow sensitivity check (added), reusing Jacek’s existing extraction comparison. |
| Full graph versus core and surviving-opinion largest components | Existing Jacek visualization/helper tasks; Georgi and Antreas consume the helper in their comparisons. |
| Explain similar unsigned/signed results and test sign placement | Existing Polly sign shuffle; Jakub’s degree-preserving reference and Georgi’s combined results. |
| Incident signs, community bridges, directed strengths/reciprocity | Existing Jakub incident-sign/static-balance tasks plus directed/bridge diagnostics (added). |
| Temporal patterns, burstiness, incident-event correlations, temporal reciprocity | Antreas, temporal diagnostics (added). |
| Explain rewiring and aggregate versus temporal preservation | Jakub, swap/reference validation (added), with Antreas’s timestamp-control checks. |
| Temporal interventions and comparable model/control evaluation | Antreas, temporal ICO/evaluation (added), using Georgi’s targeting lists. |
| Node properties versus process impact | Existing Daniel analysis; Jakub/Antreas supply the added bridge/activity columns. |
| Final presentation and reproducible result handoffs | Georgi coordinates; Antreas, Georgi, and Jakub present. |

## Optional Extensions and Broader Experiments — TBD

These items are outside the initial required workload. Ownership, schedule, and configuration are **TBD**; listing an item here does not assign it to a Phase 1 contributor. Revisit them after the main model/control and intervention results are available.

| Optional work | Proposed scope before assignment | Owner / schedule |
| --- | --- | --- |
| Subreddit embedding initialization | Check [SNAP embedding](https://snap.stanford.edu/data/web-RedditEmbeddings.html) coverage/training period, define a reproducible cluster-to-state rule, and compare against matched synthetic initialization. Similarity is not an opinion label. | TBD |
| Temporal P-O-X prediction | Test whether earlier shared-target signs predict later direct links/signs within a stated delta window; define candidate pairs, closure denominators, repeated-event handling, and a comparison baseline. Temporal ordering does not establish causality. | TBD |
| Broader parameter sensitivity | Expand core/activity thresholds, snapshot duration, updates per snapshot, update limits, realization counts, initialization rules, or intervention budget ranges one dimension at a time before considering combinations. | TBD |
| Richer temporal controls | Consider snapshot-order shuffling or constrained controls preserving per-window node activity/degrees; state feasible invariants and how each differs from timestamp permutation. | TBD |
| Alternative initial states | Historical sentiment, directed interaction roles, or topic-specific seeds with a transparent earlier-window rule and zero-event handling; no ground-truth belief claims. | TBD |
| Full title/body robustness experiment | Follow up Jacek’s assigned feasibility check with separately identified/deduplicated inputs and matched model/control reruns, if justified. | TBD |
| Weighted or multilayer variants | Compare continuous net weights, edge-level activity thresholds, or separate positive/negative layers only after the current sign-collapse baseline. | TBD |
| Text features and extended node-impact prediction | Assess supplied text features, larger node samples, or predictive modeling after the main static-balance and sampled-impact results. | TBD |
| Exhaustive temporal intervention benchmark | Expand beyond the initial temporal targeting subset and agreed grid only if runtime and initial results justify it. | TBD |
