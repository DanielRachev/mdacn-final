# Echo Chambers, Misinformation, and Targeted Fact-Checking: Testing the Inflexible Contrarian Model on Real-World Interaction Networks

**The Concept:** Study simulated opinion dynamics on Reddit interaction networks and compare stubborn-node intervention strategies. Echo chambers, misinformation, and fact-checking motivate the work; the data does not provide observed opinions or factual correctness labels.

### 1. Dataset and Label Interpretation

* **Dataset:** **SNAP Reddit Hyperlink Network** (`soc-redditHyperlinks-body.tsv`). Clean TSV containing directed, timestamped interactions between subreddits with sentiment (+1 / \-1).
* **Format:** Nodes represent subreddits; directed edges represent cross-community hyperlinks.
* **Size:** The [SNAP dataset page](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) lists 55,863 subreddits and 858,490 hyperlinks. Report the parsed body-file counts and the retained core counts separately; do not use those headline figures as the size of the body-only analysis.
* **Sentiment Granularity:** The supplied label describes the source post toward the linked target: `-1` is explicitly negative, and `+1` includes neutral and positive cases. SNAP describes crowdsourced labels and a trained text classifier; VADER/LIWC values are separate text features. Cite Kumar et al. (2018) and explain this distinction in the methods. No new NLP is required. See the [dataset documentation](https://snap.stanford.edu/data/soc-RedditHyperlinks.html).

### 2\. Technical Execution Plan (Addressing All Rubric Pillars)

#### Pillar 1: Network Characterization (Topological \+ Spectral Metrics) \- Aggregate Graph

* **Pre-processing:** Preserve directed, repeated raw events for temporal and in/out analyses. For the shared static baseline, remove self-links, merge both directions into a simple undirected graph, and retain positive/negative event counts. The sign is `sign(N_pos - N_neg)`: ties give zero signed influence while the edge remains in the unsigned topology. Use the largest connected component of the 10-core as the default; compare it with the existing node-activity-filtered LCC. Node activity filtering is distinct from filtering individual edges by their event counts.
* **Degree Distribution:** Compare descriptive power-law and Poisson fits and report their limitations. Add full-aggregate versus connected-10-core degree plots and a core layout to show what filtering removes; do not assume that a fitted tail proves a scale-free network.
* **Community Structure & Modularity:** Compute Newman’s Modularity: `m = 1/(2L) * sum_{i,j} [a_ij - (d_i * d_j)/(2L)] * delta(c_i, c_j)` and clustering. Community structure describes topological separation; it does not by itself establish opinion echo chambers.
* **Assortativity & Signed Balance:** Calculate degree correlation `rho_D` and the fraction of signed triangles with positive sign product. Report zero-sign triangles separately. Static signed-triangle counts are part of the main analysis; temporal P-O-X prediction is optional.
* **Spectral Properties:** Adjacency spectral radius `lambda_1(A)` and Laplacian algebraic connectivity `a(G) = mu_(N-1)` (measuring network cohesion and structural resistance to disconnection).
* **Signed Topology Distribution:** Report negative raw-event and negative aggregate-edge fractions separately. Compare negative-edge rates within and between fixed baseline communities, with counts and denominators, and export a per-node signed bridge ratio for the impact analysis.
* **Reciprocity and Strength:** Use the raw directed events to compute in/out-degree, positive/negative in/out-strength, and reciprocity. Distinguish frequently sending negative links from receiving them; these are interaction roles rather than verified aggressor/target identities.
* **Incident Sign Heterogeneity:** For each node, calculate the negative incident-edge fraction `d_neg(i) / d(i)` and compare within-community and bridge edges. Define the denominator on the unsigned topology and report zero-sign edges separately.
* **Temporal Sign Correlation & Burstiness:** Summarize negative-event inter-arrival times, weekly negative fractions, repeated negative exchanges, and reciprocal negative events within a fixed seven-day lag. Compare sign co-occurrence among events incident on the same node against timestamp-shuffled controls. Describe recurring patterns such as one-way negative activity, reciprocal exchanges, and concentrated bursts; these patterns alone do not prove coordination or brigading.

#### Pillar 2: Dynamic Process Modeling (Ablation: Static vs. Signed vs. Temporal NCO)

We modify the standard NCO model so positive and negative links affect opinion updates differently, modeling both social solidarity and psychological reactance:

* **Positive link (`W_ji = +1`):** Trust / Attraction. Encourages node `i` to align with neighbor `j`.
* **Negative link (`W_ji = -1`):** Hostility / Repulsion. Triggers psychological reactance: if a hostile rival holds opinion `+1`, node `i` is pushed toward `-1` ("the enemy of my enemy is my friend").

We implement an **Ablation Study across 3 model tiers** to evaluate whether adding real-world complexity alters system dynamics:
**Model 1: Static Unsigned NCO (Course Baseline):**

* Update rule:
  * If \[sigma\_i(t) \+ sum\_{j in N\_i} sigma\_j(t)\] \> 0  \--\>  sigma\_i(t+1) \= \+1
    * If \[sigma\_i(t) \+ sum\_{j in N\_i} sigma\_j(t)\] \< 0  \--\>  sigma\_i(t+1) \= \-1
    * If \[sigma\_i(t) \+ sum\_{j in N\_i} sigma\_j(t)\] \= 0  \--\>  sigma\_i(t+1) \= sigma\_i(t) (state inertia tie-break)
  * Operates on the unweighted adjacency matrix A. Assumes all interactions exert identical attractive influence.


2. **Model 2: Static Signed NCO (Structural Realism):**

   * Update rule:
     * If \[sigma\_i(t) \+ sum\_{j in N\_i} W\_ji \* sigma\_j(t)\] \> 0  \--\>  sigma\_i(t+1) \= \+1
     * If \[sigma\_i(t) \+ sum\_{j in N\_i} W\_ji \* sigma\_j(t)\] \< 0  \--\>  sigma\_i(t+1) \= \-1
     * If \[sigma\_i(t) \+ sum\_{j in N\_i} W\_ji \* sigma\_j(t)\] \= 0  \--\>  sigma\_i(t+1) \= sigma\_i(t) (state inertia tie-break)
   * Operates on the static signed matrix W, with nonzero entries +1/-1 and tied aggregate edges contributing zero influence. Attraction and repulsion are modeling assumptions, not observed psychological responses.



3. **Model 3: Temporal Signed NCO (Full Dynamic Realism):**

   * Update rule:
     * If \[sigma\_i(tau) \+ sum\_{j in N\_i(tau)} W\_ji(tau) \* sigma\_j(tau)\] \> 0  \--\>  sigma\_i(tau+1) \= \+1
     * If \[sigma\_i(tau) \+ sum\_{j in N\_i(tau)} W\_ji(tau) \* sigma\_j(tau)\] \< 0  \--\>  sigma\_i(tau+1) \= \-1
     * If sum \= 0, or if node i has no interactions during week tau  \--\>  sigma\_i(tau+1) \= sigma\_i(tau) (state inertia)
   * Operates across the full sequence of weekly snapshots on the same fixed core node set and node order. Use one synchronous update per snapshot initially. A quiet week or temporarily unchanged state is not a reason to stop before later snapshots arrive.

**Evaluation Metrics:** Keep the existing convention `s_1 = N_minus / N` and `s_2 = N_plus / N` for raw opinion fractions. Separately compute the largest connected component of each opinion-induced subgraph: `g_minus = S_minus_GCC / N` and `g_plus = S_plus_GCC / N`. Measure components on the fixed unsigned aggregate core, induced by the current opinion, so temporal inactivity does not change the connectivity reference. Plot both component fractions against `f` and intervention budget `Phi`. Record steady states, cycles, and update-limit stops separately; temporal runs report the outcome after all snapshots. Use matched initial states, recorded seeds, and comparable update horizons for model/control comparisons. Report uncertainty across realizations instead of treating small differences as established phase-transition shifts.

#### Pillar 3: Process Intervention (The ICO Model, Backfire Effects & Novel Targeting)

Introduce **Inflexible Contrarians** (the ICO model): a budget fraction `Phi` of nodes are converted into permanent fact-checkers (`sigma = +1`) that never flip.

* **The Backfire Effect in Signed Networks:** A pinned `+1` node contributes `-1` through a negative link: `W_ji * sigma_contrarian = (-1) * (+1) = -1`. Test whether this increases the opposing opinion fraction or component size relative to a matched no-intervention run. Negative influence is a model mechanism; a net backfire effect and any advantage of hub targeting must be measured.
* **Evaluation Metric (`Phi_c`):** Use the smallest tested budget for which `g_minus < 0.5` at the specified evaluation horizon. Include `Phi = 0`, keep the original core size `N` as denominator (including pinned nodes), and distinguish this cluster threshold from the raw fraction `s_1`. If the no-intervention outcome is already below the threshold, report zero; if the threshold is not reached within the tested range, report that fact rather than extrapolating.
* **Intervention Strategies Tested (Finding the Optimal Strategy under Budget `Phi`):**
  1. *Random selection* (Baseline).
  2. *Degree Centrality* (Hub targeting).
  3. *Betweenness Centrality* (Shortest-path bottlenecks).
  4. *Eigenvector Centrality* (Connected to well-connected nodes).
  5. *Dr. Wang’s Node Leverage Centrality (`l_ni`):* `l_ni = (1/d_i) * sum_{j in N_i} [(d_i - d_j) / (d_i + d_j)]`
  6. **Proposed Strategy — Signed Leverage Centrality (`l_signed`):** `l_signed(i) = (1/d_i) * sum_{j in N_i} W_ij * max(0, [(d_i - d_j) / (d_i + d_j)])`. Test whether favoring positive local leverage reduces backfire; the formula does not guarantee bridge selection or superior outcomes. Check related work before claiming novelty.
  7. **Proposed Strategy — Temporal-Activity Bridge Centrality:** Combine a defined inter-community bridge score with measured interaction frequency. Specify the normalization and observation window, and evaluate the score against the classical baselines. Treat full-window scores as retrospective; claims about acting before events require earlier-window features.
* **Relating Node Properties to Process Role (Empirical Impact `I_i`):**
  1. Measure each node's empirical intervention impact:
     `I_i = g_minus(no intervention) - g_minus(only node i pinned to +1)`, using matched initial states and seeds; positive values mean reduced opposing-cluster size.
  2. Run a simple correlation/regression between `I_i` and the node’s structural properties (degree, leverage, betweenness, signed bridge ratio, temporal activity).
  3. **Research Question:** Which topological property is the strongest statistical predictor of a node's ability to dismantle a polarized cascade?

#### Pillar 4: Scientific Depth & Controlled Null Models

To isolate whether observed opinion dynamics and intervention outcomes stem from real network structure, sign placement, or temporal ordering, we implement three controlled null models:

**1\. Null Model A — Sign Shuffling (Testing Sign Placement vs. Volume):**

* **The Method:** Keep unsigned topology, every node’s unsigned degree, and counts of positive, negative, and zero-sign aggregate edges fixed. Permute the complete sign vector over existing edges (`np.random.permutation`). This preserves global sign totals, not each node’s signed degree or the original relationship between signs and event-count weights.
* **Research Question:** Does the specific topological *placement* of hostility (e.g., concentrated along community boundaries) drive polarization, or does the system simply respond to the overall *volume* of negative links?
* **Hypothesis for Initial Observation:** The saved baseline in [nco_baseline.md](nco_baseline.md) shows modest unsigned/signed differences and substantial run variability near `f = 0.5`. A dense core or a low negative-edge fraction are possible explanations, not conclusions. Use matched sign-shuffled runs across the existing `f` sweep and compare uncertainty to test whether sign placement adds an effect.

**2\. Null Model B — Static Homophily Rewiring (Testing Echo-Chamber Insulation):**

* **The Method & Degree Preservation:** Use valid double-edge swaps (`(u, v), (x, y)` to `(u, y), (x, v)`) without self-loops or duplicate edges. Preserve each node’s unsigned aggregate degree and global positive/negative/zero sign counts. State how signs follow swapped edges and validate the invariants. Start with an unbiased degree-preserving reference, then bias swaps toward within- or between-community positive ties using fixed baseline community labels. Record attempted/accepted swaps and connectivity; do not assume that an extreme target configuration is attainable.
* **The Experiment:** Compare the empirical graph, the unbiased rewired reference, and feasible more-insular/more-integrated configurations. These are static controls: they do not preserve temporal degrees or each node’s signed degree unless explicitly constrained.
* **Research Question:** Does strong echo-chamber homophily insulate a misinformed/hostile cluster against fact-checker penetration, or does it make the cluster more vulnerable once a critical boundary node flips? We compare the critical intervention threshold `Phi_c` across rewired configurations.

**3\. Null Model C — Temporal Timestamp Shuffling (Testing Temporal Burstiness):**

* **The Method & Aggregate Preservation:** Permute timestamps among directed signed events while keeping every event’s endpoints and sign. This preserves full-window topology, aggregate net signs, node lifetime event totals, and the global timestamp histogram.
* **Degree Preservation Detail:** Node degrees and activity within individual weeks are not preserved. Randomization changes the assignment of activity and signs to times, while global calendar spikes remain. Report these limits alongside the static-rewiring invariants; neither control guarantees temporal-degree preservation.
* **Research Question:** Do empirical node/dyad timing patterns affect opinion dynamics and interventions relative to the timestamp-permuted control? This control is not a generated memoryless Poisson process and does not isolate every type of temporal correlation.

#### Configuration Checks and Literature

* **Narrow Initial Sensitivity Check:** Reuse the two existing core extraction methods. Keep the connected 10-core as the default and compare nearby `k = 8, 10, 12` values as a small pilot, alongside the current activity-LCC setting. Reuse the existing `f` sweep, seeds/realization count, sign-tie rule, and update limit; initially keep weekly snapshots fixed. Report graph sizes, sign fractions, and selected process outcomes so the configuration can be justified. These pilot values are proposed checks, not established optimal settings.
* **Scope Limit:** Avoid a full parameter cross-product. Larger threshold grids, snapshot-width tests, alternative initial states, and extended budget ranges belong to the optional TBD section in [work_distribution_2.md](work_distribution_2.md).
* **Literature & Citations:** Verify and cite the dataset, k-core extraction, community detection/modularity, spectral metrics, NCO update rule, ICO model and classical targeting strategies, leverage centrality, structural balance, and null-model methods. Distinguish the published model from this project’s signed/temporal extensions. The existing literature owners in Work Distribution 2 retain their areas; checking exact formula/source matches is a deliverable, not a claim that every reference is already verified.

### Important Methodological Note (Defense Guardrail)

The Reddit dataset tracks interaction sentiment (`+1` vs. `-1`), not ground-truth factual correctness or verified political votes. Therefore:

* `sigma_i in {+1, -1}` represents a simulated competing state; neither value is an observed subreddit belief, factual correctness label, or political vote.
* Misinformation and fact-checking serve as real-world motivating analogies. The formal scientific claims of this project focus on **opinion dynamics, psychological reactance, structural balance, and targeted network containment**.

### Optional Extension (TBD): Triadic Structural Balance and Temporal Sentiment Prediction (P-O-X Theory)

**Research Goal:**
Investigate whether the sentiment directed by two communities (`A` and `B`) toward a shared external community `C` predicts the emergence and sentiment of a direct link between `A` and `B` (`A -> B`).

**Theoretical Motivation (Lecture 3 — Motifs & Structural Balance):**
This applies Fritz Heider’s classic **P-O-X Structural Balance Theory** to inter-community macro-dynamics:

* Person `P` and Other `O` represent Subreddits `A` and `B`.
* Entity `X` represents a shared external target Subreddit `C`.
* We test whether community alliances self-organize according to structural balance (e.g., *"the enemy of my enemy is my friend"*), connecting directly to **Signed Triadic Motifs** (Lecture 3, Slides 24–29).

**Hypotheses to Test:**

1. **Triadic Balance Effect:** Subreddits with aligned sentiment toward a target (`s_AC = s_BC`) are significantly more likely to form a positive direct tie (`s_AB = +1`) than those with conflicting sentiment (`s_AC != s_BC`).
2. **The "Common Enemy" Bias:** Shared hostility toward an outside target (`s_AC = s_BC = -1`) is a stronger predictor of a positive direct alliance (`s_AB = +1`) than shared affinity (`s_AC = s_BC = +1`).

**Temporal Ordering (Temporal Motifs, Lecture 3, Slide 28):** Require a strict time window `delta`; temporal precedence is not evidence of causation:

* Filter interactions such that external sentiments strictly precede the direct link: `t_AC < t_AB` and `t_BC < t_AB` within a window `delta` (e.g., 30 days).
* **Research Question:** Does external conflict catalyze direct alliances in time, or do direct ties form independently of shared targets?

**Implementation & Effort:**

* **Optional Baseline:** Use supplied signs and timestamps to compare closed and unclosed temporal triangles, with a stated candidate-pair denominator and comparison baseline. Check repeated-event counting and computational cost before scheduling the work. Static signed-triangle analysis remains in the main plan.
* **Secondary Stretch (Optional):** If time permits, evaluate whether the text features in `PROPERTIES` (e.g., anger/affect word counts) add predictive power over binary signs.

---

### How to Utilize Each Field in the Dataset

The dataset (`soc-redditHyperlinks-body.tsv`) contains 6 fields:

| Field Name | Use It? | How to Use It Simply |
| :---- | :---: | :---- |
| **`SOURCE_SUBREDDIT`** | **YES** | Represents the source node `u` that generated the post/mention. |
| **`TARGET_SUBREDDIT`** | **YES** | Represents the target node `v` that was linked to. Combined with `SOURCE`, this builds the directed edges (`u -> v`). |
| **`LINK_SENTIMENT`** | **YES** | Supplied source-post/target label, used as an event sign. Neutral/positive is `+1`; explicitly negative is `-1`. This is distinct from the simulated state. |
| **`TIMESTAMP`** | **YES** | Format: `YYYY-MM-DD HH:MM:SS`. Used to bin interactions into weekly time slices `tau` for Model 3 (Temporal NCO/ICO) and temporal centrality calculations. |
| **`POST_ID`** | **AUDIT ONLY** | Retain when checking repeated posts/links and potential body/title overlap; not needed by the simulation engine. |
| **`PROPERTIES`** | **OPTIONAL** | Source-post text features; excluded from the main experiments. Any additional feature-based prediction belongs to TBD. |

---

### Main Edge Collapse and Optional Alternatives (Considering \+1 / \-1 Sentiment)

When Subreddit `A` links to Subreddit `B` multiple times across the observation window:

1. **Majority Rule / Net Sign (Recommended for Model 2):**
   * If `Positive Links > Negative Links` \-\> Edge is **`+1` (Constructive tie)**.
   * If `Negative Links > Positive Links` \-\> Edge is **`-1` (Hostile tie)**.
   * Equal counts give `W_AB = 0`; keep the edge in unsigned topology. Use the same rule within each temporal snapshot. This matches the current pipeline and supersedes the older Phase 1 document’s positive-tie convention without changing that document.
2. **Net Normalized Continuous Weight (Optional, TBD):** `W_AB = (N_pos - N_neg) / (N_pos + N_neg)` with values in `[-1.0, +1.0]`.
3. **Edge Activity Thresholding (Optional, TBD):** Retain an edge only if total interactions `(N_pos + N_neg) >= k` (e.g., `k = 3`) to eliminate transient one-off mentions.
4. **Multiplex / Two-Layer Representation (Optional, TBD):** Split into two parallel static graphs: Layer 1 for positive links and Layer 2 for negative links.

---

### Initial State and Optional Alternatives

1. **Synthetic Fraction Sweep (Main Experiment):** Randomly assign fraction `f` of nodes to `-1` and the rest to `+1`. Reuse the current `f = 0.1, 0.2, ..., 0.9` sweep and match initial states across comparable models and controls. Here `f` always means the initial negative fraction, even where it exceeds one half.
2. **Historical Sentiment or Interaction Role (Optional, TBD):** Explore initialization from earlier-window signed event totals or outbound/inbound negative activity. Define zero-event handling and use only a preceding window for a prospective interpretation. These are constructed states, not verified civil/hostile community labels.
3. **Topic-Specific Seeds (Optional, TBD):** Define and justify a topic-focused subgraph and a transparent seeding rule. The hyperlink labels do not establish political affiliation or observed opinions.
4. **Subreddit Embedding Initialization (Optional, TBD):** Explore whether [SNAP subreddit embeddings](https://snap.stanford.edu/data/web-RedditEmbeddings.html) can provide a reproducible clustering/seeding rule. Check core-node coverage, missing embeddings, and the training period; embedding similarity does not establish an opinion label. Ownership and scheduling remain TBD in Work Distribution 2.

### Presentation and Handoffs

* **Feedback meeting presenters:** Daniel, Jacek, and Polly.
* **Final meeting presenters:** Antreas, Georgi, and Jakub. Build the final 15-minute presentation from the shared dataset/methods, graph diagnostics, model/control comparisons, intervention results, uncertainty, and limitations; cite sources on the relevant slides.
* **Phase 2 work:** See [work_distribution_2.md](work_distribution_2.md) for owners, added tasks, feedback coverage, and the separate optional TBD section. Phase 1 assignments in [work_distribution.md](work_distribution.md) remain unchanged.
