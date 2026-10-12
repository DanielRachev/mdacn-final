# Echo Chambers, Misinformation, and Targeted Fact-Checking: Testing the Inflexible Contrarian Model on Real-World Interaction Networks

**The Concept:** Real-world social networks suffer from echo chambers and polarization. We investigate how opinions reach equilibrium on interaction topologies and evaluate intervention strategies to break polarized consensus using stubborn fact-checkers/moderators.

### 1\. Dataset (Instant Download, Zero Scraping)

* **Dataset:** **SNAP Reddit Hyperlink Network** (`soc-RedditHyperlinks-body.tsv`). Clean TSV containing directed, timestamped interactions between subreddits with sentiment (+1 / \-1).  
* **Format:** Nodes represent subreddits; directed edges represent cross-community hyperlinks.  
* **Size:** \~55,000 nodes, \~858,000 edges. Downloadable in \~10 seconds from SNAP.

### 2\. Technical Execution Plan (Addressing All Rubric Pillars)

#### Pillar 1: Network Characterization (Topological \+ Spectral Metrics) \- Aggregate Graph

* **Pre-processing:** Because spectral connectivity, Newman's modularity, and standard NCO updates assume symmetric interaction topologies, we convert the multi-edge directed graph into an aggregate undirected graph where an edge exists if either community references the other (or require mutual references for the core subgraph), summing positive and negative mentions to assign net edge signs.  
* **Degree Distribution:** Fit power-law vs. Poisson (Barabási scale-free vs. ER random graph); evaluate heavy-tail properties.  
* **Community Structure & Modularity:** Compute Newman’s Modularity: `m = 1/(2L) * sum_{i,j} [a_ij - (d_i * d_j)/(2L)] * delta(c_i, c_j)` to quantify baseline echo-chamber clustering.  
* **Assortativity & Signed Balance:** Calculate degree correlation `rho_D` and the ratio of positive/negative triangles to test Structural Balance Theory.  
* **Spectral Properties:** Adjacency spectral radius `lambda_1(A)` and Laplacian algebraic connectivity `a(G) = mu_(N-1)` (measuring network cohesion and structural resistance to disconnection).  
* **Signed Topology Distribution:** Measure the global fraction of negative interactions and test whether negative links occur disproportionately across community bridges vs. within modular clusters.  
* **Reciprocity and Strength:** Compute directed in/out-degree and in/out-strength (`N_pos` and `N_neg`) to identify asymmetrical aggressor vs. target communities.

#### Pillar 2: Dynamic Process Modeling (Ablation: Static vs. Signed vs. Temporal NCO)

We modify the standard NCO model so positive and negative links affect opinion updates differently, modeling both social solidarity and psychological reactance:

* **Positive link (`W_ji = +1`):** Trust / Attraction. Encourages node `i` to align with neighbor `j`.  
* **Negative link (`W_ji = -1`):** Hostility / Repulsion. Triggers psychological reactance: if a hostile rival holds opinion `+1`, node `i` is pushed toward `-1` ("the enemy of my enemy is my friend").

We implement an **Ablation Study across 3 model tiers** to evaluate whether adding real-world complexity alters system dynamics:  
.**Model 1: Static Unsigned NCO (Course Baseline):**

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
   * Operates on the static signed matrix W (+1/-1). Incorporates psychological reactance and structural balance.

   

3. **Model 3: Temporal Signed NCO (Full Dynamic Realism):**  
     
   * Update rule:  
     * If \[sigma\_i(tau) \+ sum\_{j in N\_i(tau)} W\_ji(tau) \* sigma\_j(tau)\] \> 0  \--\>  sigma\_i(tau+1) \= \+1  
     * If \[sigma\_i(tau) \+ sum\_{j in N\_i(tau)} W\_ji(tau) \* sigma\_j(tau)\] \< 0  \--\>  sigma\_i(tau+1) \= \-1  
     * If sum \= 0, or if node i has no interactions during week tau  \--\>  sigma\_i(tau+1) \= sigma\_i(tau) (state inertia)  
   * Operates across discrete time-slices tau (e.g., weekly snapshots).

**Evaluation Metrics:** Compare final steady-state opinion distributions (`s_1, s_2`), convergence speeds, and phase transitions as a function of the initial minority fraction `f` across all three models.

#### Pillar 3: Process Intervention (The ICO Model, Backfire Effects & Novel Targeting)

Introduce **Inflexible Contrarians** (the ICO model): a budget fraction `Phi` of nodes are converted into permanent fact-checkers (`sigma = +1`) that never flip.

* **The Backfire Effect in Signed Networks:** In standard unsigned models, placing contrarians at central hubs always aids adoption. In signed networks, if a contrarian is connected via negative links (`W_ji = -1`), their presence repels neighbors: `W_ji * sigma_contrarian = (-1) * (+1) = -1` This models the real-world **backfire effect**, where adversarial fact-checking hardens opposing beliefs.  
* **Evaluation Metric (`Phi_c`):** Measure the critical intervention fraction `Phi_c`, defined as the minimum percentage of contrarians required to break majority consensus (e.g., reducing the dominant opinion below 50% or eliminating the toxic cluster).  
* **Intervention Strategies Tested (Finding the Optimal Strategy under Budget `Phi`):**  
  1. *Random selection* (Baseline).  
  2. *Degree Centrality* (Hub targeting).  
  3. *Betweenness Centrality* (Shortest-path bottlenecks).  
  4. *Eigenvector Centrality* (Connected to well-connected nodes).  
  5. *Dr. Wang’s Node Leverage Centrality (`l_ni`):* `l_ni = (1/d_i) * sum_{j in N_i} [(d_i - d_j) / (d_i + d_j)]`  
  6. **Novel Strategy — Signed Leverage Centrality (`l_signed`):** `l_signed(i) = (1/d_i) * sum_{j in N_i} W_ij * max(0, [(d_i - d_j) / (d_i + d_j)])` *Why it is novel:* Adapts Dr. Wang’s leverage metric for signed graphs. It selects high-leverage structural bridges that maintain **predominantly positive ties** into target clusters, maximizing influence while preventing the backfire effect.  
  7. **Novel Strategy — Temporal-Activity Bridge Centrality:** Targets nodes that exhibit both high inter-community betweenness and high temporal interaction frequency (`a_i = eta * x_i`), acting before conflict cascades.  
* **Relating Node Properties to Process Role (Empirical Impact `I_i`):**  
  1. Measure each node's empirical intervention impact:  
     `I_i = (outcome without targeting node i) - (outcome when targeting node i)`  
  2. Run a simple correlation/regression between `I_i` and the node’s structural properties (degree, leverage, betweenness, signed bridge ratio, temporal activity).  
  3. **Research Question:** Which topological property is the strongest statistical predictor of a node's ability to dismantle a polarized cascade?

#### Pillar 4: Scientific Depth / Null Model Experiment (Opinion Homophily Rewiring)

Rather than standard degree-degree assortativity, we implement an **Opinion / Sign Homophily Rewiring Null Model**:

* **The Method:** We perform edge rewiring that keeps each node’s degree strictly invariant, but systematically tunes **opinion homophily** (the proportion of positive cross-talk vs. intra-group insular links).  
* **The Experiment:** Vary the network from a fully insular echo-chamber configuration (all positive links internal, only negative links across groups) to an integrated cross-cutting configuration.  
* **Research Question:** Does strong echo-chamber homophily protect a misinformed/toxic cluster against fact-checker penetration, or does it make the cluster more vulnerable once a trusted bridge node flips? Compare the critical intervention threshold `Phi_c` across rewired topologies.

**Null Model B — Sign Shuffling (Zero-Topology Modification):**

* Keep the network graph, node degrees, and total counts of positive and negative edges completely fixed, but randomly shuffle which edges receive the `+1` and `-1` signs.  
* **Research Question:** Does the specific *placement* of hostility (e.g., along community borders) matter, or does system polarization depend solely on the total *volume* of hostile interactions?

### Important Methodological Note (Defense Guardrail)

The Reddit dataset tracks interaction sentiment (`+1` vs. `-1`), not ground-truth factual correctness or verified political votes. Therefore:

* `sigma_i in {+1, -1}` represents a simulated competing state (e.g., civil vs. hostile alignment, or polarized stance).  
* Misinformation and fact-checking serve as real-world motivating analogies. The formal scientific claims of this project focus on **opinion dynamics, psychological reactance, structural balance, and targeted network containment**.

### Optional Extension: Triadic Structural Balance and Temporal Sentiment Prediction (P-O-X Theory)

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

**Temporal Causal Ordering (Temporal Motifs, Lecture 3, Slide 28):** Instead of static co-occurrence, enforce a strict time window `delta`:

* Filter interactions such that external sentiments strictly precede the direct link: `t_AC < t_AB` and `t_BC < t_AB` within a window `delta` (e.g., 30 days).  
* **Research Question:** Does external conflict catalyze direct alliances in time, or do direct ties form independently of shared targets?

**Implementation & Effort:**

* **Fast Baseline:** Test this using only the native `LINK_SENTIMENT` (`+1` / `-1`) and `TIMESTAMP` columns by counting closed vs. unclosed balanced temporal triangles. This takes under 50 lines of Python.  
* **Secondary Stretch (Optional):** If time permits, evaluate whether the text features in `PROPERTIES` (e.g., anger/affect word counts) add predictive power over binary signs.

---

### How to Utilize Each Field in the Dataset

The dataset (`soc-RedditHyperlinks-body.tsv`) contains 6 fields:

| Field Name | Use It? | How to Use It Simply |
| :---- | :---: | :---- |
| **`SOURCE_SUBREDDIT`** | **YES** | Represents the source node `u` that generated the post/mention. |
| **`TARGET_SUBREDDIT`** | **YES** | Represents the target node `v` that was linked to. Combined with `SOURCE`, this builds the directed edges (`u -> v`). |
| **`LINK_SENTIMENT`** | **YES** | Integer label (`+1` or `-1`). Used as edge sign `W_uv`: `+1` represents constructive/friendly engagement; `-1` represents hostile/brigading interaction. |
| **`TIMESTAMP`** | **YES** | Format: `YYYY-MM-DD HH:MM:SS`. Used to bin interactions into weekly time slices `tau` for Model 3 (Temporal NCO/ICO) and temporal centrality calculations. |
| **`POST_ID`** | **NO** | Internal Reddit post identifier. Safely ignore. |
| **`PROPERTIES`** | **NO** | Vector of 86 text-analysis features. Ignore completely to avoid unneeded NLP processing. |

---

### Options for Collapsing Multiple Edges (Considering \+1 / \-1 Sentiment)

When Subreddit `A` links to Subreddit `B` multiple times across the observation window:

1. **Majority Rule / Net Sign (Recommended for Model 2):**  
   * If `Positive Links > Negative Links` \-\> Edge is **`+1` (Constructive tie)**.  
   * If `Negative Links > Positive Links` \-\> Edge is **`-1` (Hostile tie)**.  
   * Ties can be broken using total historical post count or set to 0 (no edge).  
2. **Net Normalized Continuous Weight:** `W_AB = (N_pos - N_neg) / (N_pos + N_neg)` with values in `[-1.0, +1.0]`.  
3. **Activity Thresholding (Noise Reduction):** Retain an edge only if total interactions `(N_pos + N_neg) >= k` (e.g., `k = 3`) to eliminate transient one-off mentions.  
4. **Multiplex / Two-Layer Representation:** Split into two parallel static graphs: Layer 1 for positive links and Layer 2 for negative links.

---

### Initial Score Options

1. **Empirical Historical Baseline (Recommended):** Calculate each subreddit’s net historical sentiment over an initial training window: `Score(i) = [Count(+1 links) - Count(-1 links)] / Total links`  
   * If `Score(i) >= 0` \-\> `sigma_i(0) = +1` (Constructive / civil community).  
   * If `Score(i) < 0` \-\> `sigma_i(0) = -1` (Hostile / conflict-prone community).  
2. **Aggressor vs. Target Role (Asymmetry):** Calculate whether a community primarily initiates hostile links (outbound negative) or receives them (inbound negative): `Role(i) = [d_out_neg(i) - d_in_neg(i)] / [d_out_neg(i) + d_in_neg(i)]` Aggressors start as `-1`; targets/bystanders start as `+1`.  
3. **Synthetic Fraction Sweep (`f in [0, 1]`):** Randomly assign a fraction `f` of nodes to `-1` and `(1 - f)` to `+1`. Sweep `f` from `0.1` to `0.9` to produce the theoretical phase transition plots shown in Lecture 4 (Slide 34).  
4. **Topic-Specific Seeds (e.g., Political Polarization):** Filter the active subgraph to political subreddits (e.g., subreddits within 1 hop of `r/politics` and `r/conservative`). Initialize based on known community leanings (Democrat \= `+1`, Republican \= `-1`), using negative links as cross-party hostility.

Tuesday: Daniel, Jacek, Polly   
Final: Antreas, Georgi, Jakub

Use subreddit embedding for the initial state (?) \- http\://snap.stanford.edu/data/web-RedditEmbeddings.html