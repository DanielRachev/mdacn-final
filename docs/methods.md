# Methods: data, labels, and structural balance

## Data and label

We use the body file of the SNAP Reddit Hyperlink Network [1, 2]. Each row is a
hyperlink from a post in a source subreddit to a target subreddit, with a
timestamp, a label, and text features of the source post.

- **Label**: `LINK_SENTIMENT` is −1 if the source post is negative toward the
  target and +1 if it is neutral or positive [2]. It was produced by a text
  classifier trained on crowd-sourced labels [1].
- **Text features**: `PROPERTIES` is a separate 86-element vector of source-post
  text properties (counts, VADER sentiment, LIWC categories) [2] that is not utilized in this study.
- **Simulated opinions**: NCO/ICO opinions are synthetic initial states.

## Dataset size and filtering

The SNAP dataset consists of 55,863 subreddits and 858,490 hyperlinks from January
2014 to April 2017 [2].

| Stage | Rows | Posts | Nodes | Undirected edges | Zero-sign edges |
| --- | --- | --- | --- | --- | --- |
| Body file | 286,561 | 259,092 | 35,776 | — | — |
| Self-links removed (none found); directions merged by majority sign | 286,561 | 259,092 | 35,776 | 124,330 | 1,983 |
| Connected 10-core | 159,180 | 140,761 | 2,896 | 58,466 | 1,469 |

Negative fractions depend on the unit. Collapsing repeated events into one
majority-sign edge lowers the negative share:

| Unit | Body file | 10-core |
| --- | --- | --- |
| Events (rows) | 7.35% | 9.86% |
| Source posts | 7.29% | 10.04% |
| Directed pairs (majority sign) | 5.56% | 6.98% |
| Undirected edges (majority sign) | 5.68% | 7.17% |

Sources: `results/label_audit/filtering_stages.csv` and
`results/directed_roles/sign_fractions.csv`.

## Null models

Invariants of null models: [Null models](null_models.md).

## Structural balance and leverage

Heider (1946, 1958) [3, 4] introduced the concept of attitude triads to model a
person's perceived relations. A triad is balanced when it creates no pressure for
further change, which is the case when the number of negative relations is even.

Cartwright & Harary (1956) [5] applied this concept to social graphs, where the
entities are nodes and signed relations are edges. They showed that a graph is
balanced exactly when its nodes can be split into at most two groups with positive
intra- and negative inter-group relations, building on the original idea of P-O-X
theory.

Li et al. (2015) [6] showed that leverage centrality is one of the more effective
heuristics for selecting inflexible antagonists in a network, which can be applied
in our project to select contrarians.

## References

1. Kumar, S., Hamilton, W. L., Leskovec, J., and Jurafsky, D. (2018). *Community
   Interaction and Conflict on the Web*. Proceedings of the 2018 World Wide Web
   Conference (WWW '18), 933–943.
   [DOI](https://doi.org/10.1145/3178876.3186141).
2. Stanford Network Analysis Project. *Reddit Hyperlink Network*.
   [Dataset page](https://snap.stanford.edu/data/soc-RedditHyperlinks.html).
3. Heider, F. (1946). *Attitudes and Cognitive Organization*. The Journal of
   Psychology, 21(1), 107–112.
   [DOI](https://doi.org/10.1080/00223980.1946.9917275),
   [Course copy](https://snap.stanford.edu/class/cs224w-readings/heider49balance.pdf).
4. Heider, F. (1958). *The Psychology of Interpersonal Relations*. John Wiley &
   Sons. [DOI](https://doi.org/10.1037/10628-000).
5. Cartwright, D., and Harary, F. (1956). *Structural balance: A generalization of
   Heider's theory*. Psychological Review, 63(5), 277–293.
   [DOI](https://doi.org/10.1037/h0046049).
6. Li, C., Li, Q., Van Mieghem, P., Stanley, H. E., and Wang, H. (2015).
   *Correlation between centrality metrics and their application to the opinion
   model*. The European Physical Journal B, 88(3), 65.
   [DOI](https://doi.org/10.1140/epjb/e2015-50671-y),
   [Author preprint](https://arxiv.org/abs/1409.6033).
