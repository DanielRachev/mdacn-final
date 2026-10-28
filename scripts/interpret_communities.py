"""Reproduce reviewed descriptions of one saved Louvain partition, without network access.

The annotations are human interpretations, not fitted topic labels. Exact input
hashes prevent accidentally applying them to a different partition or layout.
"""

import argparse
import csv
import hashlib
import math
import os
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DATE = "2026-10-28"
BASE_COMMIT = "7f205f43dd93b0ae7809d07728521408ffc156fe"
SEED = 0
EXPECTED_HASHES = {
    "graph": "84be8ddab207881a7fd770925a34346ee4464070c6a40294a2d135e61c3314e2",
    "communities": "649b32e25d002eeda771311adca840f4e0d8277b6c9b4b51637bab338fae78d1",
    "layout": "fb23cade7454280e1790cd7aebd56d1b8d42491bf3fdc1a70925a39a12036228",
}

# Indexed by saved ID. Display numbers follow the existing figure: ID + 1.
ANNOTATIONS = {
    0: {
        "label": "Arts, media & storytelling",
        "examples": ("movies", "writingprompts"),
        "confidence": "medium",
        "interpretation": (
            "Movies, music, books, television and creative writing dominate the central "
            "members. The random sample adds music genres and fiction fandoms such as "
            "popheads, trap, asoiaf and the100. This supports a broad media and storytelling "
            "label. iama is a general interview hub, while lfg and retrofuturism broaden "
            "the interpretation; this is not a single genre."
        ),
    },
    1: {
        "label": "General discussion & advice (mixed)",
        "examples": ("askreddit", "relationships"),
        "confidence": "medium",
        "interpretation": (
            "askreddit, pics and funny sit alongside personal advice and self-disclosure "
            "forums such as legaladvice, relationships and offmychest. The random sample "
            "ranges from gradadmissions and mentalillness to occult and pittsburgh. "
            "General discussion and advice describe the central strands, but the diversity "
            "does not support a narrow subject label."
        ),
    },
    2: {
        "label": "Gaming & consumer technology",
        "examples": ("gaming", "techsupport"),
        "confidence": "medium",
        "interpretation": (
            "gaming, games, leagueoflegends, overwatch and hearthstone anchor the gaming "
            "strand; pcmasterrace, techsupport, buildapc and android add devices and "
            "support. arma, kotor, nintendoswitch, ssbm and overclocking in the random "
            "sample reinforce these themes. outoftheloop, copypasta, help and postpreview "
            "show that memes and Reddit support also belong to this group."
        ),
    },
    3: {
        "label": "Sports & team fandoms",
        "examples": ("nfl", "nba"),
        "confidence": "high",
        "interpretation": (
            "The central members name sports leagues and teams, including nfl, nba, "
            "baseball, hockey, denverbroncos and seahawks. The random sample adds habs, "
            "mls, padres and other teams. A first-person charity-contest announcement "
            "also connects the less obvious charitablebets member to fantasy football "
            "[S3]. The shared sports theme is clear in both inspected subsets; it does "
            "not imply that rival fan groups agree."
        ),
    },
    4: {
        "label": "News, politics & Reddit meta",
        "examples": ("worldnews", "subredditdrama"),
        "confidence": "medium",
        "interpretation": (
            "worldnews, news, politics, conspiracy, the_donald and shitliberalssay mix "
            "with subredditdrama, drama, circlebroke and other forums about Reddit "
            "itself. The random sample includes armenia, podemos, mhocpress, shadowban "
            "and solotravel. This is a mixture of news, political discussion and "
            "meta-commentary. The names suggest several political perspectives; "
            "ideological composition was not independently verified."
        ),
    },
    5: {
        "label": "Localities & unresolved cases (mixed)",
        "examples": ("adelaide", "gratefuldoe"),
        "confidence": "low",
        "interpretation": (
            "All 31 names were inspected. Australian and US locality names occur "
            "alongside unresolvedmysteries, gratefuldoe, lylestevik and unexplainedphotos. "
            "The current gratefuldoe rules confirm a focus on unidentified-person cases "
            "[S1], but the LyleStevik and BNEStrong lookups yielded no usable primary "
            "page. blog, everything and volunteer further weaken a single-topic reading. "
            "The label records two visible strands; it does not explain why they link."
        ),
    },
    6: {
        "label": "Elite Dangerous: Powerplay factions",
        "examples": ("aislingduval", "elitehudson"),
        "confidence": "high",
        "interpretation": (
            "All 12 names form a consistent game-faction pattern: aislingduval, "
            "eliteantal, elitehudson, elitelavigny, elitewinters, elitecg, elitemahon, "
            "elitepatreus, elitetorval, imperialhighcommand, kumocrew and elitesirius. "
            "A July 2015 player coordination post for Elite Dangerous Powerplay links "
            "directly to r/AislingDuval [S2], providing evidence from the observation "
            "period. The broad game context is well supported; each faction was not "
            "independently verified, and links alone do not establish alliances."
        ),
    },
    7: {
        "label": "Technology, crypto & practical projects",
        "examples": ("bitcoin", "learnprogramming"),
        "confidence": "medium",
        "interpretation": (
            "bitcoin, dogecoin and cryptocurrency occur with learnprogramming, sysadmin, "
            "webdev, raspberry_pi and programming. forhire, cscareerquestions and "
            "startups add work and exchange. The random sample includes asknetsec and "
            "web_design, but also gardening, homeimprovement, diyaudio and uber. A broad "
            "technology, crypto and practical-project label fits better than a claim "
            "that every member concerns cryptocurrency."
        ),
    },
    8: {
        "label": "Mixed near-clique; topic unclear",
        "examples": ("killlakill", "vmware"),
        "confidence": "low",
        "interpretation": (
            "All 13 names were inspected. They range from killlakill and thewire to "
            "vmware, nvidiashield, voxelgamedev and iupui, with several opaque names. "
            "The four targeted opaque-name checks yielded no usable page. The graph "
            "contains 75 of the 78 possible internal edges (96.2%), but this density "
            "does not establish a shared subject. Keep the topical interpretation "
            "unclear; bot activity, common ownership and coordinated linking remain "
            "untested explanations."
        ),
    },
}

# Nine distinct subreddit targets, below the agreed cap of 15. Repeated access
# attempts for a target are one check. Failed retrievals supply no semantic evidence.
LOOKUPS = (
    (
        "gratefuldoe",
        "[S1: current subreddit rules](https://www.reddit.com/r/gratefuldoe/about/)",
        "Retrieved; unidentified-person case focus. Contemporary, not an archived sidebar.",
    ),
    (
        "aislingduval",
        "[S2: player coordination post, 2015-07-16]"
        "(https://steamcommunity.com/app/359320/discussions/0/535150948597144836/)",
        "Retrieved historical primary post; explicitly links the subreddit to Powerplay. "
        "Direct wiki/old-Reddit retrieval failed.",
    ),
    (
        "charitablebets",
        "[S3: contest organiser's post, 2021-11-18]"
        "(https://www.reddit.com/r/fantasyfootball/comments/qwv5cw/)",
        "Search-index text of a primary post connects it to fantasy-football charity "
        "contests. Postdates the dataset; direct about-page retrieval failed.",
    ),
    (
        "lylestevik",
        "[Attempted subreddit page](https://www.reddit.com/r/LyleStevik/about/)",
        "No usable primary page from retrieval or targeted search. A secondary result "
        "was excluded; identity remains unverified here.",
    ),
    (
        "bnestrong",
        "[Attempted subreddit page](https://www.reddit.com/r/BNEStrong/about/)",
        "Retrieval failed; targeted search found no usable result. No meaning assigned.",
    ),
    *(
        (
            name,
            f"[Attempted subreddit page](https://www.reddit.com/r/{name}/)",
            "Retrieval failed and search gave no usable match. No meaning assigned.",
        )
        for name in ("mbotandgtron", "nomina", "cbf4life", "iwishiwaslucky")
    ),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    nodes = [row["node"] for row in rows]
    if len(nodes) != len(set(nodes)):
        raise ValueError(f"Duplicate node rows in {path}")
    return rows


def load_inputs(paths: dict[str, Path]) -> tuple[nx.Graph, dict[str, tuple[float, float]]]:
    for kind, path in paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != EXPECTED_HASHES[kind]:
            raise ValueError(
                f"Reviewed {kind} artifact differs: expected {EXPECTED_HASHES[kind]}, "
                f"got {actual}. Review the annotations before using a new artifact."
            )
    graph = nx.read_graphml(paths["graph"])
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Expected a simple undirected core without self-loops")
    community_rows = read_rows(paths["communities"])
    layout_rows = read_rows(paths["layout"])
    for rows in (community_rows, layout_rows):
        labels = {row["node"]: int(row["community"]) for row in rows}
        if set(labels) != set(graph):
            raise ValueError("Graph, community CSV and layout must contain the same nodes")
        if any(graph.nodes[node]["community"] != label for node, label in labels.items()):
            raise ValueError("Graph, community CSV and layout must agree on every label")
    if set(nx.get_node_attributes(graph, "community").values()) != set(ANNOTATIONS):
        raise ValueError("The reviewed saved IDs must be exactly 0 through 8")
    for community, annotation in ANNOTATIONS.items():
        examples = annotation["examples"]
        if len(examples) != 2 or any(
            graph.nodes.get(node, {}).get("community") != community for node in examples
        ):
            raise ValueError(f"Figure examples must belong to saved community {community}")
    positions = {row["node"]: (float(row["x"]), float(row["y"])) for row in layout_rows}
    if not all(math.isfinite(value) for point in positions.values() for value in point):
        raise ValueError("Layout coordinates must be finite")
    if (len(graph), graph.number_of_edges()) != (2896, 58466):
        raise ValueError("Expected the reviewed 2,896-node, 58,466-edge core")
    return graph, positions


def profile_communities(graph: nx.Graph) -> list[dict]:
    profiles = []
    for community in sorted({data["community"] for _, data in graph.nodes(data=True)}):
        members = sorted(n for n in graph if graph.nodes[n]["community"] == community)
        internal = graph.subgraph(members)
        degrees = dict(internal.degree())
        top = sorted(members, key=lambda node: (-degrees[node], node))[:15]
        remainder = sorted(set(members) - set(top))
        # Reset the generator per community: sampling never depends on group order.
        sample = (
            remainder
            if len(members) <= 40
            else sorted(random.Random(SEED).sample(remainder, min(10, len(remainder))))
        )
        internal_edges = internal.number_of_edges()
        boundary = sum(graph.degree(n) - degrees[n] for n in members)
        volume = 2 * internal_edges + boundary
        profiles.append(
            {
                "community": community,
                "size": len(members),
                "internal_edges": internal_edges,
                "boundary_edges": boundary,
                "outside_share": boundary / volume if volume else 0.0,
                "top": top,
                "sample": sample,
                "degrees": degrees,
            }
        )
    if sum(p["size"] for p in profiles) != len(graph):
        raise ValueError("Community sizes do not cover the graph")
    if sum(2 * p["internal_edges"] + p["boundary_edges"] for p in profiles) != 2 * len(graph.edges):
        raise ValueError("Internal and boundary counts do not reconcile with the graph")
    return profiles


def render_report(profiles: list[dict], figure_link: str) -> str:
    inspected = sum(len(p["top"]) + len(p["sample"]) for p in profiles)
    internal = sum(p["internal_edges"] for p in profiles)
    between = sum(p["boundary_edges"] for p in profiles) // 2
    lines = [
        "# Interpreting the nine Louvain communities",
        "",
        f"Reviewed {REVIEW_DATE}. These descriptions concern the saved body-link connected "
        "10-core: 2,896 subreddits, 58,466 undirected edges, nine communities.",
        "",
        "Sports and the small Elite Dangerous group have the clearest inspected themes. "
        "The larger communities combine related subjects with general discussion, support "
        "or Reddit meta-commentary. Two small groups remain mixed or unclear. These are "
        "qualitative descriptions of a structural partition, not measured topic purity.",
        "",
        f"![Labeled saved community layout]({figure_link}.png)",
        "",
        f"[PDF figure]({figure_link}.pdf). Colours and coordinates match "
        "the original plot. The legend gives two example subreddits per community. "
        "Figure numbers are saved IDs plus one; the mapping is retained in the table below.",
        "",
        "## Community profiles",
        "",
        "Internal edges join two members of a group; boundary edges join a member to "
        "another group. Outside share is `boundary / (2 * internal + boundary)`: the "
        "fraction of the group's incident **edge ends** crossing its boundary. It is "
        "unweighted and descriptive, without a null-model comparison. Boundary edges "
        "appear once in each endpoint group's row.",
        "",
        "| Figure / saved ID | Interpretation | Confidence | Nodes | Internal edges | "
        "Boundary edges | Outside share |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for profile in profiles:
        c = profile["community"]
        annotation = ANNOTATIONS[c]
        lines.append(
            f"| {c + 1} / {c} | {annotation['label']} | {annotation['confidence']} | "
            f"{profile['size']:,} | {profile['internal_edges']:,} | "
            f"{profile['boundary_edges']:,} | {profile['outside_share']:.1%} |"
        )
    lines += [
        "",
        "The Elite Dangerous group's outside share is 11.5%; sports is 27.9%. By "
        "comparison, the locality/case mixture has 79.2% outside. The 13-node unclear "
        "group is almost complete internally, yet its names do not establish one topic. "
        "Structural separation and interpretable subject matter answer different questions.",
        "",
        "Confidence refers to the proposed description: high means consistent inspected "
        "names with supporting spot-check evidence; medium means a visible theme with "
        "substantial mixing; low means ambiguity or inadequate verification. These grades "
        "are reviewer judgments, not probabilities or statistical estimates.",
        "",
        "## Evidence for each description",
    ]
    for profile in profiles:
        c = profile["community"]
        annotation = ANNOTATIONS[c]
        top = ", ".join(f"`{n}` ({profile['degrees'][n]})" for n in profile["top"])
        sample = ", ".join(f"`{n}`" for n in profile["sample"])
        lines += [
            "",
            f"### {c + 1}. {annotation['label']} (saved ID {c})",
            "",
            annotation["interpretation"],
            "",
            f"**Most internally connected:** {top}. Numbers count neighbours within this group.",
        ]
        if profile["sample"]:
            kind = "All remaining members" if profile["size"] <= 40 else "Random remainder sample"
            lines += ["", f"**{kind}:** {sample}."]
        if profile["size"] <= 40:
            lines += ["", f"All {profile['size']} member names were inspected."]
    lines += [
        "",
        "## Method, source checks and limits",
        "",
        "The partition is the saved unsigned, unweighted Louvain result (seed 0). "
        "Rank members by within-community degree descending, breaking ties by name. "
        "Inspect the top 15 and ten randomly selected remaining members for each group "
        "larger than 40. Sort the sampling population alphabetically and reset "
        "`random.Random(0)` for each group. Inspect all names for groups of at most 40. "
        f"This covers {inspected} distinct member names; it does not classify every member "
        "of the six large groups. The listed samples were reviewed before writing the annotations.",
        "",
        f"All external checks were made on {REVIEW_DATE}. Nine distinct subreddit targets "
        "were checked, below the cap of 15; a check can include search and a direct page "
        "attempt. Failed retrievals and unrelated results contribute no semantic evidence. "
        "S1 is contemporary and S3 postdates the dataset; neither reconstructs a 2014-2017 "
        "sidebar. S2 is a primary player post from the observation period. No paid API, "
        "model training, new community detection or new layout calculation was used.",
        "",
        "| Subreddit target | Source or attempted page | Outcome and time relevance |",
        "| --- | --- | --- |",
    ]
    lines += [f"| `{name}` | {source} | {outcome} |" for name, source, outcome in LOOKUPS]
    lines += [
        "",
        "Names and a small sample can conceal subtopics, satire and changes of purpose. "
        "The dataset covers historical hyperlinks, not membership overlap or original "
        "post text; its numeric text-property vectors do not supply that text "
        "([SNAP dataset description](https://snap.stanford.edu/data/soc-RedditHyperlinks.html), "
        f"checked {REVIEW_DATE}). The core covers a selected subset of Reddit. These labels "
        "do not establish shared ideology, agreement, observed opinions or the causes of "
        "links. They describe this one partition; stability across Louvain seeds was not tested.",
        "",
        "## Reproduction and provenance",
        "",
        "Run from a checkout with the three matching saved inputs available:",
        "",
        "```bash",
        "uv run python scripts/interpret_communities.py",
        "```",
        "",
        "The script also accepts `--graph`, `--communities`, `--layout`, `--report` "
        "and `--figure-dir`. The graph, labels and layout are ignored local artifacts, "
        "so a fresh checkout needs those exact files copied or independently reproduced. "
        "It verifies input hashes, unique node rows, labels, coordinates and graph counts "
        "before writing. It performs no network requests. Annotations and source notes "
        "are reviewed constants in the script; new artifacts require a fresh semantic review.",
        "",
        f"Code baseline: `{BASE_COMMIT}`. These hashes identify the actual local inputs "
        "independently of that Git revision:",
        "",
        "| Input | SHA-256 |",
        "| --- | --- |",
        f"| `data/processed/core.graphml` | `{EXPECTED_HASHES['graph']}` |",
        f"| `data/processed/communities.csv` | `{EXPECTED_HASHES['communities']}` |",
        f"| `results/topology/core_layout.csv` | `{EXPECTED_HASHES['layout']}` |",
        "",
        f"Checks: nine groups total 2,896 nodes; {internal:,} internal edges plus "
        f"{between:,} distinct between-group edges equal 58,466. Twice the internal "
        "edge total plus all boundary counts equals twice the full edge count. Every "
        "node's saved ID agrees across GraphML, community CSV and layout CSV. "
        "The regeneration preserves node positions and colour assignments.",
        "",
        "[S1]: https://www.reddit.com/r/gratefuldoe/about/",
        "[S2]: https://steamcommunity.com/app/359320/discussions/0/535150948597144836/",
        "[S3]: https://www.reddit.com/r/fantasyfootball/comments/qwv5cw/",
        "",
    ]
    return "\n".join(lines)


def plot_interpretations(graph, positions, profiles, figure_dir: Path) -> None:
    figure, (axis, legend) = plt.subplots(
        1, 2, figsize=(17, 12.8), gridspec_kw={"width_ratios": (2.2, 1), "wspace": 0.015}
    )
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.10, top=0.87)
    edges = nx.draw_networkx_edges(
        graph, positions, ax=axis, width=0.3, alpha=0.035, edge_color="slategray"
    )
    edges.set_rasterized(True)
    palette = plt.get_cmap("tab10")
    legend.set(xlim=(0, 1), ylim=(0, 1))
    for index, profile in enumerate(profiles):
        c = profile["community"]
        annotation = ANNOTATIONS[c]
        members = sorted(n for n in graph if graph.nodes[n]["community"] == c)
        colour = palette(index)
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=members,
            ax=axis,
            node_size=12,
            linewidths=0,
            node_color=[colour],
        )
        y = 0.99 - index * 0.110
        label = textwrap.fill(f"{c + 1}. {annotation['label']}", width=37)
        legend.scatter([0.025], [y - 0.012], s=95, color=[colour], linewidths=0)
        legend.text(0.075, y, label, va="top", fontsize=15, weight="bold", linespacing=1.2)
        offset = 0.026 * len(label.splitlines())
        legend.text(
            0.075,
            y - offset - 0.005,
            f"{profile['size']:,} nodes | {annotation['confidence']} confidence | "
            f"outside {profile['outside_share']:.1%}",
            va="top",
            fontsize=12,
            color="#4b5563",
        )
        legend.text(
            0.075,
            y - offset - 0.033,
            "e.g. " + ", ".join(f"r/{node}" for node in annotation["examples"]),
            va="top",
            fontsize=12,
            color="#4b5563",
        )
    axis.set_axis_off()
    legend.set_axis_off()
    figure.text(0.04, 0.955, "Interpreting the Louvain communities", fontsize=22, weight="bold")
    figure.text(
        0.04,
        0.918,
        "Connected 10-core | 2,896 subreddits | 58,466 unsigned edges | saved seed-0 layout",
        fontsize=12,
        color="#4b5563",
    )
    figure.text(
        0.04,
        0.055,
        "Descriptions are qualitative; confidence concerns the interpretation. "
        "Colours show structural groups, not observed opinions.",
        fontsize=12,
        color="#374151",
    )
    figure.text(
        0.04,
        0.030,
        "Outside share: fraction of connection ends leading to another group. "
        "Internal edges count twice. Review: " + REVIEW_DATE,
        fontsize=12,
        color="#6b7280",
    )
    for extension in ("png", "pdf"):
        figure.savefig(
            figure_dir / f"community_interpretation.{extension}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None,
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/processed/core.graphml")
    parser.add_argument("--communities", type=Path, default=ROOT / "data/processed/communities.csv")
    parser.add_argument("--layout", type=Path, default=ROOT / "results/topology/core_layout.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/community_interpretation.md")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    paths = {kind: getattr(args, kind) for kind in EXPECTED_HASHES}
    try:
        graph, positions = load_inputs(paths)
        profiles = profile_communities(graph)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_link = Path(
        os.path.relpath(args.figure_dir / "community_interpretation", args.report.parent)
    ).as_posix()
    args.report.write_text(render_report(profiles, figure_link), encoding="utf-8")
    plot_interpretations(graph, positions, profiles, args.figure_dir)
    print(f"Saved {args.report} and annotated PNG/PDF figures in {args.figure_dir}")


if __name__ == "__main__":
    main()
