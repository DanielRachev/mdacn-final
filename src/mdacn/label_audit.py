"""Audit of the supplied post labels and the counts at each filtering stage.

LINK_SENTIMENT labels a source post's stance toward the target subreddit. One post
can link to several targets, giving several rows that share the post's label and
text features. Those rows are not independent text classifications.
"""

from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from mdacn.artifacts import checksum, write_json
from mdacn.data import edge_sign_counts, load_graph, load_interactions
from mdacn.temporal_analysis import filter_core_events


def load_post_records(path: Path, *, chunksize: int = 50_000) -> pd.DataFrame:
    """Return load_interactions rows with post_id and a hash of the PROPERTIES features.

    PROPERTIES is read separately in chunks and hashed to 64 bits to keep memory
    small; equal hashes identify identical feature vectors up to a negligible
    collision probability. Rows keep file order, so the two reads align.
    """
    records = load_interactions(path, post_ids=True)
    hashes = [
        pd.util.hash_pandas_object(chunk["PROPERTIES"], index=False).to_numpy()
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["PROPERTIES"],
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
        )
    ]
    records["properties_hash"] = np.concatenate(hashes) if hashes else np.array([], np.uint64)
    return records


def post_label_audit(records: pd.DataFrame) -> dict[str, int | float]:
    """Return counts that show whether rows of one post repeat its label and features."""
    per_post = records.groupby("post_id").agg(
        rows=("target", "size"),
        sources=("source", "nunique"),
        sentiments=("sentiment", "nunique"),
        timestamps=("timestamp", "nunique"),
        properties=("properties_hash", "nunique"),
    )
    multi = per_post["rows"] > 1
    unique_pairs = records[["source", "target"]].drop_duplicates()
    return {
        "rows": len(records),
        "posts": len(per_post),
        "posts_with_multiple_rows": int(multi.sum()),
        "rows_in_multi_row_posts": int(per_post.loc[multi, "rows"].sum()),
        "max_rows_per_post": int(per_post["rows"].max()) if len(per_post) else 0,
        "duplicate_post_target_rows": int(records.duplicated(["post_id", "target"]).sum()),
        "fully_duplicate_rows": int(
            records.duplicated(["post_id", "source", "target", "timestamp", "sentiment"]).sum()
        ),
        "posts_with_mixed_sentiment": int((per_post["sentiments"] > 1).sum()),
        "posts_with_multiple_sources": int((per_post["sources"] > 1).sum()),
        "posts_with_multiple_timestamps": int((per_post["timestamps"] > 1).sum()),
        "multi_row_posts_with_differing_properties": int(
            (per_post.loc[multi, "properties"] > 1).sum()
        ),
        "directed_pairs": len(unique_pairs),
        "rows_repeating_a_directed_pair": len(records) - len(unique_pairs),
    }


def rows_per_post(records: pd.DataFrame) -> pd.DataFrame:
    """Return how many posts have each number of rows."""
    sizes = records.groupby("post_id").size()
    table = sizes.value_counts().sort_index().rename_axis("rows_per_post").reset_index(name="posts")
    table["rows"] = table["rows_per_post"] * table["posts"]
    return table


def filtering_stages(records: pd.DataFrame, core: nx.Graph) -> pd.DataFrame:
    """Return row, post, node, and edge counts from the body file to the analysed core."""
    self_links = records["source"] == records["target"]
    non_self = records.loc[~self_links]
    signs = edge_sign_counts(records)["sign"]
    core_records = filter_core_events(records, tuple(core))
    core_signs = [sign for _, _, sign in core.edges(data="sign")]
    stages = [
        {
            "stage": "body_file",
            "rows": len(records),
            "posts": records["post_id"].nunique(),
            "nodes": len(set(records["source"]) | set(records["target"])),
            "undirected_edges": np.nan,
            "zero_sign_edges": np.nan,
            "note": "All rows of the SNAP body TSV",
        },
        {
            "stage": "self_links_removed",
            "rows": len(non_self),
            "posts": non_self["post_id"].nunique(),
            "nodes": len(set(non_self["source"]) | set(non_self["target"])),
            "undirected_edges": len(signs),
            "zero_sign_edges": int((signs == 0).sum()),
            "note": f"{int(self_links.sum())} self-link rows removed; directions merged "
            "into majority-sign undirected edges",
        },
        {
            "stage": "core",
            "rows": len(core_records),
            "posts": core_records["post_id"].nunique(),
            "nodes": core.number_of_nodes(),
            "undirected_edges": core.number_of_edges(),
            "zero_sign_edges": core_signs.count(0),
            "note": "Core from core.graphml: "
            + ", ".join(
                f"{key}={core.graph[key]}"
                for key in ("core_method", "k", "min_interactions")
                if key in core.graph
            )
            + "; largest connected component",
        },
    ]
    return pd.DataFrame(stages)


def run_label_audit(
    input_path: Path,
    graph_path: Path,
    *,
    results_dir: Path = Path("results/label_audit"),
) -> dict:
    """Save the post-label audit for the body file and the core, and filtering counts."""
    records = load_post_records(input_path)
    core = load_graph(graph_path)
    core_records = filter_core_events(records, tuple(core))
    audit = {"body": post_label_audit(records), "core": post_label_audit(core_records)}
    stages = filtering_stages(records, core)
    sizes = pd.concat(
        [
            rows_per_post(records).assign(scope="body"),
            rows_per_post(core_records).assign(scope="core"),
        ],
        ignore_index=True,
    )[["scope", "rows_per_post", "posts", "rows"]]

    results_dir.mkdir(parents=True, exist_ok=True)
    stages.to_csv(results_dir / "filtering_stages.csv", index=False)
    sizes.to_csv(results_dir / "rows_per_post.csv", index=False)
    report = {
        "input_path": str(input_path.resolve()),
        "input_sha256": checksum(input_path),
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": checksum(graph_path),
        "audit": audit,
        "interpretation": {
            "label": "LINK_SENTIMENT is the source post's label toward the target",
            "text_features": "PROPERTIES describe the source post text; compared by hash",
            "simulated_opinions": "NCO opinions are synthetic initial states, not labels",
            "core_scope": "Core rows have both endpoints in the core and differ in endpoints",
        },
        "versions": {name: version(name) for name in ("mdacn-final", "pandas")},
    }
    write_json(results_dir / "label_audit.json", report)
    print(pd.DataFrame(audit).to_string(), flush=True)
    print(stages.drop(columns="note").to_string(index=False), flush=True)
    print(f"Saved label audit to {results_dir}", flush=True)
    return report
