"""Post-label audit and filtering-stage counts."""

import json

import networkx as nx
import pandas as pd

from mdacn.data import export_graph
from mdacn.label_audit import (
    filtering_stages,
    load_post_records,
    post_label_audit,
    rows_per_post,
    run_label_audit,
)


def _write_tsv(path, rows):
    pd.DataFrame(
        rows,
        columns=[
            "SOURCE_SUBREDDIT",
            "TARGET_SUBREDDIT",
            "POST_ID",
            "TIMESTAMP",
            "LINK_SENTIMENT",
            "PROPERTIES",
        ],
    ).to_csv(path, sep="\t", index=False)


ROWS = [
    ("a", "b", "p1", "2015-01-01 00:00:00", -1, "1.0,2.0"),
    ("a", "c", "p1", "2015-01-01 00:00:00", -1, "1.0,2.0"),
    ("a", "d", "p1", "2015-01-01 00:00:00", -1, "1.0,2.0"),
    ("b", "c", "p2", "2015-01-02 00:00:00", 1, "3.0,4.0"),
    ("b", "c", "p3", "2015-01-03 00:00:00", -1, "5.0,6.0"),
    ("c", "c", "p4", "2015-01-04 00:00:00", 1, "7.0,8.0"),
]


def test_post_label_audit_counts(tmp_path):
    path = tmp_path / "body.tsv"
    _write_tsv(path, ROWS)
    records = load_post_records(path, chunksize=2)
    audit = post_label_audit(records)
    assert audit["rows"] == 6 and audit["posts"] == 4
    assert audit["posts_with_multiple_rows"] == 1
    assert audit["rows_in_multi_row_posts"] == 3
    assert audit["max_rows_per_post"] == 3
    assert audit["duplicate_post_target_rows"] == 0
    assert audit["posts_with_mixed_sentiment"] == 0
    assert audit["multi_row_posts_with_differing_properties"] == 0
    assert audit["rows_repeating_a_directed_pair"] == 1

    records.loc[1, "sentiment"] = 1
    records.loc[1, "properties_hash"] = 0
    changed = post_label_audit(records)
    assert changed["posts_with_mixed_sentiment"] == 1
    assert changed["multi_row_posts_with_differing_properties"] == 1


def test_rows_per_post_and_filtering_stages(tmp_path):
    path = tmp_path / "body.tsv"
    _write_tsv(path, ROWS)
    records = load_post_records(path)
    sizes = rows_per_post(records)
    assert sizes.to_dict("list") == {"rows_per_post": [1, 3], "posts": [3, 1], "rows": [3, 3]}
    core = nx.Graph(k=1, core_method="k_core")
    core.add_edge("a", "b", sign=-1)
    core.add_edge("b", "c", sign=0)
    core.add_edge("a", "c", sign=-1)
    stages = filtering_stages(records, core).set_index("stage")
    assert stages.loc["body_file", ["rows", "nodes"]].tolist() == [6, 4]
    assert stages.loc[
        "self_links_removed", ["rows", "undirected_edges", "zero_sign_edges"]
    ].tolist() == [5, 4, 1]
    assert stages.loc["core", ["rows", "posts", "nodes", "undirected_edges"]].tolist() == [
        4,
        3,
        3,
        3,
    ]
    assert "k=1" in stages.loc["core", "note"]


def test_run_label_audit(tmp_path):
    path = tmp_path / "body.tsv"
    _write_tsv(path, ROWS)
    core = nx.Graph(k=1, core_method="k_core")
    core.add_edge("a", "b", sign=-1)
    core.add_edge("a", "c", sign=-1)
    core.add_edge("b", "c", sign=0)
    graph_path = tmp_path / "core.graphml"
    export_graph(core, graph_path)
    results_dir = tmp_path / "audit"
    report = run_label_audit(path, graph_path, results_dir=results_dir)
    assert report["audit"]["core"]["rows"] == 4
    for name in ("label_audit.json", "filtering_stages.csv", "rows_per_post.csv"):
        assert (results_dir / name).is_file()
    saved = json.loads((results_dir / "label_audit.json").read_text())
    assert saved["audit"]["body"]["posts"] == 4
