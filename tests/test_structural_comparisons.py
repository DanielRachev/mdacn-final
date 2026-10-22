"""Small, countable examples for the structural comparison commands."""

import json
from io import BytesIO
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from scipy.sparse import load_npz

from mdacn.data import DATASET_URLS, build_matrices, download_dataset, load_interactions
from mdacn.extraction_comparison import compare_extractions
from mdacn.link_comparison import compare_link_sources


def write_events(path: Path, events: list[tuple[str, str, str, int]]) -> Path:
    header = "SOURCE_SUBREDDIT\tTARGET_SUBREDDIT\tPOST_ID\tTIMESTAMP\tLINK_SENTIMENT\n"
    path.write_text(
        header
        + "".join(
            f"{source}\t{target}\t{post}\t2020-01-01 00:00:00\t{sign}\n"
            for source, target, post, sign in events
        )
    )
    return path


@pytest.fixture
def body_links(tmp_path: Path) -> Path:
    return write_events(
        tmp_path / "body.tsv",
        [
            ("a", "b", "p1", 1),
            ("a", "b", "p1", 1),
            ("b", "a", "p2", -1),
            ("b", "a", "p2", -1),
            ("a", "c", "p3", 1),
            ("b", "c", "p4", 1),
            ("d", "e", "p5", -1),
            ("d", "e", "p5", -1),
            ("d", "e", "p5", -1),
            ("z", "z", "p6", 1),
        ],
    )


def test_extractions_export_membership_signs_and_retention(
    body_links: Path, tmp_path: Path
) -> None:
    processed, results = tmp_path / "processed", tmp_path / "results"
    summary = compare_extractions(
        body_links, ks=(2, 3), min_interactions=3, processed_dir=processed, results_dir=results
    )
    rows = {row["configuration"]: row for row in summary["configurations"]}
    triangle = rows["k_core_2"]
    assert triangle["nodes"] == 3
    assert triangle["edges"] == 3
    assert triangle["node_retention"] == pytest.approx(3 / 6)
    assert triangle["edge_retention"] == pytest.approx(3 / 4)
    assert triangle["event_retention"] == pytest.approx(6 / 9)
    assert triangle["zero_edge_fraction"] == pytest.approx(1 / 3)
    assert rows["activity_lcc_3"]["non_self_interactions"] == 4
    assert rows["k_core_3"]["status"] == "empty"
    assert rows["k_core_3"]["spectral_radius"] is None
    assert json.loads((results / "comparison.json").read_text()) == summary

    for label in rows:
        graph = nx.read_graphml(processed / label / "core.graphml")
        network = build_matrices(graph)
        assert json.loads((processed / label / "nodes.json").read_text()) == list(network.nodes)
        assert (load_npz(processed / label / "signed.npz") != network.signed).nnz == 0
        assert (load_npz(processed / label / "unsigned.npz") != network.unsigned).nnz == 0
    activity_nodes = json.loads((processed / "activity_lcc_3" / "nodes.json").read_text())
    assert activity_nodes == ["a", "b"]  # Tied LCCs choose the alphabetically first.
    for name in ("topology_table", "degree_comparison", "retention_and_signs"):
        for extension in ("png", "pdf"):
            assert (results / f"{name}.{extension}").stat().st_size > 0


def test_link_overlap_keeps_event_multiplicity(body_links: Path, tmp_path: Path) -> None:
    title = write_events(
        tmp_path / "title.tsv",
        [
            ("a", "b", "p1", 1),
            ("a", "b", "p1", 1),
            ("a", "b", "p1", 1),
            ("a", "c", "other", -1),
            ("c", "a", "p7", 1),
            ("f", "a", "p8", -1),
            ("b", "c", "p4", 1),
            ("z", "z", "p6", 1),
            ("d", "e", "p5", -1),
        ],
    )
    results = tmp_path / "links"
    summary = compare_link_sources(body_links, title, k=2, results_dir=results)
    assert summary["sources"]["body"]["interactions"] == 10
    assert summary["sources"]["title"]["interactions"] == 9
    assert summary["sources"]["body"]["repeated_post_link_rows"] == 4
    assert summary["overlap"]["nodes"]["shared"] == 6
    assert summary["overlap"]["directed_pairs"]["shared"] == 4
    assert summary["overlap"]["undirected_edges"]["body_covered_by_title"] == 1
    assert summary["overlap"]["post_link_keys"]["shared"] == 4
    assert summary["post_link_occurrences"]["matched_occurrences"] == 5
    assert summary["shared_edge_signs"]["agreement_fraction"] == 0.5
    assert summary["title_coverage_of_body_core"]["edge_coverage"] == 1
    assert summary["overlap"]["core_nodes"]["shared"] == 3
    assert json.loads((results / "comparison.json").read_text()) == summary
    assert len(pd.read_csv(results / "core_membership.csv")) == 3


def test_self_links_only_report_empty_cores_and_undefined_retention(tmp_path: Path) -> None:
    source = write_events(tmp_path / "self.tsv", [("alone", "alone", "post", 1)])
    summary = compare_extractions(
        source,
        ks=(1,),
        min_interactions=1,
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
    )
    assert summary["input"]["self_interactions_removed"] == 1
    for row in summary["configurations"]:
        assert row["status"] == "empty"
        assert row["node_retention"] == 0
        assert row["edge_retention"] is None
        assert row["event_retention"] is None


def test_disjoint_title_links_keep_missing_body_nodes_as_isolates(
    body_links: Path,
    tmp_path: Path,
) -> None:
    title = write_events(tmp_path / "title.tsv", [("x", "y", "p", 1)])
    summary = compare_link_sources(body_links, title, k=2, results_dir=tmp_path / "results")
    assert summary["shared_edge_signs"]["agreement_fraction"] is None
    coverage = summary["title_coverage_of_body_core"]
    assert coverage["node_coverage"] == 0
    assert coverage["title_graph_on_fixed_body_nodes"]["nodes"] == 3
    assert coverage["title_graph_on_fixed_body_nodes"]["connected_components"] == 3
    assert coverage["active_node_fraction"] == 0
    assert summary["cores"]["title"]["nodes"] == 0
    assert summary["overlap"]["core_nodes"]["title_covered_by_body"] is None


def test_title_download_uses_title_url_and_reuses_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    urls = []

    def open_url(url: str, *, timeout: int) -> BytesIO:
        urls.append(url)
        return BytesIO(b"example dataset")

    monkeypatch.setattr("mdacn.data.urlopen", open_url)
    path = tmp_path / "title.tsv"
    assert download_dataset(path, dataset="title") == path
    assert download_dataset(path, dataset="title") == path
    assert urls == [DATASET_URLS["title"]]
    assert path.read_bytes() == b"example dataset"
    assert not path.with_name("title.tsv.part").exists()


@pytest.mark.parametrize("dataset", ["body", "title"])
def test_setup_selects_dataset_default_path(
    dataset: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mdacn import __main__ as cli

    calls = []
    monkeypatch.setattr(cli, "download_dataset", lambda path, **kw: calls.append((path, kw)))
    monkeypatch.setattr("sys.argv", ["mdacn", "setup", "--dataset", dataset])
    cli.main()
    expected = cli.DEFAULT_DATASET_PATH if dataset == "body" else cli.DEFAULT_TITLE_PATH
    assert calls == [(expected, {"dataset": dataset})]


def test_optional_post_ids_preserve_default_contract(body_links: Path) -> None:
    assert list(load_interactions(body_links).columns) == [
        "source",
        "target",
        "sentiment",
        "timestamp",
    ]
    events = load_interactions(body_links, include_post_id=True)
    assert len(events) == 10
    assert events["post_id"].tolist().count("p1") == 2
