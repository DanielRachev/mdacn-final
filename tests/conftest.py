import pandas as pd
import pytest

from mdacn.artifacts import checksum, write_json
from mdacn.centrality import leverage_centrality
from mdacn.data import build_matrices, save_matrices


@pytest.fixture
def centrality_scores():
    def scores(graph):
        return (
            pd.DataFrame(
                {
                    "degree": dict(graph.degree()),
                    "leverage": leverage_centrality(graph),
                }
            )
            .rename_axis("node")
            .reset_index()
        )

    return scores


@pytest.fixture
def saved_features(tmp_path):
    """Export feature-stage inputs for small synthetic graph tests."""

    def save(graph, graph_path):
        save_matrices(build_matrices(graph), graph_path)
        table = pd.DataFrame(
            {
                "node": ["a", "b", "c", "d"],
                "degree": [1, 2, 2, 1],
                "leverage": [-1 / 3, 1 / 6, 1 / 6, -1 / 3],
                "signed_leverage": [0, 1 / 6, 1 / 6, 0],
                "betweenness": [0, 2 / 3, 2 / 3, 0],
                "negative_incident_ratio": [0, 0, 0, 0],
                "signed_bridge_ratio": [0, 0, 0, 0],
                "temporal_activity": [4, 2, 3, 1],
            }
        )
        path = tmp_path / "features.csv"
        metadata = tmp_path / "features.json"
        table.to_csv(path, index=False)
        write_json(
            metadata,
            {
                "graph_sha256": checksum(graph_path),
                "features_sha256": checksum(path),
                "columns": table.columns.tolist(),
            },
        )
        return path, metadata

    return save
