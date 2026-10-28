"""Count negative and zero-sign core edges within and between Louvain communities."""

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge signs within vs between communities")
    parser.add_argument("--graph", type=Path, default=Path("data/processed/core.graphml"))
    parser.add_argument(
        "--communities", type=Path, default=Path("data/processed/communities.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/community_sign_fractions.csv")
    )
    args = parser.parse_args()

    graph = nx.read_graphml(args.graph)
    labels = pd.read_csv(args.communities, dtype={"node": str}).set_index("node")["community"]
    missing = set(graph) - set(labels.index)
    if missing:
        raise ValueError(f"Nodes without a community label: {sorted(missing)[:5]}")

    edges = pd.DataFrame(
        [(u, v, int(data["sign"])) for u, v, data in graph.edges(data=True)],
        columns=["u", "v", "sign"],
    )
    same = labels[edges["u"]].to_numpy() == labels[edges["v"]].to_numpy()
    edges["scope"] = pd.Series(same).map({True: "within", False: "between"})

    rows = []
    for scope, group in [*edges.groupby("scope"), ("all", edges)]:
        rows.append(
            {
                "scope": scope,
                "edges": len(group),
                "negative": int((group["sign"] == -1).sum()),
                "zero": int((group["sign"] == 0).sum()),
                "positive": int((group["sign"] == 1).sum()),
                "negative_fraction": (group["sign"] == -1).mean(),
                "zero_fraction": (group["sign"] == 0).mean(),
            }
        )
    table = pd.DataFrame(rows).set_index("scope").loc[["within", "between", "all"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
