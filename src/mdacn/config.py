"""Project settings. Paths are relative to the working directory."""

from pathlib import Path

PATHS = {
    "dataset": "data/raw/soc-RedditHyperlinks-body.tsv",
    "title_dataset": "data/raw/soc-RedditHyperlinks-title.tsv",
    "processed_dir": "data/processed",
    "results_dir": "results",
}

CHARACTERISATION = {
    "core_method": "k_core",
    "k": 10,
    "min_interactions": 100,
    "seed": 42,
}

# Shared defaults for NCO, sign shuffling, and node-impact simulations.
SIMULATION = {
    "fractions": [i / 50 for i in range(1, 51)],
    "realizations": 20,
    "seed": 0,
    "max_steps": 1000,
}


def processed_paths(processed: Path) -> dict:
    """Define shared producer and consumer paths for one processed core."""
    return {
        "aggregate": processed / "aggregate.graphml",
        "core": processed / "core.graphml",
        "unsigned": processed / "unsigned.npz",
        "signed": processed / "signed.npz",
        "nodes": processed / "nodes.json",
        "network_metadata": processed / "network_metadata.json",
        "communities": processed / "communities.csv",
        "centrality": processed / "centrality.csv",
        "centrality_metadata": processed / "centrality.json",
        "signed_leverage": processed / "signed_leverage.csv",
        "incident_signs": processed / "incident_signs.csv",
        "temporal_activity": processed / "temporal_activity.csv",
        "temporal_bridge": processed / "temporal_bridge_centrality.csv",
        "targeting_metadata": processed / "targeting_scores.json",
        "features": processed / "node_features.csv",
        "feature_metadata": processed / "node_features.json",
    }


def stage_paths() -> dict:
    """Resolve stage inputs and outputs from shared project paths."""
    processed, results = Path(PATHS["processed_dir"]), Path(PATHS["results_dir"])
    return {
        "dataset": Path(PATHS["dataset"]),
        "processed": processed,
        "results": results,
        "topology": results / "topology",
        **processed_paths(processed),
        "nco": results / "nco",
        "sign-shuffle": results / "sign_shuffle",
        "ico": results / "ico",
        "node-impact": results / "node_impact",
        "sensitivity": results / "sensitivity",
        "combined": results / "combined",
    }


ICO = {
    "budgets": [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
    "models": ["unsigned", "signed"],
    "strategies": [
        "random",
        "degree",
        "betweenness",
        "eigenvector",
        "leverage",
        "signed_leverage",
        "temporal_bridge",
    ],
    "phi_c_threshold": 0.5,
}


# Narrow extraction sensitivity: reuse the core extraction outputs and the
# simulation protocol, changing only the graph configuration.
SENSITIVITY = {
    "extraction_dir": "data/processed/extraction_comparison",
    "results_dir": "results/sensitivity",
    "configurations": ("k_core_8", "k_core_10", "k_core_12", "activity_lcc_100"),
    "fractions": SIMULATION["fractions"],
    "ico_fractions": [0.5],
    "models": ["unsigned", "signed"],
    "strategies": ["random", "degree", "leverage"],
    "budgets": ICO["budgets"],
    "realizations": SIMULATION["realizations"],
    "seed": SIMULATION["seed"],
    "max_steps": SIMULATION["max_steps"],
    "phi_c_threshold": ICO["phi_c_threshold"],
}


# Combined report: read saved outputs; no simulation settings.
COMBINED = {
    "results_dir": "results",
    "output_dir": "results/combined",
    "temporal_root": "reports/static_temporal",
    "fraction": 0.5,
}


# Node-impact overrides and settings specific to that workflow.
NODE_IMPACT = {
    "simulation": {
        **SIMULATION,
        "fractions": [0.5],
        "models": ["unsigned", "signed"],
    },
    "analysis": {"bootstrap": 1000, "seed": SIMULATION["seed"]},
}


def impact_settings() -> dict:
    """Connect node-impact to the saved core and feature-stage output."""
    paths = stage_paths()
    return {
        **NODE_IMPACT,
        "graph": str(paths["core"]),
        "results_dir": str(paths["node-impact"]),
        "features": str(paths["features"]),
        "feature_metadata": str(paths["feature_metadata"]),
    }
