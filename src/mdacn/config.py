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
        "rewire": results / "rewiring",
        "rewire-ico": results / "rewiring_ico",
        "incident-signs": results / "incident_signs",
        "signed-triads": results / "signed_triads",
        "temporal-balance": results / "temporal_balance",
        "directed-roles": results / "directed_roles",
        "label-audit": results / "label_audit",
    }


ICO = {"budgets": [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]}

# Null Model B: degree-preserving rewiring. Betas were calibrated on the 10-core
# (docs/null_models.md): insular below, near, and above the empirical score.
REWIRING = {
    "replicates": 5,
    "attempts_per_edge": 100,
    "window_per_edge": 2.0,
    "tolerance": 0.002,
    "checkpoints": 20,
    "seed": 0,
    "betas": {"insular": [1.0, 2.0, 4.0], "integrated": [0.5, 1.0, 2.0]},
}

# Signed ICO on the empirical and rewired graphs; its finer budget grid resolves
# Phi_c differences between 0.03 and 0.10.
REWIRED_ICO = {
    "budgets": [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25],
    "negative_fraction": 0.5,
    "initializations": 20,
    "strategies": ["degree", "random"],
    "seed": 0,
    "max_steps": SIMULATION["max_steps"],
}


# Signed-structure analyses. Sign shuffles share their seeds across incident signs
# and triads, so both compare against the same shuffled cores.
INCIDENT_SIGNS = {
    "realizations": 1000,
    "seed": 0,
    "bins": "width",
    "bin_width": 0.025,
    "n_bins": 20,
}
SIGNED_TRIADS = {"realizations": 1000, "er_realizations": 20, "seed": 0}
TEMPORAL_BALANCE = {"realizations": 200, "seed": 0, "frequency": "Q"}


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
