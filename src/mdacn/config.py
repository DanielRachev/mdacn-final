"""Project settings. Paths are relative to the working directory."""

from pathlib import Path

PATHS = {
    "dataset": "data/raw/soc-RedditHyperlinks-body.tsv",
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
    "fractions": [i / 10 for i in range(1, 10)],
    "realizations": 20,
    "seed": 0,
    "max_steps": 1000,
}


def stage_paths() -> dict:
    """Resolve stage inputs and outputs from shared project paths."""
    processed, results = Path(PATHS["processed_dir"]), Path(PATHS["results_dir"])
    return {
        "dataset": Path(PATHS["dataset"]),
        "processed": processed,
        "results": results,
        "core": processed / "core.graphml",
        "centrality": processed / "centrality.csv",
        "signed_leverage": processed / "signed_leverage.csv",
        "incident_signs": processed / "incident_signs.csv",
        "temporal_activity": processed / "temporal_activity.csv",
        "features": processed / "node_features.csv",
        "feature_metadata": processed / "node_features.json",
        "nco": results / "nco",
        "sign-shuffle": results / "sign_shuffle",
        "node-impact": results / "node_impact",
    }


# Node-impact overrides and settings specific to that workflow.
NODE_IMPACT = {
    "power": {
        # Prespecified target; this gives a reference budget, not network-valid power.
        "effect_size": 0.3,
        "target_power": 0.8,
        "alpha": 0.05,
        # Seven primary Pearson scores x two models at one f.
        # Spearman and multivariable regression remain descriptive.
        "comparisons": 14,
    },
    "sampling": {"bins": 3, "seed": SIMULATION["seed"]},
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
