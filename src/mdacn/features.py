"""Validate and align precomputed node scores without calculating replacements."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

PREDICTORS = (
    "degree",
    "leverage",
    "signed_leverage",
    "betweenness",
    "negative_incident_ratio",
    "signed_bridge_ratio",
    "temporal_activity",
)


def validate_features(supplied: pd.DataFrame | None, nodes: Iterable[str]) -> pd.DataFrame:
    """Require complete core scores, retaining supplied values unchanged.

    Degree must be a raw count. Every score must cover every core node.
    Extra nodes in contributor tables are excluded from the core export.
    """
    nodes = sorted(nodes)
    if not nodes:
        raise ValueError("Node features require a nonempty core")
    missing = [column for column in PREDICTORS if supplied is None or column not in supplied]
    if missing:
        raise ValueError(
            f"Missing required node scores: {', '.join(missing)}. "
            "Supply precomputed exports at the paths defined in config.py; "
            "no scores were calculated."
        )
    if "node" not in supplied or supplied.node.isna().any() or supplied.node.duplicated().any():
        raise ValueError("Feature tables require unique, nonempty node values")
    features = supplied.set_index("node").reindex(nodes)[list(PREDICTORS)].copy()
    for column in PREDICTORS:
        features[column] = pd.to_numeric(features[column], errors="raise")
        values = features[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid or incomplete core scores: {column}")
    return features.reset_index()
