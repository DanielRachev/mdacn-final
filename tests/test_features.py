import numpy as np
import pandas as pd
import pytest

from mdacn.features import PREDICTORS, validate_features


@pytest.fixture
def scores():
    return pd.DataFrame(
        {
            "node": ["c", "b", "a"],
            "degree": [1, 2, 1],
            "leverage": [-1 / 3, 1 / 3, -1 / 3],
            "signed_leverage": [0, 1 / 3, 0],
            "betweenness": [0, 1, 0],
            "negative_incident_ratio": [0, 0, 0],
            "signed_bridge_ratio": [0.2, 0.3, 0.4],
            "temporal_activity": [2, 3, 5],
        }
    )


def test_supplied_scores_align_by_node_and_remain_unchanged(scores):
    original = scores.copy(deep=True)
    actual = validate_features(scores, ["a", "b", "c"])
    pd.testing.assert_frame_equal(actual, original.sort_values("node").reset_index(drop=True))
    pd.testing.assert_frame_equal(scores, original)


@pytest.mark.parametrize("column", PREDICTORS)
def test_missing_required_column_reports_error(scores, column):
    with pytest.raises(ValueError, match=f"Missing required node scores: {column}"):
        validate_features(scores.drop(columns=column), ["a", "b", "c"])


def test_no_supplied_scores_reports_required_exports():
    with pytest.raises(ValueError, match="Supply precomputed exports"):
        validate_features(None, ["a", "b", "c"])


def test_missing_core_node_reports_error(scores):
    with pytest.raises(ValueError, match="incomplete core scores: degree"):
        validate_features(scores.iloc[:2], ["a", "b", "c"])


@pytest.mark.parametrize("value", [np.nan, np.inf, "invalid"])
@pytest.mark.parametrize("column", ["leverage", "signed_bridge_ratio", "temporal_activity"])
def test_invalid_required_score_reports_error(scores, column, value):
    scores[column] = scores[column].astype(object)
    scores.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_features(scores, ["a", "b", "c"])


def test_duplicate_node_reports_error(scores):
    scores.loc[0, "node"] = "a"
    with pytest.raises(ValueError, match="unique, nonempty node"):
        validate_features(scores, ["a", "b", "c"])
