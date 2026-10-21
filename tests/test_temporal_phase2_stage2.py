import pandas as pd

from mdacn.temporal_diagnostics import (
    negative_reciprocity,
    reciprocity_timestamp_nulls,
    signed_reciprocity,
    signed_reciprocity_timestamp_nulls,
)
from mdacn.temporal_interventions import summarize_temporal_ico


def test_seven_day_reciprocity_is_directional_and_strictly_later():
    events = pd.DataFrame(
        {
            "source": ["A", "B", "B", "A", "C"],
            "target": ["B", "A", "A", "B", "A"],
            "sentiment": [-1, -1, -1, -1, 1],
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-03", "2020-01-20", "2020-01-20", "2020-01-21"]
            ),
        }
    )
    out = negative_reciprocity(events)
    assert out["negative_events"] == 4
    assert out["reciprocated_events"] == 1
    assert negative_reciprocity(events, days=1)["reciprocated_events"] == 0
    assert len(reciprocity_timestamp_nulls(events, n_shuffles=2)) == 5


def test_ico_paired_summary_clusters_shuffles_by_initialization():
    records = []
    for initialization in range(3):
        for strategy in ("degree", "random"):
            for budget in (0, 0.05):
                for variant, shuffle_count in (("real", 1), ("global_shuffle", 4)):
                    for k in range(shuffle_count):
                        val = 0.7 if budget == 0 else (0.1 if strategy == "degree" else 0.6)
                        records.append(
                            dict(
                                f=0.7,
                                budget=budget,
                                strategy=strategy,
                                variant=variant,
                                initialization=initialization,
                                shuffle=(-1 if variant == "real" else k),
                                g_minus=val,
                            )
                        )
    outcomes, paired, crossings = summarize_temporal_ico(pd.DataFrame(records), n_bootstrap=100)
    assert (
        outcomes.loc[
            outcomes.strategy.eq("degree") & outcomes.budget.eq(0.05), "success_probability"
        ]
        .eq(1)
        .all()
    )
    assert paired.paired_initializations.eq(3).all()
    assert crossings.loc[crossings.strategy.eq("degree"), "phi_c"].eq(0.05).all()
    assert crossings.loc[crossings.strategy.eq("random"), "threshold_reached"].eq(False).all()


def test_signed_reciprocity_uses_matching_sign_and_strict_reverse():
    events = pd.DataFrame(
        {
            "source": ["A", "B", "A", "B", "A", "B"],
            "target": ["B", "A", "B", "A", "B", "A"],
            "sentiment": [1, 1, -1, 1, -1, -1],
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-10", "2020-01-20"]
            ),
        }
    )
    positive = signed_reciprocity(events, sign=1)
    negative = signed_reciprocity(events, sign=-1)
    assert positive["eligible_events"] == 3
    assert positive["reciprocated_events"] == 1
    assert negative["eligible_events"] == 3
    assert negative["reciprocated_events"] == 0  # the only reverse negative shares a timestamp
    assert negative_reciprocity(events)["negative_events"] == 3
    both = signed_reciprocity_timestamp_nulls(events, n_shuffles=2)
    assert len(both) == 10
    assert set(both.sign) == {"negative", "non_negative"}
    assert both.groupby(["variant", "shuffle"]).size().eq(2).all()
