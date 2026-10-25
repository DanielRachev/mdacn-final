import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from mdacn.features import PREDICTORS
from mdacn.impact_plots import save_impact_plots


@pytest.fixture
def plot_tables():
    summary, correlations = [], []
    for model in ("unsigned", "signed"):
        for index in range(80):
            summary.append(
                {
                    "node": f"n{index:02}",
                    "model": model,
                    "f": 0.5,
                    "common_mean_impact": 0 if model == "unsigned" else (index - 30) / 200,
                    "mean_impact": 999,  # The plot must use means from common steady runs.
                    "common_runs": 20,
                }
            )
        for method in ("pearson", "spearman"):
            for index, predictor in enumerate(PREDICTORS):
                value = (index - 3) / 5
                correlations.append(
                    {
                        "model": model,
                        "f": 0.5,
                        "method": method,
                        "predictor": predictor,
                        "coefficient": value,
                        "ci_low": value - 0.1,
                        "ci_high": value + 0.1,
                        "common_runs": 20,
                    }
                )
    return pd.DataFrame(summary), pd.DataFrame(correlations)


def test_slide_plots_use_node_means_and_saved_intervals(plot_tables, tmp_path, monkeypatch):
    matplotlib.use("Agg")
    summary, correlations = plot_tables
    # Percentile intervals need not contain the point estimate.
    correlations.loc[0, ["ci_low", "ci_high"]] = [-0.3, -0.1]
    # Undefined correlations must remain distinct from zero; missing intervals
    # must not suppress a valid coefficient.
    correlations.loc[1, ["coefficient", "ci_low", "ci_high"]] = np.nan
    correlations.loc[2, ["ci_low", "ci_high"]] = np.nan
    figures = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    legacy = tmp_path / "impact_signed_f0.5.pdf"
    legacy.write_text("previous plot")
    unrelated = tmp_path / "another_plot.png"
    unrelated.write_text("keep")
    save_impact_plots(summary, correlations, tmp_path)

    assert not legacy.exists() and unrelated.read_text() == "keep"
    assert {path.name for path in tmp_path.glob("impact_*")} == {
        "impact_distribution_f0.5.png",
        "impact_correlations_f0.5.png",
    }
    assert not list(tmp_path.glob("*.pdf"))
    distribution, intervals = figures
    points = [c for c in distribution.axes[0].collections if isinstance(c, PathCollection)]
    assert [len(c.get_offsets()) for c in points] == [80, 80]
    for model, collection in zip(("unsigned", "signed"), points, strict=True):
        np.testing.assert_allclose(
            collection.get_offsets()[:, 1],
            summary.loc[summary.model == model, "common_mean_impact"],
        )
    assert [ax.get_title() for ax in intervals.axes] == ["Pearson", "Spearman"]
    assert len(intervals.axes[0].get_yticklabels()) == len(PREDICTORS)
    assert any(text.get_text() == "n/a" for text in intervals.axes[0].texts)
    np.testing.assert_allclose(
        intervals.axes[0].collections[0].get_segments()[0][:, 0], [-0.3, -0.1]
    )
    for figure in figures:
        assert tuple(figure.get_size_inches()) == (12.8, 7.2)


def test_no_steady_outcomes_produces_empty_plots_not_zero_estimates(
    plot_tables, tmp_path, monkeypatch
):
    matplotlib.use("Agg")
    summary, correlations = plot_tables
    summary["common_mean_impact"] = np.nan
    summary["common_runs"] = 0
    correlations[["coefficient", "ci_low", "ci_high"]] = np.nan
    correlations["common_runs"] = 0
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *a, **k: figures.append(figure))
    save_impact_plots(summary, correlations, tmp_path)
    assert sum(t.get_text() == "No common steady outcomes" for t in figures[0].axes[0].texts) == 2
    for figure in figures:
        for ax in figure.axes:
            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    assert len(collection.get_offsets()) == 0
