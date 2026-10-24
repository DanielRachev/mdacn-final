import pytest

from mdacn.power import correlation_power, required_nodes


@pytest.mark.parametrize("effect,expected", [(0.2, 194), (0.3, 85), (0.5, 30)])
def test_reference_sample_sizes_reach_target_and_are_minimal(effect, expected):
    n = required_nodes(effect)
    assert n == expected
    assert correlation_power(n, effect) >= 0.8
    assert correlation_power(n - 1, effect) < 0.8


def test_two_sided_power_includes_both_tails_and_bonferroni():
    assert correlation_power(30, 0) == pytest.approx(0.05)
    assert correlation_power(30, 0, comparisons=5) == pytest.approx(0.01)
    assert correlation_power(30, -0.3) == correlation_power(30, 0.3)
    assert required_nodes(0.3, comparisons=5) > required_nodes(0.3)
    assert required_nodes(0.3, target_power=0.9) > required_nodes(0.3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"effect_size": 0},
        {"effect_size": 1},
        {"effect_size": float("nan")},
        {"effect_size": 0.3, "alpha": 0},
        {"effect_size": 0.3, "comparisons": 0},
        {"effect_size": 0.3, "comparisons": True},
        {"effect_size": 0.3, "target_power": 1},
    ],
)
def test_invalid_planning_inputs_fail(kwargs):
    with pytest.raises(ValueError):
        required_nodes(**kwargs)
