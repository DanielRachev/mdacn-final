"""Fisher-z sample sizing for a two-sided Pearson test; assumes independent observations."""

import numpy as np
from scipy.stats import norm


def _validate(effect_size: float, alpha: float, comparisons: int) -> None:
    if not np.isfinite(effect_size) or not 0 <= abs(effect_size) < 1:
        raise ValueError("effect_size must be finite with absolute value below 1")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    if isinstance(comparisons, bool) or not isinstance(comparisons, int) or comparisons < 1:
        raise ValueError("comparisons must be a positive integer")
    if alpha / comparisons == 0:
        raise ValueError("alpha/comparisons is too small for numerical calculation")


def correlation_power(
    n_nodes: int, effect_size: float, *, alpha: float = 0.05, comparisons: int = 1
) -> float:
    """Approximate Pearson power with Bonferroni adjustment and both rejection tails."""
    _validate(effect_size, alpha, comparisons)
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes < 4:
        raise ValueError("n_nodes must be an integer >= 4")
    critical = norm.isf((alpha / comparisons) / 2)
    shift = np.arctanh(abs(effect_size)) * np.sqrt(n_nodes - 3)
    return float(norm.sf(critical - shift) + norm.cdf(-critical - shift))


def required_nodes(
    effect_size: float, *, target_power: float = 0.8, alpha: float = 0.05, comparisons: int = 1
) -> int:
    """Find the smallest integer n >= 4 reaching the approximate target power."""
    _validate(effect_size, alpha, comparisons)
    if effect_size == 0:
        raise ValueError("A nonzero effect_size is required for sample-size planning")
    if not np.isfinite(target_power) or not 0.5 < target_power < 1:
        raise ValueError("target_power must be finite and in (0.5, 1)")
    critical = norm.isf((alpha / comparisons) / 2)
    estimate = 3 + ((critical + norm.ppf(target_power)) / np.arctanh(abs(effect_size))) ** 2
    if not np.isfinite(estimate) or estimate > 1e12:
        raise ValueError("Required reference sample exceeds the numerical planning limit")
    upper = max(4, int(np.ceil(estimate)))
    lower = 3
    while upper - lower > 1:
        middle = (upper + lower) // 2
        if (
            correlation_power(middle, effect_size, alpha=alpha, comparisons=comparisons)
            >= target_power
        ):
            upper = middle
        else:
            lower = middle
    return upper
