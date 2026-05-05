"""Risk metrics: VaR (historic + parametric) and CVaR / Expected Shortfall."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def value_at_risk(returns: NDArray, alpha: float = 0.05, method: str = "historic") -> float:
    """Value at Risk at confidence level ``1 - alpha``.

    Returned as a *negative* number — interpret as "expected worst-case loss
    over the bar period at the given confidence level".

    method
        "historic"   — empirical alpha-quantile of the return distribution
        "parametric" — assumes normality, uses mean/std and the inverse CDF
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if method == "historic":
        return float(np.quantile(r, alpha))
    if method == "parametric":
        return float(stats.norm.ppf(alpha, loc=r.mean(), scale=r.std(ddof=1)))
    raise ValueError(f"Unknown method {method!r}")


def cvar(returns: NDArray, alpha: float = 0.05) -> float:
    """Conditional VaR — average of returns below the VaR threshold."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    var = value_at_risk(r, alpha=alpha, method="historic")
    tail = r[r <= var]
    if len(tail) == 0:
        return float(var)
    return float(tail.mean())
