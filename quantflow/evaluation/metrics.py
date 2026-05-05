"""Financial, ML, and RL metrics — pure NumPy.

All metrics are scale-invariant under common transforms where applicable
(e.g. ``sharpe_ratio(c * r) == sharpe_ratio(r)`` for a positive constant ``c``).
NaNs are dropped before computation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _clean(returns: NDArray) -> NDArray:
    r = np.asarray(returns, dtype=np.float64)
    return r[~np.isnan(r)]


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: NDArray,
    risk_free: float = 0.0,
    annualisation: float = 252.0,
) -> float:
    """Annualised Sharpe ratio.

    Returns 0.0 when the sample is empty or has zero variance — these are
    well-defined edge cases for callers (no trades, flat strategy) and
    propagating NaN forces every consumer to special-case them.
    """
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / annualisation
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(annualisation))


def sortino_ratio(
    returns: NDArray,
    risk_free: float = 0.0,
    annualisation: float = 252.0,
) -> float:
    """Sortino ratio — Sharpe but only penalises downside volatility."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / annualisation
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    dsd = np.sqrt((downside**2).mean())
    if dsd == 0:
        return 0.0
    return float(excess.mean() / dsd * np.sqrt(annualisation))


def max_drawdown(equity: NDArray) -> float:
    """Maximum drawdown as a (negative) fraction of peak equity.

    Returns a value in [-1, 0]; -0.25 means a 25% drawdown.
    """
    e = np.asarray(equity, dtype=np.float64)
    if len(e) == 0:
        return 0.0
    peaks = np.maximum.accumulate(e)
    dd = (e - peaks) / np.where(peaks == 0, 1.0, peaks)
    return float(dd.min())


def calmar_ratio(returns: NDArray, equity: NDArray, annualisation: float = 252.0) -> float:
    """Annualised return / abs(max drawdown). Standard portfolio quality metric."""
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    annual_return = (1.0 + r).prod() ** (annualisation / len(r)) - 1.0
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(annual_return / abs(mdd))


def win_rate(returns: NDArray) -> float:
    r = _clean(returns)
    nonzero = r[r != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def profit_factor(returns: NDArray) -> float:
    """Sum of gains / |sum of losses|. >1 means net profitable."""
    r = _clean(returns)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ---------------------------------------------------------------------------
# ML metrics
# ---------------------------------------------------------------------------


def directional_accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    """Fraction of predictions whose sign matches the truth — finance-relevant."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length")
    nonzero = (yt != 0) | (yp != 0)
    if not nonzero.any():
        return 0.0
    return float((np.sign(yt[nonzero]) == np.sign(yp[nonzero])).mean())


def rmse(y_true: NDArray, y_pred: NDArray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(((yt - yp) ** 2).mean()))


# ---------------------------------------------------------------------------
# RL metrics
# ---------------------------------------------------------------------------


def cumulative_reward(rewards: NDArray) -> float:
    return float(np.asarray(rewards, dtype=np.float64).sum())


def average_episode_reward(episode_rewards: list[float]) -> float:
    if not episode_rewards:
        return 0.0
    return float(np.mean(episode_rewards))
