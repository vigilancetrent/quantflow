"""Vectorised financial indicators.

All functions operate on 1-D NumPy arrays of prices and return arrays of the
same length. The first ``period - 1`` values are NaN — callers should drop or
mask them before feeding to a model. This convention matches `pandas.rolling`.

Numerical reference:
- Wilder's RSI: https://www.investopedia.com/terms/r/rsi.asp
- MACD: https://www.investopedia.com/terms/m/macd.asp
- Bollinger Bands: https://www.investopedia.com/terms/b/bollingerbands.asp
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


def _as_float_array(x: NDArray) -> ArrayF:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}")
    return arr


def sma(prices: NDArray, period: int) -> ArrayF:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    p = _as_float_array(prices)
    out = np.full_like(p, np.nan)
    if len(p) < period:
        return out
    csum = np.cumsum(p)
    out[period - 1] = csum[period - 1] / period
    out[period:] = (csum[period:] - csum[:-period]) / period
    return out


def ema(prices: NDArray, period: int) -> ArrayF:
    """Exponential moving average using the standard ``alpha = 2/(N+1)`` form.

    The first valid value is the SMA of the first `period` observations; from
    there the recursion ``ema_t = alpha * p_t + (1 - alpha) * ema_{t-1}`` runs.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    p = _as_float_array(prices)
    out = np.full_like(p, np.nan)
    if len(p) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = p[:period].mean()
    for t in range(period, len(p)):
        out[t] = alpha * p[t] + (1.0 - alpha) * out[t - 1]
    return out


def rsi(prices: NDArray, period: int = 14) -> ArrayF:
    """Wilder's Relative Strength Index ∈ [0, 100]."""
    if period <= 0:
        raise ValueError("period must be positive")
    p = _as_float_array(prices)
    out = np.full_like(p, np.nan)
    if len(p) <= period:
        return out

    delta = np.diff(p)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for t in range(period, len(p)):
        if t > period:
            avg_gain = (avg_gain * (period - 1) + gains[t - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[t - 1]) / period
        if avg_loss == 0:
            out[t] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[t] = 100.0 - 100.0 / (1.0 + rs)
    return out


class MACDResult(NamedTuple):
    macd: ArrayF
    signal: ArrayF
    histogram: ArrayF


def macd(
    prices: NDArray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """MACD line, signal line, and histogram."""
    if fast >= slow:
        raise ValueError("fast period must be < slow period")
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = fast_ema - slow_ema
    # Signal EMA on the MACD line (skipping leading NaNs).
    valid = ~np.isnan(macd_line)
    sig_line = np.full_like(macd_line, np.nan)
    if valid.sum() >= signal_period:
        first = int(np.argmax(valid))
        sig_line[first:] = ema(macd_line[first:], signal_period)
    return MACDResult(macd=macd_line, signal=sig_line, histogram=macd_line - sig_line)


class BollingerResult(NamedTuple):
    upper: ArrayF
    middle: ArrayF
    lower: ArrayF


def bollinger_bands(
    prices: NDArray,
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    """Bollinger Bands — middle = SMA, upper/lower = ± num_std × rolling std."""
    if period <= 0:
        raise ValueError("period must be positive")
    p = _as_float_array(prices)
    middle = sma(p, period)
    std = np.full_like(p, np.nan)
    for t in range(period - 1, len(p)):
        std[t] = p[t - period + 1 : t + 1].std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return BollingerResult(upper=upper, middle=middle, lower=lower)


def atr(high: NDArray, low: NDArray, close: NDArray, period: int = 14) -> ArrayF:
    """Average True Range — Wilder's smoothing of the true range."""
    h = _as_float_array(high)
    low_ = _as_float_array(low)
    c = _as_float_array(close)
    if not (len(h) == len(low_) == len(c)):
        raise ValueError("high, low, close must have equal length")
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - low_[0]
    for t in range(1, n):
        tr[t] = max(h[t] - low_[t], abs(h[t] - c[t - 1]), abs(low_[t] - c[t - 1]))
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = tr[:period].mean()
    for t in range(period, n):
        out[t] = (out[t - 1] * (period - 1) + tr[t]) / period
    return out
