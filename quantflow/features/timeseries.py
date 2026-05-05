"""Generic time-series feature transforms.

Domain-agnostic — these are useful for finance but also energy, IoT,
clickstream, and any sequential signal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]


def returns(prices: NDArray) -> ArrayF:
    """Simple percentage returns: r_t = (p_t / p_{t-1}) - 1."""
    p = np.asarray(prices, dtype=np.float64)
    out = np.full_like(p, np.nan)
    out[1:] = p[1:] / p[:-1] - 1.0
    return out


def log_returns(prices: NDArray) -> ArrayF:
    """Continuously compounded returns: r_t = ln(p_t / p_{t-1})."""
    p = np.asarray(prices, dtype=np.float64)
    out = np.full_like(p, np.nan)
    out[1:] = np.log(p[1:] / p[:-1])
    return out


def lag(x: NDArray, k: int = 1) -> ArrayF:
    """Shift series by k steps (positive = into the past)."""
    arr = np.asarray(x, dtype=np.float64)
    if k < 0:
        raise ValueError("k must be non-negative; use forward-shift utilities for k<0")
    out = np.full_like(arr, np.nan)
    if k == 0:
        return arr.copy()
    if k < len(arr):
        out[k:] = arr[:-k]
    return out


def rolling_zscore(x: NDArray, window: int) -> ArrayF:
    """Rolling z-score: (x_t - mean_window) / std_window."""
    if window <= 1:
        raise ValueError("window must be > 1")
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        seg = arr[t - window + 1 : t + 1]
        mu = seg.mean()
        sd = seg.std(ddof=0)
        out[t] = (arr[t] - mu) / sd if sd > 0 else 0.0
    return out


def rolling_volatility(returns_arr: NDArray, window: int, annualisation: float = 252.0) -> ArrayF:
    """Annualised rolling volatility from a return series."""
    r = np.asarray(returns_arr, dtype=np.float64)
    n = len(r)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        seg = r[t - window + 1 : t + 1]
        out[t] = seg.std(ddof=1) * np.sqrt(annualisation)
    return out
