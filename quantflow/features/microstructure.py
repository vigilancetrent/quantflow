"""
quantflow.features.microstructure
=================================

Modern market-microstructure, order-flow, and liquidity indicators.

This module collects research-grade estimators from the post-2010 microstructure
literature. Unlike classical technical analysis (RSI, MACD, Bollinger), these
features attempt to recover *latent* quantities — adverse-selection cost,
toxicity of order flow, effective bid-ask spread, price-impact coefficient —
from observable trade data (price, volume, optionally OHLC). They are widely
used in execution-cost models, market-making, and short-horizon alpha at
systematic funds.

Conventions
-----------
* All public functions take 1-D NumPy arrays (or array-likes) and return a
  1-D ``np.ndarray`` of the same length as the primary input.
* Warmup observations are filled with ``np.nan`` (pandas ``rolling``-style)
  so that outputs can be aligned to the original index without lookahead.
* Inputs are validated; ``ValueError`` is raised for bad shapes / lengths /
  non-positive windows.
"""

from __future__ import annotations

from math import erf, sqrt
from typing import Optional

import numpy as np

__all__ = [
    "kyle_lambda",
    "vpin",
    "amihud_illiquidity",
    "order_flow_imbalance",
    "roll_effective_spread",
    "corwin_schultz_spread",
    "realised_quarticity",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_float_array(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return np.ascontiguousarray(a)


def _check_same_length(*arrays_with_names) -> int:
    n = arrays_with_names[0][0].size
    for arr, name in arrays_with_names[1:]:
        if arr.size != n:
            raise ValueError(
                f"length mismatch: {arrays_with_names[0][1]}={n} vs {name}={arr.size}"
            )
    return n


def _check_window(window: int, n: int, name: str = "window") -> None:
    if not isinstance(window, (int, np.integer)) or window <= 1:
        raise ValueError(f"{name} must be an integer > 1, got {window!r}")
    if window > n:
        raise ValueError(f"{name}={window} exceeds series length {n}")


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """NaN-leading rolling sum via cumulative-sum trick (O(n))."""
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if window > n:
        return out
    c = np.concatenate(([0.0], np.cumsum(x)))
    out[window - 1:] = c[window:] - c[:-window]
    return out


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Normal CDF via math.erf — vectorised."""
    return 0.5 * (1.0 + np.frompyfunc(erf, 1, 1)(z / sqrt(2.0)).astype(np.float64))


def _trade_sign_tick_rule(prices: np.ndarray) -> np.ndarray:
    """Lee-Ready tick rule. +1 on uptick, -1 on downtick, repeat last on zero-tick."""
    dp = np.diff(prices, prepend=prices[0])
    sign = np.sign(dp)
    nz = sign != 0
    idx = np.where(nz, np.arange(sign.size), 0)
    np.maximum.accumulate(idx, out=idx)
    sign = sign[idx]
    sign[0] = 0.0
    return sign


# ---------------------------------------------------------------------------
# 1. Kyle's lambda
# ---------------------------------------------------------------------------
def kyle_lambda(prices, volume, window: int = 60) -> np.ndarray:
    r"""Kyle's λ — price-impact coefficient from signed order flow.

    Equation
    --------
        Δp_t = λ · (s_t · V_t) + ε_t

    where ``s_t ∈ {-1, +1}`` is the trade sign (tick rule) and ``V_t`` is
    traded volume. λ is estimated by OLS over a rolling window of length W:

        λ̂_t = Σ Δp · q  /  Σ q²,    q := s · V,    over [t-W+1, t]

    Higher λ ⇒ lower liquidity (each unit of signed flow moves price more).

    Citation
    --------
    Kyle, A. S. (1985). "Continuous Auctions and Insider Trading."
    *Econometrica* 53(6), 1315-1335.

    Use
    ---
    Estimate adverse-selection cost / market depth at intraday horizons.
    """
    p = _as_float_array(prices, "prices")
    v = _as_float_array(volume, "volume")
    n = _check_same_length((p, "prices"), (v, "volume"))
    _check_window(window, n)
    if np.any(v < 0):
        raise ValueError("volume must be non-negative")

    dp = np.diff(p, prepend=p[0])
    dp[0] = np.nan
    s = _trade_sign_tick_rule(p)
    q = s * v

    num = dp * q
    den = q * q
    num_safe = np.where(np.isnan(num), 0.0, num)
    s_num = _rolling_sum(num_safe, window)
    s_den = _rolling_sum(den, window)

    out = np.full(n, np.nan, dtype=np.float64)
    valid = (s_den > 0) & np.isfinite(s_num)
    out[valid] = s_num[valid] / s_den[valid]
    return out


# ---------------------------------------------------------------------------
# 2. VPIN
# ---------------------------------------------------------------------------
def vpin(
    prices,
    volume,
    bucket_size: Optional[float] = None,
    n_buckets: int = 50,
) -> np.ndarray:
    r"""VPIN — Volume-Synchronized Probability of Informed Trading.

    Equation
    --------
    Trades are aggregated into equal-volume buckets of size V. Within each
    bucket τ, signed volume is split into buy / sell parts using a normal-CDF
    bulk-classification on standardised price changes:

        V^B_τ = Σ V_i · Φ( Δp_i / σ_Δp )
        V^S_τ = V_τ − V^B_τ

    VPIN over the trailing N buckets is

        VPIN_τ = ( Σ_{k=τ-N+1}^{τ} | V^B_k − V^S_k | ) / (N · V)

    Citation
    --------
    Easley, D., López de Prado, M., O'Hara, M. (2012). "Flow Toxicity and
    Liquidity in a High-Frequency World." *RFS* 25(5), 1457-1493.

    Use
    ---
    Detect order-flow toxicity / informed trading; spikes precede volatility.
    """
    p = _as_float_array(prices, "prices")
    v = _as_float_array(volume, "volume")
    n = _check_same_length((p, "prices"), (v, "volume"))
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")
    if np.any(v < 0):
        raise ValueError("volume must be non-negative")

    total_vol = float(v.sum())
    if total_vol <= 0:
        raise ValueError("volume sums to zero; cannot form buckets")

    if bucket_size is None:
        bucket_size = total_vol / max(50.0 * n_buckets, 1.0)
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    dp = np.diff(p, prepend=p[0])
    dp[0] = 0.0
    sigma = float(np.std(dp[1:], ddof=1)) if dp.size > 1 else 0.0
    if sigma <= 0:
        return np.full(n, np.nan, dtype=np.float64)

    buy_frac = _norm_cdf(dp / sigma)
    buy_vol_per_trade = v * buy_frac
    sell_vol_per_trade = v * (1.0 - buy_frac)

    bucket_buy: list[float] = []
    bucket_sell: list[float] = []
    bucket_end_idx: list[int] = []

    cur_buy = 0.0
    cur_sell = 0.0
    cur_vol = 0.0
    for i in range(n):
        cur_vol += v[i]
        cur_buy += buy_vol_per_trade[i]
        cur_sell += sell_vol_per_trade[i]
        while cur_vol >= bucket_size:
            overflow = cur_vol - bucket_size
            frac_kept = (v[i] - overflow) / v[i] if v[i] > 0 else 1.0
            kept_buy = buy_vol_per_trade[i] * frac_kept
            kept_sell = sell_vol_per_trade[i] * frac_kept
            b = cur_buy - (buy_vol_per_trade[i] - kept_buy)
            s = cur_sell - (sell_vol_per_trade[i] - kept_sell)
            bucket_buy.append(b)
            bucket_sell.append(s)
            bucket_end_idx.append(i)
            cur_buy = buy_vol_per_trade[i] - kept_buy
            cur_sell = sell_vol_per_trade[i] - kept_sell
            cur_vol = overflow

    out = np.full(n, np.nan, dtype=np.float64)
    if len(bucket_buy) < n_buckets:
        return out

    bb = np.asarray(bucket_buy)
    bs = np.asarray(bucket_sell)
    imb = np.abs(bb - bs)
    rolling = _rolling_sum(imb, n_buckets) / (n_buckets * bucket_size)

    for k, end_i in enumerate(bucket_end_idx):
        if not np.isnan(rolling[k]):
            stop = bucket_end_idx[k + 1] if k + 1 < len(bucket_end_idx) else n
            out[end_i:stop] = rolling[k]
    return out


# ---------------------------------------------------------------------------
# 3. Amihud illiquidity
# ---------------------------------------------------------------------------
def amihud_illiquidity(prices, volume, window: int = 21) -> np.ndarray:
    r"""Amihud illiquidity ratio.

    Equation
    --------
        ILLIQ_t = (1/W) · Σ_{k=t-W+1}^{t}  |r_k|  /  (P_k · V_k)

    where ``r_k`` is the simple return and ``P_k · V_k`` is the dollar volume.

    Citation
    --------
    Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and
    time-series effects." *J. Financial Markets* 5(1), 31-56.

    Use
    ---
    Daily-frequency liquidity proxy when bid-ask data are unavailable.
    """
    p = _as_float_array(prices, "prices")
    v = _as_float_array(volume, "volume")
    n = _check_same_length((p, "prices"), (v, "volume"))
    _check_window(window, n)
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    if np.any(v < 0):
        raise ValueError("volume must be non-negative")

    r = np.empty(n, dtype=np.float64)
    r[0] = np.nan
    r[1:] = p[1:] / p[:-1] - 1.0

    dollar_vol = p * v
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dollar_vol > 0, np.abs(r) / dollar_vol, np.nan)

    out = np.full(n, np.nan, dtype=np.float64)
    safe = np.where(np.isfinite(ratio), ratio, 0.0)
    cnt = np.isfinite(ratio).astype(np.float64)
    s_val = _rolling_sum(safe, window)
    s_cnt = _rolling_sum(cnt, window)
    valid = s_cnt > 0
    out[valid] = s_val[valid] / s_cnt[valid]
    return out


# ---------------------------------------------------------------------------
# 4. Order-flow imbalance
# ---------------------------------------------------------------------------
def order_flow_imbalance(prices, volume, window: int = 50) -> np.ndarray:
    r"""Rolling order-flow imbalance (signed-volume share).

    Equation
    --------
        OFI_t = Σ_{k=t-W+1}^{t} s_k V_k   /   Σ_{k=t-W+1}^{t} V_k

    with trade signs from the tick rule. Range is [-1, +1].

    Citation
    --------
    Cont, R., Kukanov, A., Stoikov, S. (2014). "The Price Impact of Order
    Book Events." *J. Financial Econometrics* 12(1), 47-88.
    """
    p = _as_float_array(prices, "prices")
    v = _as_float_array(volume, "volume")
    n = _check_same_length((p, "prices"), (v, "volume"))
    _check_window(window, n)
    if np.any(v < 0):
        raise ValueError("volume must be non-negative")

    s = _trade_sign_tick_rule(p)
    signed = s * v
    s_num = _rolling_sum(signed, window)
    s_den = _rolling_sum(v, window)
    out = np.full(n, np.nan, dtype=np.float64)
    valid = (s_den > 0) & np.isfinite(s_num)
    out[valid] = s_num[valid] / s_den[valid]
    return out


# ---------------------------------------------------------------------------
# 5. Roll's effective spread
# ---------------------------------------------------------------------------
def roll_effective_spread(prices, window: int = 60) -> np.ndarray:
    r"""Roll's implicit bid-ask spread estimator.

    Equation
    --------
        Cov_t = Cov( Δp_k, Δp_{k-1} )    over a rolling window
        S_t   = 2 · √( -Cov_t )           if Cov_t < 0, else NaN

    Citation
    --------
    Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask
    Spread in an Efficient Market." *J. Finance* 39(4), 1127-1139.
    """
    p = _as_float_array(prices, "prices")
    n = p.size
    _check_window(window, n)

    dp = np.diff(p, prepend=p[0])
    dp[0] = np.nan

    x = dp.copy()
    y = np.empty(n, dtype=np.float64)
    y[0] = np.nan
    y[1:] = dp[:-1]

    xy = x * y
    safe_x = np.where(np.isfinite(x), x, 0.0)
    safe_y = np.where(np.isfinite(y), y, 0.0)
    safe_xy = np.where(np.isfinite(xy), xy, 0.0)
    cnt = (np.isfinite(x) & np.isfinite(y)).astype(np.float64)

    s_x = _rolling_sum(safe_x, window)
    s_y = _rolling_sum(safe_y, window)
    s_xy = _rolling_sum(safe_xy, window)
    s_n = _rolling_sum(cnt, window)

    out = np.full(n, np.nan, dtype=np.float64)
    valid = s_n > 1
    with np.errstate(invalid="ignore"):
        cov = np.where(
            valid,
            s_xy / np.where(s_n > 0, s_n, 1) - (s_x * s_y) / np.where(s_n > 0, s_n * s_n, 1),
            np.nan,
        )
    neg = valid & (cov < 0)
    out[neg] = 2.0 * np.sqrt(-cov[neg])
    return out


# ---------------------------------------------------------------------------
# 6. Corwin-Schultz high-low spread
# ---------------------------------------------------------------------------
def corwin_schultz_spread(high, low) -> np.ndarray:
    r"""Corwin-Schultz high-low spread estimator.

    Equation
    --------
        β_t = [ ln(H_{t-1}/L_{t-1}) ]² + [ ln(H_t/L_t) ]²
        γ_t = [ ln( max(H_{t-1},H_t) / min(L_{t-1},L_t) ) ]²
        α_t = ( √(2β_t) − √β_t ) / (3 − 2√2)  −  √( γ_t / (3 − 2√2) )
        S_t = 2 · ( e^{α_t} − 1 ) / ( 1 + e^{α_t} )

    Citation
    --------
    Corwin, S. A., Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask
    Spreads from Daily High and Low Prices." *J. Finance* 67(2), 719-760.
    """
    h = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    n = _check_same_length((h, "high"), (low_arr, "low"))
    if np.any(h <= 0) or np.any(low_arr <= 0):
        raise ValueError("high/low must be strictly positive")
    if np.any(h < low_arr):
        raise ValueError("high must be >= low elementwise")

    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out

    h_prev, l_prev = h[:-1], low_arr[:-1]
    h_cur, l_cur = h[1:], low_arr[1:]

    log_hl_prev = np.log(h_prev / l_prev)
    log_hl_cur = np.log(h_cur / l_cur)
    beta = log_hl_prev ** 2 + log_hl_cur ** 2

    h2 = np.maximum(h_prev, h_cur)
    l2 = np.minimum(l_prev, l_cur)
    gamma = np.log(h2 / l2) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)
    with np.errstate(invalid="ignore"):
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = np.where(spread < 0, 0.0, spread)

    out[1:] = spread
    return out


# ---------------------------------------------------------------------------
# 7. Realised quarticity
# ---------------------------------------------------------------------------
def realised_quarticity(prices, window: int = 78) -> np.ndarray:
    r"""Realised quarticity — fourth-moment vol-of-vol estimator.

    Equation
    --------
        RQ_t = (W / 3) · Σ_{k=t-W+1}^{t} ( r_k )⁴

    Citation
    --------
    Barndorff-Nielsen, O. E., Shephard, N. (2002). "Estimating quadratic
    variation using realized variance." *J. Applied Econometrics* 17(5).
    """
    p = _as_float_array(prices, "prices")
    n = p.size
    _check_window(window, n)
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")

    r = np.empty(n, dtype=np.float64)
    r[0] = np.nan
    r[1:] = np.log(p[1:] / p[:-1])
    r4 = np.where(np.isfinite(r), r ** 4, 0.0)
    s4 = _rolling_sum(r4, window)
    out = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(s4)
    out[valid] = (window / 3.0) * s4[valid]
    return out
