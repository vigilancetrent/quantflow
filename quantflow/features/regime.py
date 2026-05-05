"""
quantflow.features.regime
=========================

Modern regime, volatility-decomposition, and ML-based indicators.

Given a 1-D series of log-returns r_t (or prices p_t with r_t = log(p_t/p_{t-1})),
the *quadratic variation* of the underlying log-price process decomposes as

    QV_t = IV_t + JV_t,

where IV is the continuous (integrated variance) component and JV is the
contribution of price jumps. Each indicator below isolates or characterises
one piece of this decomposition, or detects regime shifts in its dynamics.

All functions take a 1-D ``np.ndarray`` and return a 1-D ``np.ndarray`` of
the same length, with NaN-padded leading values for rolling estimators.

Dependencies: numpy, scipy, scikit-learn (GaussianMixture only).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

__all__ = [
    "log_returns",
    "realised_volatility",
    "bipower_variation",
    "realised_semivariance",
    "jump_variation",
    "cusum_change_point",
    "hmm_regime_probability",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_1d_float(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _check_window(window: int, n: int, name: str = "window") -> None:
    if not isinstance(window, (int, np.integer)) or window <= 1:
        raise ValueError(f"{name} must be an integer > 1, got {window!r}")
    if window > n:
        raise ValueError(f"{name}={window} exceeds series length {n}")


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    csum = np.concatenate(([0.0], np.cumsum(x)))
    out[window - 1:] = csum[window:] - csum[:-window]
    return out


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Convenience: r_t = log(p_t / p_{t-1}) with NaN at index 0."""
    p = _as_1d_float(prices, "prices")
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    out = np.full_like(p, np.nan)
    out[1:] = np.log(p[1:] / p[:-1])
    return out


# ---------------------------------------------------------------------------
# 1. Realised volatility
# ---------------------------------------------------------------------------

def realised_volatility(returns: np.ndarray, window: int = 78) -> np.ndarray:
    r"""Realised volatility (RV).

    Equation
    --------
        RV_t(w) = √( Σ_{i=t-w+1}^{t} r_i² )

    For 5-minute bars over a US trading day, w = 78 yields the canonical
    daily realised volatility.

    Citation
    --------
    Andersen, T. G., Bollerslev, T., Diebold, F. X., Labys, P. (2003).
    "Modeling and Forecasting Realized Volatility." *Econometrica* 71(2).

    Use
    ---
    Model-free estimate of integrated variance from evenly-spaced intraday
    returns under the no-jump assumption.
    """
    r = _as_1d_float(returns, "returns")
    _check_window(window, r.size)
    r2 = np.where(np.isnan(r), 0.0, r) ** 2
    rv = _rolling_sum(r2, window)
    return np.sqrt(rv)


# ---------------------------------------------------------------------------
# 2. Bipower variation
# ---------------------------------------------------------------------------

_MU1 = np.sqrt(2.0 / np.pi)


def bipower_variation(returns: np.ndarray, window: int = 78) -> np.ndarray:
    r"""Bipower variation (BPV) — jump-robust integrated-variance estimator.

    Equation
    --------
        BPV_t(w) = μ₁⁻² · Σ_{i=t-w+2}^{t} |r_{i-1}| · |r_i|,
        where μ₁ = E[|Z|] = √(2/π) for Z ~ N(0,1).

    BPV consistently estimates the *continuous* part of quadratic variation
    even in the presence of finitely many jumps.

    Citation
    --------
    Barndorff-Nielsen, O. E., Shephard, N. (2004). "Power and Bipower
    Variation with Stochastic Volatility and Jumps."
    *J. Financial Econometrics* 2(1).

    Use
    ---
    Estimate continuous diffusive variance while neutralising isolated price
    jumps; pair with ``realised_volatility`` to detect them.
    """
    r = _as_1d_float(returns, "returns")
    _check_window(window, r.size)
    abs_r = np.where(np.isnan(r), 0.0, np.abs(r))
    prod = np.empty_like(abs_r)
    prod[0] = 0.0
    prod[1:] = abs_r[:-1] * abs_r[1:]
    bpv_sum = _rolling_sum(prod, window)
    return bpv_sum / (_MU1 ** 2)


# ---------------------------------------------------------------------------
# 3. Realised semivariance
# ---------------------------------------------------------------------------

def realised_semivariance(
    returns: np.ndarray,
    window: int = 78,
    side: str = "both",
) -> np.ndarray:
    r"""Realised semivariance — directional decomposition of RV.

    Equations
    ---------
        RS⁻_t(w) = Σ_{i=t-w+1}^{t} r_i² · 𝟙{r_i < 0}     (downside)
        RS⁺_t(w) = Σ_{i=t-w+1}^{t} r_i² · 𝟙{r_i > 0}     (upside)
        RV_t     = RS⁺_t + RS⁻_t

    Citation
    --------
    Barndorff-Nielsen, O. E., Kinnebrock, S., Shephard, N. (2010).
    "Measuring Downside Risk: Realised Semivariance." In *Volatility and
    Time Series Econometrics*, OUP.

    Use
    ---
    Asymmetric risk; downside semivariance is the empirically dominant
    predictor of future volatility (Patton & Sheppard, 2015).

    side : 'down' returns RS⁻; 'up' returns RS⁺; 'both' returns RS⁺ − RS⁻
           (signed asymmetry; positive ⇒ net upside variance dominance).
    """
    if side not in {"up", "down", "both"}:
        raise ValueError("side must be 'up', 'down', or 'both'")
    r = _as_1d_float(returns, "returns")
    _check_window(window, r.size)
    r_clean = np.where(np.isnan(r), 0.0, r)
    r2 = r_clean ** 2
    if side == "down":
        return _rolling_sum(np.where(r_clean < 0, r2, 0.0), window)
    if side == "up":
        return _rolling_sum(np.where(r_clean > 0, r2, 0.0), window)
    rs_up = _rolling_sum(np.where(r_clean > 0, r2, 0.0), window)
    rs_dn = _rolling_sum(np.where(r_clean < 0, r2, 0.0), window)
    return rs_up - rs_dn


# ---------------------------------------------------------------------------
# 4. Jump variation
# ---------------------------------------------------------------------------

def jump_variation(returns: np.ndarray, window: int = 78) -> np.ndarray:
    r"""Jump variation — discontinuous component of quadratic variation.

    Equation
    --------
        JV_t(w) = max( RV_t(w)² − BPV_t(w), 0 )

    The non-negativity truncation is standard practice (Andersen, Bollerslev,
    Diebold, 2007) since RV² − BPV is a noisy estimator that can be slightly
    negative in finite samples.

    Citation
    --------
    Barndorff-Nielsen, O. E., Shephard, N. (2006). "Econometrics of Testing
    for Jumps in Financial Economics Using Bipower Variation."
    *J. Financial Econometrics* 4(1).

    Use
    ---
    Isolate variance attributable to discrete price jumps — feature for
    event-driven strategies and tail-risk models.
    """
    rv = realised_volatility(returns, window)
    bpv = bipower_variation(returns, window)
    jv = rv ** 2 - bpv
    mask = ~np.isnan(jv)
    jv[mask] = np.maximum(jv[mask], 0.0)
    return jv


# ---------------------------------------------------------------------------
# 5. CUSUM change-point indicator
# ---------------------------------------------------------------------------

def cusum_change_point(
    returns: np.ndarray,
    threshold: float = 5.0,
    drift: float = 0.0,
) -> np.ndarray:
    r"""Page's two-sided CUSUM change-point detector.

    Equations
    ---------
    Let z_t = (r_t − μ̂) / σ̂. The recursive statistics are

        S⁺_t = max( 0, S⁺_{t-1} + z_t − k )
        S⁻_t = max( 0, S⁻_{t-1} − z_t − k )
        c_t  = 𝟙{ S⁺_t > h  ∨  S⁻_t > h }

    where ``k = drift`` and ``h = threshold``. Both accumulators reset to
    zero after a trigger (standard Page convention).

    Citation
    --------
    Page, E. S. (1954). "Continuous Inspection Schemes."
    *Biometrika* 41(1/2).

    Use
    ---
    Flag abrupt regime shifts in the mean of a return series with minimal
    assumptions; output is a 0/1 binary signal.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if drift < 0:
        raise ValueError("drift must be non-negative")
    r = _as_1d_float(returns, "returns")
    finite = r[np.isfinite(r)]
    if finite.size < 2:
        raise ValueError("need at least 2 finite returns to standardise")
    mu, sd = float(np.mean(finite)), float(np.std(finite, ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        raise ValueError("returns have zero or non-finite standard deviation")
    z = (r - mu) / sd

    n = r.size
    out = np.zeros(n, dtype=np.int8)
    s_pos = 0.0
    s_neg = 0.0
    for t in range(n):
        zt = z[t]
        if not np.isfinite(zt):
            s_pos = 0.0
            s_neg = 0.0
            continue
        s_pos = max(0.0, s_pos + zt - drift)
        s_neg = max(0.0, s_neg - zt - drift)
        if s_pos > threshold or s_neg > threshold:
            out[t] = 1
            s_pos = 0.0
            s_neg = 0.0
    return out


# ---------------------------------------------------------------------------
# 6. Hidden Markov / GMM regime probability
# ---------------------------------------------------------------------------

def hmm_regime_probability(
    returns: np.ndarray,
    window: int = 252,
    refit_every: int = 21,
    random_state: int = 0,
) -> np.ndarray:
    r"""Online GMM proxy for the high-volatility regime probability.

    Within each rolling window we fit a 2-component univariate Gaussian
    mixture to the returns:

        f(r) = π₀ · 𝒩(r; μ₀, σ₀²) + π₁ · 𝒩(r; μ₁, σ₁²),    σ₁² > σ₀².

    The output at time t is the posterior P(high-vol regime | r_t):

        P(z_t = 1 | r_t) =  π₁ · φ(r_t; μ₁, σ₁²)
                           ───────────────────────────────────────
                            π₀ · φ(r_t; μ₀, σ₀²) + π₁ · φ(r_t; μ₁, σ₁²)

    Citation
    --------
    Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle."
    *Econometrica* 57. GMM via Dempster-Laird-Rubin EM (1977).

    Use
    ---
    Fast, stateless feature for "are we in a turbulent regime?" without the
    path-dependence and identification issues of fully filtered HMMs.
    """
    r = _as_1d_float(returns, "returns")
    if window < 30:
        raise ValueError("window must be >= 30 for stable GMM estimation")
    _check_window(window, r.size)
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")

    n = r.size
    out = np.full(n, np.nan, dtype=float)
    last_params: tuple[float, float, float, float, float, float] | None = None

    for t in range(window - 1, n):
        sample = r[t - window + 1: t + 1]
        sample = sample[np.isfinite(sample)]
        if sample.size < 30:
            continue

        need_fit = (last_params is None) or ((t - (window - 1)) % refit_every == 0)
        if need_fit:
            try:
                gm = GaussianMixture(
                    n_components=2,
                    covariance_type="full",
                    random_state=random_state,
                    reg_covar=1e-10,
                    max_iter=200,
                )
                gm.fit(sample.reshape(-1, 1))
            except Exception:
                if last_params is None:
                    continue
            else:
                vars_ = gm.covariances_.reshape(-1)
                hi = int(np.argmax(vars_))
                lo = 1 - hi
                last_params = (
                    float(gm.weights_[lo]),
                    float(gm.means_[lo, 0]),
                    float(vars_[lo]),
                    float(gm.weights_[hi]),
                    float(gm.means_[hi, 0]),
                    float(vars_[hi]),
                )

        if last_params is None:
            continue
        w_lo, m_lo, v_lo, w_hi, m_hi, v_hi = last_params
        rt = r[t]
        if not np.isfinite(rt):
            continue
        p_lo = w_lo * stats.norm.pdf(rt, loc=m_lo, scale=np.sqrt(v_lo))
        p_hi = w_hi * stats.norm.pdf(rt, loc=m_hi, scale=np.sqrt(v_hi))
        denom = p_lo + p_hi
        if denom > 0 and np.isfinite(denom):
            out[t] = p_hi / denom
    return out
