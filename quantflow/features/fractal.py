"""
quantflow.features.fractal
==========================

Modern fractal, information-theoretic and nonlinear indicators for financial
time series. These complement classical technical analysis by quantifying
*memory*, *regularity*, *chaos* and *long-range correlations* — properties
that moving averages and oscillators cannot capture.

Family overview
---------------
- Hurst exponent (R/S)            -> long memory  (H>0.5 trend, H<0.5 revert)
- Fractional differentiation FFD  -> stationarity with memory preserved
- Sample entropy                  -> signal regularity
- Permutation entropy             -> ordinal complexity, outlier-robust
- Lyapunov exponent (Rosenstein)  -> sensitive dependence (chaos)
- Detrended Fluctuation Analysis  -> scaling exponent on non-stationary data
- Multiscale entropy              -> SampEn across coarse-grained scales

All functions accept a 1-D float array. Rolling estimators return arrays of
the same length with leading NaNs; single-window estimators return a scalar.
Bad input shapes / sizes raise ``ValueError``.
"""

from __future__ import annotations

from math import factorial

import numpy as np


__all__ = [
    "hurst_rs",
    "frac_diff_ffd",
    "sample_entropy",
    "permutation_entropy",
    "lyapunov_rosenstein",
    "dfa",
    "multiscale_entropy",
]


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #

def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {a.shape}")
    if a.size == 0:
        raise ValueError("input array is empty")
    return a


def _embed(x: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
    """Time-delay embedding -> matrix of shape (N - (m-1)*tau, m)."""
    n = x.size - (m - 1) * tau
    if n <= 0:
        raise ValueError("series too short for embedding")
    idx = np.arange(m) * tau
    return x[np.arange(n)[:, None] + idx]


# --------------------------------------------------------------------------- #
#  1. Hurst exponent via Rescaled-Range (R/S) analysis                        #
# --------------------------------------------------------------------------- #

def hurst_rs(x, min_window: int = 8, max_window: int | None = None) -> float:
    r"""Hurst exponent H estimated by Rescaled-Range analysis.

    For a window of length n the rescaled range is

        R(n)/S(n) = ( max_{1<=k<=n} Y_k - min_{1<=k<=n} Y_k ) / σ(x_{1..n})

    where  Y_k = Σ_{i=1..k} (x_i - mean(x)).  Mandelbrot's law gives
    E[R(n)/S(n)] ~ c · n^H, so H is the slope of  log(R/S) vs log(n).

    H > 0.5 = persistent / trending,
    H = 0.5 = random walk,
    H < 0.5 = mean-reverting.

    Citation
    --------
    Hurst, H. E. (1951). "Long-term storage capacity of reservoirs."
    *Trans. Am. Soc. Civ. Eng.* 116, 770-808.

    Use
    ---
    Assess whether returns or prices exhibit long memory over a fixed sample.
    Returns a scalar.
    """
    x = _as_1d(x)
    N = x.size
    if max_window is None:
        max_window = N // 2
    if min_window < 4 or max_window <= min_window or max_window > N:
        raise ValueError("invalid window bounds for Hurst R/S")

    sizes = np.unique(np.logspace(np.log10(min_window),
                                  np.log10(max_window), num=20).astype(int))
    rs_vals = []
    for n in sizes:
        n_chunks = N // n
        if n_chunks < 1:
            continue
        chunks = x[: n_chunks * n].reshape(n_chunks, n)
        mean = chunks.mean(axis=1, keepdims=True)
        Y = np.cumsum(chunks - mean, axis=1)
        R = Y.max(axis=1) - Y.min(axis=1)
        S = chunks.std(axis=1, ddof=1)
        mask = S > 0
        if not mask.any():
            continue
        rs_vals.append((n, np.mean(R[mask] / S[mask])))

    if len(rs_vals) < 4:
        raise ValueError("not enough valid windows to estimate Hurst")

    ns, rss = zip(*rs_vals)
    slope, _ = np.polyfit(np.log(ns), np.log(rss), 1)
    return float(slope)


# --------------------------------------------------------------------------- #
#  2. Fractional Differentiation — Fixed-Width Window (FFD)                   #
# --------------------------------------------------------------------------- #

def _ffd_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Binomial-series weights for (1-L)^d truncated when |w_k| < threshold."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if k > 10_000:
            break
    return np.array(weights, dtype=float)


def frac_diff_ffd(x, d: float, threshold: float = 1e-4) -> np.ndarray:
    r"""Fixed-Window Fractional Differentiation.

    Defined via the binomial series of the lag operator (1-L)^d:

        x_t^{(d)} = Σ_{k=0}^{K-1} w_k · x_{t-k}
        w_0 = 1,   w_k = -w_{k-1} · (d - k + 1) / k

    Weights truncate at the smallest K for which |w_K| < threshold,
    yielding a constant-memory, stationarity-preserving filter.

    Citation
    --------
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*,
    Ch. 5. Wiley.

    Use
    ---
    Stationary feature for ML models that retains as much memory as possible
    — typically d ∈ (0, 1). Returns array same length as x with the leading
    K-1 entries set to NaN.
    """
    x = _as_1d(x)
    if not 0.0 < d < 2.0:
        raise ValueError("d should be in (0, 2)")
    w = _ffd_weights(d, threshold)
    K = w.size
    if K > x.size:
        raise ValueError("series shorter than fractional-diff window")

    out = np.full(x.size, np.nan)
    conv = np.convolve(x, w, mode="full")[: x.size]
    out[K - 1:] = conv[K - 1:]
    return out


# --------------------------------------------------------------------------- #
#  3. Sample Entropy                                                          #
# --------------------------------------------------------------------------- #

def sample_entropy(x, m: int = 2, r: float | None = None) -> float:
    r"""Sample Entropy SampEn(m, r, N).

        SampEn = -ln( A / B )
        B = #{ pairs (i,j), i<j : d(X_i^m,     X_j^m)     < r }
        A = #{ pairs (i,j), i<j : d(X_i^{m+1}, X_j^{m+1}) < r }

    where X_i^m = (x_i, ..., x_{i+m-1}) and d is the Chebyshev distance.
    Self-matches are excluded, eliminating the bias in Approximate Entropy.

    Citation
    --------
    Richman J.S., Moorman J.R. (2000). "Physiological time-series analysis
    using approximate entropy and sample entropy."
    *Am. J. Physiol. Heart Circ. Physiol.* 278, H2039-H2049.

    Use
    ---
    Score how regular / predictable a series is. Higher SampEn = more
    irregular. Returns a scalar.
    """
    x = _as_1d(x)
    if m < 1:
        raise ValueError("m must be >= 1")
    if x.size < m + 2:
        raise ValueError("series too short for sample entropy")
    if r is None:
        r = 0.2 * x.std(ddof=0)
    if r <= 0:
        raise ValueError("tolerance r must be positive (constant series?)")

    def _count(mm: int) -> int:
        Xm = _embed(x, mm)
        n = Xm.shape[0]
        count = 0
        for i in range(n - 1):
            d = np.max(np.abs(Xm[i + 1:] - Xm[i]), axis=1)
            count += int(np.sum(d < r))
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0 or A == 0:
        return float("inf")
    return float(-np.log(A / B))


# --------------------------------------------------------------------------- #
#  4. Permutation Entropy                                                     #
# --------------------------------------------------------------------------- #

def permutation_entropy(x, m: int = 3, tau: int = 1,
                        normalize: bool = True) -> float:
    r"""Permutation Entropy of order m, delay tau.

        H(m) = - Σ_{π ∈ S_m}  p(π) · log p(π)

    where p(π) is the empirical frequency of the ordinal pattern π in the
    embedded vectors (x_t, x_{t+τ}, ..., x_{t+(m-1)τ}). Optionally
    normalised by log(m!) so values lie in [0, 1].

    Citation
    --------
    Bandt C., Pompe B. (2002). "Permutation entropy: a natural complexity
    measure for time series." *Phys. Rev. Lett.* 88, 174102.

    Use
    ---
    Complexity score robust to monotone distortions, outliers and amplitude
    changes. Returns a scalar.
    """
    x = _as_1d(x)
    if m < 2:
        raise ValueError("m must be >= 2")
    X = _embed(x, m, tau)
    patterns = np.argsort(X, axis=1, kind="mergesort")
    powers = (m ** np.arange(m))[::-1]
    codes = patterns @ powers
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p))
    if normalize:
        H /= np.log(factorial(m))
    return float(H)


# --------------------------------------------------------------------------- #
#  5. Largest Lyapunov Exponent — Rosenstein method                           #
# --------------------------------------------------------------------------- #

def lyapunov_rosenstein(x, m: int = 5, tau: int = 1,
                        max_t: int = 20,
                        min_separation: int | None = None) -> float:
    r"""Approximate largest Lyapunov exponent via Rosenstein's algorithm.

    For each embedded point X_i find its nearest neighbour X_j (with
    temporal separation > Theiler window). Track the divergence

        d_i(k) = || X_{i+k} - X_{j+k} ||,
        ⟨ ln d(k) ⟩ ~ λ₁ · k · dt.

    The slope of ⟨ln d(k)⟩ vs k is the exponent estimate.

    Citation
    --------
    Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993). "A practical method
    for calculating largest Lyapunov exponents from small data sets."
    *Physica D* 65, 117-134.

    Use
    ---
    Test whether a series exhibits chaos. λ > 0 ⇒ sensitive dependence.
    Returns a scalar (per-step rate).
    """
    x = _as_1d(x)
    X = _embed(x, m, tau)
    n = X.shape[0]
    if min_separation is None:
        min_separation = (m - 1) * tau
    if n < max_t + min_separation + 2:
        raise ValueError("series too short for Lyapunov estimate")

    neighbours = np.empty(n, dtype=int)
    for i in range(n):
        d = np.linalg.norm(X - X[i], axis=1)
        d[max(0, i - min_separation): i + min_separation + 1] = np.inf
        neighbours[i] = int(np.argmin(d))

    divergence = np.full(max_t, np.nan)
    for k in range(max_t):
        ii = np.arange(n - k)
        jj = neighbours[: n - k] + k
        valid = jj < n
        ii, jj = ii[valid], jj[valid]
        d = np.linalg.norm(X[ii + k] - X[jj], axis=1)
        d = d[d > 0]
        if d.size:
            divergence[k] = np.mean(np.log(d))

    ks = np.arange(max_t)
    mask = np.isfinite(divergence)
    if mask.sum() < 4:
        raise ValueError("not enough valid divergence points")
    slope, _ = np.polyfit(ks[mask], divergence[mask], 1)
    return float(slope)


# --------------------------------------------------------------------------- #
#  6. Detrended Fluctuation Analysis (DFA)                                    #
# --------------------------------------------------------------------------- #

def dfa(x, min_window: int = 8, max_window: int | None = None,
        order: int = 1) -> float:
    r"""Detrended Fluctuation Analysis scaling exponent α.

    Steps:
        1. Y_k = Σ_{i=1..k} (x_i - mean(x))                     (profile)
        2. split Y into non-overlapping windows of size n
        3. fit a polynomial of given order in each window -> Y_n
        4. F(n) = √( mean( (Y - Y_n)² ) )
        5. F(n) ~ n^α  ⇒  α is slope on log-log plot

    α = 0.5 uncorrelated, > 0.5 persistent, < 0.5 anti-persistent.
    α > 1 indicates non-stationary processes (e.g. random walk).

    Citation
    --------
    Peng, C.-K. et al. (1994). "Mosaic organization of DNA nucleotides."
    *Phys. Rev. E* 49, 1685-1689.

    Use
    ---
    Detect long-range correlations on non-stationary signals such as raw
    price levels. Returns a scalar.
    """
    x = _as_1d(x)
    N = x.size
    if max_window is None:
        max_window = N // 4
    if min_window < order + 2 or max_window <= min_window or max_window > N:
        raise ValueError("invalid DFA window bounds")

    Y = np.cumsum(x - x.mean())
    sizes = np.unique(np.logspace(np.log10(min_window),
                                  np.log10(max_window), num=20).astype(int))
    F = []
    for n in sizes:
        n_chunks = N // n
        if n_chunks < 2:
            continue
        segs = Y[: n_chunks * n].reshape(n_chunks, n)
        t = np.arange(n)
        V = np.vander(t, order + 1)
        coefs, *_ = np.linalg.lstsq(V, segs.T, rcond=None)
        trend = (V @ coefs).T
        F.append((n, np.sqrt(np.mean((segs - trend) ** 2))))

    if len(F) < 4:
        raise ValueError("not enough valid scales for DFA")
    ns, fs = zip(*F)
    slope, _ = np.polyfit(np.log(ns), np.log(fs), 1)
    return float(slope)


# --------------------------------------------------------------------------- #
#  7. Multiscale Entropy (Costa et al., 2002)                                 #
# --------------------------------------------------------------------------- #

def multiscale_entropy(x, scales: int = 10, m: int = 2,
                       r: float | None = None) -> np.ndarray:
    r"""Multiscale Entropy MSE = { SampEn(y^{(s)}) : s = 1..S }.

    Coarse-graining at scale s averages non-overlapping blocks:

        y_j^{(s)} = (1/s) · Σ_{i=(j-1)s+1}^{j·s} x_i

    Sample entropy is computed on each coarse-grained series, with tolerance
    r fixed (by convention) to 0.15 · std of the *original* series so scales
    are comparable.

    Citation
    --------
    Costa, M., Goldberger, A.L., Peng, C.-K. (2002). "Multiscale entropy
    analysis of complex physiologic time series."
    *Phys. Rev. Lett.* 89, 068102.

    Use
    ---
    Compare complexity across temporal scales — a flat MSE curve at high
    values indicates rich, multi-scale structure (healthy markets); a
    decaying curve suggests white-noise-like behaviour. Returns a 1-D array
    of length ``scales``; entries that cannot be computed are NaN.
    """
    x = _as_1d(x)
    if scales < 1:
        raise ValueError("scales must be >= 1")
    if r is None:
        r = 0.15 * x.std(ddof=0)
    if r <= 0:
        raise ValueError("tolerance r must be positive (constant series?)")

    out = np.full(scales, np.nan)
    for s in range(1, scales + 1):
        n_blocks = x.size // s
        if n_blocks < m + 2:
            continue
        coarse = x[: n_blocks * s].reshape(n_blocks, s).mean(axis=1)
        try:
            out[s - 1] = sample_entropy(coarse, m=m, r=r)
        except ValueError:
            continue
    return out
