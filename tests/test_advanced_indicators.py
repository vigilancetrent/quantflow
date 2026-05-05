"""Smoke + invariant tests for the modern indicator families.

These don't try to replicate published numerical values — that would require
specific datasets — but they verify the API contracts, NaN-leading behaviour,
and known mathematical invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantflow.features.fractal import (
    dfa,
    frac_diff_ffd,
    hurst_rs,
    multiscale_entropy,
    permutation_entropy,
    sample_entropy,
)
from quantflow.features.microstructure import (
    amihud_illiquidity,
    corwin_schultz_spread,
    kyle_lambda,
    order_flow_imbalance,
    realised_quarticity,
    roll_effective_spread,
    vpin,
)
from quantflow.features.regime import (
    bipower_variation,
    cusum_change_point,
    hmm_regime_probability,
    jump_variation,
    log_returns,
    realised_semivariance,
    realised_volatility,
)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gbm_prices():
    rng = np.random.default_rng(42)
    n = 1500
    log_r = rng.normal(0.0001, 0.01, size=n - 1)
    return np.exp(np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(log_r)]))


@pytest.fixture
def gbm_volume(gbm_prices):
    rng = np.random.default_rng(43)
    return rng.integers(1_000, 10_000, size=len(gbm_prices)).astype(float)


@pytest.fixture
def gbm_returns(gbm_prices):
    return log_returns(gbm_prices)[1:]


# ---------------------------------------------------------------------------
# Microstructure
# ---------------------------------------------------------------------------

def test_kyle_lambda_shape_and_warmup(gbm_prices, gbm_volume):
    out = kyle_lambda(gbm_prices, gbm_volume, window=60)
    assert out.shape == gbm_prices.shape
    assert np.isnan(out[:59]).all()
    assert np.isfinite(out[100:]).any()


def test_amihud_non_negative(gbm_prices, gbm_volume):
    out = amihud_illiquidity(gbm_prices, gbm_volume, window=21)
    valid = out[~np.isnan(out)]
    assert (valid >= 0).all()


def test_ofi_in_range(gbm_prices, gbm_volume):
    out = order_flow_imbalance(gbm_prices, gbm_volume, window=50)
    valid = out[~np.isnan(out)]
    assert (valid >= -1.0 - 1e-9).all() and (valid <= 1.0 + 1e-9).all()


def test_roll_spread_non_negative(gbm_prices):
    out = roll_effective_spread(gbm_prices, window=60)
    valid = out[~np.isnan(out)]
    assert (valid >= 0).all()


def test_corwin_schultz_non_negative():
    rng = np.random.default_rng(0)
    n = 100
    close = 100 + np.cumsum(rng.normal(size=n))
    high = close + np.abs(rng.normal(size=n))
    low = close - np.abs(rng.normal(size=n))
    out = corwin_schultz_spread(high, low)
    valid = out[~np.isnan(out)]
    assert (valid >= 0).all()


def test_realised_quarticity_non_negative(gbm_prices):
    out = realised_quarticity(gbm_prices, window=78)
    valid = out[~np.isnan(out)]
    assert (valid >= 0).all()


def test_vpin_validates_inputs(gbm_prices, gbm_volume):
    with pytest.raises(ValueError):
        vpin(gbm_prices, np.zeros_like(gbm_volume))


# ---------------------------------------------------------------------------
# Fractal / info-theoretic
# ---------------------------------------------------------------------------

def test_hurst_white_noise_close_to_half():
    """R/S on stationary white noise yields H ≈ 0.5."""
    rng = np.random.default_rng(0)
    noise = rng.normal(size=4000)
    h = hurst_rs(noise, min_window=16, max_window=512)
    assert 0.35 < h < 0.65


def test_hurst_persistent_increments_above_half():
    """Increments with positive autocorrelation should yield H > 0.5."""
    rng = np.random.default_rng(1)
    n = 4000
    eps = rng.normal(size=n)
    # AR(1) with phi=0.6 — strongly persistent stationary series
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + eps[t]
    h = hurst_rs(x, min_window=16, max_window=512)
    assert h > 0.55


def test_frac_diff_ffd_shape():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=500))
    out = frac_diff_ffd(x, d=0.4, threshold=1e-3)
    assert out.shape == x.shape
    assert np.isnan(out[:5]).any()
    assert np.isfinite(out[-1])


def test_sample_entropy_constant_zero():
    """A constant series should have low entropy (∼0 — pre-tolerance check raises)."""
    x = np.zeros(200)
    with pytest.raises(ValueError):
        sample_entropy(x)


def test_permutation_entropy_in_unit_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    h = permutation_entropy(x, m=3, normalize=True)
    assert 0.0 <= h <= 1.0


def test_dfa_random_walk_around_one_and_a_half():
    """DFA on Brownian motion gives α ≈ 1.5; on white noise α ≈ 0.5."""
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.normal(size=4000))
    alpha_rw = dfa(rw, min_window=16, max_window=512)
    assert 1.3 < alpha_rw < 1.7
    noise = rng.normal(size=4000)
    alpha_wn = dfa(noise, min_window=16, max_window=512)
    assert 0.35 < alpha_wn < 0.65


def test_mse_returns_array_of_correct_length():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    out = multiscale_entropy(x, scales=5, m=2)
    assert out.shape == (5,)


# ---------------------------------------------------------------------------
# Regime / vol decomposition
# ---------------------------------------------------------------------------

def test_realised_vol_non_negative(gbm_returns):
    out = realised_volatility(gbm_returns, window=78)
    valid = out[~np.isnan(out)]
    assert (valid >= 0).all()


def test_bpv_non_negative(gbm_returns):
    out = bipower_variation(gbm_returns, window=78)
    valid = out[~np.isnan(out)]
    assert (valid >= 0).all()


def test_jump_variation_non_negative(gbm_returns):
    """JV is truncated to >= 0 by construction."""
    jv = jump_variation(gbm_returns, window=78)
    valid = jv[~np.isnan(jv)]
    assert (valid >= 0).all()


def test_semivariance_decomposition(gbm_returns):
    """RS+ + RS- ≈ RV² for the same window."""
    w = 78
    rs_up = realised_semivariance(gbm_returns, window=w, side="up")
    rs_dn = realised_semivariance(gbm_returns, window=w, side="down")
    rv = realised_volatility(gbm_returns, window=w)
    rv2 = rv ** 2
    valid = ~np.isnan(rv2) & ~np.isnan(rs_up) & ~np.isnan(rs_dn)
    diff = (rs_up + rs_dn - rv2)[valid]
    assert np.allclose(diff, 0.0, atol=1e-9)


def test_cusum_returns_binary(gbm_returns):
    out = cusum_change_point(gbm_returns, threshold=4.0, drift=0.5)
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_hmm_regime_probability_in_unit_interval(gbm_returns):
    out = hmm_regime_probability(gbm_returns, window=252, refit_every=42)
    valid = out[~np.isnan(out)]
    if valid.size:
        assert (valid >= 0).all() and (valid <= 1.0 + 1e-9).all()
