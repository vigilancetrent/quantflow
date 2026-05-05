"""Indicator correctness tests.

We use both hand-computed expected values and known invariants. The hand-
computed cases catch off-by-one errors; the invariants catch broader bugs.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantflow.features.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)


def test_sma_known_values():
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(p, period=3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert np.isclose(out[2], 2.0)
    assert np.isclose(out[3], 3.0)
    assert np.isclose(out[4], 4.0)


def test_sma_period_validation():
    with pytest.raises(ValueError):
        sma(np.array([1.0, 2.0, 3.0]), period=0)


def test_ema_first_valid_equals_sma():
    p = np.arange(1, 11, dtype=float)
    e = ema(p, period=5)
    assert np.isclose(e[4], p[:5].mean())


def test_rsi_bounds():
    rng = np.random.default_rng(0)
    p = 100 + np.cumsum(rng.normal(size=500))
    p = np.maximum(p, 1.0)  # keep positive
    out = rsi(p, period=14)
    valid = out[~np.isnan(out)]
    assert np.all(valid >= 0) and np.all(valid <= 100)


def test_rsi_all_up_is_100():
    p = np.arange(1, 31, dtype=float)
    out = rsi(p, period=14)
    assert np.isclose(out[-1], 100.0)


def test_macd_shapes():
    p = 100 + np.cumsum(np.random.default_rng(1).normal(size=200))
    p = np.maximum(p, 1.0)
    res = macd(p)
    assert res.macd.shape == res.signal.shape == res.histogram.shape == p.shape


def test_macd_fast_must_be_less_than_slow():
    with pytest.raises(ValueError):
        macd(np.arange(50, dtype=float), fast=26, slow=12)


def test_bollinger_band_ordering():
    p = 100 + np.cumsum(np.random.default_rng(2).normal(size=200))
    bb = bollinger_bands(p, period=20, num_std=2.0)
    valid = ~np.isnan(bb.middle)
    assert np.all(bb.upper[valid] >= bb.middle[valid])
    assert np.all(bb.middle[valid] >= bb.lower[valid])


def test_atr_non_negative():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(size=200))
    high = close + np.abs(rng.normal(size=200))
    low = close - np.abs(rng.normal(size=200))
    a = atr(high, low, close, period=14)
    valid = a[~np.isnan(a)]
    assert np.all(valid >= 0)
