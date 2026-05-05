"""Backtest engine tests.

Property-based invariants that must hold for every well-formed backtest.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantflow.evaluation.backtesting import BacktestEngine


def test_no_signal_means_no_trade():
    prices = 100 + np.cumsum(np.random.default_rng(0).normal(size=200))
    prices = np.maximum(prices, 1.0)
    signals = np.zeros_like(prices)
    res = BacktestEngine(initial_cash=10_000).run(prices, signals)
    assert res.n_trades == 0
    assert np.allclose(res.equity, 10_000)
    assert np.allclose(res.cash, 10_000)


def test_constant_long_signal_grows_with_uptrend():
    """Always-long signal on a strictly-rising path produces a strictly-rising equity curve.

    Constant-fraction rebalancing introduces drag vs pure buy-and-hold (the
    engine sizes by prior-bar equity, so it sells into strength). We verify
    the directional and monotonic invariants, not pure buy-and-hold parity.
    """
    prices = np.linspace(100.0, 200.0, 20)
    signals = np.ones_like(prices)
    engine = BacktestEngine(initial_cash=1_000, commission_bps=0.0, slippage_bps=0.0)
    res = engine.run(prices, signals)
    # Equity strictly increases bar-over-bar from bar 1 onwards.
    assert (np.diff(res.equity[1:]) > 0).all()
    # Final equity is meaningfully larger than starting capital.
    assert res.equity[-1] > 1.5 * 1_000


def test_short_disabled_clips_signal():
    prices = np.linspace(100, 110, 50)
    signals = -np.ones_like(prices)
    engine = BacktestEngine(initial_cash=1_000, allow_short=False)
    res = engine.run(prices, signals)
    assert (res.positions >= 0).all()
    assert res.n_trades == 0


def test_commission_reduces_terminal_equity():
    prices = np.array([100.0, 101.0, 100.0, 101.0])
    signals = np.array([1.0, -1.0, 1.0, 0.0])
    e_no_cost = BacktestEngine(commission_bps=0.0, slippage_bps=0.0).run(prices, signals)
    e_cost = BacktestEngine(commission_bps=10.0, slippage_bps=10.0).run(prices, signals)
    assert e_cost.equity[-1] < e_no_cost.equity[-1]


def test_invalid_prices_raise():
    engine = BacktestEngine()
    with pytest.raises(ValueError):
        engine.run(prices=np.array([100.0, -1.0]), signals=np.array([1.0, 1.0]))


def test_shape_mismatch_raises():
    engine = BacktestEngine()
    with pytest.raises(ValueError):
        engine.run(prices=np.array([100.0, 101.0]), signals=np.array([1.0, 1.0, 1.0]))
