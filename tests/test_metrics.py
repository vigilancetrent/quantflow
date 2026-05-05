"""Metric tests covering correctness, edge cases, and invariants."""

from __future__ import annotations

import numpy as np

from quantflow.evaluation.metrics import (
    directional_accuracy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from quantflow.evaluation.risk import cvar, value_at_risk


def test_sharpe_zero_variance_returns_zero():
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01])) == 0.0


def test_sharpe_empty_returns_zero():
    assert sharpe_ratio(np.array([])) == 0.0


def test_sharpe_scale_invariant_under_positive_constant():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=500)
    s1 = sharpe_ratio(r)
    s2 = sharpe_ratio(2.5 * r)
    assert np.isclose(s1, s2, atol=1e-9)


def test_sortino_only_penalises_downside():
    # All-positive returns → Sortino is +inf (no downside).
    r = np.array([0.01, 0.02, 0.005, 0.015])
    assert sortino_ratio(r) == float("inf")


def test_max_drawdown_known_path():
    eq = np.array([100, 120, 90, 130, 60])  # peak 130, trough 60 → -53.85%
    assert np.isclose(max_drawdown(eq), (60 - 130) / 130)


def test_max_drawdown_monotone_increasing_is_zero():
    assert max_drawdown(np.arange(1, 100, dtype=float)) == 0.0


def test_win_rate_basic():
    assert win_rate(np.array([1.0, -1.0, 1.0, 1.0])) == 0.75


def test_profit_factor_basic():
    # gains 3, losses 1 → 3.0
    assert profit_factor(np.array([1.0, 2.0, -1.0])) == 3.0


def test_directional_accuracy_basic():
    yt = np.array([1.0, -1.0, 1.0, 1.0])
    yp = np.array([1.0, -1.0, -1.0, 1.0])
    assert directional_accuracy(yt, yp) == 0.75


def test_var_historic_is_a_quantile():
    rng = np.random.default_rng(0)
    r = rng.normal(size=10_000)
    var = value_at_risk(r, alpha=0.05, method="historic")
    assert np.isclose(var, np.quantile(r, 0.05), atol=1e-9)


def test_cvar_lower_or_equal_to_var():
    rng = np.random.default_rng(0)
    r = rng.normal(size=5_000)
    var = value_at_risk(r, alpha=0.05)
    es = cvar(r, alpha=0.05)
    assert es <= var
