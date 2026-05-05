"""Quickstart: synthetic data → indicators → toy signal → backtest report.

Run:
    python examples/01_quickstart.py

This script has no third-party dependencies beyond NumPy and is the smoke test
for a fresh install.
"""

from __future__ import annotations

import numpy as np

from quantflow.data import SyntheticLoader
from quantflow.evaluation import BacktestEngine, max_drawdown, sharpe_ratio
from quantflow.features import bollinger_bands, macd, rsi


def main() -> None:
    ohlcv = SyntheticLoader(n=1000, sigma=0.25, seed=42).load()
    prices = ohlcv.close

    # Compute indicators
    rsi_vals = rsi(prices, period=14)
    macd_out = macd(prices)
    bb = bollinger_bands(prices, period=20, num_std=2.0)

    # Toy signal: long when RSI < 30 AND price below lower band; short when RSI > 70 AND above upper.
    long_cond = (rsi_vals < 30) & (prices < bb.lower)
    short_cond = (rsi_vals > 70) & (prices > bb.upper)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0)).astype(np.float64)

    # Replace NaN-driven false positives at the start with 0 (no position before warmup)
    signals[: max(20, 14, 26)] = 0

    engine = BacktestEngine(initial_cash=10_000, commission_bps=2.0, slippage_bps=1.0)
    result = engine.run(prices=prices, signals=signals)

    print("=" * 60)
    print(f"  QuantFlow quickstart  —  synthetic GBM, {len(prices)} bars")
    print("=" * 60)
    print(f"  Final equity:    ${result.equity[-1]:,.2f}")
    print(f"  Buy & hold:      ${10_000 * prices[-1] / prices[0]:,.2f}")
    print(f"  Sharpe ratio:    {sharpe_ratio(result.returns):.3f}")
    print(f"  Max drawdown:    {max_drawdown(result.equity):.2%}")
    print(f"  Trades executed: {result.n_trades}")
    print(f"  MACD last:       {macd_out.macd[-1]:.3f} (signal {macd_out.signal[-1]:.3f})")


if __name__ == "__main__":
    main()
