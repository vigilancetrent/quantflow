"""Walk-forward validated momentum strategy.

Demonstrates the right way to do time-series CV in finance: train on the past,
test on the immediate future, walk forward. Random k-fold leaks future-into-
past and is unsafe.

Run:
    python examples/02_backtest_strategy.py
"""

from __future__ import annotations

import numpy as np

from quantflow.data import SyntheticLoader
from quantflow.data.preprocessing import walk_forward_split
from quantflow.evaluation import BacktestEngine, max_drawdown, sharpe_ratio
from quantflow.features.timeseries import returns, rolling_zscore


def momentum_signal(prices: np.ndarray, lookback: int = 20, threshold: float = 1.0) -> np.ndarray:
    """Z-scored momentum: long when z > threshold, short when z < -threshold."""
    r = returns(prices)
    z = rolling_zscore(r, window=lookback)
    sig = np.where(z > threshold, 1.0, np.where(z < -threshold, -1.0, 0.0))
    sig = np.nan_to_num(sig, nan=0.0)
    return sig


def main() -> None:
    ohlcv = SyntheticLoader(n=2_000, sigma=0.3, seed=7).load()
    prices = ohlcv.close

    folds = list(walk_forward_split(n=len(prices), n_splits=5))
    print(f"Walk-forward CV: {len(folds)} folds")
    print("-" * 70)

    sharpes = []
    for k, (train_idx, test_idx) in enumerate(folds, start=1):
        # In a real ML pipeline we'd fit a model on train and freeze it for test.
        # The momentum strategy has no parameters to fit, so we just evaluate it
        # out-of-sample on the test slice.
        prices_test = prices[test_idx[0] - 20 : test_idx[-1] + 1]  # warmup buffer
        offset = test_idx[0] - (test_idx[0] - 20)
        sig = momentum_signal(prices_test)
        sig[:offset] = 0  # zero out warmup region
        engine = BacktestEngine(initial_cash=10_000, commission_bps=2.0)
        result = engine.run(prices=prices_test, signals=sig)
        s = sharpe_ratio(result.returns)
        mdd = max_drawdown(result.equity)
        sharpes.append(s)
        print(
            f"  Fold {k}: train=[{train_idx[0]}:{train_idx[-1]}]"
            f"  test=[{test_idx[0]}:{test_idx[-1]}]"
            f"  Sharpe={s:+.2f}  MaxDD={mdd:+.1%}  Trades={result.n_trades}"
        )

    print("-" * 70)
    print(f"Mean OOS Sharpe: {np.mean(sharpes):+.3f}")
    print(f"Sharpe std-dev:  {np.std(sharpes):.3f}")
    print("(Sharpe instability across folds => strategy is regime-sensitive.)")


if __name__ == "__main__":
    main()
