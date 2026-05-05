"""Event-driven backtest engine.

This is intentionally minimal but realistic enough to catch the common gotchas
that toy backtests miss:

- Trades execute on the *next* bar's open, not the same bar's close (avoids
  look-ahead bias)
- Commissions are applied to every fill (basis points of notional)
- Slippage is applied as a fraction of the fill price, in the direction adverse
  to the trader
- The signal series is interpreted as desired *position* in {-1, 0, +1} (long /
  flat / short), and the engine computes the trades needed to reach it

For exotic strategies (multi-asset, options, fractional sizing), subclass
`BacktestEngine` or compose several runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BacktestResult:
    equity: NDArray         # portfolio value at each bar (n,)
    returns: NDArray        # bar-over-bar portfolio returns (n,) — first is 0
    positions: NDArray      # held position at each bar (n,)
    trades: NDArray         # signed quantity traded at each bar (n,) — 0 if flat
    cash: NDArray           # cash balance at each bar (n,)
    n_trades: int


class BacktestEngine:
    """Vectorised next-bar-open backtest for a single instrument.

    Parameters
    ----------
    initial_cash : float
        Starting capital.
    commission_bps : float
        Round-trip commission in basis points of traded notional.
    slippage_bps : float
        Adverse fill slippage in basis points of price.
    allow_short : bool
        If False, signals < 0 are clipped to 0.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 1.0,
        allow_short: bool = True,
    ):
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("commission and slippage must be non-negative")
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.allow_short = allow_short

    def run(self, prices: NDArray, signals: NDArray) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        prices : np.ndarray of shape (n,)
            Close prices (used as next-bar fill price for the trade signaled at
            t-1; the last bar marks-to-market only).
        signals : np.ndarray of shape (n,)
            Desired position in {-1, 0, +1} at each bar. Values outside that set
            are clipped.
        """
        p = np.asarray(prices, dtype=np.float64)
        s = np.asarray(signals, dtype=np.float64)
        if p.shape != s.shape:
            raise ValueError("prices and signals must have the same shape")
        if p.ndim != 1:
            raise ValueError("prices must be 1-D")
        if (p <= 0).any():
            raise ValueError("prices must be strictly positive")

        s = np.clip(s, -1.0, 1.0)
        if not self.allow_short:
            s = np.maximum(s, 0.0)

        n = len(p)
        equity = np.zeros(n)
        cash = np.zeros(n)
        positions = np.zeros(n)        # held shares
        trades = np.zeros(n)           # signed shares traded *this* bar
        target_pos_state = 0.0         # target position carried over from prev signal

        cash[0] = self.initial_cash
        equity[0] = self.initial_cash
        n_trades = 0

        comm_rate = self.commission_bps / 10_000.0
        slip_rate = self.slippage_bps / 10_000.0

        for t in range(1, n):
            # Trade decision was made on the close of bar t-1; fill at bar t open
            # which we approximate with bar t price.
            fill_price = p[t]
            # Convert target signal to share count: full notional sizing.
            # Target shares = sign * (current_equity / fill_price)
            current_equity = cash[t - 1] + positions[t - 1] * p[t - 1]
            target_pos_state = s[t - 1]
            target_shares = target_pos_state * current_equity / fill_price
            delta = target_shares - positions[t - 1]

            if delta != 0:
                direction = np.sign(delta)
                exec_price = fill_price * (1.0 + direction * slip_rate)
                notional = abs(delta) * exec_price
                commission = notional * comm_rate
                cash[t] = cash[t - 1] - delta * exec_price - commission
                positions[t] = positions[t - 1] + delta
                trades[t] = delta
                n_trades += 1
            else:
                cash[t] = cash[t - 1]
                positions[t] = positions[t - 1]

            equity[t] = cash[t] + positions[t] * p[t]

        returns = np.zeros(n)
        returns[1:] = np.where(equity[:-1] > 0, equity[1:] / equity[:-1] - 1.0, 0.0)

        return BacktestResult(
            equity=equity,
            returns=returns,
            positions=positions,
            trades=trades,
            cash=cash,
            n_trades=n_trades,
        )
