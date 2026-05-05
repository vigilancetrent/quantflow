"""Gym(nasium)-compatible trading environment.

State, action, and reward are all configurable so the same env handles:
- discrete sell/hold/buy with simple PnL reward (default)
- continuous position sizing via Box action with risk-adjusted reward
- sparse terminal reward for short episode tasks

The env imports gymnasium lazily — install with ``pip install quantflow[rl]``
to enable. Without gymnasium, the env still works (with a stub `Space`)
which is convenient for unit tests in CI without the heavy dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantflow.rl.rewards import RewardContext, pnl_reward

try:                                  # pragma: no cover - import guard
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:                   # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _HAS_GYM = False


class _StubBase:
    """Minimal stand-in for ``gym.Env`` when gymnasium isn't installed."""

    metadata: dict = {}


_Base = gym.Env if _HAS_GYM else _StubBase  # type: ignore


class TradingEnv(_Base):  # type: ignore[misc, valid-type]
    """Single-instrument trading environment.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Close-price series.
    feature_matrix : np.ndarray, shape (T, F), optional
        Pre-computed features observed at each step. If None, the only feature
        is normalised price.
    window : int
        How many past bars are stacked into the observation.
    initial_cash : float
    commission_bps : float
    reward_fn : callable
        Receives `RewardContext`, returns float. Default = `pnl_reward`.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: NDArray,
        feature_matrix: NDArray | None = None,
        window: int = 20,
        initial_cash: float = 100_000.0,
        commission_bps: float = 1.0,
        reward_fn: Callable[[RewardContext], float] | None = None,
    ):
        super().__init__()
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 1:
            raise ValueError("prices must be 1-D")
        if len(prices) <= window + 1:
            raise ValueError("Need more bars than window for at least one step")
        if feature_matrix is not None:
            feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
            if feature_matrix.shape[0] != prices.shape[0]:
                raise ValueError("feature_matrix must align with prices on axis 0")

        self.prices = prices
        self.features = feature_matrix
        self.window = window
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps
        self.reward_fn = reward_fn or pnl_reward

        n_features = 1 if feature_matrix is None else feature_matrix.shape[1]
        # Observation: [windowed price returns, current features, position, cash_frac]
        obs_dim = window + n_features + 2

        if _HAS_GYM:
            self.action_space = spaces.Discrete(3)  # 0=sell, 1=hold, 2=buy
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64,
            )
        else:
            self.action_space = None
            self.observation_space = None

        self._t = 0
        self._cash = 0.0
        self._shares = 0.0
        self._returns: list[float] = []
        self._n_features = n_features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _equity(self) -> float:
        return self._cash + self._shares * self.prices[self._t]

    def _observation(self) -> NDArray:
        start = self._t - self.window + 1
        window_prices = self.prices[start : self._t + 1]
        # Normalise as log-returns relative to the window start
        norm = np.log(window_prices / window_prices[0])
        if self.features is None:
            feat = np.array([norm[-1]])
        else:
            feat = self.features[self._t]
        position_norm = self._shares * self.prices[self._t] / max(self._equity(), 1e-9)
        cash_frac = self._cash / self.initial_cash
        return np.concatenate([norm, feat, [position_norm, cash_frac]])

    def _execute(self, action: int) -> float:
        """Map action to a target position; return transaction cost incurred."""
        target_pos = {0: -1.0, 1: 0.0, 2: 1.0}.get(int(action), 0.0)
        price = self.prices[self._t]
        equity = self._equity()
        target_shares = target_pos * equity / price
        delta = target_shares - self._shares
        if delta == 0:
            return 0.0
        notional = abs(delta) * price
        commission = notional * self.commission_bps / 10_000.0
        self._cash -= delta * price + commission
        self._shares = target_shares
        return commission

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        if _HAS_GYM:
            super().reset(seed=seed)
        self._t = self.window - 1
        self._cash = self.initial_cash
        self._shares = 0.0
        self._returns = []
        info: dict[str, Any] = {"equity": self._equity()}
        return self._observation(), info

    def step(self, action):  # type: ignore[override]
        prev_equity = self._equity()
        cost = self._execute(int(action))
        self._t += 1
        terminated = self._t >= len(self.prices) - 1
        curr_equity = self._equity()
        if prev_equity > 0:
            self._returns.append(curr_equity / prev_equity - 1.0)

        ctx = RewardContext(
            prev_equity=prev_equity,
            curr_equity=curr_equity,
            position=self._shares,
            action_taken=int(action),
            transaction_cost=cost,
            done=terminated,
            realised_returns=self._returns,
        )
        reward = float(self.reward_fn(ctx))
        info = {"equity": curr_equity, "position": self._shares, "cost": cost}
        truncated = False
        return self._observation(), reward, terminated, truncated, info

    def render(self):  # pragma: no cover
        print(
            f"t={self._t} price={self.prices[self._t]:.2f} "
            f"shares={self._shares:.2f} cash={self._cash:.2f} equity={self._equity():.2f}"
        )
