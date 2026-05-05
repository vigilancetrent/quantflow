"""Pluggable reward functions for the trading environment.

A reward function takes a `RewardContext` and returns a float. The env passes
in everything it knows about the most recent step, and the function decides
what to optimise. This makes it trivial to A/B different reward shapings
without editing the env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardContext:
    prev_equity: float
    curr_equity: float
    position: float           # signed shares held after the action
    action_taken: int         # raw action index
    transaction_cost: float   # cost incurred this step
    done: bool                # is this the terminal step?
    realised_returns: list[float]  # series of per-step portfolio returns


def pnl_reward(ctx: RewardContext) -> float:
    """Step PnL net of costs.

    The simplest dense reward — directly maps to "make money". Vulnerable to
    high-leverage policies that take huge swings, so pair with a risk
    constraint or use `risk_adjusted_reward`.
    """
    return ctx.curr_equity - ctx.prev_equity - ctx.transaction_cost


def risk_adjusted_reward(ctx: RewardContext, vol_window: int = 20, eps: float = 1e-6) -> float:
    """PnL divided by trailing realised volatility — Sharpe-like at the step level.

    Computed only when ``vol_window`` returns are available; before that, falls
    back to raw PnL so the agent gets a learning signal from the start.
    """
    pnl = ctx.curr_equity - ctx.prev_equity - ctx.transaction_cost
    if len(ctx.realised_returns) < vol_window:
        return pnl
    window = np.asarray(ctx.realised_returns[-vol_window:], dtype=np.float64)
    sd = window.std(ddof=1)
    if sd < eps:
        return pnl
    return float(pnl / (sd * ctx.prev_equity + eps))


def sparse_terminal_reward(ctx: RewardContext) -> float:
    """Zero everywhere except on the terminal step, where it equals total return.

    Sparse rewards are realistic but make learning harder — usually only
    suitable for short episodes or paired with intrinsic motivation.
    """
    if not ctx.done:
        return 0.0
    if ctx.prev_equity == 0:
        return 0.0
    return float((ctx.curr_equity - ctx.prev_equity) / ctx.prev_equity)
