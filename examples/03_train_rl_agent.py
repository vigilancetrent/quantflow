"""Train a PPO agent on TradingEnv.

Requires:
    pip install quantflow[rl]

This is a tiny demo run (5,000 timesteps); useful policies need orders of
magnitude more training. The point is to show the wiring:
   data → features → env → agent → evaluate.
"""

from __future__ import annotations

import numpy as np

from quantflow.data import SyntheticLoader
from quantflow.evaluation import max_drawdown, sharpe_ratio
from quantflow.features.indicators import rsi
from quantflow.features.timeseries import log_returns, rolling_zscore
from quantflow.rl.envs import TradingEnv
from quantflow.rl.rewards import risk_adjusted_reward


def build_features(prices: np.ndarray) -> np.ndarray:
    r = np.nan_to_num(log_returns(prices), nan=0.0)
    z = np.nan_to_num(rolling_zscore(r, window=20), nan=0.0)
    rsi_ = np.nan_to_num(rsi(prices, period=14), nan=50.0) / 100.0
    return np.column_stack([r, z, rsi_])


def evaluate(env: TradingEnv, agent) -> dict:
    obs, _ = env.reset()
    rewards: list[float] = []
    equities: list[float] = [env.initial_cash]
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        equities.append(info["equity"])
        done = terminated or truncated
    eq = np.array(equities)
    rets = np.diff(eq) / eq[:-1]
    return {
        "final_equity": float(eq[-1]),
        "sharpe": sharpe_ratio(rets),
        "max_dd": max_drawdown(eq),
        "total_reward": float(sum(rewards)),
    }


def main() -> None:
    try:
        from quantflow.rl.agents import PPOAgent
    except ImportError as exc:
        print(f"RL extras not installed: {exc}")
        print("Install with: pip install quantflow[rl]")
        return

    ohlcv = SyntheticLoader(n=1_500, sigma=0.25, seed=11).load()
    prices = ohlcv.close
    features = build_features(prices)

    cut = int(0.7 * len(prices))
    train_env = TradingEnv(
        prices=prices[:cut],
        feature_matrix=features[:cut],
        window=20,
        reward_fn=risk_adjusted_reward,
    )
    test_env = TradingEnv(
        prices=prices[cut:],
        feature_matrix=features[cut:],
        window=20,
        reward_fn=risk_adjusted_reward,
    )

    print("Training PPO for 5,000 timesteps (smoke-test scale)…")
    agent = PPOAgent(env=train_env, verbose=0).train(total_timesteps=5_000)

    print("\n=== Out-of-sample evaluation ===")
    results = evaluate(test_env, agent)
    for k, v in results.items():
        print(f"  {k:>15}: {v:.4f}" if isinstance(v, float) else f"  {k:>15}: {v}")


if __name__ == "__main__":
    main()
