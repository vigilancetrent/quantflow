"""RL agents — thin wrappers over Stable-Baselines3 by default.

Any Gymnasium-compatible agent works with `TradingEnv`. The wrappers here
exist only so users can write `from quantflow.rl.agents import PPO` instead of
copy-pasting boilerplate, and so the rest of the library has a single import
path it can mock in tests.
"""

from quantflow.rl.agents.ppo_agent import PPOAgent

__all__ = ["PPOAgent"]
