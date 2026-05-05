"""PPO wrapper.

Stable-Baselines3's PPO is a strong default for the trading env (discrete
action space, on-policy, robust to noisy rewards). This wrapper exposes a
minimal `train` / `predict` / `save` / `load` surface so the rest of the
codebase doesn't depend on SB3's exact API surface.

If `stable-baselines3` is not installed the import raises with a helpful
message — use `pip install quantflow[rl]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3 import PPO
    _HAS_SB3 = True
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore[assignment]
    _HAS_SB3 = False


def _require_sb3() -> None:
    if not _HAS_SB3:
        raise ImportError(
            "stable-baselines3 is not installed. "
            "Install RL extras with `pip install quantflow[rl]`."
        )


class PPOAgent:
    """Thin wrapper over `stable_baselines3.PPO`.

    Parameters
    ----------
    env : gymnasium.Env
    policy : str
        SB3 policy identifier ("MlpPolicy" by default).
    **kwargs
        Forwarded to `PPO(...)`.
    """

    def __init__(self, env: Any, policy: str = "MlpPolicy", **kwargs: Any):
        _require_sb3()
        self.model = PPO(policy, env, **kwargs)

    def train(self, total_timesteps: int = 10_000, **kwargs: Any) -> PPOAgent:
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return self

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env: Any | None = None) -> PPOAgent:
        _require_sb3()
        instance = cls.__new__(cls)
        instance.model = PPO.load(str(path), env=env)
        return instance
