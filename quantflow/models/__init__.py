"""Model interface + reference implementations.

Every model — classical, ML, deep, RL — implements `BaseModel`. Backtest and
pipeline code never branches on model type.
"""

from quantflow.models.base import BaseModel, OnlineModel

__all__ = ["BaseModel", "OnlineModel"]
