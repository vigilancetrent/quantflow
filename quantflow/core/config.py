"""Lightweight configuration object.

Most users will pass parameters directly to the components. `Config` exists for
end-to-end runs where dozens of knobs need to flow together (data window, model
hyperparams, broker creds, etc.) without polluting function signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    data: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    rl: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: Config) -> Config:
        """Return a new Config with ``other`` overriding self section-wise."""
        return Config(
            data={**self.data, **other.data},
            features={**self.features, **other.features},
            model={**self.model, **other.model},
            backtest={**self.backtest, **other.backtest},
            execution={**self.execution, **other.execution},
            rl={**self.rl, **other.rl},
        )
