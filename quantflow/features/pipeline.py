"""Compose multiple feature transforms into a single matrix.

The `FeaturePipeline` collects named feature functions and applies them to a
price (or OHLCV) array, stacking the results into a 2-D feature matrix. It is
deliberately framework-free — the output is a NumPy array that any downstream
model can consume.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FeatureSpec:
    name: str
    fn: Callable[..., NDArray]
    kwargs: dict


class FeaturePipeline:
    """Apply a list of feature functions and stack outputs column-wise."""

    def __init__(self, specs: list[FeatureSpec] | None = None):
        self._specs: list[FeatureSpec] = list(specs) if specs else []

    def add(self, name: str, fn: Callable[..., NDArray], **kwargs) -> FeaturePipeline:
        self._specs.append(FeatureSpec(name=name, fn=fn, kwargs=kwargs))
        return self

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def transform(self, prices: NDArray) -> NDArray:
        """Apply every spec to ``prices``; return shape (n_obs, n_features)."""
        cols: list[NDArray] = []
        for spec in self._specs:
            out = spec.fn(prices, **spec.kwargs)
            arr = np.asarray(out, dtype=np.float64)
            if arr.ndim == 0:
                raise ValueError(f"Feature {spec.name!r} returned a scalar")
            if arr.ndim == 1:
                cols.append(arr.reshape(-1, 1))
            else:
                # Some indicators (MACD, Bollinger) return tuples — handled by caller
                # passing a wrapper that picks one component.
                cols.append(arr)
        return np.hstack(cols)
