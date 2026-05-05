"""Abstract model interface used everywhere in QuantFlow.

The contract is intentionally narrow:

- ``fit(X, y)`` — train on a feature matrix and target vector
- ``predict(X)`` — produce point predictions
- ``save(path)`` / ``load(path)`` — persist and restore (pickle by default)

Subclasses for online learning add ``partial_fit(X, y)``.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseModel(ABC):
    """Abstract supervised model."""

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> BaseModel:
        ...

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        ...

    def predict_proba(self, X: NDArray) -> NDArray:
        """Optional override for classifiers; default raises."""
        raise NotImplementedError(f"{type(self).__name__} does not support predict_proba")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> Any:
        with Path(path).open("rb") as f:
            return pickle.load(f)


class OnlineModel(BaseModel):
    """Mix in online-learning support — used for live adaptation."""

    @abstractmethod
    def partial_fit(self, X: NDArray, y: NDArray) -> OnlineModel:
        ...


class NaiveBaseline(BaseModel):
    """Predicts the last observed target — a sanity-check baseline.

    Any "useful" model should beat this on out-of-sample data. If it doesn't,
    you have a feature/leakage problem, not a model problem.
    """

    def __init__(self):
        self._last: float = 0.0

    def fit(self, X: NDArray, y: NDArray) -> NaiveBaseline:
        y = np.asarray(y, dtype=np.float64)
        if len(y) == 0:
            raise ValueError("Cannot fit on empty target")
        self._last = float(y[-1])
        return self

    def predict(self, X: NDArray) -> NDArray:
        return np.full(len(X), self._last, dtype=np.float64)
