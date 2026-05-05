"""Preprocessing utilities and validation splits.

The default split for time-series is `walk_forward_split`. Random k-fold leaks
future-into-past and is unsafe for finance — `train_test_split` here is a
*time-respecting* split (chronological), not sklearn's shuffled one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def drop_nan_rows(*arrays: NDArray) -> tuple[NDArray, ...]:
    """Drop rows where any of the input arrays has NaN at that index.

    All inputs must be 1-D or 2-D and share the same first dimension.
    """
    if not arrays:
        return ()
    n = len(arrays[0])
    mask = np.ones(n, dtype=bool)
    for a in arrays:
        a = np.asarray(a)
        if len(a) != n:
            raise ValueError("All inputs must share the same first dimension")
        if a.ndim == 1:
            mask &= ~np.isnan(a)
        else:
            mask &= ~np.isnan(a).any(axis=tuple(range(1, a.ndim)))
    return tuple(np.asarray(a)[mask] for a in arrays)


def train_test_split(
    X: NDArray,
    y: NDArray,
    train_frac: float = 0.7,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Time-respecting split — first ``train_frac`` for training, rest for test."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    n = len(X)
    if n != len(y):
        raise ValueError("X and y must have the same length")
    cut = int(n * train_frac)
    return X[:cut], X[cut:], y[:cut], y[cut:]


def walk_forward_split(
    n: int,
    n_splits: int = 5,
    min_train: int | None = None,
) -> Iterator[tuple[NDArray, NDArray]]:
    """Yield (train_idx, test_idx) for expanding-window walk-forward CV.

    Train window grows; test window is a fixed slice immediately after.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if min_train is None:
        min_train = n // (n_splits + 1)
    if min_train <= 0 or min_train >= n:
        raise ValueError("min_train must be in (0, n)")

    test_size = (n - min_train) // n_splits
    if test_size <= 0:
        raise ValueError("Not enough data for the requested n_splits")

    for k in range(n_splits):
        train_end = min_train + k * test_size
        test_end = train_end + test_size
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(test_end, n))
        if len(test_idx) == 0:
            break
        yield train_idx, test_idx
