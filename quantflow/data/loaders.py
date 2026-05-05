"""Data loaders.

QuantFlow does not bundle any market data. These loaders are thin shims over
local files and a synthetic source for tests/examples. Live broker streams live
in `quantflow.data.streams`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class OHLCV:
    """Standard OHLCV container.

    All arrays are 1-D and aligned by index. ``timestamp`` is a `datetime64[ns]`
    array (UTC). ``is_active`` is True for non-delisted assets — surfacing this
    explicitly avoids silent survivorship bias.
    """

    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    symbol: str = ""
    is_active: bool = True

    def __len__(self) -> int:
        return len(self.close)


class CSVLoader:
    """Load OHLCV from a CSV with columns: timestamp, open, high, low, close, volume."""

    REQUIRED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(self, path: str | Path, symbol: str = ""):
        self.path = Path(path)
        self.symbol = symbol

    def load(self) -> OHLCV:
        df = pd.read_csv(self.path)
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return OHLCV(
            timestamp=df["timestamp"].to_numpy(),
            open=df["open"].to_numpy(dtype=np.float64),
            high=df["high"].to_numpy(dtype=np.float64),
            low=df["low"].to_numpy(dtype=np.float64),
            close=df["close"].to_numpy(dtype=np.float64),
            volume=df["volume"].to_numpy(dtype=np.float64),
            symbol=self.symbol,
        )


class SyntheticLoader:
    """Generate a synthetic geometric Brownian motion price path.

    Useful for examples, tests, and stress-testing strategies under known
    statistical properties.
    """

    def __init__(
        self,
        n: int = 1000,
        s0: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        dt: float = 1.0 / 252.0,
        seed: int | None = 0,
    ):
        self.n = n
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.seed = seed

    def load(self) -> OHLCV:
        rng = np.random.default_rng(self.seed)
        # Daily log returns ~ N((mu - 0.5 sigma^2) dt, sigma sqrt(dt))
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        vol = self.sigma * np.sqrt(self.dt)
        log_r = rng.normal(drift, vol, size=self.n - 1)
        log_p = np.concatenate([[np.log(self.s0)], np.log(self.s0) + np.cumsum(log_r)])
        close = np.exp(log_p)
        # Synthetic OHLC: spread around close.
        spread = close * vol
        high = close + np.abs(rng.normal(0, spread))
        low = close - np.abs(rng.normal(0, spread))
        open_ = np.concatenate([[close[0]], close[:-1]])
        volume = rng.integers(1_000, 10_000, size=self.n).astype(np.float64)
        ts = (np.datetime64("2020-01-01", "D") + np.arange(self.n)).astype("datetime64[ns]")
        return OHLCV(
            timestamp=ts,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            symbol="SYNTH",
        )
