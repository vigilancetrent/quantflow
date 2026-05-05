"""QuantFlow — modular ML/RL library for financial markets.

Top-level re-exports for the most common entry points. Submodules can also be
imported directly for finer-grained control.
"""

from quantflow.core.pipeline import Pipeline, PipelineNode
from quantflow.core.registry import register, resolve
from quantflow.evaluation.backtesting import BacktestEngine, BacktestResult
from quantflow.evaluation.metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from quantflow.features.indicators import (
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)

__version__ = "0.1.0"

__all__ = [
    "Pipeline",
    "PipelineNode",
    "register",
    "resolve",
    "BacktestEngine",
    "BacktestResult",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "__version__",
]
