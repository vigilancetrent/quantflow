"""Execution layer: brokers, orders, live trading.

The `Broker` ABC abstracts away whether you're paper-trading, hitting Alpaca,
or simulating in a backtest. Strategies talk to a single interface; the
adapter is swapped at runtime.
"""

from quantflow.execution.broker import Broker, PaperBroker
from quantflow.execution.orders import Order, OrderSide, OrderStatus, OrderType

__all__ = [
    "Broker",
    "PaperBroker",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
]
