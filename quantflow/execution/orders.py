"""Order primitives shared between the backtest engine and live brokers."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field


class OrderSide(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(enum.Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("quantity must be positive (use side to indicate direction)")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
