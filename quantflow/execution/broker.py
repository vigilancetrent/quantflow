"""Broker abstraction.

A `Broker` is anything that accepts orders and reports fills + positions. The
ABC is intentionally tiny so that adapters (Alpaca, IBKR, Binance, in-house)
can be plugged in without leaking vendor-specific concepts into the strategy.

`PaperBroker` is a deterministic in-memory implementation suitable for live
paper-trading and ad-hoc what-if analysis. The full backtest engine lives in
`quantflow.evaluation.backtesting` — it does not go through this interface
because vectorised backtests are an order of magnitude faster than event-driven
ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quantflow.execution.orders import Order, OrderSide, OrderStatus


class Broker(ABC):
    """Abstract broker — minimal surface area, real adapters subclass this."""

    @abstractmethod
    def submit(self, order: Order) -> Order:
        """Submit ``order`` and return the (possibly updated) order object."""

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def position(self, symbol: str) -> float:
        ...

    @abstractmethod
    def cash(self) -> float:
        ...

    def equity(self, last_prices: dict[str, float]) -> float:
        """Cash + mark-to-market across all positions."""
        total = self.cash()
        for sym, px in last_prices.items():
            total += self.position(sym) * px
        return total


class PaperBroker(Broker):
    """In-memory paper broker. Fills market orders immediately at the last price."""

    def __init__(self, initial_cash: float = 100_000.0, commission_bps: float = 1.0):
        self._cash = initial_cash
        self._positions: dict[str, float] = {}
        self._orders: dict[str, Order] = {}
        self._last_prices: dict[str, float] = {}
        self.commission_bps = commission_bps

    def update_price(self, symbol: str, price: float) -> None:
        if price <= 0:
            raise ValueError("price must be positive")
        self._last_prices[symbol] = price

    def submit(self, order: Order) -> Order:
        if order.symbol not in self._last_prices:
            order.status = OrderStatus.REJECTED
            self._orders[order.order_id] = order
            return order
        price = self._last_prices[order.symbol]
        signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        notional = abs(signed_qty) * price
        commission = notional * self.commission_bps / 10_000.0
        cost = signed_qty * price + commission
        if cost > self._cash and order.side == OrderSide.BUY:
            order.status = OrderStatus.REJECTED
            self._orders[order.order_id] = order
            return order
        self._cash -= cost
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + signed_qty
        order.filled_quantity = order.quantity
        order.avg_fill_price = price
        order.status = OrderStatus.FILLED
        self._orders[order.order_id] = order
        return order

    def cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != OrderStatus.PENDING:
            return False
        order.status = OrderStatus.CANCELLED
        return True

    def position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    def cash(self) -> float:
        return self._cash
