"""Process-wide plugin registry.

Domain extensions (new indicators, models, environments, brokers) register
themselves here without modifying core code. The registry is keyed by
`(kind, name)`, e.g. `("indicator", "rsi")`.

Usage
-----
    from quantflow.core.registry import register, resolve

    @register("indicator", "my_custom")
    def my_custom(prices, period=14):
        ...

    fn = resolve("indicator", "my_custom")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_registry: dict[tuple[str, str], Any] = {}


def register(kind: str, name: str) -> Callable[[Any], Any]:
    """Decorator that registers a factory under `(kind, name)`."""

    def _decorator(obj: Any) -> Any:
        key = (kind, name)
        if key in _registry:
            raise ValueError(f"Duplicate registration: kind={kind!r} name={name!r}")
        _registry[key] = obj
        return obj

    return _decorator


def resolve(kind: str, name: str) -> Any:
    """Look up a previously registered factory; raise KeyError if missing."""
    try:
        return _registry[(kind, name)]
    except KeyError as exc:
        available = sorted(n for k, n in _registry if k == kind)
        raise KeyError(
            f"No {kind!r} named {name!r}. Available: {available}"
        ) from exc


def registry() -> dict[tuple[str, str], Any]:
    """Return a shallow copy of the registry — for introspection only."""
    return dict(_registry)
