"""Structured logging helper.

Live trading code should log everything: signals, orders, fills, errors —
with timestamps and an optional correlation ID. This helper standardises the
format so downstream log aggregators (Loki, Datadog, ELK) can parse it.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def get_logger(name: str = "quantflow", level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger; idempotent — repeat calls reuse the same handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
