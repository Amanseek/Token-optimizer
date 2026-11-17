"""Price table and cost calculation.

Prices are USD per million tokens, taken from public sources in Nov 2026.
Verify against Anthropic's pricing page before relying on them; override with
TOKENWISE_PRICES_JSON if needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


# Matched by longest prefix of the model id.
DEFAULT_PRICES: dict[str, Price] = {
    "claude-fable-5": Price(10.0, 50.0, 12.5, 20.0, 0.25),
    "claude-opus-5-5": Price(4.0, 20.0, 5.0, 8.0, 0.20),
    "claude-opus": Price(5.0, 25.0, 6.25, 10.0, 0.50),
    "claude-sonnet-5-5": Price(2.0, 10.0, 2.5, 4.0, 0.10),
    "claude-sonnet": Price(2.0, 10.0, 2.5, 4.0, 0.20),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.125, 0.20, 0.01),
    "claude-haiku": Price(1.0, 5.0, 1.25, 2.0, 0.10),
}


def _load_prices() -> dict[str, Price]:
    raw = os.getenv("TOKENWISE_PRICES_JSON")
    if not raw:
        return dict(DEFAULT_PRICES)
    data = json.loads(raw)
    prices = dict(DEFAULT_PRICES)
    for k, v in data.items():
        prices[k] = Price(**v)
    return prices


PRICES = _load_prices()


def price_for(model: str) -> Price | None:
    best = None
    for prefix in PRICES:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return PRICES[best] if best else None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_1h_tokens: int = 0  # subset of cache_creation billed at the 1h rate

    @classmethod
    def from_api(cls, usage: dict | None) -> "Usage":
        usage = usage or {}
        u = cls(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        )
        detail = usage.get("cache_creation") or {}
        u.cache_creation_1h_tokens = int(detail.get("ephemeral_1h_input_tokens") or 0)
        return u

    def merge(self, other: dict | None) -> None:
        """Merge a partial usage dict (e.g. from a streaming message_delta)."""
        if not other:
            return
        new = Usage.from_api(other)
        for name in vars(self):
            val = getattr(new, name)
            if val:
                setattr(self, name, val)


def cost(model: str, usage: Usage) -> float:
    p = price_for(model)
    if p is None:
        return 0.0
    write_1h = usage.cache_creation_1h_tokens
    write_5m = max(usage.cache_creation_input_tokens - write_1h, 0)
    total = (
        usage.input_tokens * p.input
        + usage.output_tokens * p.output
        + write_5m * p.cache_write_5m
        + write_1h * p.cache_write_1h
        + usage.cache_read_input_tokens * p.cache_read
    )
    return total / 1_000_000
