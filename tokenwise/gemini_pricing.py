"""Gemini prices and cost calculation from `usageMetadata`.

USD per million tokens, paid tier, from public sources in Nov 2026. Some
entries are approximate; verify against https://ai.google.dev/pricing and
override with TOKENWISE_GEMINI_PRICES_JSON.

Gemini reports tokens per modality (TEXT, AUDIO, IMAGE, VIDEO), and audio
costs several times more than text, so prices are kept per modality.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeminiPrice:
    # Input price per modality; missing modalities fall back to "TEXT".
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    # Cached input tokens cost this fraction of the normal input price.
    cache_ratio: float = 0.1


def _p(inp: dict, out: dict, cache_ratio: float = 0.1) -> GeminiPrice:
    return GeminiPrice(input=inp, output=out, cache_ratio=cache_ratio)


# Matched by longest prefix of the model id.
DEFAULT_GEMINI_PRICES: dict[str, GeminiPrice] = {
    "gemini-3.1-flash-lite": _p({"TEXT": 0.25, "AUDIO": 0.50}, {"TEXT": 1.50}),
    "gemini-3.1-flash-live": _p({"TEXT": 0.75, "AUDIO": 3.00, "IMAGE": 1.00, "VIDEO": 1.00},
                                {"TEXT": 4.50, "AUDIO": 12.00}),
    "gemini-2.5-flash-native-audio": _p({"TEXT": 0.50, "AUDIO": 3.00, "IMAGE": 3.00, "VIDEO": 3.00},
                                        {"TEXT": 2.00, "AUDIO": 12.00}),
    "gemini-2.5-flash-preview-tts": _p({"TEXT": 0.50}, {"TEXT": 10.00, "AUDIO": 10.00}),
    "gemini-2.5-flash-image": _p({"TEXT": 0.30, "IMAGE": 0.30}, {"TEXT": 2.50, "IMAGE": 30.00}),
    "gemini-2.5-flash-lite": _p({"TEXT": 0.10, "AUDIO": 0.30}, {"TEXT": 0.40}),
    "gemini-2.5-flash": _p({"TEXT": 0.30, "AUDIO": 1.00}, {"TEXT": 2.50}),
    "gemini-2.5-pro": _p({"TEXT": 1.25}, {"TEXT": 10.00}),
    "gemini-embedding": _p({"TEXT": 0.15}, {"TEXT": 0.0}),
}


def _load() -> dict[str, GeminiPrice]:
    prices = dict(DEFAULT_GEMINI_PRICES)
    raw = os.getenv("TOKENWISE_GEMINI_PRICES_JSON")
    if raw:
        for k, v in json.loads(raw).items():
            prices[k] = GeminiPrice(**v)
    return prices


GEMINI_PRICES = _load()


def gemini_price_for(model: str) -> GeminiPrice | None:
    model = model.split("/")[-1]  # accept "models/gemini-..."
    best = None
    for prefix in GEMINI_PRICES:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return GEMINI_PRICES[best] if best else None


@dataclass
class GeminiTokens:
    input_tokens: int = 0
    output_tokens: int = 0       # includes thinking tokens
    cached_tokens: int = 0
    audio_tokens: int = 0        # input + output audio, for reporting

    def add(self, other: "GeminiTokens") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.audio_tokens += other.audio_tokens


def _by_modality(details) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in details or []:
        mod = (d.get("modality") or "TEXT").upper()
        out[mod] = out.get(mod, 0) + int(d.get("tokenCount") or 0)
    return out


def _rate(table: dict, modality: str) -> float:
    return table.get(modality, table.get("TEXT", 0.0))


def gemini_cost(model: str, um: dict | None) -> tuple[float, GeminiTokens]:
    """Cost in USD for one usageMetadata block, plus a token summary."""
    um = um or {}
    prompt = int(um.get("promptTokenCount") or 0)
    tool_prompt = int(um.get("toolUsePromptTokenCount") or 0)
    # generateContent uses candidatesTokenCount; the Live API uses responseTokenCount.
    out_n = int(um.get("candidatesTokenCount") or um.get("responseTokenCount") or 0)
    thoughts = int(um.get("thoughtsTokenCount") or 0)
    cached = int(um.get("cachedContentTokenCount") or 0)

    in_mod = _by_modality(um.get("promptTokensDetails")) or {"TEXT": prompt}
    out_mod = (
        _by_modality(um.get("candidatesTokensDetails") or um.get("responseTokensDetails")) or {"TEXT": out_n}
    )
    cache_mod = _by_modality(um.get("cacheTokensDetails")) or ({"TEXT": cached} if cached else {})

    tokens = GeminiTokens(
        input_tokens=prompt + tool_prompt,
        output_tokens=out_n + thoughts,
        cached_tokens=cached,
        audio_tokens=in_mod.get("AUDIO", 0) + out_mod.get("AUDIO", 0),
    )
    p = gemini_price_for(model)
    if p is None:
        return 0.0, tokens

    total = 0.0
    for mod, n in in_mod.items():
        total += n * _rate(p.input, mod)
    total += tool_prompt * _rate(p.input, "TEXT")
    # Cached tokens are included in the prompt count; refund the discounted part.
    for mod, n in cache_mod.items():
        total -= n * _rate(p.input, mod) * (1 - p.cache_ratio)
    for mod, n in out_mod.items():
        total += n * _rate(p.output, mod)
    total += thoughts * _rate(p.output, "TEXT")
    return max(total, 0.0) / 1_000_000, tokens
