"""Settings, loaded from environment variables (all optional)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if val is None:
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


@dataclass
class Settings:
    # Where requests are forwarded.
    upstream_url: str = field(default_factory=lambda: os.getenv("TOKENWISE_UPSTREAM", "https://api.anthropic.com"))
    db_path: str = field(default_factory=lambda: os.getenv("TOKENWISE_DB", "tokenwise.db"))
    # Header used to tag requests with an engineer/team name. Stripped before forwarding.
    user_header: str = field(default_factory=lambda: os.getenv("TOKENWISE_USER_HEADER", "x-tokenwise-user"))
    team_header: str = field(default_factory=lambda: os.getenv("TOKENWISE_TEAM_HEADER", "x-tokenwise-team"))
    # Optional label for which app feature made the call (e.g. "tarot", "voice").
    feature_header: str = field(default_factory=lambda: os.getenv("TOKENWISE_FEATURE_HEADER", "x-tokenwise-feature"))
    # Send this header with any value to skip the response cache for one request.
    nocache_header: str = "x-tokenwise-no-cache"

    # Gemini (Google AI Studio API). Point the google-genai SDK here with
    # GOOGLE_GEMINI_BASE_URL=http://<tokenwise-host>:<port>
    gemini_upstream_url: str = field(
        default_factory=lambda: os.getenv("TOKENWISE_GEMINI_UPSTREAM", "https://generativelanguage.googleapis.com")
    )

    # OpenAI-compatible APIs (OpenAI, OpenRouter, Together, Groq, DeepSeek, vLLM, ...).
    # Apps set their base URL to http://<tokenwise-host>:<port>/openai/v1
    openai_upstream_url: str = field(
        default_factory=lambda: os.getenv("TOKENWISE_OPENAI_UPSTREAM", "https://api.openai.com/v1")
    )

    # --- Multi-client gateway ---
    # Reject requests that don't carry a tw- client key (otherwise provider keys pass through).
    require_client_key: bool = field(default_factory=lambda: _bool("TOKENWISE_REQUIRE_CLIENT_KEY", False))
    # Protects /dashboard and /api/* when set (send ?token=... or header x-tokenwise-admin).
    admin_token: str = field(default_factory=lambda: os.getenv("TOKENWISE_ADMIN_TOKEN", ""))

    # --- Trace capture (full prompts and answers, for training) ---
    # Client accounts have their own capture switch; this one covers traffic without a client key.
    traces_enabled: bool = field(default_factory=lambda: _bool("TOKENWISE_TRACES", False))
    traces_mask: bool = field(default_factory=lambda: _bool("TOKENWISE_TRACES_MASK", True))
    # The same user input again within this many seconds marks the earlier answer as not good.
    retry_window_s: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_RETRY_WINDOW", "600")))
    trace_max_chars: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_TRACE_MAX_CHARS", "200000")))

    # --- Response cache (exact repeats of a non-streaming request) ---
    response_cache_enabled: bool = field(default_factory=lambda: _bool("TOKENWISE_RESPONSE_CACHE", False))
    response_cache_ttl: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_RESPONSE_CACHE_TTL", str(7 * 86400))))
    # Models whose names contain any of these are never cached (you usually want a fresh image each time).
    response_cache_exclude: list[str] = field(
        default_factory=lambda: _list("TOKENWISE_RESPONSE_CACHE_EXCLUDE", ["image", "tts"])
    )

    # --- Caching ---
    cache_enabled: bool = field(default_factory=lambda: _bool("TOKENWISE_CACHE", True))
    # "5m" (default) or "1h". 1h writes cost more but survive long pauses.
    cache_ttl: str = field(default_factory=lambda: os.getenv("TOKENWISE_CACHE_TTL", "5m"))

    # --- Tool-output trimming ---
    trim_enabled: bool = field(default_factory=lambda: _bool("TOKENWISE_TRIM", True))
    # Only results of these tools are trimmed. Never add file-reading tools here:
    # agents need exact file contents to make edits.
    trim_tools: list[str] = field(default_factory=lambda: _list("TOKENWISE_TRIM_TOOLS", ["Bash", "BashOutput"]))
    trim_max_chars: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_TRIM_MAX_CHARS", "6000")))
    trim_head_lines: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_TRIM_HEAD", "40")))
    trim_tail_lines: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_TRIM_TAIL", "60")))
    trim_max_error_lines: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_TRIM_ERRORS", "60")))

    # --- Duplicate tool output removal ---
    dedupe_enabled: bool = field(default_factory=lambda: _bool("TOKENWISE_DEDUPE", True))
    dedupe_min_chars: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_DEDUPE_MIN_CHARS", "2000")))

    # --- Routing ---
    # off: never route. shadow: log what would be routed, change nothing. on: route.
    routing_mode: str = field(default_factory=lambda: os.getenv("TOKENWISE_ROUTING", "shadow"))
    cheap_model: str = field(default_factory=lambda: os.getenv("TOKENWISE_CHEAP_MODEL", "claude-haiku-5-5"))
    # Model name prefixes that are eligible to be routed down.
    routable_prefixes: list[str] = field(
        default_factory=lambda: _list("TOKENWISE_ROUTABLE", ["claude-opus", "claude-sonnet", "claude-fable"])
    )
    route_max_input_tokens: int = field(default_factory=lambda: int(os.getenv("TOKENWISE_ROUTE_MAX_TOKENS", "30000")))
    # Cascade: for non-streaming routed requests, retry on the original model if the cheap one fails.
    cascade_enabled: bool = field(default_factory=lambda: _bool("TOKENWISE_CASCADE", True))


settings = Settings()
