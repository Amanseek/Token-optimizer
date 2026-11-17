"""Rule-based model routing.

A conversation is classified once, on its first request, and the decision is
stored. Every later request in the same conversation uses the same model.
Switching models mid-conversation would throw away the prompt cache (caches
are per model) and could confuse the agent.

Modes:
  off    - never route
  shadow - decide and log what *would* happen, but send the original model
  on     - actually send simple conversations to the cheap model
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .config import Settings
from .db import Store

# Signals that a task is complex. Any match keeps the original model.
HARD_RE = re.compile(
    r"\b(debug|refactor|architect\w*|design|migrat\w*|race condition|deadlock|concurren\w*|"
    r"performance|optimi[sz]e|security|vulnerab\w*|implement|build|failing test|root cause|"
    r"multi-?file|across the (code|repo)|rewrite|port\b|upgrade)\b",
    re.IGNORECASE,
)
# Signals that a task is simple.
SIMPLE_RE = re.compile(
    r"\b(explain|what does|what is|summari[sz]e|rename|typo|spelling|format|lint|docstring|"
    r"comment|add (a )?log|translate|title|commit message|changelog|readme|list (the )?files|"
    r"which file|where is)\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    mode: str                 # off | shadow | on
    original_model: str
    target_model: str | None  # cheap model if the conversation qualifies, else None
    reason: str

    @property
    def apply(self) -> bool:
        return self.mode == "on" and self.target_model is not None

    @property
    def model_to_send(self) -> str:
        return self.target_model if self.apply else self.original_model


def estimate_tokens(body: dict) -> int:
    parts = [body.get("system"), body.get("messages"), body.get("tools")]
    return len(json.dumps(parts, default=str)) // 4


def _first_user_text(body: dict) -> str:
    for msg in body.get("messages") or []:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            return "\n".join(texts)
    return ""


def _last_user_text(body: dict) -> str:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                return "\n".join(texts)
    return ""


def _has_media(body: dict) -> bool:
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") in {"image", "document"}:
                return True
    return False


def conversation_key(body: dict, engineer: str) -> str:
    seed = json.dumps([engineer, body.get("model"), _first_user_text(body)[:4000]])
    return hashlib.sha256(seed.encode()).hexdigest()[:24]


def classify(body: dict, s: Settings) -> tuple[str | None, str]:
    """Return (target_model or None, reason) for a fresh conversation."""
    model = body.get("model") or ""
    if not any(model.startswith(p) for p in s.routable_prefixes):
        return None, "model not routable"
    if model.startswith(s.cheap_model):
        return None, "already cheap model"
    if _has_media(body):
        return None, "has images/documents"
    if estimate_tokens(body) > s.route_max_input_tokens:
        return None, "context too large"
    task = _last_user_text(body)
    if HARD_RE.search(task):
        return None, "complex-task keywords"
    if not body.get("tools") and len(task) < 2000:
        return s.cheap_model, "short request without tools"
    if SIMPLE_RE.search(task):
        return s.cheap_model, "simple-task keywords"
    return None, "no simple-task signal"


def decide(body: dict, conversation: str, s: Settings, store: Store) -> RouteDecision:
    original = body.get("model") or ""
    mode = s.routing_mode if s.routing_mode in {"off", "shadow", "on"} else "off"
    if mode == "off":
        return RouteDecision(mode, original, None, "routing off")
    found, target, reason = store.get_route(conversation)
    if not found:
        target, reason = classify(body, s)
        store.save_route(conversation, target, reason)
    return RouteDecision(mode, original, target, reason or "")


def mark_failed(conversation: str, store: Store, why: str) -> None:
    """Stop routing a conversation after the cheap model failed on it."""
    store.save_route(conversation, None, f"cheap model failed: {why}")
