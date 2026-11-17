"""Request transforms that cut tokens without changing what the model needs.

Every transform is deterministic: the same conversation history always comes
out the same way. That matters because Anthropic's prompt cache matches on an
exact prefix. If a transform produced different output for an old message on
the next request, it would break the cache and cost more, not less.

Only user-side tool_result blocks are ever modified. Assistant messages
(which may carry signed thinking blocks) are never touched.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass

from .config import Settings

ERROR_RE = re.compile(
    r"(error|exception|traceback|failed|failure|fatal|panic|assert|warning|denied|not found|cannot|undefined)",
    re.IGNORECASE,
)
TRIM_MARKER = "[tokenwise: {n} lines trimmed]"
DEDUPE_MARKER = "[tokenwise: output identical to the earlier result of tool call {ref}; see that result above]"


@dataclass
class TransformStats:
    trimmed_chars: int = 0
    deduped_chars: int = 0
    cache_breakpoints_added: int = 0


# ---------------------------------------------------------------- helpers


def _tool_names(messages: list[dict]) -> dict[str, str]:
    """Map tool_use id -> tool name from assistant messages."""
    names: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names[block.get("id", "")] = block.get("name", "")
    return names


def _tool_results(messages: list[dict]):
    """Yield every tool_result block in user messages, in order."""
    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                yield block


def _get_text(block: dict) -> str | None:
    """Text of a tool_result if it is plain text only, else None."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(
        isinstance(c, dict) and c.get("type") == "text" for c in content
    ):
        return "\n".join(c.get("text", "") for c in content)
    return None


def _set_text(block: dict, text: str) -> None:
    if isinstance(block.get("content"), str):
        block["content"] = text
    else:
        # Keep any cache_control that was on the last text block.
        cc = None
        if isinstance(block.get("content"), list) and block["content"]:
            cc = block["content"][-1].get("cache_control")
        new = {"type": "text", "text": text}
        if cc:
            new["cache_control"] = cc
        block["content"] = [new]


# ---------------------------------------------------------------- trimming


def trim_text(text: str, s: Settings) -> str:
    """Keep the head, error-looking lines, and the tail of long output."""
    if len(text) <= s.trim_max_chars:
        return text
    lines = text.splitlines()
    head_n, tail_n = s.trim_head_lines, s.trim_tail_lines
    if len(lines) <= head_n + tail_n:
        # Few but very long lines: fall back to character trimming.
        half = s.trim_max_chars // 2
        removed = len(text) - 2 * half
        return text[:half] + f"\n[tokenwise: {removed} characters trimmed]\n" + text[-half:]

    head = lines[:head_n]
    tail = lines[-tail_n:]
    middle = lines[head_n:-tail_n]
    errors = [ln for ln in middle if ERROR_RE.search(ln)][: s.trim_max_error_lines]
    removed = len(middle) - len(errors)
    out = head + [TRIM_MARKER.format(n=removed)]
    if errors:
        out += ["[tokenwise: error/warning lines kept from trimmed section:]"] + errors
    out += tail
    return "\n".join(out)


def trim_tool_outputs(messages: list[dict], s: Settings, stats: TransformStats) -> None:
    names = _tool_names(messages)
    allowed = set(s.trim_tools)
    for block in _tool_results(messages):
        if names.get(block.get("tool_use_id", "")) not in allowed:
            continue
        text = _get_text(block)
        if text is None:
            continue
        new = trim_text(text, s)
        if new != text:
            stats.trimmed_chars += len(text) - len(new)
            _set_text(block, new)


# ---------------------------------------------------------------- dedupe


def dedupe_tool_outputs(messages: list[dict], s: Settings, stats: TransformStats) -> None:
    """Replace a tool result that exactly repeats an earlier one with a short pointer.

    Only the later copy is replaced, so earlier messages (the cached prefix)
    never change.
    """
    seen: dict[str, str] = {}
    for block in _tool_results(messages):
        text = _get_text(block)
        if text is None or len(text) < s.dedupe_min_chars or block.get("is_error"):
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            new = DEDUPE_MARKER.format(ref=seen[digest])
            stats.deduped_chars += len(text) - len(new)
            _set_text(block, new)
        else:
            seen[digest] = block.get("tool_use_id", "?")


# ---------------------------------------------------------------- caching


def _has_cc(blocks) -> bool:
    return isinstance(blocks, list) and any(isinstance(b, dict) and "cache_control" in b for b in blocks)


def _cc(s: Settings) -> dict:
    cc = {"type": "ephemeral"}
    if s.cache_ttl == "1h":
        cc["ttl"] = "1h"
    return cc


def _count_breakpoints(body: dict) -> int:
    n = 0
    for t in body.get("tools") or []:
        n += "cache_control" in t
    if isinstance(body.get("system"), list):
        n += sum("cache_control" in b for b in body["system"] if isinstance(b, dict))
    for m in body.get("messages") or []:
        if isinstance(m.get("content"), list):
            n += sum("cache_control" in b for b in m["content"] if isinstance(b, dict))
    return n


def _upgrade_ttl(body: dict) -> None:
    """Set every breakpoint to 1h so ordering rules (longer TTL first) always hold."""

    def fix(block):
        if isinstance(block, dict) and isinstance(block.get("cache_control"), dict):
            block["cache_control"]["ttl"] = "1h"

    for t in body.get("tools") or []:
        fix(t)
    if isinstance(body.get("system"), list):
        for b in body["system"]:
            fix(b)
    for m in body.get("messages") or []:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                fix(b)
                # tool_result blocks can hold nested text blocks with cache_control
                if isinstance(b, dict) and isinstance(b.get("content"), list):
                    for c in b["content"]:
                        fix(c)


def add_cache_breakpoints(body: dict, s: Settings, stats: TransformStats) -> None:
    """Add prompt-cache breakpoints where the client did not set any.

    Anthropic allows at most 4 breakpoints per request.
    """
    budget = 4 - _count_breakpoints(body)

    tools = body.get("tools")
    if budget > 0 and tools and not any("cache_control" in t for t in tools):
        tools[-1]["cache_control"] = _cc(s)
        stats.cache_breakpoints_added += 1
        budget -= 1

    system = body.get("system")
    if budget > 0 and system:
        if isinstance(system, str):
            body["system"] = system = [{"type": "text", "text": system}]
        if isinstance(system, list) and not _has_cc(system):
            system[-1]["cache_control"] = _cc(s)
            stats.cache_breakpoints_added += 1
            budget -= 1

    messages = body.get("messages") or []
    if budget > 0 and messages and not any(_has_cc(m.get("content")) for m in messages):
        last = messages[-1]
        if last.get("role") == "user":
            if isinstance(last.get("content"), str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            content = last.get("content")
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                if content[-1].get("type") in {"text", "image", "document", "tool_result"}:
                    content[-1]["cache_control"] = _cc(s)
                    stats.cache_breakpoints_added += 1

    if s.cache_ttl == "1h":
        _upgrade_ttl(body)


# ---------------------------------------------------------------- entry point


def optimize(body: dict, s: Settings) -> tuple[dict, TransformStats]:
    """Return an optimized copy of a /v1/messages request body."""
    body = copy.deepcopy(body)
    stats = TransformStats()
    messages = body.get("messages") or []
    if s.dedupe_enabled:
        dedupe_tool_outputs(messages, s, stats)
    if s.trim_enabled:
        trim_tool_outputs(messages, s, stats)
    if s.cache_enabled:
        add_cache_breakpoints(body, s, stats)
    return body, stats
