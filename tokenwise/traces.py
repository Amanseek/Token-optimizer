"""Trace capture: store what the app asked, what the model answered, and whether it went well.

Works for OpenAI-compatible, Gemini and Anthropic requests. Each provider's request and
response are normalised to the same shape:

  system        system prompt text
  messages      [{"role": "user"|"assistant"|"tool", "content": str, "tool_calls"?: [...]}, ...]
  tools         names of the tools the app offered
  output_text   the answer text
  tool_calls    [{"name", "args"}] the model asked for
  finish        normalised: stop | tool_calls | length | blocked | error | other

success (automatic):
  1  finished normally (stop / tool_calls) with some output
  0  HTTP error, blocked, cut off (length), empty, or retried: the same user input
     was sent again within the retry window (the earlier trace is marked 0)
feedback (from the app): POST /tokenwise/v1/feedback with the x-tokenwise-trace-id header value.

Traces contain users' prompts: personal data is masked by default (see masking.py).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any
from urllib.parse import urlparse

from .masking import mask_obj, mask_text

TRACE_HEADER = "x-tokenwise-trace-id"
GOOD_FINISH = {"stop", "tool_calls"}

# Hosts of providers whose terms restrict training competing models on their outputs.
PROPRIETARY_HOSTS = ("api.openai.com", "generativelanguage.googleapis.com", "api.anthropic.com",
                     "aiplatform.googleapis.com")


def new_trace_id() -> str:
    return uuid.uuid4().hex


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""


def is_proprietary(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in PROPRIETARY_HOSTS)


# ------------------------------------------------------------------ normalising inputs

def _text_of(content) -> str:
    """Text from an OpenAI/Anthropic content value (string or list of parts)."""
    if isinstance(content, str):
        return content
    out = []
    for p in content or []:
        if isinstance(p, dict):
            if p.get("type") in (None, "text", "input_text", "output_text") and isinstance(p.get("text"), str):
                out.append(p["text"])
            elif p.get("type") == "tool_result":
                out.append(_text_of(p.get("content")))
    return "".join(out)


def _gemini_parts_text(parts) -> str:
    return "".join(p.get("text", "") for p in parts or [] if isinstance(p, dict) and not p.get("thought"))


def normalise_input(provider: str, body: dict) -> dict:
    system, msgs, tools = "", [], []
    if provider == "openai":
        sys_parts = []
        for m in body.get("messages") or []:
            role = m.get("role")
            if role in ("system", "developer"):
                sys_parts.append(_text_of(m.get("content")))
                continue
            entry = {"role": "assistant" if role == "assistant" else ("tool" if role == "tool" else "user"),
                     "content": _text_of(m.get("content"))}
            if m.get("tool_calls"):
                entry["tool_calls"] = [{"name": (c.get("function") or {}).get("name"),
                                        "args": (c.get("function") or {}).get("arguments")} for c in m["tool_calls"]]
            msgs.append(entry)
        system = "\n".join(x for x in sys_parts if x)
        tools = [(t.get("function") or {}).get("name") for t in body.get("tools") or [] if isinstance(t, dict)]
    elif provider == "anthropic":
        system = _text_of(body.get("system"))
        for m in body.get("messages") or []:
            content = m.get("content")
            entry = {"role": "assistant" if m.get("role") == "assistant" else "user", "content": _text_of(content)}
            calls = [{"name": b.get("name"), "args": b.get("input")} for b in content or []
                     if isinstance(content, list) and isinstance(b, dict) and b.get("type") == "tool_use"]
            if calls:
                entry["tool_calls"] = calls
            msgs.append(entry)
        tools = [t.get("name") for t in body.get("tools") or [] if isinstance(t, dict)]
    else:  # gemini
        si = body.get("systemInstruction") or body.get("system_instruction") or {}
        system = _gemini_parts_text(si.get("parts") if isinstance(si, dict) else None)
        for c in body.get("contents") or []:
            parts = c.get("parts") or []
            entry = {"role": "assistant" if c.get("role") == "model" else "user", "content": _gemini_parts_text(parts)}
            calls = [{"name": (p.get("functionCall") or {}).get("name"), "args": (p.get("functionCall") or {}).get("args")}
                     for p in parts if isinstance(p, dict) and p.get("functionCall")]
            if calls:
                entry["tool_calls"] = calls
            msgs.append(entry)
        for t in body.get("tools") or []:
            for fd in (t or {}).get("functionDeclarations") or (t or {}).get("function_declarations") or []:
                tools.append(fd.get("name"))
    last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user" and m["content"]), "")
    return {"system": system, "messages": msgs, "tools": [t for t in tools if t], "last_user": last_user}


# ------------------------------------------------------------------ normalising outputs

def _finish(provider: str, raw) -> str | None:
    if raw is None:
        return None
    r = str(raw).lower()
    table = {
        "openai": {"stop": "stop", "tool_calls": "tool_calls", "function_call": "tool_calls", "length": "length",
                   "content_filter": "blocked"},
        "anthropic": {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
                      "refusal": "blocked", "pause_turn": "other"},
        "gemini": {"stop": "stop", "max_tokens": "length", "safety": "blocked", "recitation": "blocked",
                   "blocklist": "blocked", "prohibited_content": "blocked", "spii": "blocked",
                   "malformed_function_call": "error", "finish_reason_unspecified": "stop"},
    }[provider]
    return table.get(r, "other")


class OutputCollector:
    """Builds the answer from a whole response or from streamed chunks/events."""

    def __init__(self, provider: str):
        self.provider = provider
        self.text: list[str] = []
        self.calls: dict[Any, dict] = {}
        self.finish_raw = None

    # whole JSON responses, or one parsed stream chunk/event
    def add(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        p = self.provider
        if p == "openai":
            for ch in (data.get("choices") or [])[:1]:
                part = ch.get("message") or ch.get("delta") or {}
                if isinstance(part.get("content"), str):
                    self.text.append(part["content"])
                for i, tc in enumerate(part.get("tool_calls") or []):
                    key = tc.get("index", i)
                    slot = self.calls.setdefault(key, {"name": "", "args": ""})
                    fn = tc.get("function") or {}
                    slot["name"] = slot["name"] or fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
                self.finish_raw = ch.get("finish_reason") or self.finish_raw
        elif p == "anthropic":
            if data.get("type") == "message" or "content" in data and "type" not in data:
                for i, b in enumerate(data.get("content") or []):
                    if b.get("type") == "text":
                        self.text.append(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        self.calls[i] = {"name": b.get("name"), "args": b.get("input")}
                self.finish_raw = data.get("stop_reason") or self.finish_raw
            elif data.get("type") == "content_block_start":
                b = data.get("content_block") or {}
                if b.get("type") == "tool_use":
                    self.calls[data.get("index")] = {"name": b.get("name"), "args": ""}
            elif data.get("type") == "content_block_delta":
                d = data.get("delta") or {}
                if d.get("type") == "text_delta":
                    self.text.append(d.get("text", ""))
                elif d.get("type") == "input_json_delta" and data.get("index") in self.calls:
                    self.calls[data["index"]]["args"] += d.get("partial_json", "")
            elif data.get("type") == "message_delta":
                self.finish_raw = (data.get("delta") or {}).get("stop_reason") or self.finish_raw
        else:  # gemini
            for cand in (data.get("candidates") or [])[:1]:
                for part in (cand.get("content") or {}).get("parts") or []:
                    if not isinstance(part, dict) or part.get("thought"):
                        continue
                    if "text" in part:
                        self.text.append(part["text"])
                    fc = part.get("functionCall")
                    if fc:
                        self.calls[len(self.calls)] = {"name": fc.get("name"), "args": fc.get("args") or {}}
                self.finish_raw = cand.get("finishReason") or self.finish_raw

    def result(self) -> dict:
        calls = []
        for c in self.calls.values():
            args = c.get("args")
            if isinstance(args, str) and args:
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            calls.append({"name": c.get("name"), "args": args})
        return {"text": "".join(self.text), "tool_calls": calls, "finish": _finish(self.provider, self.finish_raw)}


def sse_events(buf: str) -> tuple[list[dict], str]:
    """Parse complete server-sent events from buf; returns (parsed data objects, leftover)."""
    out = []
    buf = buf.replace("\r\n", "\n")
    while "\n\n" in buf:
        event, buf = buf.split("\n\n", 1)
        for line in event.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        out.append(json.loads(payload))
                    except ValueError:
                        pass
    return out, buf


# ------------------------------------------------------------------ building the stored row

def auto_success(status: int, out: dict) -> tuple[int, str]:
    if status != 200:
        return 0, f"http {status}"
    if out["finish"] not in GOOD_FINISH and out["finish"] is not None:
        return 0, f"finish {out['finish']}"
    if not out["text"].strip() and not out["tool_calls"]:
        return 0, "empty output"
    return 1, "ok"


def workflow_of(feature: str | None, system: str) -> tuple[str, str]:
    """Group traces by the app's feature label, or by its system prompt when there's no label."""
    if feature:
        return f"f:{feature}", feature
    if system.strip():
        norm = " ".join(system.split()).lower()[:4000]
        label = " ".join(system.split())[:60]
        return "s:" + hashlib.sha256(norm.encode()).hexdigest()[:12], label
    return "none", "(no system prompt)"


def build_row(*, trace_id: str, provider: str, upstream_url: str, client_id: str | None, who: tuple,
              model: str, status: int, stream: bool, body: dict | None, out: dict, mask: bool,
              max_chars: int, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0,
              started: float | None = None) -> dict[str, Any]:
    engineer, _team, feature = who
    inp = normalise_input(provider, body or {})
    ok, reason = auto_success(status, out)
    wf_key, wf_label = workflow_of(feature, inp["system"])
    messages, last_user, text, calls = inp["messages"], inp["last_user"], out["text"], out["tool_calls"]
    if mask:  # the system prompt is the client's own instructions, so it's kept as is
        messages, last_user, text, calls = mask_obj(messages), mask_text(last_user), mask_text(text), mask_obj(calls)

    def cap(s: str) -> str:
        return s if len(s) <= max_chars else s[:max_chars]

    def cap_json(obj) -> str:
        s = json.dumps(obj, ensure_ascii=False)
        return s if len(s) <= max_chars else json.dumps([{"truncated": True, "chars": len(s)}])

    return {
        "id": trace_id, "ts": time.time(), "client_id": client_id, "provider": provider,
        "upstream_host": host_of(upstream_url), "engineer": engineer, "feature": feature,
        "workflow_key": wf_key, "workflow_label": wf_label, "model": model, "status": status, "stream": int(stream),
        "system": cap(inp["system"]), "messages": cap_json(messages), "tools": json.dumps(inp["tools"]),
        "last_user": cap(last_user),
        "input_hash": hashlib.sha256(" ".join(inp["last_user"].lower().split()).encode()).hexdigest()[:16]
        if inp["last_user"] else "",
        "output_text": cap(text), "tool_calls": cap_json(calls), "finish": out["finish"],
        "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd,
        "latency_ms": int((time.time() - started) * 1000) if started else None,
        "success": ok, "reason": reason,
    }


def save(store, row: dict, retry_window_s: int) -> None:
    """Store a trace; if the same user input came recently, mark the earlier answer(s) as not good."""
    if row["input_hash"]:
        store.execute(
            "UPDATE traces SET success = 0, reason = 'retried' WHERE input_hash = ? AND workflow_key = ? "
            "AND COALESCE(client_id,'') = COALESCE(?,'') AND engineer = ? AND ts >= ? AND success = 1",
            (row["input_hash"], row["workflow_key"], row["client_id"], row["engineer"], row["ts"] - retry_window_s),
        )
    store.insert("traces", row)


def tool_sequence(tool_calls_json: str) -> str:
    try:
        names = [c.get("name") for c in json.loads(tool_calls_json or "[]") if isinstance(c, dict)]
    except ValueError:
        return ""
    return " > ".join(n for n in names if n) or "(no tools)"
