"""The proxy server.

Point an Anthropic client (e.g. Claude Code) at this server instead of
api.anthropic.com. Requests to /v1/messages are optimized, forwarded, and
logged. Everything else is passed through unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import dashboard, routing
from .config import Settings, settings as default_settings
from .db import Store
from .pricing import Usage, cost, price_for
from .transforms import optimize

log = logging.getLogger("tokenwise")

HOP_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "accept-encoding", "proxy-authorization", "proxy-connection", "upgrade", "te",
}
DROP_RESPONSE_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def _identity(request: Request, s: Settings) -> tuple[str, str | None]:
    engineer = request.headers.get(s.user_header)
    team = request.headers.get(s.team_header)
    if not engineer:
        key = request.headers.get("x-api-key") or request.headers.get("authorization") or ""
        engineer = "key-" + hashlib.sha256(key.encode()).hexdigest()[:8] if key else "unknown"
    return engineer, team


def _forward_headers(request: Request, s: Settings) -> dict[str, str]:
    skip = HOP_HEADERS | {s.user_header, s.team_header}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    headers["accept-encoding"] = "identity"
    return headers


def _response_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in DROP_RESPONSE_HEADERS}


class SSEUsage:
    """Pulls usage numbers out of a server-sent-event stream as it passes through."""

    def __init__(self):
        self.buf = ""
        self.usage = Usage()
        self.model: str | None = None
        self.stop_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk.decode("utf-8", errors="replace")
        while "\n\n" in self.buf:
            event, self.buf = self.buf.split("\n\n", 1)
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                except ValueError:
                    continue
                kind = data.get("type")
                if kind == "message_start":
                    msg = data.get("message") or {}
                    self.model = msg.get("model")
                    self.usage.merge(msg.get("usage"))
                elif kind == "message_delta":
                    self.usage.merge(data.get("usage"))
                    self.stop_reason = (data.get("delta") or {}).get("stop_reason") or self.stop_reason


def _saved_cost(model: str, saved_chars: int) -> float:
    """Conservative value of removed characters: priced at the cache-read rate."""
    p = price_for(model)
    if p is None or saved_chars <= 0:
        return 0.0
    return (saved_chars / 4) * p.cache_read / 1_000_000


def _cascade_needed(data: dict) -> str | None:
    """Should a non-streaming cheap-model answer be retried on the original model?"""
    content = data.get("content") or []
    if not content:
        return "empty response"
    has_tool = any(b.get("type") == "tool_use" for b in content if isinstance(b, dict))
    if data.get("stop_reason") == "max_tokens" and not has_tool:
        return "hit max_tokens"
    return None


def create_app(
    s: Settings | None = None,
    store: Store | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    s = s or default_settings
    store = store or Store(s.db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=s.upstream_url,
            transport=transport,
            timeout=httpx.Timeout(connect=15.0, read=600.0, write=120.0, pool=15.0),
        )
        yield
        await app.state.client.aclose()

    app = FastAPI(title="Tokenwise", lifespan=lifespan)
    app.state.store = store
    app.state.settings = s

    async def _log(row: dict) -> None:
        try:
            await asyncio.to_thread(store.log_request, row)
        except Exception:  # logging must never break a request
            log.exception("failed to log request")

    def _row(engineer, team, conv, decision, model_used, stream, status, started, usage, stats, route_label):
        requested = decision.original_model
        actual = cost(model_used, usage)
        saved = stats.trimmed_chars + stats.deduped_chars
        baseline = cost(requested, usage) + _saved_cost(requested, saved)
        if route_label == "cascade-attempt":
            baseline = 0.0  # the request's baseline is counted once, on the retry row
        shadow_cost = None
        if decision.mode == "shadow" and decision.target_model:
            shadow_cost = cost(decision.target_model, usage)
        return {
            "engineer": engineer,
            "team": team,
            "conversation": conv,
            "model_requested": requested,
            "model_used": model_used,
            "route_decision": route_label,
            "stream": int(bool(stream)),
            "status": status,
            "latency_ms": int((time.time() - started) * 1000),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_tokens": usage.cache_creation_input_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "cost_usd": actual,
            "baseline_cost_usd": baseline,
            "trimmed_chars": stats.trimmed_chars,
            "deduped_chars": stats.deduped_chars,
            "cache_breakpoints_added": stats.cache_breakpoints_added,
            "shadow_model": decision.target_model if decision.mode == "shadow" else None,
            "shadow_cost_usd": shadow_cost,
        }

    @app.post("/v1/messages")
    async def messages(request: Request):
        started = time.time()
        client: httpx.AsyncClient = app.state.client
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": "invalid JSON"}}, 400)

        engineer, team = _identity(request, s)
        headers = _forward_headers(request, s)
        params = dict(request.query_params)
        stream = bool(body.get("stream"))

        opt_body, stats = optimize(body, s)
        conv = routing.conversation_key(body, engineer)
        decision = await asyncio.to_thread(routing.decide, opt_body, conv, s, store)

        route_label = "none"
        if decision.target_model:
            route_label = "routed" if decision.apply else ("shadow" if decision.mode == "shadow" else "none")

        def body_for(model: str) -> bytes:
            return json.dumps({**opt_body, "model": model}).encode()

        model_used = decision.model_to_send

        # ---------------- non-streaming
        if not stream:
            resp = await client.post("/v1/messages", content=body_for(model_used), headers=headers, params=params)
            if decision.apply and resp.status_code in (400, 404):
                # Cheap model rejected the request (unsupported feature etc.): fall back.
                await asyncio.to_thread(routing.mark_failed, conv, store, f"HTTP {resp.status_code}")
                model_used, route_label = decision.original_model, "fallback"
                resp = await client.post("/v1/messages", content=body_for(model_used), headers=headers, params=params)
            usage = Usage()
            data = None
            if resp.status_code == 200:
                data = resp.json()
                usage = Usage.from_api(data.get("usage"))
                why = _cascade_needed(data) if (decision.apply and s.cascade_enabled and route_label == "routed") else None
                if why:
                    # Log the cheap attempt (it was paid for), then retry on the original model.
                    await _log(_row(engineer, team, conv, decision, model_used, False, 200, started, usage, stats, "cascade-attempt"))
                    await asyncio.to_thread(routing.mark_failed, conv, store, why)
                    model_used, route_label = decision.original_model, "cascaded"
                    resp = await client.post("/v1/messages", content=body_for(model_used), headers=headers, params=params)
                    data = resp.json() if resp.status_code == 200 else None
                    usage = Usage.from_api((data or {}).get("usage"))
            await _log(_row(engineer, team, conv, decision, model_used, False, resp.status_code, started, usage, stats, route_label))
            out_headers = _response_headers(resp)
            out_headers["x-tokenwise-model"] = model_used  # which model actually answered
            return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)

        # ---------------- streaming
        async def open_stream(model: str) -> httpx.Response:
            req = client.build_request("POST", "/v1/messages", content=body_for(model), headers=headers, params=params)
            return await client.send(req, stream=True)

        resp = await open_stream(model_used)
        if decision.apply and resp.status_code in (400, 404):
            await resp.aread()
            await resp.aclose()
            await asyncio.to_thread(routing.mark_failed, conv, store, f"HTTP {resp.status_code}")
            model_used, route_label = decision.original_model, "fallback"
            resp = await open_stream(model_used)

        if resp.status_code != 200:
            content = await resp.aread()
            await resp.aclose()
            await _log(_row(engineer, team, conv, decision, model_used, True, resp.status_code, started, Usage(), stats, route_label))
            return Response(content=content, status_code=resp.status_code, headers=_response_headers(resp))

        tracker = SSEUsage()
        out_headers = _response_headers(resp)
        out_headers["x-tokenwise-model"] = model_used
        final = {"model": model_used, "label": route_label}

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    tracker.feed(chunk)
                    yield chunk
            finally:
                await resp.aclose()
                await _log(_row(engineer, team, conv, decision, final["model"], True, 200, started,
                                tracker.usage, stats, final["label"]))

        return StreamingResponse(relay(), status_code=200, headers=out_headers,
                                 media_type=resp.headers.get("content-type", "text/event-stream"))

    # ---------------- dashboard
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dash(days: int = 30):
        return HTMLResponse(await asyncio.to_thread(dashboard.render_html, store, days))

    @app.get("/api/summary")
    async def summary(days: int = 30):
        return JSONResponse(await asyncio.to_thread(dashboard.summary, store, days))

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    # ---------------- passthrough for everything else (count_tokens, models, ...)
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def passthrough(path: str, request: Request):
        client: httpx.AsyncClient = app.state.client
        req = client.build_request(
            request.method, "/" + path, content=await request.body(),
            headers=_forward_headers(request, s), params=dict(request.query_params),
        )
        resp = await client.send(req, stream=True)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(relay(), status_code=resp.status_code, headers=_response_headers(resp))

    return app
