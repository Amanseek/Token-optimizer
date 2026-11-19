"""The proxy server.

Anthropic clients (e.g. Claude Code): set ANTHROPIC_BASE_URL to this server.
Gemini clients (google-genai SDK): set GOOGLE_GEMINI_BASE_URL to this server.

Requests are optimized, forwarded, and logged. Anything not handled
explicitly is passed through unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import dashboard, gateway, gemini, openai_compat, routing, selfhost, traces, workflows
from .common import cache_key, cacheable, response_headers
from .config import Settings, settings as default_settings
from .db import Store
from .pricing import Usage, cost, price_for
from .transforms import TransformStats, optimize

log = logging.getLogger("tokenwise")


class SSEUsage:
    """Pulls usage numbers out of an Anthropic server-sent-event stream as it passes through."""

    def __init__(self, on_event=None):
        self.buf = ""
        self.on_event = on_event
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
                if self.on_event:
                    self.on_event(data)
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
    gemini_transport: httpx.AsyncBaseTransport | None = None,
    openai_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    s = s or default_settings
    store = store or Store(s.db_path)
    timeout = httpx.Timeout(connect=15.0, read=600.0, write=120.0, pool=15.0)
    # Honour SSL_CERT_FILE (custom CA bundles, corporate proxies, self-signed test upstreams).
    verify = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"]) if os.environ.get("SSL_CERT_FILE") else True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(base_url=s.upstream_url, transport=transport, timeout=timeout, verify=verify)
        app.state.gemini_client = httpx.AsyncClient(
            base_url=s.gemini_upstream_url, transport=gemini_transport or transport, timeout=timeout, verify=verify
        )
        app.state.openai_client = httpx.AsyncClient(transport=openai_transport or transport, timeout=timeout,
                                                     verify=verify)

        async def purge_loop():
            while True:
                try:
                    n = await asyncio.to_thread(traces.purge_expired, store, s.trace_retention_days)
                    if n:
                        log.info("deleted %d traces past their retention period", n)
                except Exception:
                    log.exception("trace purge failed")
                await asyncio.sleep(3600)

        purger = asyncio.create_task(purge_loop())

        async def uptime_loop(sid: str):  # GPU hours, for the own-GPU cost on the dashboard
            while True:
                await asyncio.sleep(60)
                try:
                    await asyncio.to_thread(selfhost.beat, store, sid)
                except Exception:
                    log.exception("uptime update failed")

        sid = await asyncio.to_thread(selfhost.start_session, store) if selfhost.gpu_cost_per_hour() else None
        ticker = asyncio.create_task(uptime_loop(sid)) if sid else None
        yield
        if ticker:
            ticker.cancel()
            await asyncio.to_thread(selfhost.beat, store, sid)
        purger.cancel()
        await app.state.client.aclose()
        await app.state.gemini_client.aclose()
        await app.state.openai_client.aclose()

    app = FastAPI(title="Tokenwise", lifespan=lifespan)
    app.state.store = store
    app.state.settings = s

    async def log_row(row: dict) -> None:
        try:
            await asyncio.to_thread(store.log_request, row)
        except Exception:  # logging must never break a request
            log.exception("failed to log request")

    app.state.log_row = log_row

    def _row(who, conv, decision, model_used, stream, status, started, usage, stats, route_label):
        engineer, team, feature = who
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
            "provider": "anthropic",
            "engineer": engineer,
            "team": team,
            "feature": feature,
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

    # ================================================================ Anthropic

    @app.post("/v1/messages")
    async def messages(request: Request):
        started = time.time()
        client: httpx.AsyncClient = app.state.client
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": "invalid JSON"}}, 400)

        ctx = gateway.prepare(request, "anthropic", s, store)
        if isinstance(ctx, Response):
            return ctx
        who, headers, params = ctx.who, ctx.headers, ctx.params
        stream = bool(body.get("stream"))
        collector = traces.OutputCollector("anthropic")

        # ---- exact-repeat response cache (non-streaming only)
        rkey = None
        if not stream and cacheable(request, s, body.get("model") or ""):
            rkey = cache_key(request, "anthropic", "/v1/messages", body)
            hit = await asyncio.to_thread(store.cache_get, rkey, s.response_cache_ttl)
            if hit:
                await log_row({
                    "provider": "anthropic", "engineer": who[0], "team": who[1], "feature": who[2],
                    "model_requested": body.get("model"), "model_used": hit["model"], "route_decision": "cache-hit",
                    "stream": 0, "status": 200, "latency_ms": int((time.time() - started) * 1000),
                    "cost_usd": 0.0, "baseline_cost_usd": hit["cost_usd"],
                })
                return Response(content=hit["body"], status_code=200, media_type=hit["content_type"],
                                headers={"x-tokenwise-cache": "hit", "x-tokenwise-model": hit["model"]})

        opt_body, stats = optimize(body, s)
        conv = routing.conversation_key(body, who[0])
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
            if resp.status_code == 200:
                data = resp.json()
                usage = Usage.from_api(data.get("usage"))
                why = _cascade_needed(data) if (decision.apply and s.cascade_enabled and route_label == "routed") else None
                if why:
                    # Log the cheap attempt (it was paid for), then retry on the original model.
                    await log_row(_row(who, conv, decision, model_used, False, 200, started, usage, stats, "cascade-attempt"))
                    await asyncio.to_thread(routing.mark_failed, conv, store, why)
                    model_used, route_label = decision.original_model, "cascaded"
                    resp = await client.post("/v1/messages", content=body_for(model_used), headers=headers, params=params)
                    data = resp.json() if resp.status_code == 200 else {}
                    usage = Usage.from_api(data.get("usage"))
            row = _row(who, conv, decision, model_used, False, resp.status_code, started, usage, stats, route_label)
            await log_row(row)
            if resp.status_code == 200:
                try:
                    collector.add(resp.json())
                except ValueError:
                    pass
            await gateway.record(ctx, store, s, model=model_used, status=resp.status_code, stream=False, body=body,
                                 collector=collector, input_tokens=usage.input_tokens + usage.cache_read_input_tokens
                                 + usage.cache_creation_input_tokens, output_tokens=usage.output_tokens,
                                 cost_usd=row["cost_usd"], started=started)
            if rkey and resp.status_code == 200:
                await asyncio.to_thread(store.cache_put, rkey, resp.content,
                                        resp.headers.get("content-type", "application/json"), model_used, row["baseline_cost_usd"])
            out_headers = response_headers(resp)
            out_headers["x-tokenwise-model"] = model_used  # which model actually answered
            out_headers.update(gateway.trace_headers(ctx))
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
            await log_row(_row(who, conv, decision, model_used, True, resp.status_code, started, Usage(), stats, route_label))
            await gateway.record(ctx, store, s, model=model_used, status=resp.status_code, stream=True, body=body,
                                 collector=collector, started=started)
            return Response(content=content, status_code=resp.status_code, headers=response_headers(resp))

        tracker = SSEUsage(on_event=collector.add)
        out_headers = response_headers(resp)
        out_headers["x-tokenwise-model"] = model_used
        out_headers.update(gateway.trace_headers(ctx))

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    tracker.feed(chunk)
                    yield chunk
            finally:
                await resp.aclose()
                srow = _row(who, conv, decision, model_used, True, 200, started, tracker.usage, stats, route_label)
                await log_row(srow)
                u = tracker.usage
                await gateway.record(ctx, store, s, model=model_used, status=200, stream=True, body=body,
                                     collector=collector, input_tokens=u.input_tokens + u.cache_read_input_tokens
                                     + u.cache_creation_input_tokens, output_tokens=u.output_tokens,
                                     cost_usd=srow["cost_usd"], started=started)

        return StreamingResponse(relay(), status_code=200, headers=out_headers,
                                 media_type=resp.headers.get("content-type", "text/event-stream"))

    # ================================================================ Gemini
    gemini.register(app, s, store)
    openai_compat.register(app, s, store)
    workflows.register(app, s, store)

    def admin_ok(request: Request) -> bool:
        if not s.admin_token:
            return True
        return s.admin_token in (request.query_params.get("token"), request.headers.get("x-tokenwise-admin"))

    # ================================================================ dashboard
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dash(request: Request, days: int = 30):
        if not admin_ok(request):
            return HTMLResponse("Forbidden: add ?token=<TOKENWISE_ADMIN_TOKEN>", status_code=403)
        await asyncio.to_thread(store.audit, "admin", "view_dashboard", request.client.host if request.client else "")
        return HTMLResponse(await asyncio.to_thread(dashboard.render_html, store, days))

    @app.get("/api/summary")
    async def summary(request: Request, days: int = 30):
        if not admin_ok(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        return JSONResponse(await asyncio.to_thread(dashboard.summary, store, days))

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    # ================================================================ passthrough
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def passthrough(path: str, request: Request):
        is_gemini = gemini.is_gemini_request(request, path)
        client: httpx.AsyncClient = app.state.gemini_client if is_gemini else app.state.client
        ctx = gateway.prepare(request, "gemini" if is_gemini else "anthropic", s, store)
        if isinstance(ctx, Response):
            return ctx
        req = client.build_request(
            request.method, "/" + path, content=await request.body(), headers=ctx.headers, params=ctx.params,
        )
        resp = await client.send(req, stream=True)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(relay(), status_code=resp.status_code, headers=response_headers(resp))

    return app


__all__ = ["create_app", "TransformStats"]
