"""OpenAI-compatible APIs: OpenAI, OpenRouter, Together, Groq, DeepSeek, Fireworks, vLLM, Ollama...

Point the app's OpenAI SDK at Tokenwise with no code changes:

    OPENAI_BASE_URL=http://<tokenwise-host>:8787/openai/v1
    OPENAI_API_KEY=tw-...          # client key; or the provider's own key in pass-through mode

POST /openai/v1/chat/completions is logged, priced and (optionally) traced, streamed or not.
Everything else under /openai/v1/ (models, embeddings, ...) is passed through.

Prices: model names vary by host, so set them yourself (USD per million tokens):
    TOKENWISE_OPENAI_PRICES='{"gpt-5-mini": [0.25, 2.0], "deepseek-chat": [0.27, 1.1]}'
Unknown models are logged with tokens and a cost of 0.
"""

from __future__ import annotations

import json
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import gateway, traces
from .common import response_headers
from .config import Settings
from .db import Store

PREFIX = "/openai/v1"


def _prices() -> dict[str, tuple[float, float]]:
    try:
        raw = json.loads(os.getenv("TOKENWISE_OPENAI_PRICES", "{}"))
        return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
    except (ValueError, TypeError, IndexError):
        return {}


def price(model: str, usage: dict | None) -> tuple[float, int, int, int]:
    """(cost_usd, input_tokens, output_tokens, cached_tokens) from an OpenAI usage object."""
    u = usage or {}
    inp, out = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    table = _prices()
    p = table.get(model)
    if p is None:  # longest matching prefix, e.g. "gpt-5-mini" for "gpt-5-mini-2026-08-01"
        matches = [k for k in table if model.startswith(k)]
        p = table[max(matches, key=len)] if matches else None
    if p is None:
        return 0.0, inp, out, cached
    return (inp * p[0] + out * p[1]) / 1_000_000, inp, out, cached


def register(app: FastAPI, s: Settings, store: Store) -> None:

    def log_row(ctx: gateway.Ctx, model: str, status: int, started: float, stream: bool, usage: dict | None):
        engineer, team, feature = ctx.who
        c, inp, out, cached = price(model, usage)
        return {
            "provider": "openai", "engineer": engineer, "team": team, "feature": feature,
            "model_requested": model, "model_used": model, "route_decision": "none", "stream": int(stream),
            "status": status, "latency_ms": int((time.time() - started) * 1000),
            "input_tokens": inp, "output_tokens": out, "cache_read_tokens": cached,
            "cost_usd": c, "baseline_cost_usd": c,
        }, c, inp, out

    @app.post(PREFIX + "/chat/completions")
    async def chat_completions(request: Request):
        started = time.time()
        ctx = gateway.prepare(request, "openai", s, store)
        if isinstance(ctx, Response):
            return ctx
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON", "type": "invalid_request_error"}}, 400)
        model = body.get("model") or ""
        stream = bool(body.get("stream"))
        if stream and s_stream_usage():
            # Ask for a final usage chunk so streamed calls can be priced.
            body = {**body, "stream_options": {**(body.get("stream_options") or {}), "include_usage": True}}
            raw = json.dumps(body).encode()
        client: httpx.AsyncClient = app.state.openai_client
        url = ctx.upstream_url + "/chat/completions"
        ctx.headers["content-type"] = "application/json"
        collector = traces.OutputCollector("openai")

        if not stream:
            resp = await client.post(url, content=raw, headers=ctx.headers, params=ctx.params)
            usage = None
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    usage = data.get("usage")
                    model = data.get("model") or model
                    collector.add(data)
                except ValueError:
                    pass
            row, c, inp, out = log_row(ctx, model, resp.status_code, started, False, usage)
            await app.state.log_row(row)
            await gateway.record(ctx, store, s, model=model, status=resp.status_code, stream=False, body=body,
                                 collector=collector, input_tokens=inp, output_tokens=out, cost_usd=c, started=started)
            return Response(content=resp.content, status_code=resp.status_code,
                            headers={**response_headers(resp), **gateway.trace_headers(ctx)})

        req = client.build_request("POST", url, content=raw, headers=ctx.headers, params=ctx.params)
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            content = await resp.aread()
            await resp.aclose()
            row, *_ = log_row(ctx, model, resp.status_code, started, True, None)
            await app.state.log_row(row)
            await gateway.record(ctx, store, s, model=model, status=resp.status_code, stream=True, body=body,
                                 collector=collector, started=started)
            return Response(content=content, status_code=resp.status_code, headers=response_headers(resp))

        state = {"buf": "", "usage": None, "model": model}

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    state["buf"] += chunk.decode("utf-8", errors="replace")
                    events, state["buf"] = traces.sse_events(state["buf"])
                    for ev in events:
                        collector.add(ev)
                        state["usage"] = ev.get("usage") or state["usage"]
                        state["model"] = ev.get("model") or state["model"]
                    yield chunk
            finally:
                await resp.aclose()
                events, _ = traces.sse_events(state["buf"] + "\n\n")
                for ev in events:
                    collector.add(ev)
                    state["usage"] = ev.get("usage") or state["usage"]
                row, c, inp, out = log_row(ctx, state["model"], 200, started, True, state["usage"])
                await app.state.log_row(row)
                await gateway.record(ctx, store, s, model=state["model"], status=200, stream=True, body=body,
                                     collector=collector, input_tokens=inp, output_tokens=out, cost_usd=c,
                                     started=started)

        return StreamingResponse(relay(), status_code=200,
                                 headers={**response_headers(resp), **gateway.trace_headers(ctx)},
                                 media_type=resp.headers.get("content-type", "text/event-stream"))

    @app.api_route(PREFIX + "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def openai_passthrough(path: str, request: Request):
        ctx = gateway.prepare(request, "openai", s, store)
        if isinstance(ctx, Response):
            return ctx
        client: httpx.AsyncClient = app.state.openai_client
        req = client.build_request(request.method, f"{ctx.upstream_url}/{path}", content=await request.body(),
                                   headers=ctx.headers, params=ctx.params)
        resp = await client.send(req, stream=True)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(relay(), status_code=resp.status_code, headers=response_headers(resp))


def s_stream_usage() -> bool:
    return os.getenv("TOKENWISE_OPENAI_STREAM_USAGE", "1").strip().lower() not in {"0", "false", "no", "off"}
