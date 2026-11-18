"""What every proxied request goes through first: who is it, which upstream, capture or not."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field

from fastapi.responses import JSONResponse
from starlette.requests import HTTPConnection

from . import clients, traces
from .common import forward_headers, identity
from .config import Settings
from .db import Store

log = logging.getLogger("tokenwise.gateway")


@dataclass
class Ctx:
    provider: str
    client: clients.Client | None
    who: tuple
    headers: dict[str, str]
    params: dict[str, str]
    upstream_url: str
    capture: bool
    mask: bool
    trace_id: str = field(default_factory=traces.new_trace_id)

    @property
    def client_id(self) -> str | None:
        return self.client.id if self.client else None


def _error(provider: str, status: int, message: str) -> JSONResponse:
    if provider == "anthropic":
        body = {"type": "error", "error": {"type": "authentication_error" if status == 401 else "invalid_request_error",
                                           "message": message}}
    elif provider == "gemini":
        body = {"error": {"code": status, "message": message,
                          "status": "UNAUTHENTICATED" if status == 401 else "INVALID_ARGUMENT"}}
    else:
        body = {"error": {"message": message, "type": "invalid_request_error",
                          "code": "invalid_api_key" if status == 401 else None}}
    return JSONResponse(body, status_code=status)


def prepare(conn: HTTPConnection, provider: str, s: Settings, store: Store) -> Ctx | JSONResponse:
    client, sent_gateway_key = clients.resolve(conn, store)
    if sent_gateway_key and client is None:
        return _error(provider, 401, "Unknown or disabled Tokenwise key.")
    if client is None and s.require_client_key:
        return _error(provider, 401, "This gateway needs a Tokenwise client key (tw-...).")

    headers = forward_headers(conn, s)
    params = dict(conn.query_params)
    if provider == "openai":
        upstream = (client.openai_base_url if client and client.openai_base_url else s.openai_upstream_url).rstrip("/")
    elif provider == "gemini":
        upstream = s.gemini_upstream_url
    else:
        upstream = s.upstream_url

    engineer, team, feature = identity(conn, s)
    if client:
        try:
            key = client.upstream_key(provider)
        except clients.ConfigError as e:
            log.error("%s", e)
            return _error(provider, 500, "Gateway is missing TOKENWISE_SECRET.")
        if not key:
            return _error(provider, 400, f"No {provider} key is set for client '{client.name}'.")
        clients.swap_credentials(provider, key, headers, params)
        if not conn.headers.get(s.user_header):
            engineer = client.name
        capture, mask = client.capture, client.mask
        if capture and client.capture_rate < 1.0 and random.random() >= client.capture_rate:
            capture = False  # sampled out: logged for cost, conversation not stored
    else:
        capture, mask = s.traces_enabled, s.traces_mask

    return Ctx(provider, client, (engineer, team, feature), headers, params, upstream, capture, mask)


async def record(ctx: Ctx, store: Store, s: Settings, *, model: str, status: int, stream: bool, body: dict | None,
                 collector: traces.OutputCollector, input_tokens: int = 0, output_tokens: int = 0,
                 cost_usd: float = 0.0, started: float | None = None) -> None:
    """Store the trace if capture is on. Never raises: a trace must not break a request."""
    if not ctx.capture:
        return
    try:
        row = traces.build_row(
            trace_id=ctx.trace_id, provider=ctx.provider, upstream_url=ctx.upstream_url, client_id=ctx.client_id,
            who=ctx.who, model=model, status=status, stream=stream, body=body, out=collector.result(),
            mask=ctx.mask, max_chars=s.trace_max_chars, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, started=started)
        await asyncio.to_thread(traces.save, store, row, s.retry_window_s)
    except Exception:
        log.exception("failed to store trace")


def trace_headers(ctx: Ctx) -> dict[str, str]:
    return {traces.TRACE_HEADER: ctx.trace_id} if ctx.capture else {}
