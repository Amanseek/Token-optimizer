"""Helpers shared by the Anthropic and Gemini handlers."""

from __future__ import annotations

import hashlib
import json

import httpx
from starlette.requests import HTTPConnection

from .config import Settings

HOP_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "accept-encoding", "proxy-authorization", "proxy-connection", "upgrade", "te",
}
DROP_RESPONSE_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def credential(conn: HTTPConnection) -> str:
    """Whatever API credential the client sent (header or ?key=)."""
    h = conn.headers
    return (
        h.get("x-api-key") or h.get("x-goog-api-key") or h.get("authorization")
        or conn.query_params.get("key") or ""
    )


def credential_hash(conn: HTTPConnection) -> str:
    cred = credential(conn)
    return hashlib.sha256(cred.encode()).hexdigest()[:8] if cred else ""


def identity(conn: HTTPConnection, s: Settings) -> tuple[str, str | None, str | None]:
    """(engineer, team, feature) for logging."""
    engineer = conn.headers.get(s.user_header)
    if not engineer:
        h = credential_hash(conn)
        engineer = f"key-{h}" if h else "unknown"
    return engineer, conn.headers.get(s.team_header), conn.headers.get(s.feature_header)


def our_headers(s: Settings) -> set[str]:
    return {s.user_header, s.team_header, s.feature_header, s.nocache_header}


def forward_headers(conn: HTTPConnection, s: Settings) -> dict[str, str]:
    skip = HOP_HEADERS | our_headers(s)
    headers = {k: v for k, v in conn.headers.items() if k.lower() not in skip}
    headers["accept-encoding"] = "identity"
    return headers


def response_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() not in DROP_RESPONSE_HEADERS}


# ---------------------------------------------------------------- response cache


def cacheable(conn: HTTPConnection, s: Settings, model: str) -> bool:
    if not s.response_cache_enabled or conn.headers.get(s.nocache_header):
        return False
    return not any(x and x in model for x in s.response_cache_exclude)


def cache_key(conn: HTTPConnection, provider: str, path: str, body: dict) -> str:
    """Exact-match key. Includes the credential so different customers never share answers."""
    extra = conn.headers.get("anthropic-beta", "")
    seed = json.dumps([provider, path, credential_hash(conn), extra, body], sort_keys=True, default=str)
    return hashlib.sha256(seed.encode()).hexdigest()
