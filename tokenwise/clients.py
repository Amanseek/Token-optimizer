"""Client accounts for the multi-client gateway.

Each client gets a gateway key ("tw-..."). Their app sends that key instead of the
provider's key; Tokenwise looks up the client, swaps in the client's real upstream key
(kept encrypted on the server) and forwards the request. The app never holds the
provider key, so it can't bypass the gateway.

Bring your own key: a client that won't hand over its provider key keeps sending it as
usual and adds its tw- key in the x-tokenwise-key header. Tokenwise uses their key for
that one request and never stores it (only usage is logged).

Requests that carry a normal provider key (not tw-) still pass through unchanged,
unless TOKENWISE_REQUIRE_CLIENT_KEY=1.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
import uuid
from dataclasses import dataclass

from starlette.requests import HTTPConnection

from .db import Store

KEY_PREFIX = "tw-"
PROVIDERS = ("openai", "gemini", "anthropic")


class ConfigError(RuntimeError):
    pass


def _fernet():
    from cryptography.fernet import Fernet

    secret = os.getenv("TOKENWISE_SECRET")
    if not secret:
        raise ConfigError("Set TOKENWISE_SECRET (run: python -m tokenwise.admin new-secret) to store upstream keys.")
    return Fernet(secret.encode())


def new_secret() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    return _fernet().decrypt(value.encode()).decode()


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class Client:
    id: str
    name: str
    capture: bool
    mask: bool
    openai_base_url: str | None
    _row: dict
    capture_rate: float = 1.0

    def upstream_key(self, provider: str) -> str | None:
        enc = self._row.get(f"{provider}_key_enc")
        return decrypt(enc) if enc else None


def create_client(store: Store, name: str, upstream_keys: dict[str, str] | None = None,
                  openai_base_url: str | None = None, capture: bool = True, mask: bool = True) -> tuple[str, str]:
    """Returns (client_id, gateway_key). The gateway key is shown once and not stored."""
    key = KEY_PREFIX + base64.urlsafe_b64encode(secrets.token_bytes(30)).decode().rstrip("=")
    cid = uuid.uuid4().hex[:12]
    row = {"id": cid, "name": name, "key_hash": hash_key(key), "key_prefix": key[:9], "created": time.time(),
           "active": 1, "capture": int(capture), "mask": int(mask), "openai_base_url": openai_base_url}
    for prov, val in (upstream_keys or {}).items():
        if prov not in PROVIDERS:
            raise ValueError(f"unknown provider {prov}")
        if val:
            row[f"{prov}_key_enc"] = encrypt(val)
    store.insert("clients", row)
    return cid, key


def rotate_key(store: Store, name: str) -> str:
    key = KEY_PREFIX + base64.urlsafe_b64encode(secrets.token_bytes(30)).decode().rstrip("=")
    if not store.execute("UPDATE clients SET key_hash = ?, key_prefix = ? WHERE name = ?",
                         (hash_key(key), key[:9], name)):
        raise ValueError(f"no client named {name}")
    return key


def set_upstream_key(store: Store, name: str, provider: str, value: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider}")
    if not store.execute(f"UPDATE clients SET {provider}_key_enc = ? WHERE name = ?", (encrypt(value), name)):
        raise ValueError(f"no client named {name}")


def find_by_key(store: Store, key: str) -> Client | None:
    rows = store.query("SELECT * FROM clients WHERE key_hash = ? AND active = 1", (hash_key(key),))
    if not rows:
        return None
    r = rows[0]
    rate = r.get("capture_rate")
    return Client(r["id"], r["name"], bool(r["capture"]), bool(r["mask"]), r["openai_base_url"], r,
                  1.0 if rate is None else float(rate))


def presented_key(conn: HTTPConnection) -> str:
    """The credential the app sent, in whichever place its SDK puts it."""
    h = conn.headers
    auth = h.get("authorization", "")
    if auth.lower().startswith("bearer "):
        auth = auth[7:].strip()
    return h.get("x-api-key") or h.get("x-goog-api-key") or auth or conn.query_params.get("key") or ""


GATEWAY_KEY_HEADER = "x-tokenwise-key"  # tw- key sent beside the client's own provider key


def brings_own_key(conn: HTTPConnection) -> bool:
    """True when the app sends its own provider key and identifies itself with x-tokenwise-key."""
    return (conn.headers.get(GATEWAY_KEY_HEADER, "").startswith(KEY_PREFIX)
            and not presented_key(conn).startswith(KEY_PREFIX))


def resolve(conn: HTTPConnection, store: Store) -> tuple[Client | None, bool]:
    """(client, sent_gateway_key). A tw- key that matches no active client gives (None, True)."""
    key = presented_key(conn)
    if not key.startswith(KEY_PREFIX):
        key = conn.headers.get(GATEWAY_KEY_HEADER, "")
        if not key.startswith(KEY_PREFIX):
            return None, False
    return find_by_key(store, key), True


def swap_credentials(provider: str, upstream_key: str, headers: dict[str, str], params: dict[str, str]) -> None:
    """Replace the gateway key with the client's real provider key, in the place that provider expects."""
    for h in ("authorization", "x-api-key", "x-goog-api-key"):
        for k in [k for k in headers if k.lower() == h]:
            del headers[k]
    params.pop("key", None)
    if provider == "openai":
        headers["authorization"] = f"Bearer {upstream_key}"
    elif provider == "anthropic":
        headers["x-api-key"] = upstream_key
    else:
        headers["x-goog-api-key"] = upstream_key
