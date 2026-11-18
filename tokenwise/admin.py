"""Manage client accounts.

    python -m tokenwise.admin new-secret                     # once; set the output as TOKENWISE_SECRET
    python -m tokenwise.admin create acme --openai-key-env ACME_OPENAI_KEY \\
        --openai-base-url https://openrouter.ai/api/v1       # prints the client's tw- key (shown once)
    python -m tokenwise.admin set-key acme gemini --key-env ACME_GEMINI_KEY
    python -m tokenwise.admin list
    python -m tokenwise.admin rotate acme                    # new tw- key, old one stops working
    python -m tokenwise.admin disable acme | enable acme
    python -m tokenwise.admin capture acme on|off | mask acme on|off
    python -m tokenwise.admin sample acme 0.1                # store 10% of conversations
    python -m tokenwise.admin retention acme 30              # delete traces older than 30 days
    python -m tokenwise.admin purge acme --user U123 | --all # delete stored conversations now
    python -m tokenwise.admin audit [--limit 50]             # who viewed, exported or deleted what

Upstream provider keys are read from environment variables (never typed on the
command line, where they'd end up in shell history) and stored encrypted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

from . import clients, traces
from .db import Store


def _env(name: str | None) -> str | None:
    if not name:
        return None
    val = os.getenv(name)
    if not val:
        raise SystemExit(f"environment variable {name} is empty or not set")
    return val


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.getenv("TOKENWISE_DB", "tokenwise.db"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("new-secret")
    c = sub.add_parser("create")
    c.add_argument("name")
    for p in clients.PROVIDERS:
        c.add_argument(f"--{p}-key-env", help=f"env var holding the client's {p} key")
    c.add_argument("--openai-base-url", help="OpenAI-compatible host, e.g. https://openrouter.ai/api/v1")
    c.add_argument("--no-capture", action="store_true")
    c.add_argument("--no-mask", action="store_true")
    k = sub.add_parser("set-key")
    k.add_argument("name")
    k.add_argument("provider", choices=clients.PROVIDERS)
    k.add_argument("--key-env", required=True)
    sub.add_parser("list")
    for cmd in ("rotate", "disable", "enable"):
        sub.add_parser(cmd).add_argument("name")
    for cmd in ("capture", "mask"):
        x = sub.add_parser(cmd)
        x.add_argument("name")
        x.add_argument("state", choices=["on", "off"])
    x = sub.add_parser("sample")
    x.add_argument("name")
    x.add_argument("rate", type=float)
    x = sub.add_parser("retention")
    x.add_argument("name")
    x.add_argument("days", type=int)
    x = sub.add_parser("purge")
    x.add_argument("name")
    g = x.add_mutually_exclusive_group(required=True)
    g.add_argument("--user")
    g.add_argument("--all", action="store_true")
    sub.add_parser("audit").add_argument("--limit", type=int, default=50)
    a = ap.parse_args(argv)

    if a.cmd == "new-secret":
        print(clients.new_secret())
        print("Set this as TOKENWISE_SECRET on the server. Keep it safe: losing it means re-entering "
              "upstream keys.", file=sys.stderr)
        return 0

    store = Store(a.db)

    def client_id(name: str) -> str:
        rows = store.query("SELECT id FROM clients WHERE name = ?", (name,))
        if not rows:
            raise ValueError(f"no client named {name}")
        return rows[0]["id"]

    try:
        if a.cmd == "create":
            keys = {p: _env(getattr(a, f"{p}_key_env")) for p in clients.PROVIDERS}
            _, key = clients.create_client(store, a.name, {p: v for p, v in keys.items() if v},
                                           openai_base_url=a.openai_base_url, capture=not a.no_capture,
                                           mask=not a.no_mask)
            print(key)
            print(f"Client '{a.name}' created. Give the key above to the client; it is not shown again.",
                  file=sys.stderr)
        elif a.cmd == "set-key":
            clients.set_upstream_key(store, a.name, a.provider, _env(a.key_env))
            print(f"{a.provider} key updated for {a.name}")
        elif a.cmd == "list":
            for r in store.query("SELECT * FROM clients ORDER BY created"):
                provs = ",".join(p for p in clients.PROVIDERS if r[f"{p}_key_enc"]) or "-"
                when = dt.datetime.fromtimestamp(r["created"]).strftime("%Y-%m-%d")
                print(f"{r['name']:<20} {r['key_prefix']}…  {'active' if r['active'] else 'DISABLED':<8} "
                      f"capture={'on' if r['capture'] else 'off'} sample={r['capture_rate']:.0%} "
                      f"retention={r['retention_days']}d mask={'on' if r['mask'] else 'off'} "
                      f"providers={provs} openai_base={r['openai_base_url'] or '-'} created={when}")
        elif a.cmd == "rotate":
            print(clients.rotate_key(store, a.name))
            print("New key above; the old key no longer works.", file=sys.stderr)
        elif a.cmd in ("disable", "enable"):
            if not store.execute("UPDATE clients SET active = ? WHERE name = ?", (int(a.cmd == "enable"), a.name)):
                raise ValueError(f"no client named {a.name}")
            print(f"{a.name} {a.cmd}d")
        elif a.cmd in ("capture", "mask"):
            if not store.execute(f"UPDATE clients SET {a.cmd} = ? WHERE name = ?", (int(a.state == "on"), a.name)):
                raise ValueError(f"no client named {a.name}")
            print(f"{a.cmd} {a.state} for {a.name}")
        elif a.cmd == "sample":
            if not 0 <= a.rate <= 1:
                raise ValueError("rate must be between 0 and 1")
            client_id(a.name)
            store.execute("UPDATE clients SET capture_rate = ? WHERE name = ?", (a.rate, a.name))
            print(f"{a.name}: storing {a.rate:.0%} of conversations")
        elif a.cmd == "retention":
            if a.days < 0:
                raise ValueError("days must be 0 or more")
            client_id(a.name)
            store.execute("UPDATE clients SET retention_days = ? WHERE name = ?", (a.days, a.name))
            print(f"{a.name}: traces kept {a.days} days")
        elif a.cmd == "purge":
            n = traces.delete_for(store, client_id(a.name), user=a.user, everything=a.all)
            store.audit("cli", "delete", f"{a.name}:" + ("all" if a.all else f"user:{a.user}"), f"{n} traces deleted")
            print(f"deleted {n} traces")
        elif a.cmd == "audit":
            for r in store.query("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (a.limit,)):
                when = dt.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M")
                print(f"{when}  {r['actor']:<16} {r['action']:<16} {r['target'] or '':<24} {r['detail'] or ''}")
        if a.cmd in ("create", "set-key", "rotate", "disable", "enable", "capture", "mask", "sample", "retention"):
            store.audit("cli", a.cmd, a.name)
    except (ValueError, clients.ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
