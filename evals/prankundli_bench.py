"""Run PranKundali's real Gemini code through Tokenwise and compare usage.

It imports PranKundali's own tarot module (unchanged) and the google-genai
SDK's Live client, and plays a seeded sequence of user sessions twice:

  run A  direct      the SDK talks straight to the Gemini upstream
  run B  tokenwise   GOOGLE_GEMINI_BASE_URL points at Tokenwise (response cache on)

Default upstream is evals/fake_gemini.py (no API key needed; token counts are
estimates). With --real it uses the real Gemini API and GEMINI_API_KEY.

    python evals/prankundli_bench.py --prankundli ../prankundli
    GEMINI_API_KEY=... python evals/prankundli_bench.py --prankundli ../prankundli --real
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tokenwise.gemini_pricing import gemini_cost  # noqa: E402

QUESTIONS = [
    "Will I get a new job this year?", "Is my relationship going to last?", "Should I move to another city?",
    "", "When will I get married?", "Will my business grow?", "", "Is this a good time to buy a house?",
]
FOLLOWUPS = ["What should I do next?", "Can you explain the middle card more?", "Is there any warning?"]
LIVE_MODEL = "gemini-3.1-flash-live-preview"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http() -> httpx.Client:
    import ssl
    return httpx.Client(verify=ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"]), timeout=30)


def wait_up(url: str) -> None:
    for _ in range(100):
        try:
            http().get(url, timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"{url} did not start")


def make_local_tls(tmp: Path) -> tuple[str, str]:
    """Self-signed cert for 127.0.0.1, added to a CA bundle via SSL_CERT_FILE.

    Needed because the google-genai SDK always opens Live sessions over wss://.
    """
    import certifi

    cert, key = tmp / "cert.pem", tmp / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    bundle = tmp / "bundle.pem"
    base = os.environ.get("SSL_CERT_FILE") or certifi.where()
    bundle.write_text(Path(base).read_text() + "\n" + cert.read_text())
    os.environ["SSL_CERT_FILE"] = str(bundle)
    return str(cert), str(key)


def start(cmd: list[str], env: dict, url: str) -> subprocess.Popen:
    p = subprocess.Popen(cmd, env={**os.environ, **env}, cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_up(url)
    return p


# ------------------------------------------------------------------ the workload

def play_sessions(tarot, n: int, seed: int) -> dict:
    """Seeded sessions modelled on PranKundali's tarot flow (web/app.py)."""
    rng = random.Random(seed)
    random.seed(seed)  # tarot.draw_major uses the global RNG
    calls, outputs, latencies = 0, [], []

    def timed(fn, *a, **kw):
        nonlocal calls
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        latencies.append(time.perf_counter() - t0)
        calls += 1
        outputs.append(out)
        return out

    for _ in range(n):
        q = rng.choice(QUESTIONS)
        qtype = tarot.guess_type(q) if q else "general"
        cards = tarot.draw_major(3)
        lang = rng.choice(["en", "en", "en", "hi"])
        timed(tarot.teaser_reading, qtype, cards, lang, q)
        if rng.random() < 0.15:                       # page refresh / app retry: identical request
            timed(tarot.teaser_reading, qtype, cards, lang, q)
        if rng.random() < 0.5:                        # buys the full reading
            reading = timed(tarot.full_reading, qtype, cards, lang, q)
            if rng.random() < 0.1:                    # reopens it / network retry
                timed(tarot.full_reading, qtype, cards, lang, q)
            if rng.random() < 0.3:                    # asks a follow-up
                timed(tarot.clarify, qtype, cards, reading, rng.choice(FOLLOWUPS), lang)
    return {"calls": calls, "outputs": outputs, "avg_latency_ms": 1000 * mean(latencies)}


async def play_live(sessions: int, turns: int) -> dict:
    """Voice sessions through the same SDK call PranKundali uses (client.aio.live.connect)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    replies = 0
    for _ in range(sessions):
        cfg = types.LiveConnectConfig(response_modalities=["TEXT"])
        async with client.aio.live.connect(model=LIVE_MODEL, config=cfg) as ses:
            for t in range(turns):
                await ses.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=f"Question {t} about my chart")]),
                    turn_complete=True)
                async for msg in ses.receive():
                    if msg.server_content and msg.server_content.turn_complete:
                        replies += 1
                        break
    return {"live_replies": replies}


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prankundli", required=True, help="path to the prankundli repo")
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--live-sessions", type=int, default=3)
    ap.add_argument("--live-turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real", action="store_true", help="use the real Gemini API (needs GEMINI_API_KEY)")
    args = ap.parse_args()

    pk = Path(args.prankundli).resolve()
    sys.path[:0] = [str(pk), str(pk / "web")]
    procs = []
    tmp = Path(tempfile.mkdtemp())
    cert, key = make_local_tls(tmp)
    tls = ["--ssl-certfile", cert, "--ssl-keyfile", key]
    try:
        if args.real:
            if not os.environ.get("GEMINI_API_KEY"):
                sys.exit("Set GEMINI_API_KEY for --real.")
            upstream = "https://generativelanguage.googleapis.com"
        else:
            os.environ["GEMINI_API_KEY"] = "fake-key-for-benchmark"
            port = free_port()
            upstream = f"https://127.0.0.1:{port}"
            procs.append(start([sys.executable, "-m", "uvicorn", "evals.fake_gemini:app", "--port", str(port), *tls],
                               {}, upstream + "/stats"))

        tw_port = free_port()
        tw_url = f"https://127.0.0.1:{tw_port}"
        procs.append(start([sys.executable, "-m", "tokenwise.main"], {
            "TOKENWISE_PORT": str(tw_port), "TOKENWISE_DB": str(tmp / "bench.db"),
            "TOKENWISE_GEMINI_UPSTREAM": upstream, "TOKENWISE_RESPONSE_CACHE": "true",
            "TOKENWISE_SSL_CERTFILE": cert, "TOKENWISE_SSL_KEYFILE": key,
        }, tw_url + "/healthz"))

        import tarot  # PranKundali's own module, unchanged

        results = {}
        for label, base in (("direct", None if args.real else upstream), ("tokenwise", tw_url)):
            if base:
                os.environ["GOOGLE_GEMINI_BASE_URL"] = base
            else:
                os.environ.pop("GOOGLE_GEMINI_BASE_URL", None)
            if not args.real:
                http().post(upstream + "/reset")
            r = play_sessions(tarot, args.sessions, args.seed)
            r.update(asyncio.run(play_live(args.live_sessions, args.live_turns)))
            if not args.real:
                r["upstream"] = http().get(upstream + "/stats").json()
            results[label] = r

        summary = http().get(tw_url + "/api/summary").json()
    finally:
        for p in procs:
            p.terminate()

    a, b = results["direct"], results["tokenwise"]
    same = a["outputs"] == b["outputs"]
    rows = {r["feature"] if False else r["provider"]: r for r in summary["by_provider"]}
    routing = {r["decision"]: r for r in summary["routing"]}
    live = routing.get("live", {})
    text_cost = summary["totals"]["cost"] - (live.get("cost") or 0)
    text_baseline = summary["totals"]["baseline"] - (live.get("cost") or 0)

    print("\n=== PranKundali through Tokenwise ===")
    print(f"Tarot sessions: {args.sessions} (seed {args.seed}); voice: {args.live_sessions} x {args.live_turns} turns")
    print(f"App-level Gemini calls (both runs): {a['calls']}")
    print(f"Answers identical in both runs: {'yes' if same else 'NO'}")
    print(f"Voice replies received: direct {a['live_replies']}, via Tokenwise {b['live_replies']}")
    if not args.real:
        ua, ub = a["upstream"], b["upstream"]
        print("\nWhat reached Gemini (the bill):")
        print(f"  {'':18}{'direct':>12}{'tokenwise':>12}")
        for k in ("calls", "prompt_tokens", "output_tokens", "live_turns"):
            print(f"  {k:18}{ua[k]:>12,}{ub[k]:>12,}")
    print("\nTokenwise dashboard (run B):")
    print(f"  text spend ${text_cost:.5f} vs ${text_baseline:.5f} without cache "
          f"({summary['response_cache']['hits']} cache hits)")
    print(f"  voice spend ${live.get('cost') or 0:.5f} over {live.get('requests') or 0} sessions")
    print(f"  avg latency per call: direct {a['avg_latency_ms']:.1f} ms, via Tokenwise {b['avg_latency_ms']:.1f} ms")
    print(json.dumps({"direct": {k: v for k, v in a.items() if k != "outputs"},
                      "tokenwise": {k: v for k, v in b.items() if k != "outputs"},
                      "summary_totals": summary["totals"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
