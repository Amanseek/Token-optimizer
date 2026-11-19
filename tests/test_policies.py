import json
import time

import httpx
import pytest

from tokenwise import clients, dashboard, policies
from tokenwise.admin import main as admin_main

OA = "/openai/v1/chat/completions"
ASK = {"model": "demo-scam", "messages": [{"role": "user", "content": "SBI KYC expires today, click bit.ly/x"}]}
GOOD = '{"v": "scam", "c": "kyc_bank"}'


def reply(content, finish="stop", model="m", status=200):
    return httpx.Response(status, json={"id": "c1", "object": "chat.completion", "model": model,
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                                     "finish_reason": finish}],
                                        "usage": {"prompt_tokens": 100, "completion_tokens": 10}})


def setup_client(store, monkeypatch, *, fallback=True, shadow_rate=None, require_json=True):
    _, key = clients.create_client(store, "acme", {"openai": "mk-gpu"}, openai_base_url="https://gpu.test/v1")
    monkeypatch.setenv("BACKUP_KEY", "AIza-backup")
    monkeypatch.setenv("SHADOW_KEY", "mk-shadow")
    if fallback:
        policies.set_target(store, "acme", "fallback", "https://backup.test/v1", "gemini-2.5-flash", "AIza-backup",
                            require_json=require_json)
    if shadow_rate:
        policies.set_target(store, "acme", "shadow", "https://shadow.test/v1", "acme-small", "mk-shadow",
                            rate=shadow_rate)
    return {"authorization": f"Bearer {key}"}


# ------------------------------------------------------------------ pure functions

def test_failure_reasons():
    ok = reply(GOOD).json()
    assert policies.failure(None, None, False) == "unreachable"
    assert policies.failure(503, None, False) == "http_503"
    assert policies.failure(400, None, False) is None  # the app's own mistake: a backup won't help
    assert policies.failure(200, ok, True) is None
    assert policies.failure(200, reply("").json(), False) == "empty"
    assert policies.failure(200, reply("partial", finish="length").json(), False) == "cut_off"
    assert policies.failure(200, reply("not json").json(), True) == "bad_json"
    assert policies.failure(200, reply("not json").json(), False) is None
    assert policies.failure(200, reply("```json\n" + GOOD + "\n```").json(), True) is None


def test_agreement_scores():
    assert policies.agreement(GOOD, '{"c": "KYC_bank", "v": "scam"}') == (1.0, True)
    assert policies.agreement(GOOD, '{"v": "scam", "c": "other"}') == (0.5, False)
    assert policies.agreement(GOOD, "it is a scam") == (0.0, False)
    assert policies.agreement('{"r": ["a", "b"]}', '{"r": ["b", "a"]}') == (1.0, True)  # order doesn't matter
    score, exact = policies.agreement("Your parcel ships today", "your parcel ships tomorrow")
    assert 0.5 < score < 1 and not exact


# ------------------------------------------------------------------ safety net through the gateway

@pytest.mark.parametrize("main, reason", [
    (lambda: (_ for _ in ()).throw(httpx.ConnectError("refused")), "unreachable"),
    (lambda: httpx.Response(503, json={"error": "loading"}), "http_503"),
    (lambda: reply("Sorry, I think it is a scam"), "bad_json"),
    (lambda: reply('{"v": "sc', finish="length"), "cut_off"),
])
def test_fallback_answers_when_main_model_fails(make_client, openai_upstream, monkeypatch, main, reason):
    client, store = make_client()
    headers = setup_client(store, monkeypatch)
    seen = []

    def handler(request, body):
        seen.append((request.url.host, body["model"], request.headers["authorization"]))
        return main() if request.url.host == "gpu.test" else reply(GOOD, model=body["model"])

    openai_upstream.handler = handler
    r = client.post(OA, json=ASK, headers=headers)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == GOOD
    assert r.headers["x-tokenwise-fallback"] == reason
    assert seen[-1] == ("backup.test", "gemini-2.5-flash", "Bearer AIza-backup")
    row = store.query("SELECT * FROM requests")[-1]
    assert row["route_decision"] == f"fallback:{reason}"
    assert row["model_requested"] == "demo-scam" and row["model_used"] == "gemini-2.5-flash"


def test_no_fallback_when_main_answer_is_fine_or_not_configured(make_client, openai_upstream, monkeypatch):
    client, store = make_client()
    headers = setup_client(store, monkeypatch)
    openai_upstream.handler = lambda request, body: reply(GOOD, model=body["model"])
    r = client.post(OA, json=ASK, headers=headers)
    assert "x-tokenwise-fallback" not in r.headers and len(openai_upstream.requests) == 1

    policies.clear(store, "acme", "fallback")
    openai_upstream.handler = lambda request, body: httpx.Response(503, json={"error": "down"})
    assert client.post(OA, json=ASK, headers=headers).status_code == 503  # no backup: error passes through


def test_stream_falls_back_when_main_model_is_down(make_client, openai_upstream, monkeypatch):
    client, store = make_client()
    headers = setup_client(store, monkeypatch)
    default = openai_upstream.default

    def handler(request, body):
        if request.url.host == "gpu.test":
            raise httpx.ConnectError("GPU stopped")
        return default(request, body)

    openai_upstream.handler = handler
    r = client.post(OA, json={**ASK, "stream": True}, headers=headers)
    assert r.status_code == 200 and "reading" in r.text and r.headers["x-tokenwise-fallback"] == "unreachable"
    assert store.query("SELECT route_decision FROM requests")[-1]["route_decision"] == "fallback:unreachable"


def test_both_down_gives_clear_error(make_client, openai_upstream, monkeypatch):
    client, store = make_client()
    headers = setup_client(store, monkeypatch)

    def handler(request, body):
        raise httpx.ConnectError("all down")

    openai_upstream.handler = handler
    r = client.post(OA, json=ASK, headers=headers)
    assert r.status_code == 502 and "both failed" in r.json()["error"]["message"]


# ------------------------------------------------------------------ shadow tests

def wait_for_shadow(store, n=1, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        store.execute(policies.SHADOW_SQL)
        rows = store.query("SELECT * FROM shadow_results")
        if len(rows) >= n:
            return rows
        time.sleep(0.05)
    raise AssertionError("no shadow result")


def test_shadow_scores_agreement_without_storing_text(make_client, openai_upstream, monkeypatch):
    client, store = make_client()
    headers = setup_client(store, monkeypatch, fallback=False, shadow_rate=1.0)

    def handler(request, body):
        if request.url.host == "shadow.test":
            assert "stream" not in body and request.headers["authorization"] == "Bearer mk-shadow"
            return reply('{"v": "scam", "c": "other"}', model=body["model"])
        return reply(GOOD, model=body["model"])

    openai_upstream.handler = handler
    r = client.post(OA, json=ASK, headers={**headers, "x-tokenwise-feature": "scam-check"})
    assert r.json()["choices"][0]["message"]["content"] == GOOD  # the app gets the main answer
    [row] = wait_for_shadow(store)
    assert row["agreement"] == 0.5 and row["exact"] == 0 and row["shadow_json"] == 1
    assert row["feature"] == "scam-check" and row["shadow_model"] == "acme-small"
    assert "kyc" not in json.dumps(row).lower()  # scores only, no text

    page = dashboard.render_html(store)
    assert "Shadow tests" in page and "50%" in page and "demo-scam → acme-small" in page


def test_shadow_respects_rate_and_failed_shadow_is_counted(make_client, openai_upstream, monkeypatch):
    client, store = make_client()
    headers = setup_client(store, monkeypatch, fallback=False, shadow_rate=1.0)

    def handler(request, body):
        if request.url.host == "shadow.test":
            raise httpx.ConnectError("shadow GPU off")
        return reply(GOOD, model=body["model"])

    openai_upstream.handler = handler
    assert client.post(OA, json=ASK, headers=headers).status_code == 200  # shadow failure never hurts the app
    [row] = wait_for_shadow(store)
    assert row["status"] == 0 and row["agreement"] is None


def test_admin_commands_store_keys_encrypted(make_client, monkeypatch, capsys):
    client, store = make_client()
    clients.create_client(store, "acme", {"openai": "sk"})
    monkeypatch.setenv("GEM", "AIza-secret-123")
    db = store.path
    assert admin_main(["--db", db, "fallback", "acme", "--base-url", "https://backup.test/v1",
                       "--model", "gemini-2.5-flash", "--key-env", "GEM", "--require-json"]) == 0
    assert admin_main(["--db", db, "shadow", "acme", "--base-url", "https://shadow.test/v1",
                       "--model", "acme-small", "--key-env", "GEM", "--rate", "0.2"]) == 0
    raw = store.query("SELECT policy FROM clients WHERE name = 'acme'")[0]["policy"]
    assert "AIza-secret-123" not in raw
    capsys.readouterr()
    admin_main(["--db", db, "policy", "acme"])
    shown = json.loads(capsys.readouterr().out)
    assert shown["fallback"]["model"] == "gemini-2.5-flash" and shown["shadow"]["rate"] == 0.2
    assert "key_enc" not in json.dumps(shown)
    assert admin_main(["--db", db, "shadow", "acme", "--off"]) == 0
    assert "shadow" not in policies.describe(store, "acme")
