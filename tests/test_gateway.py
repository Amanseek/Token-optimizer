import json

import pytest

from tokenwise import clients, export, traces
from tokenwise.admin import main as admin_main
from tokenwise.masking import mask_text

OA = "/openai/v1/chat/completions"
GEM = "/v1beta/models/gemini-3.1-flash-lite:generateContent"
CHAT = {"model": "deepseek-chat", "messages": [
    {"role": "system", "content": "You write support replies for Acme."},
    {"role": "user", "content": "My order is late, call me on 9876543210 or mail a.b@x.com"}]}


def traces_of(store):
    return [traces.unseal_row(r) for r in store.query("SELECT * FROM traces ORDER BY ts")]


def make_acme(store, **kw):
    return clients.create_client(store, "acme", {"openai": "sk-real-acme", "gemini": "AIza-real-acme",
                                                 "anthropic": "sk-ant-real-acme"},
                                 openai_base_url="https://openrouter.test/api/v1", **kw)


# ------------------------------------------------------------------ masking

def test_masking_covers_indian_personal_data():
    s = mask_text("Call +91 98765 43210, mail ravi.k@gmail.com, UPI ravi@okhdfc, PAN ABCDE1234F, "
                  "Aadhaar 1234 5678 9012, card 4111 1111 1111 1111, OTP 482913")
    for secret in ("98765", "ravi.k@gmail.com", "ravi@okhdfc", "ABCDE1234F", "1234 5678 9012", "4111", "482913"):
        assert secret not in s, secret
    assert "[phone]" in s and "[email]" in s and "[pan]" in s


# ------------------------------------------------------------------ OpenAI-compatible proxy

def test_openai_passthrough_mode_logs_without_capture(make_client, openai_upstream):
    client, store = make_client()
    r = client.post(OA, json=CHAT, headers={"authorization": "Bearer sk-own"})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Your reading"
    sent = openai_upstream.requests[0]
    assert str(sent.url) == "https://api.openai.com/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer sk-own"
    [row] = store.query("SELECT * FROM requests")
    assert row["provider"] == "openai" and row["input_tokens"] == 1200 and row["output_tokens"] == 300
    assert traces_of(store) == [] and traces.TRACE_HEADER not in r.headers  # capture is opt-in


def test_openai_prices_from_env(make_client, monkeypatch):
    monkeypatch.setenv("TOKENWISE_OPENAI_PRICES", json.dumps({"deepseek-chat": [1.0, 2.0]}))
    client, store = make_client()
    client.post(OA, json=CHAT, headers={"authorization": "Bearer sk-own"})
    [row] = store.query("SELECT * FROM requests")
    assert row["cost_usd"] == pytest.approx((1200 * 1.0 + 300 * 2.0) / 1e6)


def test_client_key_swapped_for_real_key_and_trace_captured(make_client, openai_upstream):
    client, store = make_client()
    _, key = make_acme(store)
    r = client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}", "x-tokenwise-feature": "support"})
    assert r.status_code == 200
    sent = openai_upstream.requests[0]
    assert str(sent.url) == "https://openrouter.test/api/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer sk-real-acme"   # the app never held this key
    assert key not in json.dumps(dict(sent.headers))
    [t] = traces_of(store)
    assert r.headers[traces.TRACE_HEADER] == t["id"]
    assert (t["engineer"], t["workflow_key"], t["success"], t["finish"]) == ("acme", "f:support", 1, "stop")
    assert t["system"] == "You write support replies for Acme."
    assert "9876543210" not in t["messages"] and "a.b@x.com" not in t["last_user"] and "[phone]" in t["last_user"]
    assert t["output_text"] == "Your reading" and t["upstream_host"] == "openrouter.test"


def test_streaming_openai_trace_and_usage(make_client, openai_upstream):
    client, store = make_client()
    _, key = make_acme(store)
    r = client.post(OA, json={**CHAT, "stream": True}, headers={"authorization": f"Bearer {key}"})
    assert r.status_code == 200 and "reading" in r.text
    assert openai_upstream.bodies[0]["stream_options"] == {"include_usage": True}
    [row] = store.query("SELECT * FROM requests")
    assert row["stream"] == 1 and row["output_tokens"] == 300
    [t] = traces_of(store)
    assert t["output_text"] == "Your reading" and t["success"] == 1


def test_unknown_or_missing_client_key_rejected(make_client):
    client, store = make_client()
    r = client.post(OA, json=CHAT, headers={"authorization": "Bearer tw-notreal"})
    assert r.status_code == 401
    strict, _ = make_client(require_client_key=True)
    assert strict.post(OA, json=CHAT, headers={"authorization": "Bearer sk-own"}).status_code == 401


def test_client_without_provider_key_gets_clear_error(make_client):
    client, store = make_client()
    _, key = clients.create_client(store, "solo", {"openai": "sk-x"})
    r = client.post(GEM, json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                    headers={"x-goog-api-key": key})
    assert r.status_code == 400 and "gemini" in r.json()["error"]["message"]


def test_disabled_client_and_rotation(make_client, tmp_path):
    client, store = make_client()
    _, key = make_acme(store)
    new_key = clients.rotate_key(store, "acme")
    assert client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"}).status_code == 401
    assert client.post(OA, json=CHAT, headers={"authorization": f"Bearer {new_key}"}).status_code == 200
    store.execute("UPDATE clients SET active = 0 WHERE name = 'acme'")
    assert client.post(OA, json=CHAT, headers={"authorization": f"Bearer {new_key}"}).status_code == 401


def test_upstream_keys_need_secret(monkeypatch, tmp_path):
    from tokenwise.db import Store
    monkeypatch.delenv("TOKENWISE_SECRET")
    with pytest.raises(clients.ConfigError):
        clients.create_client(Store(str(tmp_path / "x.db")), "x", {"openai": "sk"})


# ------------------------------------------------------------------ success signals

def test_cut_off_answer_marked_not_good(make_client, openai_upstream):
    openai_upstream.finish = "length"
    client, store = make_client()
    _, key = make_acme(store)
    client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    [t] = traces_of(store)
    assert t["success"] == 0 and t["reason"] == "finish length"


def test_retry_marks_previous_answer_bad(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    for _ in range(2):
        client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    first, second = traces_of(store)
    assert (first["success"], first["reason"]) == (0, "retried")
    assert second["success"] == 1


def test_upstream_error_traced_as_failure(make_client, openai_upstream):
    import httpx
    openai_upstream.handler = lambda req, body: httpx.Response(429, json={"error": {"message": "rate limit"}})
    client, store = make_client()
    _, key = make_acme(store)
    assert client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"}).status_code == 429
    [t] = traces_of(store)
    assert t["success"] == 0 and t["reason"] == "http 429"


# ------------------------------------------------------------------ feedback

def test_feedback_good_bad_and_corrected(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    r = client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    tid = r.headers[traces.TRACE_HEADER]
    auth = {"authorization": f"Bearer {key}"}
    assert client.post("/tokenwise/v1/feedback", json={"trace_id": tid, "rating": "bad"}, headers=auth).json() == {"ok": True}
    assert client.post("/tokenwise/v1/feedback", headers=auth, json={
        "trace_id": tid, "corrected_output": "Sorry! It ships today. We'll call 9876543210."}).status_code == 200
    [t] = traces_of(store)
    assert t["feedback"] == 0 and "[phone]" in t["corrected_output"]
    assert client.post("/tokenwise/v1/feedback", json={"trace_id": tid, "rating": "meh"}, headers=auth).status_code == 400


def test_feedback_cannot_touch_another_clients_trace(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    _, other = clients.create_client(store, "other", {"openai": "sk-o"})
    tid = client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"}).headers[traces.TRACE_HEADER]
    r = client.post("/tokenwise/v1/feedback", json={"trace_id": tid, "rating": "good"},
                    headers={"authorization": f"Bearer {other}"})
    assert r.status_code == 404


# ------------------------------------------------------------------ other providers

def test_gemini_trace_with_client_key(make_client, gemini_upstream):
    client, store = make_client()
    _, key = make_acme(store)
    body = {"systemInstruction": {"parts": [{"text": "You are a tarot reader."}]},
            "contents": [{"role": "user", "parts": [{"text": "Read my cards"}]}]}
    r = client.post(GEM, json=body, headers={"x-goog-api-key": key})
    assert r.status_code == 200
    assert gemini_upstream.requests[0].headers["x-goog-api-key"] == "AIza-real-acme"
    [t] = traces_of(store)
    assert t["provider"] == "gemini" and t["output_text"] == "Your reading" and t["success"] == 1
    assert t["workflow_key"].startswith("s:") and t["workflow_label"] == "You are a tarot reader."


def test_gemini_streaming_trace(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    path = "/v1beta/models/gemini-3.1-flash-lite:streamGenerateContent?alt=sse"
    client.post(path, json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}, headers={"x-goog-api-key": key})
    [t] = traces_of(store)
    assert t["output_text"] == "Your reading" and t["stream"] == 1


def test_anthropic_trace_with_client_key(make_client, upstream):
    client, store = make_client()
    _, key = make_acme(store)
    body = {"model": "claude-haiku-5-5", "max_tokens": 50, "system": "Be brief.",
            "messages": [{"role": "user", "content": "hello"}]}
    assert client.post("/v1/messages", json=body, headers={"x-api-key": key}).status_code == 200
    assert upstream.requests[0].headers["x-api-key"] == "sk-ant-real-acme"
    assert client.post("/v1/messages", json={**body, "stream": True}, headers={"x-api-key": key}).status_code == 200
    a, b = traces_of(store)
    assert a["output_text"] == "hello" and b["output_text"] == "hi" and b["stream"] == 1
    assert a["provider"] == "anthropic" and a["system"] == "Be brief."


def test_global_capture_for_passthrough_traffic(make_client):
    client, store = make_client(traces_enabled=True)
    r = client.post(OA, json=CHAT, headers={"authorization": "Bearer sk-own"})
    [t] = traces_of(store)
    assert t["client_id"] is None and r.headers[traces.TRACE_HEADER] == t["id"]


# ------------------------------------------------------------------ workflows, dashboard, export

def test_workflows_summary_and_dashboard(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    for q in ("late order", "refund please", "late order"):
        client.post(OA, json={**CHAT, "messages": [CHAT["messages"][0], {"role": "user", "content": q}]},
                    headers={"authorization": f"Bearer {key}", "x-tokenwise-feature": "support"})
    [wf] = client.get("/api/workflows").json()
    assert (wf["client"], wf["workflow_key"], wf["requests"], wf["retried"], wf["trainable"]) == \
        ("acme", "f:support", 3, 1, 2)
    assert "Workflows (captured traces)" in client.get("/dashboard").text


def test_admin_token_protects_dashboard(make_client):
    client, _ = make_client(admin_token="s3cret")
    assert client.get("/dashboard").status_code == 403
    assert client.get("/api/workflows").status_code == 403
    assert client.get("/dashboard?token=s3cret").status_code == 200


def test_export_respects_provider_terms(make_client, tmp_path):
    client, store = make_client()
    _, key = clients.create_client(store, "acme", {"openai": "sk"})  # default host = api.openai.com
    for q in [f"question {i}" for i in range(12)]:
        client.post(OA, json={**CHAT, "messages": [{"role": "user", "content": q}]},
                    headers={"authorization": f"Bearer {key}"})
    out = tmp_path / "ex"
    db = store.path
    assert export.main(["--db", db, "--out", str(out)]) == 1           # OpenAI answers blocked by default
    tid = traces_of(store)[0]["id"]
    store.execute("UPDATE traces SET corrected_output = 'A human answer' WHERE id = ?", (tid,))
    assert export.main(["--db", db, "--out", str(out)]) == 0           # corrected answer always allowed
    rows = [json.loads(x) for f in ("train.jsonl", "test.jsonl") for x in (out / f).read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["messages"][-1] == {"role": "assistant", "content": "A human answer"}
    assert export.main(["--db", db, "--out", str(out), "--allow-provider-outputs"]) == 0
    n = sum(len((out / f).read_text().splitlines()) for f in ("train.jsonl", "test.jsonl"))
    assert n == 12


def test_export_open_model_outputs_by_default(make_client, tmp_path):
    client, store = make_client()
    _, key = make_acme(store)  # openrouter.test: not a restricted host
    client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    assert export.main(["--db", store.path, "--out", str(tmp_path / "o")]) == 0
    [ex] = [json.loads(x) for x in (tmp_path / "o" / "train.jsonl").read_text().splitlines()]
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert "[phone]" in ex["messages"][1]["content"]


def test_admin_cli(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "a.db")
    monkeypatch.setenv("ACME_KEY", "sk-real")
    assert admin_main(["--db", db, "create", "acme", "--openai-key-env", "ACME_KEY",
                       "--openai-base-url", "https://openrouter.test/api/v1"]) == 0
    key = capsys.readouterr().out.strip()
    assert key.startswith("tw-")
    assert admin_main(["--db", db, "list"]) == 0
    listing = capsys.readouterr().out
    assert "acme" in listing and "providers=openai" in listing and "sk-real" not in listing
    assert admin_main(["--db", db, "capture", "acme", "off"]) == 0
    with pytest.raises(SystemExit):
        admin_main(["--db", db, "create", "acme2", "--gemini-key-env", "MISSING_VAR"])


# ------------------------------------------------------------------ data protection (milestone 1.5)

def test_trace_content_encrypted_at_rest(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    [raw] = store.query("SELECT * FROM traces")
    for col in ("system", "messages", "last_user", "output_text"):
        assert raw[col].startswith(traces.SEAL_PREFIX), col
    dumped = json.dumps({k: v for k, v in raw.items() if k != "workflow_label"})
    assert "Your reading" not in dumped and "support replies" not in dumped
    assert traces.unseal_row(raw)["output_text"] == "Your reading"


def test_no_secret_means_plain_storage(make_client, monkeypatch):
    client, store = make_client(traces_enabled=True)
    monkeypatch.delenv("TOKENWISE_SECRET")
    client.post(OA, json=CHAT, headers={"authorization": "Bearer sk-own"})
    assert store.query("SELECT output_text FROM traces")[0]["output_text"] == "Your reading"


def test_sampling_rate(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    store.execute("UPDATE clients SET capture_rate = 0 WHERE name = 'acme'")
    r = client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    assert r.status_code == 200 and traces.TRACE_HEADER not in r.headers
    assert traces_of(store) == [] and len(store.query("SELECT * FROM requests")) == 1  # cost still logged


def test_retention_purge(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    assert traces.purge_expired(store, 90) == 0
    store.execute("UPDATE clients SET retention_days = 0 WHERE name = 'acme'")
    assert traces.purge_expired(store, 90) == 1 and traces_of(store) == []


def test_delete_on_request_by_user_and_all(make_client):
    client, store = make_client()
    _, key = make_acme(store)
    _, other = clients.create_client(store, "other", {"openai": "sk-o"})
    auth = {"authorization": f"Bearer {key}"}
    for user in ("u1", "u1", "u2"):
        client.post(OA, json={**CHAT, "messages": [{"role": "user", "content": f"hi from {user} {len(user)}"}]},
                    headers={**auth, "x-tokenwise-user": user})
    client.post(OA, json=CHAT, headers={"authorization": f"Bearer {other}", "x-tokenwise-user": "u1"})
    r = client.post("/tokenwise/v1/delete", json={"user": "u1"}, headers=auth)
    assert r.json() == {"ok": True, "deleted": 2}
    left = traces_of(store)
    assert sorted(t["engineer"] for t in left) == ["u1", "u2"]  # other client's u1 untouched
    assert client.post("/tokenwise/v1/delete", json={"all": True}, headers=auth).json()["deleted"] == 1
    assert client.post("/tokenwise/v1/delete", json={}, headers=auth).status_code == 400
    [entry] = store.query("SELECT * FROM audit WHERE action = 'delete' AND target = 'user:u1'")
    assert entry["actor"] == "client:acme" and entry["detail"] == "2 traces deleted"


def test_audit_log_for_dashboard_and_export(make_client, tmp_path):
    client, store = make_client()
    _, key = make_acme(store)
    client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    client.get("/dashboard")
    export.main(["--db", store.path, "--out", str(tmp_path / "e")])
    actions = [r["action"] for r in store.query("SELECT action FROM audit ORDER BY id")]
    assert "view_dashboard" in actions and "export" in actions


def test_export_decrypts_sealed_traces(make_client, tmp_path):
    client, store = make_client()
    _, key = make_acme(store)
    client.post(OA, json=CHAT, headers={"authorization": f"Bearer {key}"})
    assert export.main(["--db", store.path, "--out", str(tmp_path / "d")]) == 0
    [ex] = [json.loads(x) for x in (tmp_path / "d" / "train.jsonl").read_text().splitlines()]
    assert ex["messages"][-1]["content"] == "Your reading"


def test_admin_cli_data_controls(tmp_path, monkeypatch, capsys):
    from tokenwise.db import Store
    db = str(tmp_path / "c.db")
    monkeypatch.setenv("K", "sk")
    admin_main(["--db", db, "create", "acme", "--openai-key-env", "K"])
    assert admin_main(["--db", db, "sample", "acme", "0.25"]) == 0
    assert admin_main(["--db", db, "retention", "acme", "30"]) == 0
    assert admin_main(["--db", db, "sample", "acme", "2"]) == 1
    capsys.readouterr()
    admin_main(["--db", db, "list"])
    assert "sample=25%" in capsys.readouterr().out
    store = Store(db)
    cid = store.query("SELECT id FROM clients")[0]["id"]
    store.insert("traces", {"id": "t1", "ts": 0, "client_id": cid, "engineer": "u9", "workflow_key": "w"})
    assert admin_main(["--db", db, "purge", "acme", "--user", "u9"]) == 0
    assert store.query("SELECT COUNT(*) AS n FROM traces")[0]["n"] == 0
    capsys.readouterr()
    admin_main(["--db", db, "audit"])
    out = capsys.readouterr().out
    assert "delete" in out and "sample" in out and "create" in out


# ------------------------------------------------------------------ bring your own key

def test_bring_your_own_key_is_used_but_never_stored(make_client, openai_upstream, monkeypatch):
    monkeypatch.setenv("TOKENWISE_REQUIRE_CLIENT_KEY", "1")
    client, store = make_client(require_client_key=True)
    _, tw = clients.create_client(store, "byok", {}, openai_base_url="https://openrouter.test/api/v1")
    r = client.post(OA, json=CHAT, headers={"authorization": "Bearer sk-their-own", "x-tokenwise-key": tw})
    assert r.status_code == 200
    sent = openai_upstream.requests[-1]
    assert sent.headers["authorization"] == "Bearer sk-their-own"
    assert "x-tokenwise-key" not in sent.headers and str(sent.url).startswith("https://openrouter.test/")
    [row] = store.query("SELECT * FROM requests")
    assert row["engineer"] == "byok"  # usage counted for the client
    with open(store.path, "rb") as f:
        assert b"sk-their-own" not in f.read()  # their key is never written down


def test_bring_your_own_key_needs_a_valid_gateway_key(make_client):
    client, store = make_client(require_client_key=True)
    bad = client.post(OA, json=CHAT, headers={"authorization": "Bearer sk-x", "x-tokenwise-key": "tw-nope"})
    assert bad.status_code == 401
    none = client.post(OA, json=CHAT, headers={"authorization": "Bearer sk-x"})
    assert none.status_code == 401  # require_client_key: a bare provider key is refused
