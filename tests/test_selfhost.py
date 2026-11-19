import time

from tokenwise import dashboard, selfhost
from tokenwise.db import Store

PRICES = '{"gemini-2.5-flash": [0.30, 2.50]}'


def log(store, model, inp, out, cost, baseline, ts=None):
    store.log_request({"engineer": "app", "provider": "openai", "model_requested": model, "model_used": model, "status": 200,
                       "input_tokens": inp, "output_tokens": out, "cost_usd": cost, "baseline_cost_usd": baseline,
                       **({"ts": ts} if ts else {})})


def test_compare_to_forms(monkeypatch):
    monkeypatch.delenv("TOKENWISE_OPENAI_COMPARE_TO", raising=False)
    assert selfhost.compare_to("demo-scam") is None
    monkeypatch.setenv("TOKENWISE_OPENAI_COMPARE_TO", "gemini-2.5-flash")
    assert selfhost.compare_to("anything") == "gemini-2.5-flash"
    monkeypatch.setenv("TOKENWISE_OPENAI_COMPARE_TO", '{"demo-scam": "gpt-5-mini"}')
    assert selfhost.compare_to("demo-scam") == "gpt-5-mini" and selfhost.compare_to("gpt-4o") is None


def test_own_model_is_priced_against_the_api_it_replaces(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import httpx

    from tokenwise.app import create_app

    monkeypatch.setenv("TOKENWISE_OPENAI_PRICES", PRICES)
    monkeypatch.setenv("TOKENWISE_OPENAI_COMPARE_TO", "gemini-2.5-flash")
    monkeypatch.setenv("TOKENWISE_DB", str(tmp_path / "t.db"))

    def upstream(request):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 200}})

    app = create_app(openai_transport=httpx.MockTransport(upstream))
    with TestClient(app) as c:
        r = c.post("/openai/v1/chat/completions", headers={"authorization": "Bearer sk-x",
                                                           "x-tokenwise-upstream": "http://gpu:8000/v1"},
                   json={"model": "demo-scam", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
    row = app.state.store.query("SELECT cost_usd, baseline_cost_usd FROM requests")[-1]
    assert row["cost_usd"] == 0  # own model: no per-token price
    assert abs(row["baseline_cost_usd"] - (1000 * 0.30 + 200 * 2.50) / 1e6) < 1e-12


def test_report_counts_gpu_and_breakeven(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKENWISE_OPENAI_COMPARE_TO", '{"demo-scam": "gemini-2.5-flash"}')
    monkeypatch.setenv("TOKENWISE_GPU_COST_PER_HOUR", "1.40")
    monkeypatch.setenv("TOKENWISE_INR_PER_USD", "88")
    store = Store(str(tmp_path / "t.db"))
    now = time.time()
    for _ in range(3):
        log(store, "demo-scam", 235, 42, 0.0, 0.0002, ts=now - 60)
    log(store, "gpt-5-mini", 100, 10, 0.01, 0.01)  # not self-hosted: left out of the report
    sid = selfhost.start_session(store)
    store.execute("UPDATE uptime SET started = ? WHERE id = ?", (now - 1800, sid))  # ran for half an hour

    r = selfhost.report(store, now - 86400)
    assert r["requests"] == 3 and r["models"] == ["demo-scam"] and r["compare_to"] == ["gemini-2.5-flash"]
    assert abs(r["gpu_hours"] - 0.5) < 0.01 and abs(r["gpu_cost"] - 0.70) < 0.02
    assert r["saved"] < 0  # 3 requests can't pay for half an hour of GPU
    assert round(r["breakeven_per_hour"]) == 7000  # 1.40 / 0.0002
    assert r["busiest_hour"] == 3

    s = dashboard.summary(store)
    assert abs(s["totals"]["cost"] - (0.01 + r["gpu_cost"])) < 0.02  # GPU counted in Spend
    page = dashboard.render_html(store)
    assert "Own GPU vs hosted API" in page and "7,000 requests per hour" in page and "₹" in page
    assert "the API would have been cheaper" in page


def test_no_section_without_config(monkeypatch, tmp_path):
    monkeypatch.delenv("TOKENWISE_OPENAI_COMPARE_TO", raising=False)
    store = Store(str(tmp_path / "t.db"))
    log(store, "demo-scam", 10, 10, 0, 0)
    assert selfhost.report(store, 0) is None
    assert "Own GPU" not in dashboard.render_html(store)
