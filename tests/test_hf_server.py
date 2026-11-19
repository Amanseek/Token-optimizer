import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "serve"))

import hf_server  # noqa: E402

KEY = "test-key"
AUTH = {"authorization": f"Bearer {KEY}"}


class FakeEngine:
    def __init__(self):
        self.calls = []

    def models(self):
        return ["Qwen/base", "demo-scam"]

    def generate(self, name, messages, max_tokens, temperature, top_p, stop):
        self.calls.append((name, messages, max_tokens))
        return '{"v":"scam"} END extra', 12, 5, "stop"


@pytest.fixture
def client():
    eng = FakeEngine()
    return TestClient(hf_server.create_app(eng, KEY)), eng


def test_needs_key(client):
    c, _ = client
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert [m["id"] for m in c.get("/v1/models", headers=AUTH).json()["data"]] == ["Qwen/base", "demo-scam"]
    with pytest.raises(SystemExit):
        hf_server.create_app(FakeEngine(), "")


def test_chat_picks_adapter_and_returns_openai_shape(client):
    c, eng = client
    r = c.post("/v1/chat/completions", headers=AUTH, json={
        "model": "demo-scam", "max_tokens": 50, "stop": "END",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == '{"v":"scam"} '
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    assert eng.calls == [("demo-scam", [{"role": "user", "content": "hi"}], 50)]


def test_unknown_model_and_stream(client):
    c, _ = client
    msgs = [{"role": "user", "content": "hi"}]
    assert c.post("/v1/chat/completions", headers=AUTH, json={"model": "nope", "messages": msgs}).status_code == 404
    r = c.post("/v1/chat/completions", headers=AUTH, json={
        "model": "demo-scam", "messages": msgs, "stream": True, "stream_options": {"include_usage": True}})
    lines = [x[6:] for x in r.text.splitlines() if x.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(x) for x in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["content"].startswith('{"v":"scam"}')
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and chunks[-1]["usage"]["total_tokens"] == 17


def test_finds_adapters_and_base(tmp_path):
    for name, base in [("b-astro", "Qwen/x"), ("a-scam", "Qwen/x")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    (tmp_path / "not-an-adapter").mkdir()
    found = hf_server.find_adapters(tmp_path)
    assert list(found) == ["a-scam", "b-astro"]
    assert hf_server.base_of(found) == "Qwen/x"
    (tmp_path / "b-astro" / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Other"}))
    with pytest.raises(SystemExit):
        hf_server.base_of(hf_server.find_adapters(tmp_path))
