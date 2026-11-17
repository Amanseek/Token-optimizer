from tokenwise.config import Settings
from tokenwise.db import Store
from tokenwise.routing import classify, conversation_key, decide

S = Settings(db_path=":memory:", routing_mode="on")
TOOLS = [{"name": "Bash", "input_schema": {}}]


def body(text, tools=TOOLS, model="claude-opus-5-5"):
    return {"model": model, "max_tokens": 10, "tools": tools, "messages": [{"role": "user", "content": text}]}


def test_simple_task_routes_to_cheap():
    target, reason = classify(body("Can you explain what this function does?"), S)
    assert target == "claude-haiku-5-5"


def test_hard_task_stays():
    target, _ = classify(body("Explain and then refactor the auth module"), S)
    assert target is None


def test_unknown_task_stays():
    target, reason = classify(body("look at main.py"), S)
    assert target is None and reason == "no simple-task signal"


def test_short_toolless_request_routes():
    target, _ = classify(body("Write a 5 word title for this chat", tools=None), S)
    assert target == "claude-haiku-5-5"


def test_large_context_stays():
    target, reason = classify(body("explain " + "x" * 200_000), S)
    assert target is None and reason == "context too large"


def test_non_routable_model():
    target, _ = classify(body("explain this", model="claude-haiku-5-5"), S)
    assert target is None


def test_images_stay():
    b = body("explain this")
    b["messages"][0]["content"] = [{"type": "image", "source": {}}, {"type": "text", "text": "explain this"}]
    assert classify(b, S)[0] is None


def test_decision_is_sticky(tmp_path):
    store = Store(str(tmp_path / "r.db"))
    first = body("explain this code")
    conv = conversation_key(first, "aman")
    d1 = decide(first, conv, S, store)
    assert d1.apply and d1.model_to_send == "claude-haiku-5-5"

    # Later in the same conversation the user asks for something hard; model must not flip.
    later = body("explain this code")
    later["messages"] += [{"role": "assistant", "content": "ok"},
                          {"role": "user", "content": "now refactor everything"}]
    assert conversation_key(later, "aman") == conv
    d2 = decide(later, conv, S, store)
    assert d2.model_to_send == "claude-haiku-5-5"


def test_shadow_mode_never_changes_model(tmp_path):
    store = Store(str(tmp_path / "r.db"))
    s = Settings(db_path=":memory:", routing_mode="shadow")
    b = body("explain this")
    d = decide(b, conversation_key(b, "x"), s, store)
    assert d.target_model == "claude-haiku-5-5"
    assert not d.apply and d.model_to_send == "claude-opus-5-5"
