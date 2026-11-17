import copy

from tokenwise.config import Settings
from tokenwise.transforms import optimize, trim_text

S = Settings(db_path=":memory:")


def convo(tool_name: str, output: str, extra_turns: int = 0):
    msgs = [{"role": "user", "content": "run the tests"}]
    for i in range(extra_turns + 1):
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": tool_name, "input": {}}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": output}]})
    return {"model": "claude-opus-5-5", "max_tokens": 100, "messages": msgs}


def long_log(n=2000):
    lines = [f"ok line {i}" for i in range(n)]
    lines[1000] = "ERROR: something broke at foo.py:12"
    return "\n".join(lines)


def test_trim_keeps_head_tail_and_errors():
    out = trim_text(long_log(), S)
    assert out.startswith("ok line 0")
    assert out.endswith("ok line 1999")
    assert "ERROR: something broke" in out
    assert "lines trimmed" in out
    assert len(out) < len(long_log()) / 5


def test_short_output_untouched():
    assert trim_text("short", S) == "short"


def test_bash_output_trimmed_but_file_reads_never():
    bash, stats = optimize(convo("Bash", long_log()), S)
    assert "lines trimmed" in bash["messages"][2]["content"][0]["content"]
    assert stats.trimmed_chars > 0

    read, stats = optimize(convo("Read", long_log()), S)
    assert read["messages"][2]["content"][0]["content"] == long_log()
    assert stats.trimmed_chars == 0


def test_dedupe_replaces_only_later_copy():
    body = convo("Read", "x" * 3000, extra_turns=1)
    out, stats = optimize(body, S)
    first = out["messages"][2]["content"][0]["content"]
    second = out["messages"][4]["content"][0]
    assert first == "x" * 3000
    assert "identical to the earlier result" in second["content"]
    assert stats.deduped_chars > 2000


def test_original_body_not_mutated():
    body = convo("Bash", long_log())
    before = copy.deepcopy(body)
    optimize(body, S)
    assert body == before


def test_transform_is_deterministic_for_cache_prefix():
    """Older messages must come out identical on the next request, or the cache breaks."""
    turn1 = convo("Bash", long_log())
    turn2 = copy.deepcopy(turn1)
    turn2["messages"] += [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t9", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": long_log(3000)}]},
    ]
    s = Settings(db_path=":memory:", cache_enabled=False)
    out1, _ = optimize(turn1, s)
    out2, _ = optimize(turn2, s)
    assert out2["messages"][: len(out1["messages"])] == out1["messages"]


def test_cache_breakpoints_added():
    body = {
        "model": "claude-opus-5-5", "max_tokens": 10,
        "system": "You are helpful",
        "tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    out, stats = optimize(body, S)
    assert stats.cache_breakpoints_added == 3
    assert out["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert out["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert out["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_existing_breakpoints_respected():
    body = {
        "model": "claude-opus-5-5", "max_tokens": 10,
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
    }
    out, stats = optimize(body, S)
    assert stats.cache_breakpoints_added == 0


def test_never_more_than_four_breakpoints():
    cc = {"type": "ephemeral"}
    body = {
        "model": "claude-opus-5-5", "max_tokens": 10,
        "tools": [{"name": "a", "input_schema": {}}],
        "system": [{"type": "text", "text": "s"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": str(i), "cache_control": cc} for i in range(4)]}],
    }
    out, stats = optimize(body, S)
    assert stats.cache_breakpoints_added == 0


def test_one_hour_ttl_upgrades_all():
    s = Settings(db_path=":memory:", cache_ttl="1h")
    body = {
        "model": "claude-opus-5-5", "max_tokens": 10,
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    out, _ = optimize(body, s)
    assert out["system"][0]["cache_control"]["ttl"] == "1h"
