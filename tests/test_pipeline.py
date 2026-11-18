import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import evaluate  # noqa: E402
import train  # noqa: E402


def write(tmp_path, rows):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"messages": m}) for m in rows))
    return str(p)


CONVO = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def test_train_examples_are_prompt_completion(tmp_path):
    ex = train.load_examples(write(tmp_path, [CONVO, CONVO[:2]]))
    assert ex == [{"prompt": CONVO[:2], "completion": [CONVO[2]]}]  # the one without an answer is skipped


def test_scores_free_text():
    assert evaluate.f1("the parcel ships today", "Parcel ships today!") > 0.8
    assert evaluate.f1("", "x") == 0.0
    rows = [{"prompt": CONVO[:2], "reference": "ships today"}] * 2
    s = evaluate.score(rows, [{"answer": "ships today", "error": None, "ms": 10},
                              {"answer": "no idea", "error": None, "ms": 30}])
    assert s["exact"] == 0.5 and s["main_metric"] == "f1" and s["latency_ms"] == 20


def test_scores_json_fields():
    ref = json.dumps({"v": "scam", "c": "kyc_bank"})
    rows = [{"prompt": CONVO[:2], "reference": ref}] * 2
    outs = [{"answer": '```json\n{"v": "scam", "c": "upi_otp"}\n```', "error": None, "ms": 5},
            {"answer": "garbage", "error": None, "ms": 5}]
    s = evaluate.score(rows, outs)
    assert s["main_metric"] == "json_fields" and s["json_valid"] == 0.5 and s["json_fields"] == 0.25


def test_load_test(tmp_path):
    [r] = evaluate.load_test(write(tmp_path, [CONVO]), None)
    assert r["reference"] == "a" and r["prompt"] == CONVO[:2]
