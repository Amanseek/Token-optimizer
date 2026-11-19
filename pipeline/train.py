"""Train a LoRA adapter for one client workflow from a Tokenwise export.

    python pipeline/train.py --train data/train.jsonl --out adapters/acme-support \\
        --base Qwen/Qwen3-4B-Instruct-2507

Input: chat JSONL as written by `python -m tokenwise.export` ({"messages": [...]}).
Each example's last assistant message is the target; the loss is computed only on it.

Plain transformers + PEFT + TRL (no unsloth), so the saved adapter loads the same way
in vLLM, PEFT or any other server. Use --qlora on small GPUs (T4 / L4) to train
the base model in 4-bit; on A100/H100 leave it off.

Fail-fast: --check N generates answers for N held-out examples before and after training
and exits with code 2 if the trained model's answers are empty, garbled or (when the targets
are JSON) not JSON. Training also stops at once if the loss becomes NaN/inf. Run a short
smoke test first (--max-steps 30 --check 4) so a broken setup costs minutes, not an hour.

Typical: 4B base, ~5,000 examples, 1 epoch -> ~20-40 min on an A100, ~60-90 min on a T4.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def load_examples(path: str) -> list[dict]:
    """Chat JSONL -> TRL conversational prompt/completion pairs."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            continue
        out.append({"prompt": msgs[:-1], "completion": [msgs[-1]]})
    return out


def looks_garbled(text: str) -> str | None:
    """Return a reason if a generated answer is obviously broken, else None."""
    t = text.strip()
    if not t:
        return "empty"
    odd = sum(1 for ch in t if not (ch.isprintable() or ch in "\n\t"))
    if odd / len(t) > 0.05:
        return "unprintable characters"
    words = t.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.2:
        return "repeating the same words"
    if len(t) >= 40 and max(t.count(t[i:i + 4]) for i in range(0, len(t) - 4, 4)) * 4 > 0.6 * len(t):
        return "repeating the same characters"
    return None


def is_json(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except ValueError:
        return False


def sanity_check(outputs: list[str], references: list[str]) -> list[str]:
    """Problems with the generated answers; an empty list means they look healthy."""
    problems = [f"answer {i + 1}: {why}" for i, o in enumerate(outputs) if (why := looks_garbled(o))]
    if references and all(is_json(r) for r in references):
        ok = sum(is_json(o) for o in outputs)
        if ok * 2 < len(outputs):
            problems.append(f"targets are JSON but only {ok}/{len(outputs)} answers are")
    return problems


def generate(model, tok, prompts: list[list[dict]], max_new_tokens: int) -> list[str]:
    import torch
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = True
    out = []
    for msgs in prompts:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        out.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    model.config.use_cache = use_cache
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (for quick tests)")
    ap.add_argument("--qlora", action="store_true", help="load the base in 4-bit (small GPUs)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="hold out N examples; fail (exit 2) if the trained model's answers to them look broken")
    ap.add_argument("--check-tokens", type=int, default=120)
    a = ap.parse_args(argv)

    import os
    # Plain `python train.py` on a multi-GPU box (Kaggle T4 x2): TRL refuses nn.DataParallel,
    # so use one GPU unless launched with accelerate/torchrun (which set LOCAL_RANK).
    if "LOCAL_RANK" not in os.environ:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class LossGuard(TrainerCallback):
        """Stop at once if the loss stops being a real number."""
        def __init__(self):
            self.losses, self.bad = [], None

        def on_log(self, args, state, control, logs=None, **kw):
            loss = (logs or {}).get("loss")
            if loss is None:
                return
            if not math.isfinite(loss):
                self.bad = f"loss became {loss} at step {state.global_step}"
                control.should_training_stop = True
            self.losses.append(loss)

    examples = load_examples(a.train)
    if not examples:
        raise SystemExit(f"no usable examples in {a.train}")
    held = []
    if a.check:
        held, examples = examples[-a.check:], examples[:-a.check]
    # Real bf16 needs compute capability 8.0+ (A100/L4/H100). Newer torch reports bf16 as
    # "supported" on a T4 (7.5) through emulation, but matmuls then fail, so check the hardware.
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"dtype": dtype, "device_map": {"": 0} if torch.cuda.is_available() else None}
    if a.qlora:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, **kwargs)
    if a.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    steps_per_epoch = math.ceil(len(examples) / (a.batch * a.grad_accum))
    import inspect
    # transformers 5 dropped warmup_ratio; a fractional warmup_steps means the same thing there.
    warmup = ({"warmup_ratio": 0.03} if "warmup_ratio" in inspect.signature(SFTConfig).parameters
              else {"warmup_steps": 0.03})
    args = SFTConfig(
        output_dir=str(Path(a.out) / "_checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", weight_decay=0.0, **warmup,
        logging_steps=max(1, (a.max_steps if a.max_steps > 0 else int(steps_per_epoch * a.epochs)) // 20), save_strategy="no", report_to="none", seed=a.seed,
        bf16=bf16, fp16=not bf16 and torch.cuda.is_available(), gradient_checkpointing=True,
        max_length=a.max_len, packing=False, padding_free=False, completion_only_loss=True,
    )
    guard = LossGuard()
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(examples),
                         processing_class=tok, peft_config=lora, callbacks=[guard])
    # fp16 mixed precision (T4) needs float32 trainable weights; some library versions create the
    # LoRA weights in the model's bf16, which makes the fp16 grad-scaler crash. Upcast them.
    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable), "dtypes:", sorted({str(p.dtype) for p in trainable}))
    if not bf16:
        for p in trainable:
            if p.dtype != torch.float32:
                p.data = p.data.float()
    prompts = [e["prompt"] for e in held]
    refs = [e["completion"][0]["content"] for e in held]
    if held:
        print("== before training (base model) ==")
        for o in generate(trainer.model, tok, prompts, a.check_tokens):
            print(" ", repr(o[:300]))
    t0 = time.time()
    stats = trainer.train()
    if guard.bad:
        print(f"CHECK FAILED: {guard.bad}")
        raise SystemExit(2)
    if len(guard.losses) >= 4 and guard.losses[-1] >= guard.losses[0]:
        print(f"WARNING: loss did not go down ({guard.losses[0]:.3f} -> {guard.losses[-1]:.3f})")
    out = Path(a.out)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    meta = {"base": a.base, "examples": len(examples), "epochs": a.epochs, "rank": a.rank, "qlora": a.qlora,
            "minutes": round((time.time() - t0) / 60, 1), "final_loss": round(stats.training_loss, 4)}
    if held:
        outputs = generate(trainer.model, tok, prompts, a.check_tokens)
        print("== after training ==")
        for o, r in zip(outputs, refs):
            print("  want", repr(r[:300])); print("  got ", repr(o[:300]))
        problems = sanity_check(outputs, refs)
        meta["check"] = {"answers": len(outputs), "problems": problems,
                         "exact": sum(o.strip() == r.strip() for o, r in zip(outputs, refs))}
    (out / "training.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    if held and meta["check"]["problems"]:
        print("CHECK FAILED:", "; ".join(meta["check"]["problems"]))
        raise SystemExit(2)
    if held:
        print("CHECK PASSED")


if __name__ == "__main__":
    main()
