"""Train a LoRA adapter for one client workflow from a Tokenwise export.

    python pipeline/train.py --train data/train.jsonl --out adapters/acme-support \\
        --base Qwen/Qwen3-4B-Instruct-2507

Input: chat JSONL as written by `python -m tokenwise.export` ({"messages": [...]}).
Each example's last assistant message is the target; the loss is computed only on it.

Plain transformers + PEFT + TRL (no unsloth), so the saved adapter loads the same way
in vLLM, PEFT or any other server. Use --qlora on small GPUs (T4 / L4) to train
the base model in 4-bit; on A100/H100 leave it off.

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
    a = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    examples = load_examples(a.train)
    if not examples:
        raise SystemExit(f"no usable examples in {a.train}")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
    args = SFTConfig(
        output_dir=str(Path(a.out) / "_checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        logging_steps=max(1, steps_per_epoch // 20), save_strategy="no", report_to="none", seed=a.seed,
        bf16=bf16, fp16=not bf16 and torch.cuda.is_available(), gradient_checkpointing=True,
        max_length=a.max_len, packing=False, padding_free=False, completion_only_loss=True,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(examples),
                         processing_class=tok, peft_config=lora)
    t0 = time.time()
    stats = trainer.train()
    out = Path(a.out)
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    meta = {"base": a.base, "examples": len(examples), "epochs": a.epochs, "rank": a.rank, "qlora": a.qlora,
            "minutes": round((time.time() - t0) / 60, 1), "final_loss": round(stats.training_loss, 4)}
    (out / "training.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
