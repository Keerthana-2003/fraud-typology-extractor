"""QLoRA fine-tuning with Hugging Face transformers + peft + trl (SFTTrainer).

Fits on a free Colab/Kaggle T4 (16 GB) for 1-4B models. With recent TRL versions the loss is computed on the
assistant (JSON) tokens only; older versions fall back to training on the full conversation.

    python -m typology.train_qlora --model Qwen/Qwen2.5-3B-Instruct --out outputs/qwen3b-qlora
"""
from __future__ import annotations

import argparse
import inspect
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="outputs/qwen3b-qlora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="none", help="none | wandb | tensorboard")
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype),
        torch_dtype=compute_dtype, device_map="auto")
    model.config.use_cache = False

    peft_config = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                             task_type="CAUSAL_LM", target_modules="all-linear")

    ds = load_dataset("json", data_files={"train": a.train, "validation": a.val})
    ds = ds.remove_columns([c for c in ds["train"].column_names if c != "messages"])

    # TRL renamed a few arguments across versions; pick whichever this install supports.
    cfg_params = inspect.signature(SFTConfig.__init__).parameters
    cfg = dict(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
               warmup_ratio=0.03, per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
               gradient_accumulation_steps=a.grad_accum, gradient_checkpointing=True, logging_steps=10,
               eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100, save_total_limit=2,
               load_best_model_at_end=True, metric_for_best_model="eval_loss", bf16=bf16, fp16=not bf16,
               report_to=a.report_to, seed=a.seed)
    cfg["max_length" if "max_length" in cfg_params else "max_seq_length"] = a.max_length
    if "completion_only_loss" in cfg_params:
        # conversational prompt/completion format: loss only on the assistant's JSON answer
        ds = ds.map(lambda ex: {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]},
                    remove_columns=["messages"])
        cfg["completion_only_loss"] = True
    if "eval_strategy" not in cfg_params:
        cfg["evaluation_strategy"] = cfg.pop("eval_strategy")
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    tok_kw = {"processing_class": tokenizer} if "processing_class" in trainer_params else {"tokenizer": tokenizer}

    trainer = SFTTrainer(model=model, args=SFTConfig(**cfg), train_dataset=ds["train"],
                         eval_dataset=ds["validation"], peft_config=peft_config, **tok_kw)
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(a.out)
    tokenizer.save_pretrained(a.out)
    with open(f"{a.out}/train_config.json", "w") as f:
        json.dump({**vars(a), "final_eval": trainer.evaluate()}, f, indent=2)
    print(f"Adapter saved to {a.out}")


if __name__ == "__main__":
    main()
