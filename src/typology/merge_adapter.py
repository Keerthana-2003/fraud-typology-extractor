"""Merge the LoRA adapter into the base weights so vLLM can serve a single model directory.

    python -m typology.merge_adapter --base Qwen/Qwen2.5-3B-Instruct --adapter outputs/qwen3b-qlora \
        --out models/qwen3b-fraud-typology
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen3b-qlora")
    ap.add_argument("--out", default="models/qwen3b-fraud-typology")
    ap.add_argument("--push-to-hub", default=None, help="optional repo id, e.g. your-name/qwen3b-fraud-typology")
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16, device_map="auto")
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
    if a.push_to_hub:
        merged.push_to_hub(a.push_to_hub)
        AutoTokenizer.from_pretrained(a.adapter).push_to_hub(a.push_to_hub)
    print(f"Merged model saved to {a.out}")


if __name__ == "__main__":
    main()
