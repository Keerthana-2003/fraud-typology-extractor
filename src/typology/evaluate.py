"""Evaluate any OpenAI-compatible endpoint on the verified test set.

Works for the frontier API (OpenAI), the base model served by vLLM (zero-/few-shot baselines), and the
fine-tuned model served by vLLM. Each run appends a row to results/results.json and rewrites results/results.md.

    # frontier model, few-shot
    python -m typology.evaluate --name "gpt-4o (few-shot)" --model gpt-4o --few-shot \
        --base-url https://api.openai.com/v1 --price-in 2.50 --price-out 10.00

    # base model on vLLM, zero-shot (GPU cost for cost-per-1k)
    python -m typology.evaluate --name "Qwen2.5-3B base (zero-shot)" --model base \
        --base-url http://localhost:8001/v1 --gpu-cost-per-hour 0.80 --concurrency 32

    # fine-tuned model on vLLM
    python -m typology.evaluate --name "Qwen2.5-3B QLoRA (ours)" --model ft \
        --base-url http://localhost:8001/v1 --gpu-cost-per-hour 0.80 --concurrency 32
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from .metrics import score
from .prompts import messages


def call(base_url: str, model: str, msgs: list[dict], api_key: str, max_tokens: int = 256,
         timeout: float = 120) -> tuple[str, float, tuple[int, int]]:
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            r = requests.post(f"{base_url.rstrip('/')}/chat/completions", timeout=timeout,
                              headers={"Authorization": f"Bearer {api_key}"},
                              json={"model": model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens})
            r.raise_for_status()
            body = r.json()
            u = body.get("usage") or {}
            return (body["choices"][0]["message"]["content"], time.perf_counter() - t0,
                    (u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
        except requests.RequestException:
            if attempt == 2:
                return "", time.perf_counter() - t0, (0, 0)
            time.sleep(2 ** attempt)
    return "", time.perf_counter() - t0, (0, 0)


def run(test_path: Path, name: str, model: str, base_url: str, few_shot: bool, concurrency: int,
        price: tuple[float, float] | None, gpu_cost_per_hour: float | None, limit: int | None,
        results_dir: Path) -> dict[str, Any]:
    recs = [json.loads(line) for line in open(test_path)][:limit]
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outs = list(pool.map(lambda r: call(base_url, model, messages(r["narrative"], few_shot=few_shot), api_key),
                             recs))
    wall = time.perf_counter() - t0
    metrics = score([r["label"] for r in recs], [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs],
                    price, gpu_cost_per_hour, wall)
    metrics.update(name=name, model=model, few_shot=few_shot, concurrency=concurrency,
                   throughput_rps=len(recs) / wall, date=time.strftime("%Y-%m-%d"))

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"predictions_{name.replace(' ', '_').replace('/', '-')}.jsonl", "w") as f:
        for r, o in zip(recs, outs):
            f.write(json.dumps({"id": r["id"], "gold": r["label"], "raw_output": o[0]}) + "\n")
    path = results_dir / "results.json"
    allres = json.loads(path.read_text()) if path.exists() else []
    allres = [x for x in allres if x["name"] != name] + [metrics]
    path.write_text(json.dumps(allres, indent=2))
    (results_dir / "results.md").write_text(to_markdown(allres))
    print(json.dumps(metrics, indent=2))
    return metrics


def to_markdown(rows: list[dict[str, Any]]) -> str:
    head = ("| Model | Typology acc. | Macro-F1 | JSON validity | All-fields EM | Red-flag F1 | p95 latency | "
            "Cost / 1K requests |\n|---|---|---|---|---|---|---|---|")
    lines = ["# Results\n", f"Test set: hand-verified held-out examples (n = {rows[0]['n']})\n", head]
    for r in rows:
        cost = r.get("cost_per_1k_requests_usd")
        lines.append(f"| {r['name']} | {r['typology_accuracy']:.1%} | {r['typology_macro_f1']:.3f} | "
                     f"{r['json_validity']:.1%} | {r['all_fields_exact_match']:.1%} | {r['red_flag_f1']:.3f} | "
                     f"{r.get('latency_p95_ms', 0):.0f} ms | " + (f"${cost:.3f}" if cost is not None else "n/a") + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="row label in the results table")
    ap.add_argument("--model", required=True, help="model name at the endpoint (vLLM: --served-model-name)")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--few-shot", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (API models)")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens (API models)")
    ap.add_argument("--gpu-cost-per-hour", type=float, default=None, help="for self-hosted models")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    a = ap.parse_args()
    price = (a.price_in, a.price_out) if a.price_in is not None and a.price_out is not None else None
    run(a.test, a.name, a.model, a.base_url, a.few_shot, a.concurrency, price, a.gpu_cost_per_hour, a.limit,
        a.results_dir)
