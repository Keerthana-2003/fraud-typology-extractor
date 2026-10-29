"""Parsing and metrics: typology accuracy + macro-F1, JSON validity, field-level exact match, latency, cost."""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from .schema import TYPOLOGIES, Extraction

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_output(text: str) -> tuple[dict[str, Any] | None, str]:
    """Return (validated dict or None, status). status: ok | invalid_json | schema_error."""
    t = FENCE_RE.sub("", (text or "").strip())
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None, "invalid_json"
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None, "invalid_json"
    try:
        return Extraction(**obj).model_dump(), "ok"
    except Exception:
        return None, "schema_error"


def macro_f1(gold: list[str], pred: list[str | None], labels: list[str] = TYPOLOGIES) -> float:
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def _set_f1(gold: set, pred: set) -> tuple[int, int, int]:
    return len(gold & pred), len(pred - gold), len(gold - pred)


def score(golds: list[dict[str, Any]], raw_outputs: list[str], latencies_s: list[float] | None = None,
          usage: list[tuple[int, int]] | None = None, price_per_1m: tuple[float, float] | None = None,
          gpu_cost_per_hour: float | None = None, wall_time_s: float | None = None) -> dict[str, Any]:
    parsed = [parse_output(o) for o in raw_outputs]
    preds = [p for p, _ in parsed]
    n = len(golds)
    gold_typ = [g["typology"] for g in golds]
    pred_typ = [p["typology"] if p else None for p in preds]
    em = {f: 0 for f in ["typology", "red_flags", "amounts", "channel", "urgency"]}
    all_exact = 0
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        if p is None:
            fn += len(g["red_flags"])
            continue
        checks = {
            "typology": p["typology"] == g["typology"],
            "red_flags": set(p["red_flags"]) == set(g["red_flags"]),
            "amounts": sorted(p["amounts"]) == sorted(round(a, 2) for a in g["amounts"]),
            "channel": p["channel"] == g["channel"],
            "urgency": p["urgency"] == g["urgency"],
        }
        for k, v in checks.items():
            em[k] += v
        all_exact += all(checks.values())
        a, b, c = _set_f1(set(g["red_flags"]), set(p["red_flags"]))
        tp, fp, fn = tp + a, fp + b, fn + c
    out: dict[str, Any] = {
        "n": n,
        "typology_accuracy": sum(p == g for p, g in zip(pred_typ, gold_typ)) / n,
        "typology_macro_f1": macro_f1(gold_typ, pred_typ),
        "json_validity": sum(s == "ok" for _, s in parsed) / n,
        "invalid_json_rate": sum(s == "invalid_json" for _, s in parsed) / n,
        "schema_error_rate": sum(s == "schema_error" for _, s in parsed) / n,
        "field_exact_match": {k: v / n for k, v in em.items()},
        "all_fields_exact_match": all_exact / n,
        "red_flag_f1": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0,
    }
    if latencies_s:
        lat = np.array(latencies_s)
        out.update(latency_mean_ms=float(lat.mean() * 1000), latency_p50_ms=float(np.percentile(lat, 50) * 1000),
                   latency_p95_ms=float(np.percentile(lat, 95) * 1000))
    if usage and price_per_1m:
        tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
        out["cost_per_1k_requests_usd"] = (tin * price_per_1m[0] + tout * price_per_1m[1]) / 1e6 / n * 1000
    elif gpu_cost_per_hour and wall_time_s:
        out["cost_per_1k_requests_usd"] = gpu_cost_per_hour / 3600 * wall_time_s / n * 1000
    return out
