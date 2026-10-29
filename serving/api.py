"""FastAPI wrapper around the vLLM OpenAI-compatible server.

Adds request validation, output validation against the Extraction schema (one repair retry on invalid JSON),
timeouts, and latency reporting. This is the endpoint the investigation agent calls as `typology_classifier`.

    VLLM_URL=http://localhost:8001/v1 MODEL_NAME=ft uvicorn serving.api:app --port 8000
"""
from __future__ import annotations

import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from typology.metrics import parse_output
from typology.prompts import messages
from typology.schema import Extraction, ExtractRequest

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "ft")
TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S", "30"))

app = FastAPI(title="Fraud Typology Extractor", version="0.1.0")
client = httpx.AsyncClient(timeout=TIMEOUT_S)


class ExtractResponse(BaseModel):
    result: Extraction
    model: str
    latency_ms: float
    repaired: bool


async def _complete(msgs: list[dict]) -> str:
    r = await client.post(f"{VLLM_URL}/chat/completions",
                          json={"model": MODEL_NAME, "messages": msgs, "temperature": 0, "max_tokens": 256})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


@app.get("/health")
async def health() -> dict:
    try:
        r = await client.get(f"{VLLM_URL}/models", timeout=5)
        return {"status": "ok" if r.status_code == 200 else "degraded", "model": MODEL_NAME}
    except httpx.HTTPError:
        return {"status": "upstream_unavailable", "model": MODEL_NAME}


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    t0 = time.perf_counter()
    msgs = messages(req.text)
    try:
        raw = await _complete(msgs)
        parsed, status = parse_output(raw)
        repaired = False
        if parsed is None:  # one repair attempt with the error fed back
            msgs = msgs + [{"role": "assistant", "content": raw},
                           {"role": "user", "content": f"That output was {status}. Return only the valid JSON object."}]
            raw = await _complete(msgs)
            parsed, status = parse_output(raw)
            repaired = True
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"model backend unavailable: {type(e).__name__}") from e
    if parsed is None:
        raise HTTPException(status_code=502, detail=f"model returned {status} output")
    return ExtractResponse(result=Extraction(**parsed), model=MODEL_NAME,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 1), repaired=repaired)
