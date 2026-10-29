"""Synthetic case-note generation.

Labels come first: we sample a label spec (typology, red flags, amounts, channel, urgency), then write a
narrative that expresses it. The label is therefore known by construction, and a quality filter rejects any
narrative whose dollar amounts don't match the spec.

Two writers:
  --writer llm       a strong LLM via any OpenAI-compatible API (recommended for the real dataset)
  --writer template  offline templates with noise (typos, abbreviations, vague wording); good for
                     pipeline tests and as extra hard examples

    python -m typology.generate_data --writer llm --n 4000 --out data/raw.jsonl
    python -m typology.generate_data --writer template --n 4000 --out data/raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .schema import Extraction

SPEC: dict[str, dict[str, Any]] = {
    "account_takeover": dict(flags=["password reset", "new device", "unrecognized login", "new payee",
                                    "shipping address change"], k=(2, 4), channels=["online_banking", "mobile_app"],
                             urgency=["high", "high", "medium"], n_amounts=(1, 2), amount=(500, 9000)),
    "authorized_push_payment_scam": dict(flags=["social engineering call", "bank impersonation", "urgency pressure",
                                                "new payee", "remote access software", "crypto request"], k=(2, 4),
                                         channels=["phone", "phone", "sms", "email"], urgency=["high", "medium"],
                                         n_amounts=(1, 3), amount=(300, 15000)),
    "card_not_present_fraud": dict(flags=["denies authorizing transaction", "multiple merchants",
                                          "shipping address change"], k=(1, 2), channels=["card"],
                                   urgency=["medium", "low"], n_amounts=(1, 3), amount=(20, 1500)),
    "card_testing": dict(flags=["small test charges", "multiple merchants"], k=(1, 2), channels=["card"],
                         urgency=["medium", "high"], n_amounts=(1, 2), amount=(0.5, 4.99)),
    "money_mule": dict(flags=["multiple unrelated senders", "rapid fund movement", "recently opened account",
                              "account shared with third party"], k=(2, 3), channels=["online_banking", "mobile_app"],
                       urgency=["high", "medium"], n_amounts=(1, 3), amount=(500, 9000)),
    "synthetic_identity": dict(flags=["mismatched identity details", "recently opened account"], k=(1, 2),
                               channels=["online_banking", "branch"], urgency=["medium", "low"], n_amounts=(0, 2),
                               amount=(1000, 20000)),
    "first_party_fraud": dict(flags=["item received but disputed", "denies authorizing transaction"], k=(1, 1),
                              channels=["card", "online_banking"], urgency=["low", "medium"], n_amounts=(1, 1),
                              amount=(20, 2500)),
    "check_fraud": dict(flags=["altered check", "duplicate deposit"], k=(1, 2), channels=["check", "mobile_app",
                                                                                         "branch"],
                        urgency=["medium", "high"], n_amounts=(1, 2), amount=(200, 12000)),
    "elder_financial_exploitation": dict(flags=["caregiver involvement", "isolation from family", "gift card request",
                                                "urgency pressure"], k=(1, 3), channels=["branch", "phone"],
                                         urgency=["high", "medium"], n_amounts=(1, 3), amount=(200, 8000)),
    "not_fraud": dict(flags=[], k=(0, 0), channels=["card", "online_banking", "branch", "unknown"], urgency=["low"],
                      n_amounts=(0, 1), amount=(10, 500)),
}

STYLES = ["terse call-center note with abbreviations", "customer's own words, emotional and rambling",
          "investigator summary, formal", "chat transcript excerpt", "note with typos and missing punctuation",
          "vague note where the agent was unsure what happened"]


def sample_label(rng: random.Random, typology: str) -> dict[str, Any]:
    s = SPEC[typology]
    k = rng.randint(*s["k"])
    flags = sorted(rng.sample(s["flags"], k)) if k else []
    lo, hi = s["amount"]
    amounts = [round(rng.uniform(lo, hi), 2 if hi < 100 else 0) for _ in range(rng.randint(*s["n_amounts"]))]
    label = {"typology": typology, "red_flags": flags, "amounts": amounts, "channel": rng.choice(s["channels"]),
             "urgency": rng.choice(s["urgency"])}
    return Extraction(**label).model_dump()


# --------------------------------------------------------------------------- template writer
FLAG_PHRASES = {
    "new payee": ["a new payee was added right before", "sent to a payee set up that same day", "brand new recipient"],
    "social engineering call": ["got a call from someone saying", "caller walked them through it", "rang her up and said"],
    "bank impersonation": ["said they were from our fraud dept", "claimed to be the bank", "pretended to be from security team"],
    "urgency pressure": ["told to act right away or lose the money", "kept pressuring, said it had to be today",
                         "was rushed, 'no time to call anyone'"],
    "password reset": ["pw was reset that morning", "password changed without them knowing", "a reset email they never asked for"],
    "new device": ["login from a phone we've never seen", "new device on the acct", "unknown android device"],
    "unrecognized login": ["doesn't recognize the login", "sign-in from another state", "login they didn't do"],
    "small test charges": ["bunch of tiny charges", "several charges under five bucks", "lots of little one-dollar-ish txns"],
    "multiple merchants": ["at different online stores", "across many merchants", "from a bunch of sites"],
    "rapid fund movement": ["money came in and went right back out", "funds moved out within hours", "in and out same day"],
    "multiple unrelated senders": ["deposits from people he doesn't know", "several strangers sent money",
                                   "incoming transfers from unrelated names"],
    "mismatched identity details": ["SSN and DOB don't line up", "ID info doesn't match records", "address history inconsistent"],
    "recently opened account": ["acct only opened a few weeks ago", "new account, under 30 days", "recently opened"],
    "denies authorizing transaction": ["says they never made it", "did not authorize this", "claims it wasn't them"],
    "item received but disputed": ["carrier shows delivered", "tracking says signed for", "item was delivered per merchant"],
    "altered check": ["payee name on the check looks changed", "amount on check was altered", "check looks washed"],
    "duplicate deposit": ["same check deposited twice", "check was also cashed at another bank", "duplicate mobile deposit"],
    "caregiver involvement": ["her caregiver came with her", "home aide does the banking now", "caretaker handles her card"],
    "isolation from family": ["family says they can't reach her", "son was told not to call", "she's been cut off from relatives"],
    "gift card request": ["asked to buy gift cards", "paid with store gift cards", "read gift card numbers over phone"],
    "crypto request": ["told to put it in bitcoin", "moved it to a crypto ATM", "crypto wallet they were given"],
    "remote access software": ["installed AnyDesk for them", "let them remote into the laptop", "screen sharing app installed"],
    "shipping address change": ["shipping address was changed", "orders going to a new address", "ship-to changed last week"],
    "account shared with third party": ["let a 'friend' use the account", "gave login to someone online", "someone else uses the acct"],
}
OPENERS = ["Cust called in.", "Customer reports:", "Member states", "Per chat w/ cust -", "Branch referral:",
           "Investigator note:", "cx says"]
TYPO_BODY = {
    "account_takeover": "they lost access and money was sent out",
    "authorized_push_payment_scam": "they sent the money themselves because they believed the caller",
    "card_not_present_fraud": "online charges showed up but card is still in their wallet",
    "card_testing": "card got hit with test charges",
    "money_mule": "account is being used to pass money through",
    "synthetic_identity": "application details look stitched together",
    "first_party_fraud": "dispute filed but our records show the purchase was legit",
    "check_fraud": "issue with a deposited check",
    "elder_financial_exploitation": "elderly customer, spending pattern changed a lot",
    "not_fraud": "turned out to be a charge they forgot about, recurring subscription",
}
CHANNEL_HINT = {"phone": "over the phone", "sms": "via text message", "email": "through an email",
                "online_banking": "in online banking", "mobile_app": "on the mobile app", "card": "on the debit card",
                "check": "by check", "branch": "at the branch", "unknown": ""}
URGENCY_HINT = {"high": ["wants it stopped now, transfer may still be pending", "please expedite"],
                "medium": ["asked for a callback", "follow up this week"], "low": ["no rush", "fyi only"]}


def _fmt_amount(rng: random.Random, x: float) -> str:
    if x < 100:
        return f"${x:.2f}"
    choice = rng.random()
    if choice < 0.6:
        return f"${x:,.0f}" if x == int(x) else f"${x:,.2f}"
    if choice < 0.85:
        return f"{x:.0f} dollars" if x == int(x) else f"{x:.2f} dollars"
    return f"${x:.0f}" if x == int(x) else f"${x:.2f}"


def _noise(rng: random.Random, text: str) -> str:
    if rng.random() < 0.4:
        text = text.lower()
    if rng.random() < 0.4:
        for a, b in [("customer", "cust"), ("account", "acct"), ("transaction", "txn"), ("because", "bc")]:
            text = text.replace(a, b)
    if rng.random() < 0.5:  # typos, never inside numbers
        words = text.split(" ")
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(words))
            w = words[i]
            if len(w) > 4 and not any(ch.isdigit() for ch in w):
                j = rng.randrange(1, len(w) - 2)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        text = " ".join(words)
    return text


def template_narrative(rng: random.Random, label: dict[str, Any]) -> str:
    parts = [rng.choice(OPENERS), TYPO_BODY[label["typology"]]]
    if CHANNEL_HINT[label["channel"]]:
        parts[-1] += " " + CHANNEL_HINT[label["channel"]]
    parts[-1] += "."
    flag_parts = [rng.choice(FLAG_PHRASES[f]) + "." for f in label["red_flags"]]
    # shuffle sentence slots, but keep amounts in label order so "amounts in order of appearance" holds
    slots = ["body"] + ["flag"] * len(flag_parts) + ["amount"] * len(label["amounts"])
    rng.shuffle(slots)
    out, fi, ai = [rng.choice(OPENERS)], 0, 0
    for slot in slots:
        if slot == "body":
            out.append(parts[1])
        elif slot == "flag":
            out.append(flag_parts[fi])
            fi += 1
        else:
            a = label["amounts"][ai]
            out.append(rng.choice(["Amount", "Txn of", "Total of", "Another one for" if ai else "First was"]) + " "
                       + _fmt_amount(rng, a) + ".")
            ai += 1
    out.append(rng.choice(URGENCY_HINT[label["urgency"]]) + ".")
    return _noise(rng, " ".join(out))


# --------------------------------------------------------------------------- LLM writer
LLM_INSTRUCTIONS = """Write one realistic bank case note (60-150 words). Style: {style}.
It must clearly reflect this situation without naming the category:
- fraud typology: {typology}
- red flags, each expressed in your own words: {flags}
- dollar amounts, each mentioned exactly once and in this order: {amounts}
- channel: {channel}
- urgency: {urgency}
Do not mention any other dollar amounts. Do not use labels or bullet points. Output only the note."""


def llm_narrative(label: dict[str, Any], style: str, model: str, base_url: str, api_key: str,
                  temperature: float = 0.9) -> str:
    import requests

    prompt = LLM_INSTRUCTIONS.format(style=style, typology=label["typology"].replace("_", " "),
                                     flags=", ".join(label["red_flags"]) or "none (this may be a false alarm)",
                                     amounts=", ".join(f"${a:,.2f}" for a in label["amounts"]) or "none",
                                     channel=label["channel"], urgency=label["urgency"])
    for attempt in range(4):
        try:
            r = requests.post(f"{base_url.rstrip('/')}/chat/completions", timeout=60,
                              headers={"Authorization": f"Bearer {api_key}"},
                              json={"model": model, "temperature": temperature,
                                    "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("generation failed after retries")


# --------------------------------------------------------------------------- quality filter
AMOUNT_RE = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(k\b|dollars?)?", re.I)


DOLLAR_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(k\b)?", re.I)


def amounts_match(text: str, amounts: list[float]) -> bool:
    """Every labeled amount appears in the text (handles $4,200 / 4200 dollars / $4.2k), and every
    $-prefixed number in the text is a labeled amount (no stray, unlabeled dollar figures)."""
    def parse(pairs):
        return [round(float(n.replace(",", "")) * (1000 if s and s.lower() == "k" else 1), 2) for n, s in pairs]

    found = parse(AMOUNT_RE.findall(text))
    dollars = parse(DOLLAR_RE.findall(text))
    labeled_present = all(any(abs(f - a) < 0.01 for f in found) for a in amounts)
    no_extra = all(any(abs(d - a) < 0.01 for a in amounts) for d in dollars)
    return labeled_present and no_extra


def generate(n: int, writer: str, out: Path, seed: int = 13, model: str | None = None, workers: int = 8) -> int:
    rng = random.Random(seed)
    from .schema import TYPOLOGIES

    specs = []
    for i in range(n):
        typ = TYPOLOGIES[i % len(TYPOLOGIES)]
        specs.append((f"S{i:05d}", sample_label(rng, typ), rng.choice(STYLES), random.Random(seed * 100003 + i)))

    def work(spec):
        sid, label, style, r = spec
        if writer == "template":
            text = template_narrative(r, label)
        else:
            text = llm_narrative(label, style, model or os.getenv("GEN_MODEL", "gpt-4o-mini"),
                                 os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), os.environ["OPENAI_API_KEY"])
        ok = amounts_match(text, label["amounts"])
        return {"id": sid, "narrative": text, "label": label, "writer": writer, "style": style, "passed_filter": ok}

    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with ThreadPoolExecutor(max_workers=1 if writer == "template" else workers) as pool, open(out, "w") as f:
        for rec in pool.map(work, specs):
            if rec["passed_filter"]:
                f.write(json.dumps(rec) + "\n")
                kept += 1
    print(f"Wrote {kept}/{n} records to {out} (rest failed the amount filter)")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--writer", choices=["llm", "template"], default="llm")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("data/raw.jsonl"))
    ap.add_argument("--model", default=None, help="generation model (default GEN_MODEL or gpt-4o-mini)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    generate(a.n, a.writer, a.out, a.seed, a.model, a.workers)
