"""Prompt templates shared by training, baselines, evaluation, and serving (so they never drift apart)."""
from __future__ import annotations

import json

from .schema import CHANNELS, RED_FLAGS, TYPOLOGIES

SYSTEM_PROMPT = f"""You extract structured data from fraud case notes and dispute descriptions.
Return ONLY a JSON object with these keys:
- "typology": one of {json.dumps(TYPOLOGIES)}
- "red_flags": list of zero or more of {json.dumps(RED_FLAGS)}
- "amounts": list of dollar amounts mentioned, as numbers, in the order they appear
- "channel": one of {json.dumps(CHANNELS)} (how the fraud reached the customer or account)
- "urgency": "low", "medium", or "high" (high if money is still moving, the customer is vulnerable, or funds can still be recovered)
No explanations, no markdown."""

FEW_SHOT = [
    ("Cust says she got a call from 'fraud dept' at the bank, they said her acct was compromised and told her to "
     "move $4,200 to a 'safe account'. New payee set up same day. She now thinks it was a scam, wants $ back asap.",
     {"typology": "authorized_push_payment_scam", "red_flags": ["bank impersonation", "new payee",
      "social engineering call", "urgency pressure"], "amounts": [4200.0], "channel": "phone", "urgency": "high"}),
    ("17 charges between $0.50 and $2.99 on the card overnight at different online stores, customer still has the "
     "card. No larger charges yet.",
     {"typology": "card_testing", "red_flags": ["multiple merchants", "small test charges"], "amounts": [0.5, 2.99],
      "channel": "card", "urgency": "medium"}),
    ("Member disputes a $129.99 charge for headphones, says package never came. Carrier shows delivered and signed "
     "for at his address. Third dispute this year.",
     {"typology": "first_party_fraud", "red_flags": ["item received but disputed"], "amounts": [129.99],
      "channel": "card", "urgency": "low"}),
]


def user_prompt(narrative: str) -> str:
    return f"Case note:\n{narrative.strip()}"


def messages(narrative: str, few_shot: bool = False, target: dict | None = None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for text, out in FEW_SHOT:
            msgs += [{"role": "user", "content": user_prompt(text)},
                     {"role": "assistant", "content": json.dumps(out)}]
    msgs.append({"role": "user", "content": user_prompt(narrative)})
    if target is not None:
        msgs.append({"role": "assistant", "content": json.dumps(target)})
    return msgs
