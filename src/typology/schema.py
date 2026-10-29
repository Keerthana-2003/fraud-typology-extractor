"""Output schema and controlled vocabularies. Controlled values make field-level exact match meaningful."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

TYPOLOGIES = [
    "account_takeover",
    "authorized_push_payment_scam",
    "card_not_present_fraud",
    "card_testing",
    "money_mule",
    "synthetic_identity",
    "first_party_fraud",
    "check_fraud",
    "elder_financial_exploitation",
    "not_fraud",
]

RED_FLAGS = [
    "new payee", "social engineering call", "bank impersonation", "urgency pressure", "password reset",
    "new device", "unrecognized login", "small test charges", "multiple merchants", "rapid fund movement",
    "multiple unrelated senders", "mismatched identity details", "recently opened account",
    "denies authorizing transaction", "item received but disputed", "altered check", "duplicate deposit",
    "caregiver involvement", "isolation from family", "gift card request", "crypto request",
    "remote access software", "shipping address change", "account shared with third party",
]

CHANNELS = ["phone", "sms", "email", "online_banking", "mobile_app", "card", "check", "branch", "unknown"]

Typology = Literal[tuple(TYPOLOGIES)]  # type: ignore[valid-type]
Channel = Literal[tuple(CHANNELS)]  # type: ignore[valid-type]
RedFlag = Literal[tuple(RED_FLAGS)]  # type: ignore[valid-type]


class Extraction(BaseModel):
    typology: Typology
    red_flags: list[RedFlag] = Field(default_factory=list)
    amounts: list[float] = Field(default_factory=list, description="Dollar amounts mentioned, in order")
    channel: Channel = "unknown"
    urgency: Literal["low", "medium", "high"]

    @field_validator("red_flags")
    @classmethod
    def unique_sorted(cls, v: list[str]) -> list[str]:
        return sorted(set(v))

    @field_validator("amounts")
    @classmethod
    def round_amounts(cls, v: list[float]) -> list[float]:
        return [round(float(x), 2) for x in v]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=10, max_length=4000)
