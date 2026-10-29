"""Offline tests for data generation, contamination control, parsing, and metrics (no GPU needed)."""
from __future__ import annotations

import json
import random

import pytest

from typology.generate_data import amounts_match, sample_label, template_narrative
from typology.metrics import macro_f1, parse_output, score
from typology.prepare_data import drop_near_duplicates, jaccard, _shingles
from typology.prompts import FEW_SHOT, messages
from typology.schema import TYPOLOGIES, Extraction


def test_labels_are_valid_for_every_typology():
    rng = random.Random(0)
    for t in TYPOLOGIES:
        for _ in range(20):
            Extraction(**sample_label(rng, t))


def test_template_narratives_contain_labeled_amounts():
    rng = random.Random(1)
    for t in TYPOLOGIES:
        for _ in range(30):
            lab = sample_label(rng, t)
            assert amounts_match(template_narrative(rng, lab), lab["amounts"])


def test_amount_filter_rejects_missing_and_extra_amounts():
    assert amounts_match("sent $4,200 then 300 dollars", [4200, 300])
    assert amounts_match("about $4.2k gone", [4200])
    assert not amounts_match("sent $4,200", [4200, 300])
    assert not amounts_match("sent $4,200 and $99 fee", [4200])


def test_few_shot_examples_match_schema():
    for _, out in FEW_SHOT:
        Extraction(**out)


def test_messages_format():
    m = messages("note", few_shot=True, target={"typology": "not_fraud"})
    assert m[0]["role"] == "system" and m[-1]["role"] == "assistant"
    assert len(m) == 1 + 2 * len(FEW_SHOT) + 2


@pytest.mark.parametrize("raw,status", [
    ('{"typology": "card_testing", "red_flags": [], "amounts": [], "channel": "card", "urgency": "low"}', "ok"),
    ('```json\n{"typology": "card_testing", "amounts": [1.5], "urgency": "low"}\n```', "ok"),
    ('Sure! {"typology": "card_testing", "urgency": "low"} hope this helps', "ok"),
    ('{"typology": "made_up", "urgency": "low"}', "schema_error"),
    ('{"typology": "card_testing", "red_flags": ["not a flag"], "urgency": "low"}', "schema_error"),
    ("not json at all", "invalid_json"),
    ('{"typology": "card_testing", ', "invalid_json"),
])
def test_parse_output(raw, status):
    assert parse_output(raw)[1] == status


def test_macro_f1_perfect_and_partial():
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    assert 0 < macro_f1(["a", "b", "b"], ["a", "a", "b"], ["a", "b"]) < 1


def test_score_end_to_end():
    gold = [{"typology": "card_testing", "red_flags": ["small test charges"], "amounts": [1.5], "channel": "card",
             "urgency": "medium"},
            {"typology": "not_fraud", "red_flags": [], "amounts": [], "channel": "unknown", "urgency": "low"}]
    outs = [json.dumps(gold[0]), "garbage"]
    m = score(gold, outs, [0.1, 0.3], gpu_cost_per_hour=1.0, wall_time_s=3.6)
    assert m["json_validity"] == 0.5 and m["typology_accuracy"] == 0.5 and m["all_fields_exact_match"] == 0.5
    assert m["cost_per_1k_requests_usd"] == pytest.approx(0.5)


def test_contamination_filter_drops_near_duplicates():
    test = [{"narrative": "customer got a call from the bank fraud team and sent 4200 dollars to a new payee today"}]
    pool = [{"narrative": "customer got a call from the bank fraud team and sent 4200 dollars to a new payee today!"},
            {"narrative": "seventeen tiny card charges at many different online merchants overnight, card in wallet"}]
    kept, dropped = drop_near_duplicates(pool, test, 0.8)
    assert dropped == 1 and len(kept) == 1
    assert jaccard(_shingles("a b c d e f"), _shingles("a b c d e f")) == 1.0


def test_api_with_mocked_backend(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import serving.api as api

    good = '{"typology": "card_testing", "red_flags": ["small test charges"], "amounts": [2.5], ' \
           '"channel": "card", "urgency": "medium"}'
    replies = iter(["not json", good])

    async def fake_complete(msgs):
        return next(replies)

    monkeypatch.setattr(api, "_complete", fake_complete)
    c = TestClient(api.app)
    r = c.post("/extract", json={"text": "lots of tiny charges on my card at random websites"})
    assert r.status_code == 200 and r.json()["repaired"] is True
    assert c.post("/extract", json={"text": "short"}).status_code == 422
