"""Load test for the /extract endpoint.

    locust -f serving/locustfile.py --host http://localhost:8000 --users 32 --spawn-rate 8 \
        --run-time 3m --headless --csv results/locust
Report requests/sec and p95 latency from results/locust_stats.csv.
"""
import json
import random
from pathlib import Path

from locust import HttpUser, between, task

TEST = Path(__file__).resolve().parents[1] / "data" / "test.jsonl"
NARRATIVES = [json.loads(line)["narrative"] for line in open(TEST)] if TEST.exists() else [
    "Customer got a call from someone claiming to be the bank and sent $2,400 to a new payee. Wants it reversed.",
    "Twelve charges under $3 at different websites overnight, card still in wallet.",
]


class ExtractUser(HttpUser):
    wait_time = between(0.0, 0.2)

    @task
    def extract(self):
        self.client.post("/extract", json={"text": random.choice(NARRATIVES)}, name="/extract")
