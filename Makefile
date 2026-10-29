.PHONY: install test data-llm data-template split import train merge serve api loadtest

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest -q

data-llm:
	python -m typology.generate_data --writer llm --n 4000 --out data/raw.jsonl

data-template:
	python -m typology.generate_data --writer template --n 4000 --out data/raw.jsonl

split:
	python -m typology.prepare_data split

import:
	python -m typology.prepare_data import

train:
	python -m typology.train_qlora --model Qwen/Qwen2.5-3B-Instruct --out outputs/qwen3b-qlora

merge:
	python -m typology.merge_adapter --adapter outputs/qwen3b-qlora --out models/qwen3b-fraud-typology

serve:
	vllm serve models/qwen3b-fraud-typology --served-model-name ft --port 8001 --max-model-len 2048

api:
	uvicorn serving.api:app --port 8000

loadtest:
	locust -f serving/locustfile.py --host http://localhost:8000 --users 32 --spawn-rate 8 --run-time 3m --headless --csv results/locust
