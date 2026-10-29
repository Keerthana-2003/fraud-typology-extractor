# API layer only (CPU). The model itself is served by the vLLM container (see docker-compose.yml).
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt pyproject.toml ./
RUN pip install fastapi "uvicorn[standard]" httpx pydantic numpy
COPY src ./src
COPY serving ./serving
RUN pip install -e .
EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
