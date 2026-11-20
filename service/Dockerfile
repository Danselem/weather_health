# Build from the repository root:
#   docker build -f service/Dockerfile -t weather-health:latest .
FROM python:3.10-slim

ARG MODEL_VERSION=dev

WORKDIR /code

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/app.py .
COPY models/weather_disease_model.pkl .
COPY data/processed/feature_order.json .
COPY data/processed/label_encoder.pkl .

ENV MODEL_PATH=/code/weather_disease_model.pkl \
    FEATURE_ORDER_PATH=/code/feature_order.json \
    ENCODER_PATH=/code/label_encoder.pkl \
    MODEL_VERSION=${MODEL_VERSION}

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /code
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
