FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml .python-version api_requirements.txt ./

RUN uv add -r api_requirements.txt

COPY app.py weather_disease_model.pkl minmax_scaler.pkl label_encoder.pkl ./

EXPOSE 8080

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
