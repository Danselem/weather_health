init:
	uv venv --python 3.10
	uv init && rm hello.py
	uv tool install black

install:
	. .venv/bin/activate
	uv pip install --all-extras --requirement pyproject.toml
	# uv pip sync requirements.txt
	# uv add -r requirements.txt

delete:
	rm uv.lock pyproject.toml .python-version && rm -rf .venv

env:
	cp .env.example .env

awscreds:
	uv run -m src.utils.create_aws_block

visualise:
	uv run -m src.visualise

clean:
	uv run -m src.clean_data

transform:
	uv run -m src.transform 

train:
	uv run -m src.train

pipeline:
	uv run -m src.pipeline

fetch-best-model:
	uv run -m src.fetch_best_model

sample:
	uv run -m src.create_input_sample

serve_local:
	uv run -m src.serve_local

serve:
	uv run -m src.serve

observe:
	uv run -m src.evidently_metrics

quality_checks:
	@echo "Running quality checks"
	uv run -m isort .
	uv run -m black .
	uv run -m ruff check .
	uv run -m mypy .

dvc:
	uv run dvc repro

prefect:
	uv run prefect server start &

prefect-init:
	uv run prefect init

worker:
	uv run prefect worker start -p weather -t process &

deploy:
	uv run prefect deploy src/train.py:main -n weather-health -p weather 

deployment:
	uv run prefect deployment run 'train_model/weather-health'


build:
	docker build -t weather-disease:v1.0.0 .

run:
	docker run -d -p 9696:9696 weather-disease:v1.0.0

start-monitoring:
	cd monitoring && docker-compose up -d