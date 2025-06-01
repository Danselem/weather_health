init:
	uv venv --python 3.10
	uv init && rm hello.py
	uv tool install black

install:
	. .venv/bin/activate
	# uv pip install --all-extras --requirement pyproject.toml
	# uv pip sync requirements.txt
	uv add -r requirements.txt

delete:
	rm uv.lock pyproject.toml .python-version && rm -rf .venv

env:
	cp .env.example .env

visualise:
	uv run -m src.visualise

clean:
	uv run -m src.clean_data

transform:
	uv run -m src.transform 

train:
	uv run -m src.train

run:
	uv run -m src.train

dl:
	uv run src/download.py

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