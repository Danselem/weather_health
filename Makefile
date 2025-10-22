init:
	uv venv --python 3.10
	uv init && rm hello.py
	uv tool install black

install:
	. .venv/bin/activate
# 	uv pip install --all-extras --requirement pyproject.toml
# 	uv pip sync requirements.txt
	uv add -r requirements.txt

delete:
	rm uv.lock pyproject.toml .python-version && rm -rf .venv

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "Cleanup completed"

env:
	cp .env.example .env

awscreds:
	uv run -m src.utils.create_aws_block

visualise:
	uv run -m src.visualise

clean-data:
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
	uv run -m ruff check . --fix
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
	docker build -t weather-health:v1.0 service/

run:
	docker run -it --rm -p 8080:8080 weather-health:v1.0

load-kind:
	kind load docker-image weather-health:v1.0 --name weather-health

start-monitoring:
	cd monitoring && docker-compose up -d

create-cluster:
	uv run kind create cluster --name weather-health

deploy-k8s:
	uv run kubectl apply -f k8s/deployment.yaml
	uv run kubectl apply -f k8s/service.yaml
	uv run kubectl apply -f k8s/hpa.yaml

check-k8s:
	kubectl get deployments
	kubectl get pods
	kubectl describe deployment weather-health

check-services:
	kubectl get services
	kubectl describe service weather-health

kube-forward:
	uv run kubectl port-forward svc/weather-health 30080:8080

metric-server:
	kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
	kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

hpa:
	kubectl get hpa
	kubectl describe hpa weather-health-hpa

cleanup:
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f k8s/service.yaml
	kubectl delete -f k8s/hpa.yaml

autoclean:
	kubectl delete all -l app=weather-health
	kubectl delete hpa weather-health-hpa

del-cluster:
	kind delete cluster --name weather-health