# =============================================================================
# Weather-Health MLOps Project Makefile
# =============================================================================
# This file provides convenient commands for common development tasks.
# Usage: make <target> [VAR=value]
# 
# Prerequisites Checks:
#   - Prefects server must be running for training/pipeline commands
#   - Docker must be running for monitoring commands
# =============================================================================

# -----------------------------------------------------------------------------
# Service Checks (Internal)
# -----------------------------------------------------------------------------

# Check if Prefect server is running
prefect-check:
	@curl -s http://127.0.0.1:4200/api/health > /dev/null 2>&1 || (echo "❌ Prefect server not running. Run 'make prefect' first." && exit 1)
	@echo "✅ Prefect server is running"

# Check if Docker is running
docker-check:
	@docker info > /dev/null 2>&1 || (echo "❌ Docker not running. Start Docker first." && exit 1)
	@echo "✅ Docker is running"

# Check if environment variables are set for MLflow
mlflow-check:
	@uv run python -c "import os; import dotenv; dotenv.load_dotenv(); os.getenv('DAGSHUB_REPO_OWNER') or os.getenv('MLFLOW_TRACKING_URI') or exit(print('❌ Set DAGSHUB_REPO_OWNER or MLFLOW_TRACKING_URI in .env'))" || (echo "❌ MLflow env vars missing. Check .env" && exit 1)
	@echo "✅ MLflow configured"

# -----------------------------------------------------------------------------
# Project Initialization
# -----------------------------------------------------------------------------

init:                          ## Initialize new project with uv (creates .venv)
	uv venv --python 3.10
	uv init && rm hello.py
	uv tool install black

install:                      ## Install dependencies from requirements.txt into .venv
	. .venv/bin/activate
	uv add -r requirements.txt

delete:                       ## Remove all project files (uv.lock, pyproject.toml, .venv)
	rm uv.lock pyproject.toml .python-version && rm -rf .venv

# -----------------------------------------------------------------------------
# Data Pipeline (Requires Prefect for pipeline, not for individual steps)
# -----------------------------------------------------------------------------

# Run with Hydra config - examples:
#   make train                              # Default: env=dev, model=logistic_regression
#   make train-with MODEL=lightgbm          # LightGBM in dev
#   make train-with MODEL=lightgbm ENV=prod  # LightGBM in prod
#   make train-with ENV=prod MODEL=random_forest n_trials=20
#   make train-prod                         # Production config (logistic regression)
#   make train-prod-lgbm                    # Production + LightGBM

visualise:                    ## Run EDA/visualization (src.visualise module)
	uv run -m src.visualise

clean-data:                  ## Clean raw data and save to parquet (src.clean_data module)
	uv run -m src.clean_data

clean-data-prod:             ## Clean data using production config
	uv run -m src.clean_data env=prod

transform:                   ## Preprocess data: split, scale, encode (src.transform module)
	uv run -m src.transform 

transform-prod:              ## Transform data using production config
	uv run -m src.transform env=prod

train:                       ## Train model with MLflow tracking (src.train module) - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run -m src.train

train-lightgbm:            ## Train with LightGBM - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run python -m src.train modeling.model_family=lightgbm

train-rf:                  ## Train with Random Forest - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run python -m src.train modeling.model_family=random_forest

train-gb:                  ## Train with Gradient Boosting - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run python -m src.train modeling.model_family=gradient_boosting

train-prod:                  ## Train all models with production config - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run -m src.train_all env=prod

train-prod-lgbm:             ## Train with production config + LightGBM only - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run python -m src.train env=prod modeling.model_family=lightgbm

train-prod-rf:             ## Train with production config + Random Forest only - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run python -m src.train env=prod modeling.model_family=random_forest

train-staging:               ## Train all models with staging config - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run -m src.train_all env=staging

pipeline:                    ## Run full pipeline: EDA -> clean -> transform -> train - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run -m src.pipeline

pipeline-prod:               ## Run full pipeline with production config - REQUIRES PREFECT
	@make prefect-check || exit 1
	uv run -m src.pipeline env=prod

fetch-best-model:            ## Retrieve best model from MLflow registry
	uv run -m src.fetch_best_model

sample:                      ## Create sample input JSON for API testing
	uv run -m src.create_input_sample

# -----------------------------------------------------------------------------
# Model Serving
# -----------------------------------------------------------------------------

serve_local:                 ## Run FastAPI server locally on port 8000
	uv run -m src.serve_local

serve:                      ## Run model serving (production mode)
	uv run -m src.serve

# -----------------------------------------------------------------------------
# Monitoring & Observability (Requires Docker)
# -----------------------------------------------------------------------------

observe:                     ## Run Evidently metrics for data drift detection
	uv run -m src.evidently_metrics

start-monitoring:           ## Start Grafana + PostgreSQL via Docker Compose - REQUIRES DOCKER
	@make docker-check || exit 1
	cd monitoring && docker-compose up -d

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------

test:                        ## Run all unit tests with pytest
	uv run pytest -v

test-cov:                   ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=html --cov-report=term

test-watch:                 ## Run tests in watch mode (reruns on file changes)
	uv run pytest -v --watch

quality_checks:              ## Run all linters and type checkers (isort, black, ruff, mypy)
	@echo "Running quality checks"
	uv run -m isort .
	uv run -m black .
	uv run -m ruff check . --fix
	uv run -m mypy .

# Clean generated files (cache, coverage, etc.)
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "Cleanup completed"

env:                        ## Copy .env.example to .env for environment variables
	cp .env.example .env

awscreds:                   ## Create AWS credentials block for Prefect
	uv run -m src.utils.create_aws_block

# -----------------------------------------------------------------------------
# DVC (Data Version Control)
# -----------------------------------------------------------------------------

dvc:                        ## Reproduce DVC pipeline (clean -> transform -> train -> evaluate)
	uv run dvc repro

dvc-status:                ## Show DVC pipeline status (which stages need to rerun)
	uv run dvc status

dvc-push:                 ## Push data/artifacts to DVC remote (Dagshub by default)
	uv run dvc push

dvc-pull:                 ## Pull data/artifacts from DVC remote
	uv run dvc pull

dvc-add:                 ## Add a file or directory to DVC tracking (use: make dvc-add FILE=path/to/file)
	uv run dvc add $(FILE)

dvc-init:                ## Initialize DVC in project (creates .dvc directory)
	uv run dvc init

dvc-gc:                  ## Garbage collection: clean unused cache from .dvc/cache
	uv run dvc gc

# -----------------------------------------------------------------------------
# Prefect Orchestration
# -----------------------------------------------------------------------------

prefect:                    ## Start Prefect server locally (background)
	@echo "Checking Prefect server..."
	@if curl -s http://127.0.0.1:4200/api/health > /dev/null 2>&1; then \
		echo "✅ Prefect server is already running at http://127.0.0.1:4200"; \
	else \
		echo "Starting Prefect server..."; \
		uv run prefect server start & \
	fi

prefect-stop:               ## Stop Prefect server
	@echo "Stopping Prefect server..."
	@pkill -f "prefect server" 2>/dev/null || true
	@echo "✅ Prefect server stopped."

prefect-reset:             ## Reset Prefect database (fixes migration errors)
	@echo "Resetting Prefect database..."
	@pkill -f "prefect server" 2>/dev/null || true
	@rm -f ~/.prefect/prefect.db
	@echo "✅ Prefect database reset. Run 'make prefect' to start fresh."

prefect-force:             ## Force start Prefect server (resets DB if needed)
	@echo "Force starting Prefect server..."
	@pkill -f "prefect server" 2>/dev/null || true
	@rm -f ~/.prefect/prefect.db 2>/dev/null || true
	@sleep 1
	@echo "Starting Prefect server..."
	uv run prefect server start &
	@echo "Waiting for server to start..."
	@sleep 5
	@curl -s http://127.0.0.1:4200/api/health > /dev/null 2>&1 && echo "✅ Prefect server running at http://127.0.0.1:4200" || echo "❌ Failed to start Prefect"

prefect-init:               ## Initialize Prefect in current directory
	uv run prefect init

worker:                    ## Start Prefect worker for 'weather' pool
	uv run prefect worker start -p weather -t process &

deploy:                    ## Deploy train.py flow to Prefect
	uv run prefect deploy src/train.py:main -n weather-health -p weather 

deployment:                ## Trigger a Prefect deployment run
	uv run prefect deployment run 'train_model/weather-health'

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

build:                      ## Build Docker image from service/Dockerfile
	docker build -t weather-health:v1.0 service/

run:                        ## Run containerized service on port 8080
	docker run -it --rm -p 8080:8080 weather-health:v1.0

load-kind:                  ## Load Docker image into kind cluster
	kind load docker-image weather-health:v1.0 --name weather-health

# -----------------------------------------------------------------------------
# Kubernetes
# -----------------------------------------------------------------------------

create-cluster:             ## Create kind cluster named 'weather-health'
	uv run kind create cluster --name weather-health

deploy-k8s:                 ## Deploy application to Kubernetes (deployment, service, HPA)
	uv run kubectl apply -f k8s/deployment.yaml
	uv run kubectl apply -f k8s/service.yaml
	uv run kubectl apply -f k8s/hpa.yaml

check-k8s:                  ## Show Kubernetes deployment status and details
	kubectl get deployments
	kubectl get pods
	kubectl describe deployment weather-health

check-services:             ## Show Kubernetes service status and details
	kubectl get services
	kubectl describe service weather-health

kube-forward:               ## Port-forward service to localhost:30080
	uv run kubectl port-forward svc/weather-health 30080:8080

metric-server:              ## Install metrics-server for HPA (HorizontalPodAutoscaler)
	kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
	kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

hpa:                        ## Show HorizontalPodAutoscaler status
	kubectl get hpa
	kubectl describe hpa weather-health-hpa

cleanup:                    ## Remove Kubernetes resources (deployment, service, HPA)
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f k8s/service.yaml
	kubectl delete -f k8s/hpa.yaml

autoclean:                  ## Quick cleanup: delete all resources with app=weather-health label
	kubectl delete all -l app=weather-health
	kubectl delete hpa weather-health-hpa

del-cluster:               ## Delete kind cluster
	kind delete cluster --name weather-health

# -----------------------------------------------------------------------------
# Cloud Deployment (AWS, GCP)
# -----------------------------------------------------------------------------

fetch-model:                ## Fetch best model from MLflow to models/
	@make mlflow-check || exit 1
	@echo "Fetching best model from MLflow..."
	uv run -m src.fetch_best_model

docker-build:               ## Build Docker image locally
	docker build -t weather-health:latest .

docker-build-prod:          ## Build Docker image with production model
	@make fetch-model || exit 1
	docker build -t weather-health:prod .

# === AWS Deployment ===

aws-init:                  ## Initialize Terraform for AWS (only needed first time)
	@echo "Initializing Terraform for AWS..."
	@rm -f infra-aws/.terraform.tfstate.lock.info 2>/dev/null || true
	cd infra-aws && terraform init -upgrade

aws-plan:                 ## Plan AWS infrastructure changes
	@make fetch-model || exit 1
	@rm -f infra-aws/.terraform.tfstate.lock.info 2>/dev/null || true
	cd infra-aws && terraform plan -var-file=environments/prod/terraform.tfvars -lock=false

aws-deploy:               ## Build and push to ECR and deploy to ECS
	@make fetch-model || exit 1
	@echo "=== Step 1: Build Docker image ==="
	docker build -t weather-health:latest .
	@echo "=== Step 2: Create infrastructure (ECR, ECS, VPC) ==="
	@rm -f infra-aws/.terraform.tfstate.lock.info 2>/dev/null || true
	cd infra-aws && terraform apply -var-file=environments/prod/terraform.tfvars -auto-approve -lock=false
	@echo "=== Step 3: Push image to ECR ==="
	@ECR_URI=$$(aws ecr describe-repositories --region us-east-1 --repository-names weather-health --query 'repositories[0].repositoryUri' --output text); \
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$${ECR_URI%/*}"; \
	docker tag weather-health:latest "$${ECR_URI}:v1"; \
	docker push "$${ECR_URI}:v1"
	@echo "=== Step 4: Update ECS service with new image ==="
	cd infra-aws && terraform apply -var-file=environments/prod/terraform.tfvars -var "container_image=$${ECR_URI}:v1" -auto-approve -lock=false
	@echo "=== Deployment complete! ==="

aws-destroy:              ## Destroy AWS infrastructure
	@rm -f infra-aws/.terraform.tfstate.lock.info 2>/dev/null || true
	@aws ecr delete-repository --region us-east-1 --repository-name weather-health --force 2>/dev/null || true
	cd infra-aws && terraform destroy -var-file=environments/prod/terraform.tfvars -auto-approve -lock=false

# === GCP Deployment ===

gcp-init:                 ## Initialize Terraform for GCP (only needed first time)
	@echo "Initializing Terraform for GCP..."
	cd infra-gcp && terraform init -upgrade

gcp-plan:                ## Plan GCP infrastructure changes
	@make fetch-model || exit 1
	cd infra-gcp && terraform plan -var-file=environments/prod/terraform.tfvars

gcp-deploy:              ## Deploy to GCP Artifact Registry + Cloud Run
	@make fetch-model || exit 1
	@echo "Building Docker image..."
	docker build -t weather-health:latest .
	@echo "Configuring GCP auth..."
	gcloud auth configure-docker
	@echo "Tagging for GCP Artifact Registry..."
	docker tag weather-health:latest us-central1-docker.pkg.dev/weather-health-prod/weather-health-prod/weather-health:latest
	@echo "Pushing to GCP Artifact Registry..."
	docker push us-central1-docker.pkg.dev/weather-health-prod/weather-health-prod/weather-health:latest
	@echo "Deploying to Cloud Run..."
	cd infra-gcp && terraform apply -var-file=environments/prod/terraform.tfvars -auto-approve

gcp-destroy:             ## Destroy GCP infrastructure
	cd infra-gcp && terraform destroy -var-file=environments/prod/terraform.tfvars -auto-approve