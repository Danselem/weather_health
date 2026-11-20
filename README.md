# 🌦️ Weather Disease Prediction - MLOps Project

## 👤 Author
**Daniel Egbo** – [@Danselem](https://github.com/Danselem)

---

## 📖 Project Description

An end-to-end MLOps pipeline for predicting weather-sensitive diseases using machine learning.

**Scientific note:** the 45 features are mostly symptom indicators. A feature
ablation shows weather-only reaches 37.6% accuracy (22.9% macro F1) while
symptoms-only reaches 98.7% (98.8% macro F1). The model therefore learns
symptom-to-disease patterns; weather adds little standalone signal. The
dataset is synthetic and deterministic, so the headline metrics are not a
claim about real clinical performance. Five-fold CV gives accuracy
0.9794 ± 0.0035 and macro F1 0.9790 ± 0.0041; calibration is strong
(Brier 0.0405, ECE 0.0182). See `MODEL_CARD.md` for slices, SHAP, and
limitations.

The project demonstrates professional MLOps practices including:

- **Configuration Management** with Hydra (multi-environment support)
- **Data Versioning** with DVC
- **Experiment Tracking** with MLflow
- **Orchestration** with Prefect
- **CI/CD** with GitHub Actions
- **Infrastructure as Code** with Terraform (AWS & GCP)
- **Container Orchestration** (ECS Fargate, Cloud Run, Kubernetes)
- **Monitoring** with Evidently, Grafana

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Data     │────▶│   Training  │────▶│  Deployment │
│  (DVC/GCS)  │     │ (MLflow)    │     │ (AWS/GCP)   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │  Monitoring │
                    │ (Evidently) │
                    └─────────────┘
```

---

## 📁 Project Structure

```
weather-health/
├── config/                    # Hydra configuration
│   ├── config.yaml           # Base config
│   ├── env/                  # Environment configs (dev/staging/prod)
│   ├── model/                # Model configs (logistic_regression, lightgbm, etc.)
│   ├── cloud/                # Cloud configs (aws.yaml, gcp.yaml)
│   └── mlflow.yaml           # MLflow settings
│
├── src/                      # Source code
│   ├── clean_data.py         # Data cleaning
│   ├── transform.py          # Data preprocessing
│   ├── train.py              # Model training
│   ├── pipeline.py           # Full pipeline orchestration
│   └── utils/                # Utilities (MLflow, optimization)
│
├── tests/                   # Unit tests
│   ├── test_clean_data.py
│   ├── test_transform.py
│   └── test_utils.py
│
├── infra-aws/               # AWS Terraform (separate repo recommended)
│   ├── modules/             # Reusable Terraform modules
│   │   ├── ecr/             # ECR container registry
│   │   ├── ecs-fargate/    # ECS Fargate service
│   │   └── s3-artifacts/   # S3 storage
│   └── environments/        # dev/staging/prod
│
├── infra-gcp/               # GCP Terraform (separate repo recommended)
│   ├── modules/
│   │   ├── cloudrun/        # Cloud Run service
│   │   ├── artifact-registry/
│   │   └── gcs-artifacts/   # GCS storage
│   └── environments/
│
├── .github/workflows/       # CI/CD pipelines
│   ├── ci.yml               # Lint, type-check, test
│   └── (deploy-aws.yml, deploy-gcp.yml in infra repos)
│
└── Makefile                 # Development commands
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Danselem/weather-health.git
cd weather-health
make init
make install
```

### 2. Configure Environment

```bash
make env  # Creates .env from .env.example
# Edit .env with your MLflow/DVC credentials
```

### 3. Run Pipeline

```bash
# Development (default)
make pipeline

# Production
make pipeline-prod

# Specific model
make train MODEL=lightgbm
make train ENV=prod MODEL=random_forest
```

---

## ⚙️ Configuration (Hydra)

### Environments

```bash
make train ENV=dev         # Dev environment
make train ENV=staging    # Staging environment  
make train ENV=prod       # Production environment
```

### Models

```bash
make train MODEL=logistic_regression
make train MODEL=random_forest
make train MODEL=gradient_boosting
make train MODEL=lightgbm
```

### Combined

```bash
make train ENV=prod MODEL=lightgbm n_trials=10
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_train.py -v
```

---

## 🔧 Available Make Commands

| Command | Description |
|---------|-------------|
| `make init` | Initialize project with uv |
| `make install` | Install dependencies |
| `make pipeline` | Run full ML pipeline |
| `make train` | Train model |
| `make test` | Run tests |
| `make validate-data` | Validate the cleaned dataset contract |
| `make ablation` | Compare weather-only, symptoms-only, and full features |
| `make check-metrics` | Fail on model or data-contract regression |
| `make analysis` | Calibration, slices, CV, and SHAP reports |
| `make quality_checks` | Run linters (ruff, black, mypy) |
| `make dvc` | Run DVC pipeline |
| `make build` | Build Docker image |
| `make serve_local` | Run FastAPI locally |
| `make start-monitoring` | Start API + Grafana + PostgreSQL |
| `make stop-monitoring` | Stop the monitoring stack |
| `make observe` | Run the Evidently drift report |
| `make drift-demo` | Simulate drift and run the gate |
| `make performance` | Compute accuracy/F1 from feedback labels |
| `make monitoring-deploy` | Deploy scheduled monitoring to Prefect |
| `make security-scan` | Checkov scan for high-severity Terraform issues |
| `make rollback` | Roll ECS back to a task-definition revision |

### Prefect Server

```bash
make prefect          # Start Prefect server
make prefect-pool     # Create the 'weather' work pool
make prefect-stop    # Stop Prefect server
make prefect-reset  # Reset Prefect database
```

---

## ☁️ Cloud Deployment

### AWS (ECS Fargate)

The AWS path uses private ECS subnets, remote Terraform state, Secrets Manager,
and an optional monthly budget.

```bash
# Step 0: Create the S3 state bucket + DynamoDB lock table (once per account)
make aws-bootstrap TF_STATE_BUCKET=<globally-unique-bucket> \
  TF_LOCK_TABLE=weather-health-terraform-locks AWS_REGION=us-east-1

# Create infra-aws/backend.hcl from backend.hcl.example, then:
make aws-init

# Step 2: Plan infrastructure changes
make aws-plan

# Step 3: Deploy (builds Docker, creates infra, pushes to ECR, deploys to ECS)
make aws-deploy

# Step 4: Destroy infrastructure when done
make aws-destroy
```

**Workflow:**
1. `aws-init` - Initialize Terraform providers
2. `aws-plan` - Fetch model from MLflow and preview infrastructure changes  
3. `aws-deploy` - Build Docker image → Create ECR/VPC/ECS → Push to ECR → Update ECS service
4. `aws-destroy` - Delete ECR repository and destroy all infrastructure

**Outputs:**
- ALB DNS: Check Terraform output for `alb_dns_name`
- ECR: Check Terraform output for `ecr_repository_url`
- DB password: generated in Secrets Manager; never in tfvars

### GCP (Cloud Run) — alternative path

> The supported production path is AWS. GCP is kept as a reference and uses
> local Terraform state.

```bash
# Initialize
make gcp-init

# Plan
make gcp-plan

# Deploy
make gcp-deploy

# Destroy
make gcp-destroy
```

---

## 📊 Monitoring

```bash
make start-monitoring   # API + PostgreSQL + Grafana + Adminer
make observe            # Evidently: logged predictions vs pinned reference
make drift-demo         # simulated drift + drift gate
```

- **API**: http://localhost:8000
- **Grafana**: http://localhost:3000
- **Adminer**: http://localhost:8081
- **Reports/alerts**: `reports/monitoring/`
- **Runbook**: [`monitoring/README.md`](monitoring/README.md)

The API logs predictions to SQLite by default, or PostgreSQL via
`PREDICTION_DB_URL`. Drift reports use the pinned reference written by
`src.evaluate` (`data/processed/reference.parquet`), not training data.
The daily Prefect deployment sends optional webhook alerts and can dispatch a
retrain after repeated failures; see [`monitoring/README.md`](monitoring/README.md).

---

## 🔄 CI/CD Pipeline

`.github/workflows/ci.yml` gates every pull request:

1. Ruff and mypy (core modules)
2. `dvc pull` + pytest with coverage (`--cov-fail-under=42`)
3. `dvc repro` + data contract + model quality gate (`src.check_metrics`)
4. Docker build, container `/ready` + `/predict` smoke test
5. `terraform fmt -check` and `terraform validate` for AWS and GCP

Required CI/CD secrets: `DAGSHUB_USER`, `DAGSHUB_TOKEN` (DVC data),
`AWS_ROLE_ARN`, `MLFLOW_TRACKING_URI`. Required repository variables:
`ECS_CLUSTER`, `ECS_SERVICE`, `ECS_TASK_DEFINITION`, `ECR_REPOSITORY`,
`ALB_DNS_NAME`, `AWS_REGION`.

Deployment is manual and environment-gated:

```bash
# GitHub Actions -> Deploy -> Run workflow -> choose dev/staging/prod
```

It pulls DVC artifacts, fetches `lightgbm_best_model@champion` from MLflow,
builds/pushes an image tagged with the model version, registers a new ECS task
definition, updates the service, and runs `/ready` + `/predict` smoke tests.
A separate `Retrain` workflow runs on `repository_dispatch` or manually; it
trains, evaluates, runs the quality gate, and registers a new champion, but
never deploys automatically.
Rollback:

```bash
./scripts/rollback.sh <cluster> <service> <task-family> <revision>
```

---

## ✅ Requirements

- Python 3.10+
- uv (package manager)
- Docker
- Terraform (for cloud deployment)
- AWS CLI / GCP CLI (for cloud deployment)

---

## 📝 Notes

- Use Hydra for multi-environment configuration
- DVC handles data versioning
- MLflow tracks experiments
- Separate infra repos recommended for production

---

## 📜 License

MIT License - See [LICENSE](LICENSE)

---

## 🙋 Contact

Created by [Daniel Egbo](mailto:danoegbo@egmail.com)