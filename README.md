# 🌦️ Weather Disease Prediction - MLOps Project

## 👤 Author
**Daniel Egbo** – [@Danselem](https://github.com/Danselem)

---

## 📖 Project Description

An end-to-end MLOps pipeline for predicting weather-sensitive diseases using machine learning. The project demonstrates professional DevOps practices including:

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
| `make quality_checks` | Run linters (ruff, black, mypy) |
| `make dvc` | Run DVC pipeline |
| `make build` | Build Docker image |
| `make serve_local` | Run FastAPI locally |
| `make start-monitoring` | Start Grafana/PostgreSQL |

### Prefect Server

```bash
make prefect          # Start Prefect server
make prefect-stop    # Stop Prefect server
make prefect-reset  # Reset Prefect database
```

---

## ☁️ Cloud Deployment

### AWS (ECS Fargate)

```bash
# Step 1: Initialize Terraform (first time only)
make aws-init

# Step 2: Plan infrastructure changes
make aws-plan

# Step 3: Deploy (builds Docker, creates infra, pushes to ECR, deploys to ECS)
make aws-destroy  # First clean up any existing resources
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
- ECR: `828221019178.dkr.ecr.us-east-1.amazonaws.com/weather-health`

### GCP (Cloud Run)

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
# Start monitoring stack
make start-monitoring
```

- **Grafana**: http://localhost:3000
- **Adminer**: http://localhost:8080
- **Evidently**: Run `make observe` for drift detection

---

## 🔄 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml runs:
1. Lint (ruff)
2. Type Check (mypy)
3. Test (pytest --cov)
4. Docker Build
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