# Weather Health - GCP Infrastructure

**Parent Project:** [weather-health](https://github.com/Danselem/weather-health)

Terraform infrastructure as code for deploying the Weather Health ML application on Google Cloud Platform.

> **Note:** This folder can be extracted to a separate repo (`weather-health-infra-gcp`) for production use.

---

## 🏗️ Architecture

```
┌──────────────────┐     ┌─────────────┐     ┌─────────────┐
│ Artifact Registry│────▶│  Cloud Run  │────▶│     GCS     │
│   (Images)       │     │  (FastAPI)  │     │ (Artifacts) │
└──────────────────┘     └─────────────┘     └─────────────┘
```

---

## 📁 Directory Structure

```
infra-gcp/
├── backend.tf                    # GCS remote state
├── versions.tf                   # Provider versions
├── main.tf                       # Root module
├── modules/
│   ├── cloudrun/                # Cloud Run service
│   ├── artifact-registry/       # Artifact Registry
│   └── gcs-artifacts/           # GCS bucket for models
├── environments/
│   ├── dev/                     # Development config
│   ├── staging/                 # Staging config
│   └── prod/                    # Production config
├── .github/workflows/           # CI/CD (separate repo)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- GCP Project
- Terraform >= 1.7.0
- Google Cloud SDK configured
- Docker (for building images)

### Deploy

```bash
# 1. Authenticate
gcloud auth login
gcloud auth application-default login

# 2. Navigate to environment
cd environments/dev

# 3. Initialize Terraform
terraform init

# 4. Plan changes
terraform plan

# 5. Apply (or use CI/CD)
terraform apply
```

---

## 🌈 Environments

| Environment | CPU | Memory | Instances | Monthly Cost |
|-------------|-----|--------|-----------|--------------|
| Dev | 1 | 1Gi | 0-5 | ~$0-25 |
| Staging | 1 | 1Gi | 0-8 | ~$25-50 |
| Prod | 2 | 2Gi | 2-10 | ~$50-100 |

---

## 🔧 Configuration

Edit `environments/*/terraform.tfvars`:

```hcl
project_id      = "weather-health-dev"
region          = "us-central1"
environment     = "dev"
container_image = "us-central1-docker.pkg.dev/PROJECT/weather-health:v1.0"
```

---

## 🔐 Security

- ✅ Artifact Registry with Docker format
- ✅ GCS with uniform bucket-level access
- ✅ Cloud Run with IAM invoker
- ✅ Service accounts with least privilege

---

## 📦 Modules

### Cloud Run (`modules/cloudrun/`)
- Serverless container deployment
- Auto-scaling (0 to N instances)
- Custom CPU and memory allocation
- Environment variables support
- Public URL with HTTPS

### Artifact Registry (`modules/artifact-registry/`)
- Docker container registry
- Regional storage
- Public read access

### GCS Artifacts (`modules/gcs-artifacts/`)
- Versioning enabled
- Lifecycle rules (30-day expiry)
- Uniform bucket-level access

---

## 🧪 Testing

```bash
# Validate Terraform syntax
terraform fmt -recursive
terraform validate

# Plan (dry run)
terraform plan
```

---

## 💰 Cost Estimate

| Resource | Monthly Cost |
|----------|--------------|
| Cloud Run | ~$0.000024/second |
| Artifact Registry | ~$0.10/GB |
| GCS | ~$0.020/GB |

---

## 🔄 CI/CD Integration

```yaml
# .github/workflows/deploy.yml (add to separate infra repo)
jobs:
  terraform:
    - run: terraform init
    - run: terraform plan
    - run: terraform apply
  
  deploy-cloudrun:
    - run: gcloud run deploy --image <image> --platform managed
```

---

## 🆘 Troubleshooting

```bash
# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50

# Check service status
gcloud run describe weather-health-dev --region us-central1

# View revision details
gcloud run revisions list --service weather-health-dev --region us-central1
```

---

## 📝 License

MIT License - See [LICENSE](../LICENSE)