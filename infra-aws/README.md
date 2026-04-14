# Weather Health - AWS Infrastructure

**Parent Project:** [weather-health](https://github.com/Danselem/weather-health)

Terraform infrastructure as code for deploying the Weather Health ML application on AWS.

> **Note:** This folder can be extracted to a separate repo (`weather-health-infra-aws`) for production use.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   ECR       │────▶│  ECS Task   │────▶│     S3      │
│ (Images)    │     │ (FastAPI)   │     │ (Artifacts) │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                    ┌──────┴──────┐
                    │     ALB     │
                    └─────────────┘
```

---

## 📁 Directory Structure

```
infra-aws/
├── backend.tf                    # S3 remote state
├── versions.tf                   # Provider versions
├── main.tf                       # Root module
├── modules/
│   ├── ecr/                     # ECR repository
│   ├── ecs-fargate/             # ECS cluster & service + ALB
│   └── s3-artifacts/            # S3 bucket for models
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

- AWS Account
- Terraform >= 1.7.0
- AWS CLI configured
- Docker (for building images)

### Deploy

```bash
# 1. Navigate to environment
cd environments/dev

# 2. Initialize Terraform
terraform init

# 3. Plan changes
terraform plan

# 4. Apply (or use CI/CD)
terraform apply
```

---

## 🌈 Environments

| Environment | Replicas | Auto-scale | Cost |
|-------------|----------|------------|------|
| Dev | 1 | 1-5 | $25-50/mo |
| Staging | 2 | 2-8 | $50-100/mo |
| Prod | 3 | 2-10 | $100-200/mo |

---

## 🔧 Configuration

Edit `environments/*/terraform.tfvars`:

```hcl
aws_region      = "us-east-1"
environment     = "dev"
container_image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/weather-health:v1.0"
```

---

## 🔐 Security

- ✅ ECR image scanning on push
- ✅ S3 versioning enabled
- ✅ S3 encryption (AES256)
- ✅ S3 block public access
- ✅ IAM roles with least privilege
- ✅ ALB with security groups

---

## 📦 Modules

### ECR (`modules/ecr/`)
- Docker container registry
- Image scanning on push
- Lifecycle policy (keep 10 images)

### ECS Fargate (`modules/ecs-fargate/`)
- ECS Cluster with Container Insights
- Fargate tasks with proper IAM roles
- Application Load Balancer
- Auto-scaling (CPU-based)
- CloudWatch logging

### S3 Artifacts (`modules/s3-artifacts/`)
- Versioning enabled
- Server-side encryption
- Lifecycle rules
- Public access block

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
| ECR Storage | ~$0.10/GB |
| ECS Fargate | ~$25-50/task-month |
| ALB | ~$16/month |
| S3 | ~$0.02/GB |
| Data Transfer | ~$0.09/GB |

---

## 🔄 CI/CD Integration

```yaml
# .github/workflows/deploy.yml (add to separate infra repo)
jobs:
  terraform:
    - run: terraform init
    - run: terraform plan
    - run: terraform apply
  
  deploy-ecs:
    - run: aws ecs update-service --force-new-deployment
```

---

## 🆘 Troubleshooting

```bash
# View ECS logs
aws logs tail /ecs/weather-health --follow

# Check service status
aws ecs describe-services \
  --cluster weather-health-dev \
  --services weather-health-service-dev

# Check ALB target health
aws elbv1 describe-target-health \
  --target-group-arn <arn>
```

---

## 📝 License

MIT License - See [LICENSE](../LICENSE)