# MLflow Infrastructure with Terraform

This Terraform setup provisions the necessary AWS resources for running an MLflow Tracking Server, including:

- An EC2 Ubuntu instance
- An RDS PostgreSQL database
- An S3 bucket for storing MLflow artifacts
- A VPC with public subnets and internet access

## Prerequisites

- Terraform installed
- AWS CLI configured (`aws configure`)
- An EC2 key pair created in AWS Console
- A `terraform.tfvars` file with the required variables

## Required `terraform.tfvars` Example

```hcl
aws_region           = "us-east-1"
aws_profile          = "default" or your profile name
ec2_key_name         = "mlfkp"
db_password          = "StrongPassword123"
artifact_bucket_name = "your-unique-bucket-name"
```
**Note:** Ensure your AWS credentials are in `~/.aws/credentials` and properly configured using the AWS CLI:

```bash
aws configure
```

## Commands

### Initialize Terraform
```bash
terraform init
```

### Validate configuration
```bash
terraform validate
```

### Preview plan
```bash
terraform plan -var-file="terraform.tfvars"
```

### Apply infrastructure
```bash
terraform apply -var-file="terraform.tfvars"
```

### Destroy infrastructure
```bash
terraform destroy -var-file="terraform.tfvars"
```

