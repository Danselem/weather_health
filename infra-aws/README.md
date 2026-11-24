> **Optional reference stack.** AWS ECS is defined here for IaC review and CI validation. It is **not deployed**. The live demo runs on GCP Cloud Run.

# AWS Infrastructure

Supported AWS deployment path for the weather-health API.

## Architecture

```
Internet -> ALB (public subnets) -> ECS Fargate (private subnets)
                                      |-> Secrets Manager
                                      |-> CloudWatch Logs
ECR -> ECS task image                 |-> S3 artifacts
```

- ECS tasks run in private subnets behind a NAT gateway.
- The ALB is the only public entry point.
- The database password is generated and stored in Secrets Manager; it is
  injected into the task as a `secrets` entry, never as plaintext environment.
- Terraform state is remote (S3 + DynamoDB locking).
- Optional monthly AWS budget with email notifications.

## One-time state bootstrap

```bash
cd infra-aws/bootstrap
terraform init
terraform apply \
  -var="bucket_name=<globally-unique-bucket>" \
  -var="lock_table_name=weather-health-terraform-locks"
```

Create `infra-aws/backend.hcl` from `backend.hcl.example` using the returned
bucket and table.

## Deploy

```bash
cd infra-aws
terraform init -backend-config=backend.hcl
terraform plan -var-file=environments/dev/terraform.tfvars
terraform apply -var-file=environments/dev/terraform.tfvars
```

Set `container_image` after pushing the image to ECR, or let the deploy
workflow register a new task definition with the new image.

## Cost guard

Set `monthly_budget_usd` and `budget_email` in the environment tfvars. A value
of `0` disables the budget resource.

## Security notes

- Remote state may contain the generated DB password; restrict bucket access
  and keep encryption enabled.
- The ALB currently terminates HTTP for the demo. Add an ACM certificate and an
  HTTPS listener before production use.
- `infra/` is a legacy/archived stack and is not supported.
- `infra-gcp/` is an alternative path, not the primary deployment.
