# Terraform State Bootstrap

Creates the S3 bucket and DynamoDB lock table used by `infra-aws/backend.tf`.
Run this once per AWS account, from a machine with AWS credentials.

```bash
cd infra-aws/bootstrap
terraform init
terraform apply \
  -var="bucket_name=<globally-unique-name>" \
  -var="lock_table_name=weather-health-terraform-locks"
```

Then create `infra-aws/backend.hcl` from `backend.hcl.example` with the
returned values.
