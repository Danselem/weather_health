# Remote state. Initialize with:
#   terraform init -backend-config=backend.hcl
# See backend.hcl.example and bootstrap/.
terraform {
  backend "s3" {}
}
