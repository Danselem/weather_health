# =============================================================================
# Terraform Backend Configuration
# =============================================================================
# For local development: uses local state file.
# For production/team: change to S3 backend (see backend.s3.tf.example).
# =============================================================================

terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}