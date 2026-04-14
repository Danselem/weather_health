# =============================================================================
# GCP Backend Configuration
# =============================================================================
# For local development: uses local state file.
# For production/team: change to GCS backend.
# =============================================================================

terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}