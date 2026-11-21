# =============================================================================
# Main GCP Terraform Configuration
# =============================================================================
# Root module combining all GCP modules.
# =============================================================================

module "artifact_registry" {
  source = "./modules/artifact-registry"

  name       = var.app_name
  location   = var.region
  project_id = var.project_id
}

module "cloudsql" {
  source = "./modules/cloudsql"

  name          = "weather-health-db"
  project_id    = var.project_id
  region        = var.region
  root_password = random_password.db.result
}

module "gcs_artifacts" {
  source = "./modules/gcs-artifacts"

  name       = "${var.app_name}-${var.environment}-artifacts"
  location   = var.region
  project_id = var.project_id
}

module "cloudrun" {
  source = "./modules/cloudrun"

  name       = "${var.app_name}-${var.environment}"
  location   = var.region
  project_id = var.project_id
  image      = var.container_image != "" ? var.container_image : "nginx:latest"

  port          = var.container_port
  cpu           = var.cpu
  memory        = var.memory
  min_instances = var.environment == "prod" ? var.min_instances : 0
  max_instances = var.max_instances
  concurrency   = var.concurrency

  service_account_email = google_service_account.runtime.email
  cloudsql_instances    = [module.cloudsql.connection_name]

  environment_variables = {
    ENVIRONMENT   = var.environment
    MODEL_VERSION = var.model_version
  }

  secret_environment_variables = {
    PREDICTION_DB_URL = google_secret_manager_secret.db_url.secret_id
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "artifact_registry_url" {
  description = "Artifact Registry URL"
  value       = module.artifact_registry.full_url
}

output "gcs_bucket" {
  description = "GCS artifacts bucket name"
  value       = module.gcs_artifacts.bucket_name
}

output "cloudrun_service_name" {
  description = "Cloud Run service name"
  value       = module.cloudrun.service_name
}

output "cloudrun_service_url" {
  description = "Cloud Run service URL"
  value       = module.cloudrun.service_url
}