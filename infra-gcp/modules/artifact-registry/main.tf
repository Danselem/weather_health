# =============================================================================
# Artifact Registry Module - GCP Container Registry
# =============================================================================
# Creates a Docker container registry on GCP.
# =============================================================================

variable "name" {
  description = "Repository name"
  type        = string
}

variable "location" {
  description = "GCP location/region"
  type        = string
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "description" {
  description = "Repository description"
  type        = string
  default     = "Container registry for weather-health"
}

variable "labels" {
  description = "Labels to apply"
  type        = map(string)
  default     = {}
}

# =============================================================================
# Artifact Registry Repository
# =============================================================================

resource "google_artifact_registry_repository" "this" {
  provider = google-beta

  repository_id = var.name
  location      = var.location
  description   = var.description
  format        = "DOCKER"

  labels = var.labels
}

# =============================================================================
# Outputs
# =============================================================================

output "repository_id" {
  description = "Artifact Registry repository ID"
  value       = google_artifact_registry_repository.this.repository_id
}

output "repository_url" {
  description = "Artifact Registry URL"
  value       = google_artifact_registry_repository.this.repository_id
}

output "full_url" {
  description = "Full repository URL"
  value       = "${var.location}-docker.pkg.dev/${var.project_id}/${var.name}"
}