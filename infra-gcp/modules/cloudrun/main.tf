# =============================================================================
# Cloud Run Module - Serverless Containers on GCP
# =============================================================================
# Deploys FastAPI application on Google Cloud Run.
# =============================================================================

variable "name" {
  description = "Service name"
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

variable "image" {
  description = "Container image URL"
  type        = string
}

variable "port" {
  description = "Container port"
  type        = number
  default     = 8080
}

variable "cpu" {
  description = "CPU allocation (1, 2, or 4)"
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Memory allocation"
  type        = string
  default     = "1Gi"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "concurrency" {
  description = "Max concurrent requests per instance"
  type        = number
  default     = 80
}

variable "service_account_email" {
  description = "Service account to run as"
  type        = string
  default     = ""
}

variable "vpc_connector" {
  description = "VPC connector for private connectivity"
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to apply"
  type        = map(string)
  default     = {}
}

variable "environment_variables" {
  description = "Environment variables"
  type        = map(string)
  default     = {}
}

# =============================================================================
# Cloud Run Service
# =============================================================================

resource "google_cloud_run_service" "this" {
  name     = var.name
  location = var.location
  project  = var.project_id

  template {
    spec {
      containers {
        image = var.image
        ports {
          container_port = var.port
        }
        resources {
          limits = {
            cpu    = var.cpu
            memory = var.memory
          }
        }
        dynamic "env" {
          for_each = var.environment_variables
          content {
            name  = env.key
            value = env.value
          }
        }
      }

      service_account_name = var.service_account_email != "" ? var.service_account_email : null
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = var.min_instances
        "autoscaling.knative.dev/maxScale" = var.max_instances
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image,
    ]
  }

  metadata {
    labels = var.labels
  }
}

# =============================================================================
# IAM - Allow public access to Cloud Run
# =============================================================================

resource "google_cloud_run_service_iam_member" "this" {
  service  = google_cloud_run_service.this.name
  location = google_cloud_run_service.this.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# =============================================================================
# Outputs
# =============================================================================

output "service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_service.this.name
}

output "service_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_service.this.status[0].url
}

output "service_arn" {
  description = "Cloud Run service ARN"
  value       = google_cloud_run_service.this.arn
}