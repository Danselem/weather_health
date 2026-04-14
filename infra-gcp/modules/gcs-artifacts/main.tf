# =============================================================================
# GCS Artifacts Module - Google Cloud Storage
# =============================================================================
# Creates a GCS bucket for storing ML artifacts (models, data, etc.)
# =============================================================================

variable "name" {
  description = "Bucket name"
  type        = string
}

variable "location" {
  description = "GCS bucket location"
  type        = string
  default     = "US"
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "labels" {
  description = "Labels to apply"
  type        = map(string)
  default     = {}
}

variable "lifecycle_rules" {
  description = "Lifecycle rules for object expiration"
  type        = list(map(any))
  default     = []
}

variable "versioning" {
  description = "Enable object versioning"
  type        = bool
  default     = true
}

# =============================================================================
# GCS Bucket
# =============================================================================

resource "google_storage_bucket" "this" {
  name     = var.name
  location = var.location
  project  = var.project_id

  uniform_bucket_level_access = true

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30 # Delete objects older than 30 days
    }
  }

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_rules
    content {
      action {
        type = lifecycle_rule.value["action_type"]
      }
      condition {
        age = lifecycle_rule.value["age"]
      }
    }
  }

  labels = var.labels
}

# =============================================================================
# Bucket IAM - Storage Object Viewer for allUsers (read-only)
# =============================================================================

resource "google_storage_bucket_iam_member" "viewer" {
  bucket = google_storage_bucket.this.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# =============================================================================
# Outputs
# =============================================================================

output "bucket_name" {
  description = "GCS bucket name"
  value       = google_storage_bucket.this.name
}

output "bucket_url" {
  description = "GCS bucket URL"
  value       = "gs://${google_storage_bucket.this.name}"
}

output "bucket_self_link" {
  description = "GCS bucket self link"
  value       = google_storage_bucket.this.self_link
}