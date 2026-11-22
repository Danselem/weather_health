# =============================================================================
# Cloud SQL Postgres for prediction logging
# =============================================================================

variable "name" {
  description = "Cloud SQL instance name"
  type        = string
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Cloud SQL region"
  type        = string
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "weatherhealth"
}

variable "tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-f1-micro"
}

variable "root_password" {
  description = "Postgres root password"
  type        = string
  sensitive   = true
}

variable "deletion_protection" {
  description = "Enable Cloud SQL deletion protection"
  type        = bool
  default     = true
}

variable "backups_enabled" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

resource "google_sql_database_instance" "this" {
  name                = var.name
  project             = var.project_id
  region              = var.region
  database_version    = "POSTGRES_16"
  deletion_protection = var.deletion_protection

  lifecycle {
    ignore_changes = [root_password]
  }

  settings {
    tier                        = var.tier
    edition                     = "ENTERPRISE"
    availability_type           = "ZONAL"
    deletion_protection_enabled = var.deletion_protection
    disk_size                   = 10
    disk_type                   = "PD_SSD"

    backup_configuration {
      enabled = var.backups_enabled
    }

    ip_configuration {
      ipv4_enabled = true
    }
  }
}

resource "google_sql_database" "this" {
  name     = var.database_name
  instance = google_sql_database_instance.this.name
  project  = var.project_id
}

output "connection_name" {
  description = "Cloud SQL connection name used by the Cloud Run connector"
  value       = google_sql_database_instance.this.connection_name
}

output "public_ip_address" {
  description = "Public IP address of the Cloud SQL instance"
  value       = google_sql_database_instance.this.public_ip_address
}
