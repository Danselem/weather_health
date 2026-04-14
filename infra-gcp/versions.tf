# =============================================================================
# Terraform and Provider Versions
# =============================================================================

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# Variable Definitions
# =============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "weather-health"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "weather-health"
}

variable "container_image" {
  description = "Container image URL"
  type        = string
  default     = ""
}

variable "container_port" {
  description = "Container port"
  type        = number
  default     = 8080
}

variable "min_instances" {
  description = "Minimum Cloud Run instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum Cloud Run instances"
  type        = number
  default     = 10
}

variable "concurrency" {
  description = "Max concurrent requests per instance"
  type        = number
  default     = 80
}

variable "cpu" {
  description = "CPU allocation (1, 2, or 4)"
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Memory allocation (512Mi, 1Gi, 2Gi, etc.)"
  type        = string
  default     = "1Gi"
}

variable "service_account_email" {
  description = "Service account email for Cloud Run"
  type        = string
  default     = ""
}

# =============================================================================
# Locals
# =============================================================================

locals {
  name_prefix = "${var.app_name}-${var.environment}"
  common_labels = {
    project     = var.app_name
    environment = var.environment
    managed_by  = "terraform"
  }
}