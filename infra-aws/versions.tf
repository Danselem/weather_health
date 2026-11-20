# =============================================================================
# Terraform and Provider Versions
# =============================================================================
# Specifies required versions and providers for this infrastructure.
# =============================================================================

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
  }
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "aws" {
  region = var.aws_region

  # Use AWS_PROFILE from environment or default
  profile = var.aws_profile != "" ? var.aws_profile : null

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# =============================================================================
# Variable Definitions
# =============================================================================

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS profile to use (from ~/.aws/config)"
  type        = string
  default     = ""
}

variable "project" {
  description = "Project name"
  type        = string
  default     = "weather-health"
}

variable "environment" {
  description = "Environment name (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "weather-health"
}

variable "container_image" {
  description = "ECR image URI for the container"
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  description = "Optional monthly AWS budget in USD; 0 disables the budget"
  type        = number
  default     = 0
}

variable "budget_email" {
  description = "Email for budget alerts; required when monthly_budget_usd > 0"
  type        = string
  default     = ""
}

variable "vpc_id" {
  description = "VPC ID (uses default VPC if not provided)"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
  default     = []
}

# =============================================================================
# Local Values
# =============================================================================

locals {
  name_prefix = "${var.project}-${var.environment}"
  common_tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}