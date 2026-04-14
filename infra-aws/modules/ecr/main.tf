# =============================================================================
# ECR Module - Elastic Container Registry
# =============================================================================
# Creates a Docker container registry for storing application images.
# =============================================================================

variable "name" {
  description = "Repository name"
  type        = string
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
}

variable "project" {
  description = "Project name"
  type        = string
  default     = "weather-health"
}

variable "enable_scan_on_push" {
  description = "Enable image scanning on push"
  type        = bool
  default     = true
}

variable "enable_immutable_tags" {
  description = "Make image tags immutable"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# =============================================================================
# ECR Repository - create if not exists (idempotent)
# =============================================================================

resource "aws_ecr_repository" "this" {
  name                 = var.name
  image_tag_mutability = var.enable_immutable_tags ? "IMMUTABLE" : "MUTABLE"

  image_scanning_configuration {
    scan_on_push = var.enable_scan_on_push
  }

  tags = merge(
    var.tags,
    {
      Name        = var.name
      Environment = var.environment
    }
  )
}

# =============================================================================
# Lifecycle Policy - Keep only recent images
# =============================================================================

resource "aws_ecr_lifecycle_policy" "this" {
  repository = aws_ecr_repository.this.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        action = {
          type = "expire"
        }
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
      }
    ]
  })
}

# =============================================================================
# Outputs
# =============================================================================

output "repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.this.repository_url
}

output "repository_arn" {
  description = "ECR repository ARN"
  value       = aws_ecr_repository.this.arn
}