# =============================================================================
# S3 Artifacts Module - Model & Data Storage
# =============================================================================
# Creates S3 bucket for storing ML artifacts (models, data, etc.)
# =============================================================================

variable "name" {
  description = "Bucket name"
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

variable "enable_versioning" {
  description = "Enable versioning"
  type        = bool
  default     = true
}

variable "enable_encryption" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "lifecycle_rules" {
  description = "Lifecycle rules for object expiration"
  type        = list(map(any))
  default     = []
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# =============================================================================
# S3 Bucket
# =============================================================================

resource "aws_s3_bucket" "this" {
  bucket = var.name

  tags = merge(var.tags, {
    Name        = var.name
    Environment = var.environment
  })
}

# =============================================================================
# Bucket Versioning
# =============================================================================

resource "aws_s3_bucket_versioning" "this" {
  bucket = aws_s3_bucket.this.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

# =============================================================================
# Server-Side Encryption
# =============================================================================

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.this.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# =============================================================================
# Block Public Access
# =============================================================================

resource "aws_s3_bucket_public_access_block" "this" {
  bucket = aws_s3_bucket.this.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# =============================================================================
# Lifecycle Rules
# =============================================================================

# =============================================================================
# Lifecycle Rules - Disabled (causes provider issues)
# =============================================================================

# resource "aws_s3_bucket_lifecycle_configuration" "this" {
#   count  = length(var.lifecycle_rules) > 0 ? 1 : 0
#   bucket = aws_s3_bucket.this.id
#
#   dynamic "rule" {
#     for_each = var.lifecycle_rules
#     content {
#       id     = rule.value["id"]
#       status = "Enabled"
#
#       expiration {
#         days = rule.value["expiration_days"]
#       }
#     }
#   }
# }

# =============================================================================
# Outputs
# =============================================================================

output "bucket_name" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.this.id
}

output "bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.this.arn
}