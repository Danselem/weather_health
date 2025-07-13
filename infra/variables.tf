variable "aws_region" {
  default = "us-east-1"
}

variable "aws_profile" {
  default = "default"
}

variable "ec2_key_name" {
  description = "EC2 key pair name"
  type        = string
}

variable "db_password" {
  description = "Password for the RDS PostgreSQL instance"
  type        = string
  sensitive   = true
}

variable "artifact_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  type        = string
}

variable "project" {
  description = "Project name or identifier"
  type        = string
}

variable "environment" {
  type        = string
  description = "Deployment environment (dev/prod)"
  default     = "dev"
}